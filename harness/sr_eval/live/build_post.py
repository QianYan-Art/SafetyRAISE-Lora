"""第2轮离线后训练构建入口；真实教师调用由run_revisions单独驱动。"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

from app.report_harness.contracts import canonical_digest

from .revise import (
    ROOT, call_hash, evaluate_response, load_traces, revise_call,
    revision_instruction, workspace_path,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value) -> None:
    workspace_path(path, output=True).write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _lines(path: Path, records: list[dict]) -> None:
    workspace_path(path, output=True).write_text(
        "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                for record in records), encoding="utf-8")


def _read_revisions(directory) -> tuple[dict, list[dict]]:
    if directory is None:
        return {}, []
    directory = workspace_path(directory)
    if not directory.is_dir():
        raise ValueError("修订结果目录不存在。")
    records, errors = {}, []
    for path in sorted(directory.rglob("*.json")):
        workspace_path(path)
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if path.name == "summary.json":
                continue
            digest = record["source_call_sha256"]
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError("源调用摘要格式不合法。")
            if digest in records:
                raise ValueError("重复源调用修订结果。")
            records[digest] = record
        except (OSError, UnicodeError, ValueError, KeyError, TypeError) as exc:
            errors.append({"path": str(path.relative_to(ROOT)), "reason": type(exc).__name__})
    return records, errors


def _checked_revision(record: dict, trace: dict, call: dict) -> tuple[dict | None, str | None]:
    if (record.get("source_call_sha256") != call_hash(trace, call)
            or record.get("trace_sha256") != canonical_digest(trace)
            or record.get("payload") != call["payload"]
            or record.get("reasoning_content") != call.get("reasoning_content", "")
            or record.get("synthetic") is not True):
        return None, "revision_binding_invalid"
    if record.get("training_eligible") is not True:
        return None, record.get("discard_reason", "revision_not_eligible")
    content = record.get("chosen_content")
    if not isinstance(content, str):
        return None, "revision_content_missing"
    if record.get("derivation") == "student_original":
        if content != call.get("content", ""):
            return None, "original_content_changed"
    elif record.get("derivation") == "minimax_revision":
        original_validation = evaluate_response(trace, call)
        if original_validation["training_eligible"]:
            return None, "revision_without_student_failure"
        teachers = record.get("teacher_calls", [])
        if not teachers:
            return None, "teacher_proof_missing"
        validation = original_validation
        current = call.get("content", "")
        revision_index = 0
        for teacher in teachers:
            if "response" not in teacher:
                continue
            if (canonical_digest(teacher["response"]) != teacher.get("response_sha256")
                    or canonical_digest(teacher.get("payload")) != teacher.get("request_sha256")):
                return None, "teacher_digest_invalid"
            if record.get("teacher_backend") == "MiniMaxBackend" and (
                not teacher.get("ledger_id")
                or teacher["response"].get("live_ledger", {}).get("call_id") != teacher["ledger_id"]
            ):
                return None, "teacher_ledger_binding_invalid"
            expected_messages = deepcopy(call["payload"]["messages"]) + [
                {"role": "assistant", "content": call.get("content", ""),
                 "reasoning_content": call.get("reasoning_content", "")},
                {"role": "user", "content": revision_instruction(validation)
                 + ("\n上次修订仍不合格：" + current if revision_index else "")},
            ]
            if teacher["payload"].get("messages") != expected_messages or revision_index >= 2:
                return None, "teacher_messages_binding_invalid"
            revision_index += 1
            try:
                choice = teacher["response"]["choices"][0]
                current = choice["message"]["content"]
                if not isinstance(current, str) or choice.get("finish_reason") == "length":
                    raise ValueError("教师响应缺失或截断。")
            except (KeyError, IndexError, TypeError, ValueError):
                current = current if isinstance(current, str) else call.get("content", "")
                validation = {**validation, "training_eligible": False,
                              "fail_signals": ["teacher_invalid_or_truncated_response"]}
            else:
                validation = evaluate_response(trace, call, current)
        last = teachers[-1]
        try:
            choice = last["response"]["choices"][0]
            if choice["message"]["content"] != content or choice.get("finish_reason") == "length":
                return None, "teacher_content_binding_invalid"
            payload = last["payload"]
            if (payload["messages"][:3] != call["payload"]["messages"]
                    or payload["response_format"] != {"type": "json_object"}
                    or payload["reasoning"]["effort"] != "max"):
                return None, "teacher_request_binding_invalid"
        except (KeyError, IndexError, TypeError):
            return None, "teacher_proof_invalid"
        if record.get("rejected_content") != call.get("content", ""):
            return None, "rejected_student_binding_invalid"
    else:
        return None, "unknown_derivation"
    validation = evaluate_response(trace, call, content)
    if not validation["training_eligible"]:
        return None, "revision_revalidation_failed"
    return {**record, "validation": validation}, None


def _select_unique(candidates: list[dict], *, anchors: int, pairs: int, seeded_cap: int,
                   original_anchor_cap: int, min_long: int, long_tokens: int, seed: int) -> tuple[list, Counter]:
    """时间受限的选择策略:每个学生调用至多一行(同一调用不再同时出锚点+偏好对+种子对),
    偏好对优先取"学生真实失败→修订通过"(pair_revised),种子对限额,锚点优先取修订版、学生原样锚点限额;
    先保证至少 min_long 行是 >long_tokens 的长行(28000 窗口下会被丢掉的那部分)。"""
    def key(c):
        return hashlib.sha256((c["record"]["source_call_sha256"] + str(seed) + c["row"]["pair_id"]).encode()).hexdigest()

    def is_pair(c):
        return not c["row"]["kind"].startswith("anchor_")

    def is_seeded(c):
        return c["row"]["kind"].startswith("pair_seeded_")

    def is_original(c):
        return c["record"].get("derivation") == "student_original"

    ordered = sorted(candidates, key=key)
    selected, drops, used = [], Counter(), set()
    n = Counter()

    def can_take(c):
        call = c["record"]["source_call_sha256"]
        if call in used:
            return False
        if is_pair(c):
            return n["pair"] < pairs and (not is_seeded(c) or n["seeded"] < seeded_cap)
        return n["anchor"] < anchors and (not is_original(c) or n["original"] < original_anchor_cap)

    def take(c):
        used.add(c["record"]["source_call_sha256"])
        selected.append(c)
        n["pair" if is_pair(c) else "anchor"] += 1
        n["seeded"] += is_seeded(c)
        n["original"] += (not is_pair(c)) and is_original(c)
        n["long"] += c["tokens"] > long_tokens

    def passes(pool):
        # 偏好对(修订)→ 修订锚点 → 种子对 → 学生原样锚点
        for pred in (lambda c: is_pair(c) and not is_seeded(c),
                     lambda c: (not is_pair(c)) and not is_original(c),
                     is_seeded,
                     lambda c: (not is_pair(c)) and is_original(c)):
            for c in pool:
                if pred(c) and can_take(c):
                    yield c

    longs = [c for c in ordered if c["tokens"] > long_tokens]
    for c in passes(longs):
        if n["long"] >= min_long:
            break
        take(c)
    for c in passes(ordered):
        take(c)
    chosen = {id(c) for c in selected}
    for c in candidates:
        if id(c) not in chosen:
            drops["unique_policy:" + c["row"]["kind"]] += 1
    return selected, drops


def _select(candidates: list[dict], *, anchors: int, pairs: int, quotas: dict) -> tuple[list, Counter]:
    """按类别轮转取样，避免先遇到的单案/类别吃满配额。"""
    groups = defaultdict(list)
    for candidate in candidates:
        groups[candidate["row"]["kind"]].append(candidate)
    selected, drops = [], Counter()
    counts, families = Counter(), Counter()
    while groups:
        for kind in sorted(list(groups)):
            item = groups[kind].pop(0)
            family = "anchor" if kind.startswith("anchor_") else "pair"
            cap = anchors if family == "anchor" else pairs
            if counts[kind] >= quotas.get(kind, cap) or families[family] >= cap:
                drops["quota:" + kind] += 1
            else:
                selected.append(item)
                counts[kind] += 1
                families[family] += 1
            if not groups[kind]:
                del groups[kind]
    return selected, drops


def _percentiles(values: list[int]) -> dict:
    if not values:
        return {"count": 0, "p50": None, "p90": None, "max": None}
    values = sorted(values)
    return {"count": len(values), "p50": values[math.ceil(len(values) * .5) - 1],
            "p90": values[math.ceil(len(values) * .9) - 1], "max": values[-1],
            "method": "nearest_rank"}


def _audit_status(path, audit: dict, rows_sha256: str) -> dict:
    pending = {"status": "pending", "batch_training_eligible": False, "rollback": False}
    if path is None:
        return pending
    path = workspace_path(path)
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["rows_sha256"] != rows_sha256:
            return {**pending, "reason": "audit_rows_digest_mismatch"}
        sampled = set(audit["sampled_pair_ids"])
        answers = {}
        if audit.get("answers_path"):
            answers = json.loads(workspace_path(audit["answers_path"]).read_text(encoding="utf-8"))
        by_id = {entry["pair_id"]: entry for entry in answers.values()}
        reviews = report["reviews"]
        if not isinstance(reviews, list):
            raise ValueError("抽检结果必须是数组。")
        seen = set()
        for review in reviews:
            ident = review.get("pair_id") or answers[review["pair"]]["pair_id"]
            if ident in seen or ident not in sampled:
                raise ValueError("抽检结果ID重复或不在样本中。")
            seen.add(ident)
            if (review.get("reviewer_kind") not in {"codex", "kimi", "human"}
                    or not isinstance(review.get("reviewer_id"), str)
                    or not review["reviewer_id"].strip()
                    or review.get("offline") is not True or review.get("blind") is not True
                    or type(review.get("passed")) is not bool):
                raise ValueError("缺少独立离线盲评证明。")
            if review["passed"] is False:
                return {"status": "failed", "batch_training_eligible": False, "rollback": True,
                        "failed_pair_id": ident}
            mapping = by_id.get(ident, {})
            if str(mapping.get("kind", "")).startswith("pair_"):
                winner = review.get("winner")
                if winner not in {"A", "B", "tie"}:
                    raise ValueError("偏好盲评须给出A/B/tie结论。")
                if winner == "tie" or mapping.get(winner) != "chosen":
                    return {"status": "failed", "batch_training_eligible": False, "rollback": True,
                            "failed_pair_id": ident, "reason": "preference_not_confirmed"}
        if seen != sampled or not sampled:
            return {**pending, "reason": "audit_incomplete"}
        if audit.get("shortfall", 0):
            return {**pending, "reason": "audit_minimum_not_met"}
        return {"status": "passed", "batch_training_eligible": True, "rollback": False}
    except (OSError, UnicodeError, ValueError, TypeError, KeyError):
        return {**pending, "reason": "audit_result_invalid"}


def build_post(traces_dir, out_dir, *, revise_results=None, dry_run=False,
               template=None, max_len=28000, anchor_quota=70, pair_quota=30,
               quotas=None, seed=7, seed_errors_enabled=True, audit_results=None,
               unique_calls=False, seeded_cap=4, original_anchor_cap=6, min_long=0, long_tokens=28000,
               audit_fraction=.2, anchor_allowlist=None, exclude_calls=None) -> dict:
    """产出混合训练行与盲评包；离线构建绝不发起教师调用。"""
    from .audit import build_audit
    from .rows import build_row
    from .seeded import ERROR_CLASSES, seed_errors

    if (type(max_len) is not int or not 0 < max_len <= 32768
            or type(anchor_quota) is not int or anchor_quota < 0
            or type(pair_quota) is not int or pair_quota < 0
            or any(type(value) is not int or value < 0 for value in (quotas or {}).values())):
        raise ValueError("窗口至多32768，配额须为非负整数。")
    output = workspace_path(out_dir, output=True)
    if output.exists() and any(output.iterdir()):
        raise ValueError("输出目录必须为空，避免覆盖既有交付。")
    traces, load_errors = load_traces(traces_dir)
    revisions, revision_errors = _read_revisions(revise_results)
    candidates, drops, sources, unknowns = [], Counter(), [], Counter()
    measured_totals = []
    seen_calls = set()
    used_revisions = set()
    discarded = []
    for path, trace in traces:
        sources.append({"path": str(path.relative_to(ROOT)), "sha256": _sha(path),
                        "trace_digest": canonical_digest(trace), "case_id": trace["case_id"],
                        "split": trace["case"].get("split"),
                        "retrieval_mode": trace.get("retrieval", {}).get("mode")})
        if trace["case"].get("split") != "train" and not dry_run:
            drops["split_not_train"] += 1
            continue
        for index, call in enumerate(trace["calls"]):
            if call.get("role") != "generator":
                continue
            if call.get("not_sent"):
                drops["student_not_sent"] += 1
                continue
            digest = call_hash(trace, call)
            if digest in seen_calls:
                drops["duplicate_student_call"] += 1
                continue
            seen_calls.add(digest)
            if digest in revisions:
                used_revisions.add(digest)
                record, reason = _checked_revision(revisions[digest], trace, call)
            else:
                record = asyncio.run(revise_call(trace, call))
                reason = record.get("discard_reason") if not record["training_eligible"] else None
            if reason:
                drops[reason] += 1
                discarded.append({"source_call_sha256": digest, "reason": reason})
                continue
            record = {**record, "call_index": index, "split": trace["case"].get("split"),
                      "trace_path": str(path.relative_to(ROOT)), "trace_sha256": canonical_digest(trace)}
            for name, count in record["validation"].get("unknown_metrics", {}).items():
                unknowns[name] += max(1, count)
            variants = []
            anchor = deepcopy(record)
            anchor.pop("rejected_content", None)
            variants.append(anchor)
            if record.get("rejected_content"):
                variants.append(record)
            if seed_errors_enabled:
                offset = (int(digest[:8], 16) + seed) % len(ERROR_CLASSES)
                error_order = ERROR_CLASSES[offset:] + ERROR_CLASSES[:offset]
                seeds = seed_errors(trace, call, record["chosen_content"],
                                    error_classes=error_order, template=template, max_pairs=4)
                for item in seeds:
                    variants.append({**record, **item, "derivation": "seeded",
                                     "training_eligible": True,
                                     "seed_validation": item.get("validation", {}),
                                     "validation": record["validation"]})
                if not seeds:
                    drops["seed_no_isolated_error"] += 1
            for variant in variants:
                result = build_row(variant, template=template, max_len=max_len)
                if isinstance(result.get("tokens"), int):
                    measured_totals.append(result["tokens"])
                if result["row"] is None:
                    drops[result.get("reason") or "row_rejected"] += 1
                    continue
                candidates.append({**result, "record": variant})
    drops.update("trace_load:" + error["reason"] for error in load_errors)
    drops.update("revision_load:" + error["reason"] for error in revision_errors)
    drops["unused_revision"] += len(set(revisions) - used_revisions)
    if exclude_calls:  # 第二轮续训:排除前一轮已用过的源调用
        blocked = set(exclude_calls)
        before = len(candidates)
        candidates = [c for c in candidates if c["record"]["source_call_sha256"] not in blocked]
        drops["excluded_used_call"] += before - len(candidates)
    if anchor_allowlist is not None:  # 锚点只取经全量严审通过的调用;偏好对不受限(方向由抽检确认)
        allowed = set(anchor_allowlist)
        kept = [c for c in candidates if not c["row"]["kind"].startswith("anchor_")
                or c["record"]["source_call_sha256"] in allowed]
        drops["anchor_not_in_allowlist"] += len(candidates) - len(kept)
        candidates = kept
    if unique_calls:
        selected, quota_drops = _select_unique(
            candidates, anchors=anchor_quota, pairs=pair_quota, seeded_cap=seeded_cap,
            original_anchor_cap=original_anchor_cap, min_long=min_long, long_tokens=long_tokens, seed=seed)
    else:
        selected, quota_drops = _select(candidates, anchors=anchor_quota, pairs=pair_quota, quotas=quotas or {})
    drops.update(quota_drops)
    output.mkdir(parents=True, exist_ok=True)
    rows = [item["row"] for item in selected]
    records_by_id = {item["row"]["pair_id"]: {**item["record"], **item["metadata"]}
                     for item in selected}
    _lines(output / "rows.jsonl", rows)
    row_sha = _sha(output / "rows.jsonl")
    audit = build_audit(rows, records_by_id, output / "audit", fraction=audit_fraction, minimum=15, seed=seed)
    status = _audit_status(audit_results, audit, row_sha)
    if status["rollback"]:
        _lines(output / "rows.isolated.jsonl", rows)
        _lines(output / "rows.jsonl", [])
    provenance = [{
        "pair_id": item["row"]["pair_id"], "kind": item["row"]["kind"],
        "case_id": item["record"]["case_id"], "source_call_sha256": item["record"]["source_call_sha256"],
        "trace_sha256": item["record"]["trace_sha256"],
        "derivation": item["record"]["derivation"], "synthetic": True,
        "teacher_backend": item["record"].get("teacher_backend"),
        "teacher_protocol_fixture": item["record"].get("teacher_protocol_fixture", False),
        "teacher_calls": [{"request_sha256": teacher.get("request_sha256"),
                           "response_sha256": teacher.get("response_sha256"),
                           "ledger_id": teacher.get("ledger_id")}
                          for teacher in item["record"].get("teacher_calls", [])],
        "unknown_metrics": item["record"]["validation"].get("unknown_metrics", {}),
        "seed_evidence": item["record"].get("seed_validation") if item["record"]["derivation"] == "seeded" else None,
        "diff": item["record"].get("diff"),
    } for item in selected]
    _lines(output / "provenance.jsonl", provenance)
    hashes = sorted({(item["metadata"].get("tokenizer_sha256"),
                      item["metadata"].get("template_sha256")) for item in selected}, key=str)
    kinds = Counter(row["kind"] for row in rows)
    family = {"anchors": sum(value for kind, value in kinds.items() if kind.startswith("anchor_")),
              "pairs": sum(value for kind, value in kinds.items() if not kind.startswith("anchor_"))}
    manifest = {
        "schema_version": 2, "source_brief": "harness/reports/codex-brief-live-env-r2.md",
        "synthetic": True, "human_verified": False, "dry_run": dry_run,
        "training_forbidden": dry_run or not status["batch_training_eligible"],
        "independent_review": "sampled", "independent_review_status": status,
        "release_eligible": False, "metric_unknown_allowed": True,
        "unknown_metrics": dict(unknowns), "reasoning_effort": "compact",
        "stable_limit": max_len, "max_len": max_len, "hard_window": 32768,
        "counts": {"selected": len(rows), "written": 0 if status["rollback"] else len(rows), **family},
        "kinds": dict(kinds), "discard_reasons": {key: value for key, value in drops.items() if value},
        "discarded_calls": discarded, "sources": sources, "load_errors": load_errors,
        "revision_errors": revision_errors,
        "quotas": {"anchors": anchor_quota, "pairs": pair_quota, "kinds": quotas or {}},
        "selection_policy": ({"unique_calls": True, "seeded_cap": seeded_cap, "original_anchor_cap": original_anchor_cap,
                              "min_long": min_long, "long_tokens": long_tokens} if unique_calls else {"unique_calls": False}),
        "long_rows_selected": sum(item["tokens"] > long_tokens for item in selected),
        "target_shortfall": {"anchors_below_60": max(0, 60 - family["anchors"]),
                             "pairs_below_25": max(0, 25 - family["pairs"])},
        "window_distribution": _percentiles([item["tokens"] for item in selected]),
        "window_candidates": _percentiles(measured_totals),
        "tokenizer_template_hashes": [
            {"tokenizer_sha256": tokenizer, "template_sha256": template_hash}
            for tokenizer, template_hash in hashes],
        "audit": audit, "pre_rollback_rows_sha256": row_sha,
        "files": {"rows.jsonl": _sha(output / "rows.jsonl"),
                  "provenance.jsonl": _sha(output / "provenance.jsonl")},
        "network_calls_by_builder": 0,
        "quality_policy": "确定性硬门全通过且解读无fail；unknown单列；独立盲评抽检通过才接受整批训练。",
    }
    _write(output / "manifest.json", manifest)
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="离线构建学生轨迹的第2轮混合后训练行。")
    parser.add_argument("--traces", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--revise-results")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--anchor-quota", type=int, default=70)
    parser.add_argument("--pair-quota", type=int, default=30)
    parser.add_argument("--quota", action="append", default=[], metavar="KIND=N")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-len", type=int, default=28000)
    parser.add_argument("--no-seeded", action="store_true")
    parser.add_argument("--unique-calls", action="store_true", help="每个学生调用至多一行(时间受限的训练集选择)")
    parser.add_argument("--seeded-cap", type=int, default=4)
    parser.add_argument("--original-anchor-cap", type=int, default=6)
    parser.add_argument("--min-long", type=int, default=0, help="至少选这么多行总长 >long-tokens(28000 窗口下会被丢弃的长行)")
    parser.add_argument("--long-tokens", type=int, default=28000)
    parser.add_argument("--audit-fraction", type=float, default=.2, help="抽检比例;1.0=全量(用于全量严审筛锚点)")
    parser.add_argument("--anchor-allowlist", help="JSON 数组:允许作为锚点的 source_call_sha256")
    parser.add_argument("--exclude-calls", help="JSON 数组:要排除的 source_call_sha256(前一轮已用)")
    parser.add_argument("--audit-results")
    args = parser.parse_args(argv)
    quotas = {}
    try:
        for item in args.quota:
            kind, count = item.split("=", 1)
            if kind not in {"anchor_final", "anchor_tool", "pair_revised"} and not kind.startswith("pair_seeded_"):
                raise ValueError("未知行类别配额。")
            quotas[kind] = int(count)
        manifest = build_post(
            args.traces, args.out, revise_results=args.revise_results, dry_run=args.dry_run,
            max_len=args.max_len, anchor_quota=args.anchor_quota, pair_quota=args.pair_quota,
            quotas=quotas, seed=args.seed, seed_errors_enabled=not args.no_seeded,
            audit_results=args.audit_results, unique_calls=args.unique_calls, seeded_cap=args.seeded_cap,
            original_anchor_cap=args.original_anchor_cap, min_long=args.min_long, long_tokens=args.long_tokens,
            audit_fraction=args.audit_fraction,
            anchor_allowlist=(json.loads(Path(args.anchor_allowlist).read_text(encoding="utf-8")) if args.anchor_allowlist else None),
            exclude_calls=(json.loads(Path(args.exclude_calls).read_text(encoding="utf-8")) if args.exclude_calls else None),
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"out": str(workspace_path(args.out)), "counts": manifest["counts"],
                      "discard_reasons": manifest["discard_reasons"],
                      "independent_review_status": manifest["independent_review_status"]["status"],
                      "training_forbidden": manifest["training_forbidden"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
