"""live trace 到后训练偏好数据的保守构建器。

构建器只接受同一案件、同一完整 prompt/context 指纹的候选，避免把不同
上下文的候选错误拼成偏好对。没有独立复核的候选默认只能进入 dry-run
隔离区，不会进入正式可训练的 ``pairs_env.jsonl``。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from app.report_harness.contracts import (
    candidate_contract_problems, candidate_round_problems, enforce_semantic_severity,
    enforce_source_authority, validate_publication,
)
from app.report_harness.errors import HarnessError
from app.report_harness.controlled_tools import ControlledTools
from app.report_harness.quoted_roles import resolve_quotes
from app.report_harness.role_loop import role_context
from app.report_harness.source_access import generator_access_problems
from app.schemas.report_run import CandidateReport, ReviewResult
from pydantic import ValidationError

from .metrics import FIELD_NAMES, evaluate_candidate
from .samples import normalize_call, render_call_sample


MAX_LEN_DEFAULT = 32768
STABLE_KINDS = {"quality", "gate", "terminal", "protocol", "seed_error"}
ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = (ROOT / "harness" / "reports" / "live-dataset").resolve()


def _json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump(value: Any) -> Any:
    if isinstance(value, dict):
        return value
    method = getattr(value, "model_dump", None)
    if callable(method):
        return method(mode="json")
    return value


def _json_from_text(text: Any) -> dict[str, Any] | None:
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        return None
    candidate = text.strip()
    # 线上 json_object 需要完整 JSON；不抓取代码块或外层文本中的内层对象。
    if not candidate or candidate.startswith("```") or candidate.endswith("```"):
        return None
    try:
        obj = json.loads(candidate)
    except (TypeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def _case(trace: dict[str, Any]) -> dict[str, Any]:
    value = trace.get("case") or trace.get("input_case") or trace.get("snapshot") or {}
    value = _dump(value)
    return value if isinstance(value, dict) else {}


def _candidate_from_trace(trace: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("candidate", "final_candidate", "published_candidate"):
        candidate = _json_from_text(trace.get(key))
        if candidate and ("report_markdown" in candidate or "claims" in candidate):
            return candidate
    rounds = trace.get("rounds") or []
    if isinstance(rounds, list):
        for item in reversed(rounds):
            item = _dump(item) or {}
            if not isinstance(item, dict):
                continue
            candidate = _json_from_text(item.get("candidate"))
            if candidate and ("report_markdown" in candidate or "claims" in candidate):
                return candidate
    return None


def _candidate_call(trace: dict[str, Any]) -> tuple[dict[str, Any] | None, int | None]:
    calls = trace.get("calls") or []
    for index in range(len(calls) - 1, -1, -1):
        call = _dump(calls[index]) or {}
        if not isinstance(call, dict):
            continue
        role = str(call.get("role", "")).lower()
        if role not in {"generator", "reviser"}:
            continue
        # 最后一次生成调用才是配对目标；不能越过尾部失败响应复用旧候选。
        return call, index
    return None, None


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _independent_reviewed(trace: dict[str, Any], candidate: dict[str, Any] | None) -> bool:
    """独立复核必须绑定候选摘要、复核者身份和盲评标记。"""
    review = _dump(trace.get("independent_review") or trace.get("quality_review"))
    if not isinstance(review, dict) or review.get("passed") is not True or not isinstance(candidate, dict):
        return False
    expected = _canonical_digest(candidate)
    digest = review.get("candidate_digest") or review.get("candidate_sha256")
    reviewer = review.get("reviewer_id") or review.get("reviewer") or review.get("source")
    blind = review.get("blind") is True or review.get("blind_review") is True
    if digest != expected or not isinstance(reviewer, str) or not reviewer.strip() or not blind:
        return False
    if reviewer.strip().lower() in {"student", "generator", "same_model", "deterministic", "程序", "学生"}:
        return False
    if review.get("reviewer_kind") not in {"codex", "kimi", "human"} or review.get("offline") is not True:
        return False
    checks = review.get("checks") or {}
    return all(checks.get(name) is True for name in
               ("facts", "citations", "privacy", "leakage", "tool_authenticity", "source_authorization"))


def _prompt_signature(trace: dict[str, Any], call: dict[str, Any]) -> str:
    normalized = normalize_call(trace, call)
    raw_payload = normalized.get("payload") or {}
    payload = {
        "messages": normalized.get("messages") or [],
        "tools": normalized.get("tools"),
        "with_tools": normalized.get("with_tools"),
        "context_digest": _canonical_digest(normalized.get("messages") or []),
        "snapshot_digest": _canonical_digest(trace.get("snapshot")),
        "knowledge_manifest_digest": (trace.get("snapshot") or {}).get("knowledge_manifest_digest"),
        "role": normalized.get("role"),
        "response_format": raw_payload.get("response_format"),
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _trace_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_output_dir(output_dir: Path) -> None:
    """只允许写入本任务登记的 reports/live-dataset 子目录，并拒绝 symlink。"""
    try:
        candidate = output_dir.resolve()
    except OSError as exc:
        raise ValueError(f"输出目录无法解析: {output_dir}") from exc
    if candidate != OUTPUT_ROOT and OUTPUT_ROOT not in candidate.parents:
        raise ValueError(f"输出目录必须位于 {OUTPUT_ROOT} 下。")
    cursor = output_dir
    while True:
        if cursor.exists() and cursor.is_symlink():
            raise ValueError(f"输出目录或父目录不能是符号链接: {cursor}")
        if cursor == OUTPUT_ROOT or cursor.parent == cursor:
            break
        cursor = cursor.parent


def _published_problems(trace: dict[str, Any], call: dict[str, Any],
                        candidate: dict[str, Any] | None) -> list[str]:
    """重新执行冻结硬门；缺少控制器取证记录的旧 trace 仅能隔离。"""
    try:
        snapshot = trace["snapshot"]
        if (set(snapshot["accident_data"]) != set(FIELD_NAMES)
                or snapshot["accident_data"] != _case(trace).get("accident_data")
                or trace["snapshot_digest"] != _canonical_digest(snapshot)):
            return ["snapshot_binding_invalid"]
        draft = CandidateReport.model_validate(candidate)
        round_data = next(item for item in reversed(trace["rounds"])
                          if _canonical_digest(item.get("candidate")) == _canonical_digest(candidate))
        context = json.loads(call["payload"]["messages"][-1]["content"])
        expected_snapshot, _ = role_context(snapshot)
        if context.get("snapshot") != expected_snapshot:
            return ["generator_snapshot_context_invalid"]
        expected_context = _canonical_digest(context)
        if call.get("context_digest") not in (None, expected_context):
            return ["context_digest_invalid"]
        if call.get("request_sha256") not in (None, _canonical_digest(call["payload"])):
            return ["request_digest_invalid"]
        obligations = [item["obligation_id"] for item in snapshot["fact_obligations"]]
        generator = round_data["generator_access"]
        reviewer = round_data["reviewer_access"]
        problems = [
            *candidate_contract_problems(draft, obligations),
            *candidate_round_problems(
                draft, version=context["candidate_version"],
                open_issue_ids=[item["issue_id"] for item in context.get("unresolved_issues", [])],
            ),
            *generator_access_problems(draft, set(generator["evidence"]), set(generator["knowledge"])),
        ]
        if problems:
            return ["candidate_contract_invalid"]
        # 访问集合来自控制器，不接受模型自报；知识必须同时存在于冻结注册表。
        source = trace["knowledge_registry"]
        source_ids = {item["id"] for item in source}
        if not (set(generator["knowledge"]) | set(reviewer["knowledge"])) <= source_ids:
            return ["source_registry_invalid"]
        if generator != _verified_access(trace, context, "generator"):
            return ["generator_access_proof_invalid"]
        reviewer_context = next(
            json.loads(item["payload"]["messages"][-1]["content"])
            for item in reversed(trace["calls"])
            if item.get("role") == "reviewer"
            and _canonical_digest(json.loads(item["payload"]["messages"][-1]["content"]).get("candidate"))
            == _canonical_digest(candidate)
        )
        if (reviewer_context.get("snapshot") != expected_snapshot
                or reviewer_context.get("snapshot_digest") != trace["snapshot_digest"]
                or reviewer_context.get("candidate_digest") != _canonical_digest(candidate)):
            return ["reviewer_snapshot_context_invalid"]
        if reviewer != _verified_access(trace, reviewer_context, "reviewer"):
            return ["reviewer_access_proof_invalid"]
        review = ReviewResult.model_validate(round_data["review_enforced"])
        enforced = enforce_semantic_severity(enforce_source_authority(
            draft, review, source, trace.get("issue_history", []),
        ))
        if _canonical_digest(enforced.model_dump(mode="json")) != _canonical_digest(review.model_dump(mode="json")):
            return ["source_authority_invalid"]
        publication = trace["publication"]
        if publication != {
            "candidate_digest": _canonical_digest(candidate),
            "review_digest": _canonical_digest(review.model_dump(mode="json")),
            "snapshot_digest": trace["snapshot_digest"],
        }:
            return ["publication_binding_invalid"]
        validate_publication(
            draft, review, trace["snapshot_digest"], obligations,
            set(reviewer["evidence"]), set(reviewer["knowledge"]),
            resolved_issue_ids=frozenset(round_data["resolved_issue_ids"]),
        )
    except (KeyError, TypeError, ValueError, StopIteration, ValidationError, HarnessError):
        return ["publication_proof_invalid"]
    return []


def _verified_access(trace: dict[str, Any], context: dict[str, Any], role: str) -> dict[str, Any]:
    """从真实回合里的工具结果重放只读操作，不信任模型或访问集合自报。"""
    source_by_id = {item["id"]: item for item in trace["knowledge_registry"]}
    search_candidates: list[dict[str, Any]] = []
    tools = ControlledTools(
        trace["snapshot"], trace["knowledge_registry"], compact_registry=True,
        search=lambda query, top_k: list(search_candidates),
    )
    _, inline = role_context(trace["snapshot"])
    if role == "generator":
        tools.include_knowledge(role, context.get("initial_knowledge_snippets", []))
    for result in context.get("tool_results", []):
        name = result["name"]
        event = next(
            item for item in trace["tools"]
            if item.get("role") == role and item.get("name") == name
            and item.get("private", {}).get("call_id") == result["call_id"]
            and "result" in item.get("private", {})
        )
        private = event["private"]
        if (private["result"] != result["result"]
                or event["result_digest"] != _canonical_digest(result["result"])):
            raise ValueError("工具结果摘要不匹配。")
        if name == "search_knowledge":
            search_candidates = [source_by_id[item["id"]] for item in result["result"]["items"]]
            normalized_query = " ".join(str(private["arguments"]["query"]).split())
            matching_event = any(
                item.get("step") == "additional" and item.get("query") == normalized_query
                and item.get("approved_ids") == [source["id"] for source in search_candidates]
                and item.get("approved_projection_digest") == _canonical_digest(search_candidates)
                and 0 < item.get("top_k", 0) <= private["arguments"]["top_k"]
                for item in trace["retrieval"]["events"]
            )
            if not matching_event:
                raise ValueError("追加检索结果未绑定控制器批准的检索事件。")
        replayed = tools.execute(role, name, private["arguments"])
        if replayed != result["result"]:
            raise ValueError("工具来源或分页结果无法重放。")
    return {"evidence": sorted(inline | tools.accessed_evidence(role)),
            "knowledge": sorted(tools.accessed_knowledge(role))}


def _candidate_record(path: Path, trace: dict[str, Any], template: Any, reasoning_effort: str) -> dict[str, Any] | None:
    call, index = _candidate_call(trace)
    if call is None:
        return None
    sample = render_call_sample(trace, call, call_index=index or 0, template=template, reasoning_effort=reasoning_effort)
    candidate = _candidate_from_trace(trace)
    wire_candidate = _json_from_text(normalize_call(trace, call).get("content"))
    wire_problems: list[str] = []
    if wire_candidate is None or not ("report_markdown" in wire_candidate or "claims" in wire_candidate):
        wire_candidate = None
        candidate = None
        wire_problems.append("no_candidate")
    if wire_candidate is not None:
        try:
            wire_candidate = CandidateReport.model_validate(resolve_quotes(wire_candidate)).model_dump(mode="json")
        except (ValidationError, HarnessError, ValueError, TypeError):
            wire_problems.append("wire_contract_invalid")
    candidate_mismatch = bool(candidate and wire_candidate and _canonical_digest(candidate) != _canonical_digest(wire_candidate))
    if candidate is None:
        candidate = wire_candidate
    case = _case(trace)
    case_kind = str(case.get("kind") or trace.get("case_kind") or "")
    provenance = _dump(trace.get("provenance")) or {}
    provenance_known = (provenance.get("synthetic") is True
                        and isinstance(provenance.get("source"), str)
                        and bool(provenance.get("source").strip()))
    metrics = evaluate_candidate(candidate or {}, case) if candidate else {"overall": "unknown", "strict_pass": False, "checks": {}}
    status = str(trace.get("status") or trace.get("review_status") or "failed")
    response = _dump(call.get("response")) or {}
    choices = response.get("choices") if isinstance(response, dict) else None
    choice_finish = choices[0].get("finish_reason") if isinstance(choices, list) and choices and isinstance(choices[0], dict) else None
    finish_reason = str(call.get("finish_reason") or choice_finish or "")
    reasons: list[str] = list(wire_problems)
    case_id = trace.get("case_id")
    if not isinstance(case_id, str) or not case_id.strip():
        reasons.append("case_id_missing")
    if status not in {"published", "ok", "passed"}:
        reasons.append(status or "not_published")
    if finish_reason == "length":
        reasons.append("truncated")
    if not candidate:
        reasons.append("no_candidate")
    if candidate_mismatch:
        reasons.append("candidate_mismatch")
    if case_kind != "synthetic":
        reasons.append("non_synthetic_case")
    if not provenance_known:
        reasons.append("provenance_unknown")
    if metrics.get("overall") == "fail":
        reasons.append("metric_fail")
    elif metrics.get("overall") == "unknown" and status in {"published", "ok", "passed"}:
        reasons.append("metric_unknown")
    if status in {"published", "ok", "passed"}:
        reasons.extend(_published_problems(trace, call, candidate))
    kind = str(trace.get("pair_kind") or trace.get("kind") or "")
    if kind not in STABLE_KINDS:
        kind = "terminal" if ("truncated" in reasons or status in {"failed", "needs_review"}) else ("gate" if "metric_fail" in reasons else "quality")
    return {
        "path": str(path), "trace": trace, "trace_sha256": _trace_hash(path), "case_id": str(trace.get("case_id") or ""),
        "call": call, "call_index": index, "sample": sample, "candidate": candidate,
        "metrics": metrics, "status": status, "reasons": reasons, "kind": kind,
        "case_kind": case_kind, "provenance_known": provenance_known,
        "prompt_signature": _prompt_signature(trace, call),
        "independent_reviewed": _independent_reviewed(trace, candidate),
    }


def _read_records(trace_dir: Path, template: Any, reasoning_effort: str) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    sources: list[str] = []
    input_errors: list[dict[str, Any]] = []
    for path in sorted(trace_dir.rglob("*.json")):
        if path.name in {"manifest.json", "conformance.json"}:
            continue
        try:
            trace = _json_load(path)
        except (OSError, ValueError) as exc:
            input_errors.append({"path": str(path), "reason": "input_unreadable",
                                 "error_type": type(exc).__name__})
            if isinstance(exc, ValueError):
                input_errors[-1]["sha256"] = _trace_hash(path)
            continue
        if not isinstance(trace, dict) or not trace.get("calls"):
            input_errors.append({"path": str(path), "reason": "not_a_trace",
                                 "sha256": _trace_hash(path)})
            continue
        sources.append(_trace_hash(path))
        try:
            record = _candidate_record(path, trace, template, reasoning_effort)
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            input_errors.append({"path": str(path), "reason": "trace_shape_invalid",
                                 "error_type": type(exc).__name__, "sha256": _trace_hash(path)})
            continue
        if record is not None:
            records.append(record)
        else:
            input_errors.append({"path": str(path), "reason": "no_candidate_call",
                                 "sha256": _trace_hash(path)})
    return records, sources, input_errors


def _pair_id(chosen: dict[str, Any], rejected: dict[str, Any], kind: str) -> str:
    raw = "|".join((chosen["case_id"], chosen["prompt_signature"], rejected["prompt_signature"], kind,
                     chosen.get("trace_sha256", ""), rejected.get("trace_sha256", ""),
                     str(chosen.get("call_index")), str(rejected.get("call_index"))))
    return "env-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _pair_row(chosen: dict[str, Any], rejected: dict[str, Any], kind: str) -> dict[str, Any] | None:
    cs, rs = chosen["sample"], rejected["sample"]
    if cs.get("status") != "exact" or rs.get("status") != "exact":
        return None
    if not cs.get("response_ids") or not rs.get("response_ids") or cs.get("prompt_ids") != rs.get("prompt_ids"):
        return None
    if any(not isinstance(token, int) or isinstance(token, bool) for token in cs["prompt_ids"] + cs["response_ids"] + rs["response_ids"]):
        return None
    if cs.get("reasoning_effort") != rs.get("reasoning_effort"):
        return None
    if cs.get("tokenizer", {}).get("template_sha256") != rs.get("tokenizer", {}).get("template_sha256") or cs.get("tokenizer", {}).get("tokenizer_sha256") != rs.get("tokenizer", {}).get("tokenizer_sha256"):
        return None
    if cs.get("c_start") is None or rs.get("c_start") is None:
        return None
    if cs["response_ids"] == rs["response_ids"]:
        return None
    # 偏好信号只能落在最终 JSON/报告段；思考前缀必须逐 token 相同。
    if cs["response_ids"][:cs["c_start"]] != rs["response_ids"][:rs["r_start"]]:
        return None
    return {
        "pair_id": _pair_id(chosen, rejected, kind),
        "kind": kind,
        "case_id": chosen["case_id"],
        "prompt_ids": cs["prompt_ids"],
        "chosen_ids": cs["response_ids"],
        "rejected_ids": rs["response_ids"],
        "c_start": cs["c_start"],
        "r_start": rs["c_start"],
        "chosen_source": chosen["path"],
        "rejected_source": rejected["path"],
        "chosen_status": chosen["status"],
        "rejected_status": rejected["status"],
        "chosen_metric_status": chosen["metrics"].get("overall"),
        "rejected_metric_status": rejected["metrics"].get("overall"),
    }


def build_dataset(
    trace_dir: str | Path,
    output_dir: str | Path,
    *,
    template: Any = None,
    reasoning_effort: str = "xhigh",
    max_len: int = MAX_LEN_DEFAULT,
    dry_run: bool = False,
    allow_unverified: bool = False,
    write_isolated: bool = True,
    make_anchor: bool = True,
) -> dict[str, Any]:
    """构建 pairs/anchor/manifest；返回与 manifest 相同的统计对象。"""
    trace_dir, output_dir = Path(trace_dir), Path(output_dir)
    if max_len <= 0 or max_len > MAX_LEN_DEFAULT:
        raise ValueError(f"max_len 必须在 1..{MAX_LEN_DEFAULT} 范围内。")
    if allow_unverified and not dry_run:
        raise ValueError("allow_unverified 只能用于 dry-run 隔离，不能正式放行。")
    _assert_output_dir(output_dir)
    records, source_hashes, input_errors = _read_records(trace_dir, template, reasoning_effort)
    by_case: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_case.setdefault(record["case_id"], []).append(record)
    pairs: list[dict[str, Any]] = []
    isolated: list[dict[str, Any]] = []
    stats = {"trace_files": len(source_hashes), "candidate_records": len(records), "cases": len(by_case),
             "formal_pairs": 0, "isolated_pairs": 0, "unverified_chosen": 0, "non_synthetic_rejected": 0, "provenance_unknown": 0, "token_unavailable": 0,
             "too_long": 0, "stable_over_24k": 0, "hard_dropped_over_32768": 0,
             "prompt_mismatch": 0, "thought_mismatch": 0, "no_pair": 0,
             "input_errors": len(input_errors)}
    for record in records:
        if "non_synthetic_case" in record["reasons"]:
            stats["non_synthetic_rejected"] += 1
        if "provenance_unknown" in record["reasons"]:
            stats["provenance_unknown"] += 1
    for case_id, candidates in sorted(by_case.items()):
        blocking_reasons = {"metric_fail", "candidate_mismatch", "no_candidate", "truncated",
                            "non_synthetic_case", "provenance_unknown", "case_id_missing"}
        # metric_unknown 可在 dry-run 形成隔离 pair，但永远不能正式放行。
        chosen_pool = [item for item in candidates
                       if item["status"] in {"published", "ok", "passed"}
                       and not (blocking_reasons & set(item["reasons"]))]
        rejected_pool = [item for item in candidates if item not in chosen_pool
                         and item["case_kind"] == "synthetic" and item["provenance_known"]
                         and "case_id_missing" not in item["reasons"]]
        for chosen in chosen_pool:
            if not chosen["independent_reviewed"]:
                stats["unverified_chosen"] += 1
            rejected = [item for item in rejected_pool if item["prompt_signature"] == chosen["prompt_signature"]]
            if not rejected:
                stats["no_pair"] += 1
                continue
            rejected.sort(key=lambda item: (item["kind"] != "terminal", item["path"]))
            row = _pair_row(chosen, rejected[0], rejected[0]["kind"])
            if row is None:
                if chosen["sample"].get("status") != "exact" or rejected[0]["sample"].get("status") != "exact":
                    stats["token_unavailable"] += 1
                elif chosen["sample"].get("prompt_ids") == rejected[0]["sample"].get("prompt_ids") and chosen["sample"].get("c_start") is not None and rejected[0]["sample"].get("r_start") is not None and chosen["sample"]["response_ids"][:chosen["sample"]["c_start"]] != rejected[0]["sample"]["response_ids"][:rejected[0]["sample"]["r_start"]]:
                    stats["thought_mismatch"] += 1
                else:
                    stats["prompt_mismatch"] += 1
                continue
            total_tokens = len(row["prompt_ids"]) + max(len(row["chosen_ids"]), len(row["rejected_ids"]))
            if total_tokens > 24000:
                stats["stable_over_24k"] += 1
            if total_tokens > max_len:
                stats["hard_dropped_over_32768"] += 1
                stats["too_long"] += 1
                continue
            verified = (chosen["independent_reviewed"] and chosen["metrics"].get("strict_pass") is True
                        and not chosen["reasons"])
            if not verified:
                row["isolated"] = True
                row["isolation_reason"] = "chosen 未通过全部硬门、解读指标或独立复核"
                row["isolation_reasons"] = chosen["reasons"] + (
                    [] if chosen["independent_reviewed"] else ["independent_review_missing"])
                isolated.append(row)
                continue
            pairs.append(row)
    stats["formal_pairs"] = len(pairs)
    stats["isolated_pairs"] = len(isolated)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_to_write = pairs
    pairs_path = output_dir / "pairs_env.jsonl"
    with pairs_path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows_to_write:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    isolated_path = output_dir / "pairs_env.isolated.jsonl"
    if dry_run and write_isolated:
        with isolated_path.open("w", encoding="utf-8", newline="\n") as fh:
            for row in isolated:
                fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    anchor_rows = [{"pair_id": row["pair_id"], "kind": "anchor", "case_id": row["case_id"],
                    "prompt_ids": row["prompt_ids"], "chosen_ids": row["chosen_ids"],
                    "rejected_ids": [], "c_start": row["c_start"], "r_start": 0}
                   for row in pairs] if make_anchor else []
    anchor_path = output_dir / "anchor_env.jsonl"
    if make_anchor:
        with anchor_path.open("w", encoding="utf-8", newline="\n") as fh:
            for row in anchor_rows:
                fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    manifest = {
        "schema_version": 1,
        "builder": "sr_eval.live.dataset",
        "mode": "dry-run" if dry_run else "formal",
        "release_eligible": bool(not dry_run and pairs and stats["unverified_chosen"] == 0 and stats["non_synthetic_rejected"] == 0 and stats["provenance_unknown"] == 0 and not isolated and not input_errors),
        "max_len": max_len,
        "template_effort": reasoning_effort,
        "stats": stats,
        "kind_counts": {
            "formal": dict(Counter(row["kind"] for row in pairs)),
            "isolated": dict(Counter(row["kind"] for row in isolated)),
        },
        "files": {"pairs_env.jsonl": {"sha256": _trace_hash(pairs_path), "rows": len(rows_to_write)},
                  **({"anchor_env.jsonl": {"sha256": _trace_hash(anchor_path), "rows": len(anchor_rows)}} if make_anchor else {})},
        "source_trace_sha256": sorted(source_hashes),
        "input_errors": input_errors,
        "record_audit": [{"path": item["path"], "trace_sha256": item["trace_sha256"],
                          "case_id": item["case_id"], "reasons": item["reasons"],
                          "retrieval_mode": (item["trace"].get("retrieval") or {}).get("mode", "unknown"),
                          "metric_status": item["metrics"].get("overall"),
                          "independent_reviewed": item["independent_reviewed"],
                          "tokenizer": item["sample"].get("tokenizer")}
                         for item in records],
        "provenance": {"synthetic": bool(records) and all(item.get("case_kind") == "synthetic" and item.get("provenance_known") for item in records), "human_verified": False,
                       "independent_review_required": True, "isolated_unverified_rows": len(isolated)},
        "training_guard": {"formal_pairs_file": "pairs_env.jsonl", "isolated_file": "pairs_env.isolated.jsonl" if dry_run and write_isolated else None,
                           "formal_only": True, "reject_isolated": True, "allow_unverified_formal": False},
        "notes": ["token 数必须来自真实 Qwen tokenizer；不可用时样本被隔离。",
                  "仅同一 case_id + prompt/context 指纹的候选允许组成偏好对。",
                  "dry-run 的隔离行不得直接用于训练。"],
    }
    if dry_run and write_isolated:
        manifest["files"]["pairs_env.isolated.jsonl"] = {"sha256": _trace_hash(isolated_path), "rows": len(isolated), "training_forbidden": True}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="从 live trace 构建偏好数据")
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-unverified", action="store_true")
    parser.add_argument("--max-len", type=int, default=MAX_LEN_DEFAULT)
    parser.add_argument("--no-anchor", action="store_true")
    args = parser.parse_args(argv)
    manifest = build_dataset(args.trace_dir, args.output_dir, dry_run=args.dry_run,
                             allow_unverified=args.allow_unverified, max_len=args.max_len,
                             make_anchor=not args.no_anchor)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["build_dataset", "main", "MAX_LEN_DEFAULT"]
