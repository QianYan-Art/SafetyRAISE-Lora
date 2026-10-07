from __future__ import annotations

import json
import hashlib
import shutil
from pathlib import Path
from uuid import uuid4

import pytest

from sr_eval.live.dataset import build_dataset
from sr_eval.live.metrics import EVALUATION_FIELDS, FIELD_NAMES
from sr_eval.live.samples import count_payload, window_guard
from app.report_harness.contracts import canonical_digest
from app.report_harness.evidence import freeze_snapshot
from app.report_harness.quoted_roles import resolve_quotes
from sr_eval.live.scripted import passing_review


def _trace(root: Path, name: str, status: str, *, independent: bool = False) -> None:
    accident = {field: f"值-{index}" for index, field in enumerate(FIELD_NAMES)}
    snapshot = freeze_snapshot(accident, [], 0, canonical_digest([]))
    obligations = snapshot["fact_obligations"]
    lines = [f"{field}：事故信息记载{value}。" for field, value in accident.items()]
    report = "\n".join(lines)
    wire = {
        "version": 1 if status == "published" else 2,
        "report_markdown": report,
        "claims": [{"claim_id": f"c{index}", "type": "fact", "quote": lines[index],
                    "evidence_refs": obligations[index]["source_refs"], "knowledge_refs": []}
                   for index, field in enumerate(FIELD_NAMES)],
        "obligation_resolutions": [{"obligation_id": item["obligation_id"], "treatment": "covered",
                                    "resolution": "已在正文覆盖"} for item in obligations],
    }
    candidate = resolve_quotes(wire)
    candidate["issue_responses"] = []
    digest = canonical_digest(candidate)
    review = passing_review({"snapshot": snapshot, "candidate": candidate,
                             "snapshot_digest": canonical_digest(snapshot), "candidate_digest": digest,
                             "unresolved_issues": []})
    context = {"candidate_version": 1, "snapshot": snapshot, "unresolved_issues": []}
    access = {"evidence": sorted(ref for item in obligations for ref in item["source_refs"]), "knowledge": []}
    data = {
        "case_id": "syn_train_live_001",
        "case": {"kind": "synthetic", "accident_data": accident},
        "status": status,
        "snapshot": snapshot, "snapshot_digest": canonical_digest(snapshot),
        "knowledge_registry": [], "tools": [], "issue_history": [],
        "rounds": [{"candidate": candidate, "review_enforced": review, "generator_access": access,
                    "reviewer_access": access, "resolved_issue_ids": []}],
        "publication": {"candidate_digest": digest, "review_digest": canonical_digest(review),
                        "snapshot_digest": canonical_digest(snapshot)},
        "provenance": {"synthetic": True, "human_verified": False, "source": "测试夹具"},
        "independent_review": {"passed": True, "candidate_digest": digest, "reviewer_id": "blind-codex",
                               "reviewer_kind": "codex", "offline": True,
                               "blind": True, "checks": {name: True for name in
                                ("facts", "citations", "privacy", "leakage", "tool_authenticity",
                                 "source_authorization")}} if independent else None,
        "calls": [{
            "call_index": 0,
            "role": "generator",
            "payload": {"messages": [{"role": "system", "content": "系统"},
                                    {"role": "user", "content": "输入"},
                                    {"role": "user", "content": json.dumps(context, ensure_ascii=False)}],
                        "response_format": {"type": "json_object"}},
            "response": {"choices": [{"message": {"content": json.dumps(wire, ensure_ascii=False),
                                                       "reasoning_content": "简短思考"}, "finish_reason": "stop"}]},
            "content": json.dumps(wire, ensure_ascii=False),
            "reasoning_content": "简短思考",
            "finish_reason": "stop",
        }, {"role": "reviewer", "payload": {"messages": [{"role": "user", "content":
             json.dumps({"candidate": candidate, "snapshot": snapshot,
                         "snapshot_digest": canonical_digest(snapshot),
                         "candidate_digest": digest}, ensure_ascii=False)}]}}],
    }
    (root / name).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_bridge_count_is_exact_or_explicitly_unavailable() -> None:
    result = count_payload({"messages": [{"role": "user", "content": "请返回 JSON"}]},
                           content="{\"ok\":true}", reasoning_content="思考")
    assert result["exact"] is (result["status"] == "exact")
    if result["exact"]:
        assert result["token_count"] == result["prompt_tokens"] + result["response_tokens"]
        assert window_guard({"status": result["status"], "prompt_tokens": result["prompt_tokens"],
                             "response_tokens": result["response_tokens"]})["hard_ok"] is True
    else:
        assert result["token_count"] is None


def test_dry_run_builds_same_context_pair_and_manifest() -> None:
    root = Path("harness") / "reports" / "live-dataset" / ".scratch-live-tests" / str(uuid4())
    traces = root / "traces"
    out = root / "out"
    traces.mkdir(parents=True)
    try:
        _trace(traces, "chosen.json", "published", independent=True)
        _trace(traces, "rejected.json", "needs_review")
        manifest = build_dataset(traces, out, dry_run=True)
        assert manifest["mode"] == "dry-run"
        assert manifest["stats"]["trace_files"] == 2
        pairs = [json.loads(line) for line in (out / "pairs_env.jsonl").read_text(encoding="utf-8").splitlines() if line]
        if pairs:
            row = pairs[0]
            assert {"prompt_ids", "chosen_ids", "rejected_ids", "c_start", "r_start", "pair_id", "kind"} <= row.keys()
            assert row["prompt_ids"]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_unverified_chosen_isolated_from_formal_pairs() -> None:
    root = Path("harness") / "reports" / "live-dataset" / ".scratch-live-tests" / str(uuid4())
    traces = root / "traces"
    out = root / "out"
    traces.mkdir(parents=True)
    try:
        _trace(traces, "chosen.json", "published", independent=False)
        _trace(traces, "rejected.json", "needs_review")
        manifest = build_dataset(traces, out, dry_run=True)
        assert manifest["stats"]["unverified_chosen"] == 1
        assert manifest["stats"]["formal_pairs"] == 0
        assert manifest["stats"]["isolated_pairs"] == 1
        assert (out / "pairs_env.jsonl").read_text(encoding="utf-8") == ""
        assert (out / "pairs_env.isolated.jsonl").exists()
        assert manifest["release_eligible"] is False
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_non_synthetic_and_unknown_provenance_are_rejected() -> None:
    root = Path("harness") / "reports" / "live-dataset" / ".scratch-live-tests" / str(uuid4())
    traces = root / "traces"
    out = root / "out"
    traces.mkdir(parents=True)
    try:
        _trace(traces, "real.json", "published", independent=True)
        value = json.loads((traces / "real.json").read_text(encoding="utf-8"))
        value["case"]["kind"] = "real"
        value["provenance"] = {"source": "unknown"}
        (traces / "real.json").write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        manifest = build_dataset(traces, out, dry_run=True)
        assert manifest["stats"]["non_synthetic_rejected"] == 1
        assert manifest["stats"]["provenance_unknown"] == 1
        assert manifest["release_eligible"] is False
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_formal_unverified_and_window_overflow_are_hard_rejected() -> None:
    root = Path("harness") / "reports" / "live-dataset" / ".scratch-live-tests" / str(uuid4())
    traces = root / "traces"
    out = root / "out"
    traces.mkdir(parents=True)
    try:
        _trace(traces, "chosen.json", "published", independent=False)
        with pytest.raises(ValueError, match="只能用于 dry-run"):
            build_dataset(traces, out, allow_unverified=True)
        with pytest.raises(ValueError, match="1..32768"):
            build_dataset(traces, out, dry_run=True, max_len=32769)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_different_student_thought_prefix_cannot_form_pair() -> None:
    root = Path("harness") / "reports" / "live-dataset" / ".scratch-live-tests" / str(uuid4())
    traces = root / "traces"
    out = root / "out"
    traces.mkdir(parents=True)
    try:
        _trace(traces, "chosen.json", "published", independent=False)
        _trace(traces, "rejected.json", "needs_review")
        value = json.loads((traces / "rejected.json").read_text(encoding="utf-8"))
        value["calls"][0]["reasoning_content"] = "不同学生思考前缀"
        value["calls"][0]["response"]["choices"][0]["message"]["reasoning_content"] = "不同学生思考前缀"
        (traces / "rejected.json").write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        manifest = build_dataset(traces, out, dry_run=True)
        assert manifest["stats"]["thought_mismatch"] == 1
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.mark.parametrize("mutation", ["publication", "duplicate_obligation", "forged_access", "case_id",
                                    "terminal_call", "snapshot_context"])
def test_published_label_cannot_bypass_contract_and_proof(mutation) -> None:
    root = Path("harness/reports/live-dataset/.scratch-live-tests") / str(uuid4())
    traces, out = root / "traces", root / "out"
    traces.mkdir(parents=True)
    try:
        _trace(traces, "chosen.json", "published", independent=True)
        _trace(traces, "rejected.json", "needs_review")
        path = traces / "chosen.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        if mutation == "publication":
            value["publication"]["candidate_digest"] = "0" * 64
        elif mutation == "duplicate_obligation":
            wire = json.loads(value["calls"][0]["content"])
            wire["obligation_resolutions"].append(wire["obligation_resolutions"][0])
            value["calls"][0]["content"] = json.dumps(wire, ensure_ascii=False)
        elif mutation == "forged_access":
            value["rounds"][0]["generator_access"]["knowledge"] = ["unread-source"]
        elif mutation == "terminal_call":
            last = json.loads(json.dumps(value["calls"][0], ensure_ascii=False))
            last["content"] = "{"
            last["response"]["choices"][0]["message"]["content"] = "{"
            value["calls"].append(last)
        elif mutation == "snapshot_context":
            context = json.loads(value["calls"][0]["payload"]["messages"][-1]["content"])
            context["snapshot"]["accident_data"]["天气"] = "另一合成案件的字段值"
            value["calls"][0]["payload"]["messages"][-1]["content"] = json.dumps(context, ensure_ascii=False)
            value["calls"][0]["context_digest"] = canonical_digest(context)
        else:
            value["case_id"] = ""
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        manifest = build_dataset(traces, out)
        assert manifest["stats"]["formal_pairs"] == 0
        assert manifest["release_eligible"] is False
        assert next(row for row in manifest["record_audit"] if row["path"] == str(path))["reasons"]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_unknown_metric_and_unbound_independent_review_never_formal(monkeypatch) -> None:
    root = Path("harness/reports/live-dataset/.scratch-live-tests") / str(uuid4())
    traces, out = root / "traces", root / "out"
    traces.mkdir(parents=True)
    try:
        _trace(traces, "chosen.json", "published", independent=True)
        _trace(traces, "rejected.json", "needs_review")
        import sr_eval.live.dataset as dataset
        monkeypatch.setattr(dataset, "evaluate_candidate",
                            lambda *args: {"overall": "unknown", "strict_pass": False})
        manifest = build_dataset(traces, out, dry_run=True)
        assert manifest["stats"]["formal_pairs"] == 0
        assert manifest["stats"]["isolated_pairs"] == 1
        path = traces / "chosen.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["independent_review"]["candidate_digest"] = "0" * 64
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        manifest = build_dataset(traces, out, dry_run=True)
        assert manifest["stats"]["unverified_chosen"] == 1
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_bad_json_is_audited_and_blocks_release() -> None:
    root = Path("harness/reports/live-dataset/.scratch-live-tests") / str(uuid4())
    traces, out = root / "traces", root / "out"
    traces.mkdir(parents=True)
    try:
        _trace(traces, "chosen.json", "published", independent=True)
        _trace(traces, "rejected.json", "needs_review")
        (traces / "bad.json").write_text("{", encoding="utf-8")
        manifest = build_dataset(traces, out)
        assert manifest["stats"]["input_errors"] == 1
        assert manifest["input_errors"][0]["sha256"] == hashlib.sha256(b"{").hexdigest()
        assert manifest["release_eligible"] is False
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_formal_pair_and_anchor_match_training_row_parser() -> None:
    root = Path("harness/reports/live-dataset/.scratch-live-tests") / str(uuid4())
    traces, out = root / "traces", root / "out"
    traces.mkdir(parents=True)
    try:
        _trace(traces, "chosen.json", "published", independent=True)
        _trace(traces, "rejected.json", "needs_review")
        manifest = build_dataset(traces, out)
        assert manifest["stats"]["formal_pairs"] == 1, manifest["record_audit"]
        assert manifest["release_eligible"] is True
        pair = json.loads((out / "pairs_env.jsonl").read_text(encoding="utf-8"))
        anchor = json.loads((out / "anchor_env.jsonl").read_text(encoding="utf-8"))
        # 复核训练脚本的输入表达式，不导入会立即启动 GPU 的训练模块。
        for row in (pair, anchor):
            assert row["chosen_ids"] and (row["rejected_ids"] or row["kind"].startswith("anchor"))
            assert len(row["prompt_ids"]) + max(len(row["chosen_ids"]), len(row["rejected_ids"])) <= 32768
            assert 0 <= row["c_start"] < len(row["chosen_ids"])
        assert anchor["rejected_ids"] == []
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_full_search_result_access_replays_like_frozen_tools() -> None:
    import asyncio
    from copy import deepcopy
    from sr_eval.live.backends import ScriptedBackend
    from sr_eval.live.env import LiveEnvironment
    from sr_eval.live.dataset import _candidate_call, _candidate_from_trace, _published_problems
    from sr_eval.live.scripted import SyntheticRetrieval, wire_candidate

    case = {"case_id": "fixture-search-access", "kind": "synthetic", "guidance": {},
            "accident_data": {field: f"值-{index}" for index, field in enumerate(FIELD_NAMES)}}
    retrieval = SyntheticRetrieval()
    retrieval.search = lambda query, top_k: deepcopy(retrieval.knowledge_source[-1:])

    def script(role, payload, index):
        context = json.loads(payload["messages"][-1]["content"])
        if not context["tool_results"]:
            return {"tool_calls": [{"name": "search_knowledge", "call_id": f"{role}-search",
                                   "arguments": {"query": "合成追加查询", "top_k": 1}}]}
        if role == "reviewer":
            return passing_review(context)
        candidate = wire_candidate(context)
        candidate["report_markdown"] += "\n合成知识校验。"
        candidate["claims"].append({"claim_id": "knowledge-extra", "type": "knowledge",
                                    "quote": "合成知识校验。", "evidence_refs": [],
                                    "knowledge_refs": [retrieval.knowledge_source[-1]["id"]]})
        return candidate

    trace = asyncio.run(LiveEnvironment(
        ScriptedBackend(script), retrieval, payload_guard=lambda payload: {"unit_test": True},
    ).run(case))
    assert trace["status"] == "published"
    call, _ = _candidate_call(trace)
    assert _published_problems(trace, call, _candidate_from_trace(trace)) == []
    for event in trace["retrieval"]["events"]:
        if event["step"] == "additional":
            event["approved_projection_digest"] = "0" * 64
    assert _published_problems(trace, call, _candidate_from_trace(trace)) != []
