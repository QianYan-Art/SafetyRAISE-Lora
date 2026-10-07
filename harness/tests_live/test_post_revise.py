"""第2轮修订单测，全部响应由本地明确合成，不使用真实模型响应。"""

import asyncio
import json
from copy import deepcopy

import pytest

from app.report_harness.contracts import canonical_digest
from sr_eval.llm import LLMError
from sr_eval.live.backends import ScriptedBackend, openai_response
from sr_eval.live.env import LiveEnvironment
from sr_eval.live.metrics import FIELD_NAMES
from sr_eval.live.revise import evaluate_response, revise_call, run_revisions
from sr_eval.live.scripted import SyntheticRetrieval, passing_review, wire_candidate


def fixture_trace():
    accident = dict.fromkeys(FIELD_NAMES, "")
    accident.update({"事故标题": "明确合成测试事故", "天气": "晴", "事故形态": "侧面碰擦",
                     "事故认定原因": "未注意观察"})
    case = {"case_id": "r2-synthetic-fixture", "kind": "synthetic", "split": "train",
            "accident_data": accident, "guidance": {"建议": "仅按合成输入分析"}}

    def script(role, payload, index):
        context = json.loads(payload["messages"][-1]["content"])
        if role == "reviewer":
            return passing_review(context)
        candidate = wire_candidate(context)
        for claim in candidate["claims"]:
            if claim["evidence_refs"] == ["accident:/事故认定原因"]:
                before = claim["quote"]
                claim["quote"] = "事故信息记载" + before
                candidate["report_markdown"] = candidate["report_markdown"].replace(before, claim["quote"])
        return openai_response(candidate, reasoning="本地合成学生思考，仅用于协议测试。")

    return asyncio.run(LiveEnvironment(
        ScriptedBackend(script), SyntheticRetrieval(),
        payload_guard=lambda payload: {"status": "单测不计token"},
    ).run(case))


def test_original_accepted_unknown_counted_without_teacher():
    trace = fixture_trace()
    call = trace["calls"][0]
    validation = evaluate_response(trace, call)
    assert validation["training_eligible"], validation["fail_signals"]
    assert validation["kind"] == "final"
    assert validation["unknown_metrics"]
    result = asyncio.run(revise_call(trace, call))
    assert result["derivation"] == "student_original"
    assert result["chosen_content"] == call["content"]
    assert result["teacher_calls"] == []
    assert result["diff"]["similarity"] == 1


def test_minimal_revision_payload_and_ledger_binding():
    trace = fixture_trace()
    call = deepcopy(trace["calls"][0])
    original = call["content"]
    bad = json.loads(original)
    bad["version"] = 9
    call["content"] = json.dumps(bad, ensure_ascii=False)
    call["response"] = openai_response(call["content"], reasoning=call["reasoning_content"])
    response = openai_response(original)
    response["live_ledger"] = {"call_id": "synthetic-ledger-not-provider"}
    backend = ScriptedBackend([response])
    result = asyncio.run(revise_call(trace, call, backend))
    assert result["training_eligible"]
    assert result["derivation"] == "minimax_revision"
    assert result["rejected_content"] == call["content"]
    assert result["teacher_calls"][0]["ledger_id"] == "synthetic-ledger-not-provider"
    request = backend.requests[0]
    assert request["role"] == "reviser"
    assert request["payload"]["messages"][:3] == call["payload"]["messages"]
    assert request["payload"]["messages"][3]["reasoning_content"] == call["reasoning_content"]
    assert all(name in request["payload"]["messages"][-1]["content"] for name in FIELD_NAMES)
    assert request["payload"]["reasoning"]["effort"] == "max"
    assert request["payload"]["response_format"] == {"type": "json_object"}


def test_two_revisions_exhausted():
    trace = fixture_trace()
    call = deepcopy(trace["calls"][0])
    call["content"] = "{invalid"
    call["response"] = openai_response(call["content"])
    backend = ScriptedBackend(["{invalid", "{still-invalid"])
    result = asyncio.run(revise_call(trace, call, backend))
    assert result["discard_reason"] == "revision_exhausted"
    assert len(result["teacher_calls"]) == 2
    assert not result["training_eligible"]


def test_truncated_teacher_never_accepted():
    trace = fixture_trace()
    call = deepcopy(trace["calls"][0])
    original = call["content"]
    call["content"] = "{invalid"
    backend = ScriptedBackend([openai_response(original, finish_reason="length")] * 2)
    result = asyncio.run(revise_call(trace, call, backend))
    assert not result["training_eligible"]
    assert result["discard_reason"] == "revision_exhausted"


def test_missing_student_response_and_tampered_snapshot_forbidden():
    trace = fixture_trace()
    call = deepcopy(trace["calls"][0])
    call.pop("response")
    assert not asyncio.run(revise_call(trace, call))["training_eligible"]
    trace["snapshot"]["accident_data"]["天气"] = "雨"
    assert not evaluate_response(trace, trace["calls"][0])["training_eligible"]


def test_rule_excerpt_and_unread_source_hard_gates():
    trace = fixture_trace()
    call = trace["calls"][0]
    parsed = json.loads(call["content"])
    parsed["claims"][0]["knowledge_refs"] = ["fixture#chunk#2"]
    result = evaluate_response(trace, call, json.dumps(parsed, ensure_ascii=False))
    assert any(x.startswith("rule_excerpt_as_evidence") for x in result["hard_problems"])
    parsed["claims"][0]["knowledge_refs"] = ["never-read"]
    result = evaluate_response(trace, call, json.dumps(parsed, ensure_ascii=False))
    assert any(x.startswith("source_not_read") for x in result["hard_problems"])


def test_tool_name_args_and_cumulative_search_limit():
    trace = fixture_trace()
    call = trace["calls"][0]
    calls = [{"call_id": "t1", "name": "search_knowledge",
              "arguments": {"query": "明确合成测试", "top_k": 3}}]
    good = evaluate_response(trace, call, json.dumps({"tool_calls": calls}, ensure_ascii=False))
    assert good["training_eligible"], good["fail_signals"]
    calls[0]["arguments"]["top_k"] = 4
    assert "tool:retrieval_policy_exceeded" in evaluate_response(
        trace, call, json.dumps({"tool_calls": calls}))["hard_problems"]
    calls[0]["name"] = "execute"
    assert "tool:unknown_tool" in evaluate_response(
        trace, call, json.dumps({"tool_calls": calls}))["hard_problems"]
    calls = [{"call_id": f"t{i}", "name": "search_knowledge",
              "arguments": {"query": "明确合成测试", "top_k": 1}} for i in range(3)]
    assert "tool:retrieval_request_budget_exhausted" in evaluate_response(
        trace, call, json.dumps({"tool_calls": calls}))["hard_problems"]


def test_retry_rejected_transient_but_not_unknown():
    trace = fixture_trace()
    call = deepcopy(trace["calls"][0])
    original = call["content"]
    call["content"] = "{invalid"
    backend = ScriptedBackend([LLMError("overload", billing_state="rejected", http_status=529),
                               openai_response(original)])
    assert asyncio.run(revise_call(trace, call, backend, retries=1, retry_delay=0))["training_eligible"]
    backend = ScriptedBackend([LLMError("timeout", billing_state="unknown")])
    result = asyncio.run(revise_call(trace, call, backend, retries=1))
    assert result["discard_reason"] == "teacher_error:timeout"
    assert len(backend.requests) == 1


def test_run_revisions_writes_bound_records_and_limits_workers(tmp_path):
    source = tmp_path / "traces"
    source.mkdir()
    trace = fixture_trace()
    (source / "synthetic.json").write_text(json.dumps(trace, ensure_ascii=False), encoding="utf-8")
    summary = run_revisions(source, tmp_path / "derived", ScriptedBackend([]), workers=4)
    assert summary["accepted"] == 1
    results = list((tmp_path / "derived").glob("*.json"))
    result = next(json.loads(path.read_text(encoding="utf-8")) for path in results
                  if path.name != "summary.json")
    assert result["trace_sha256"] == canonical_digest(trace)
    assert run_revisions(source, tmp_path / "derived", ScriptedBackend([]), workers=1)["accepted"] == 1
    with pytest.raises(ValueError):
        run_revisions(source, tmp_path / "derived", ScriptedBackend([]), workers=5)


def test_later_call_access_is_replayed_and_denial_is_not_a_read():
    base = fixture_trace()
    case = base["case"]

    def script(role, payload, index):
        context = json.loads(payload["messages"][-1]["content"])
        if role == "reviewer":
            return passing_review(context)
        if index == 0:
            return {"tool_calls": [{"call_id": "read", "name": "read_knowledge",
                                    "arguments": {"chunk_ids": ["fixture#chunk#0"]}}]}
        if index == 1:
            return {"tool_calls": [{"call_id": "denied", "name": "search_knowledge",
                                    "arguments": {"query": "合成测试", "top_k": 4}}]}
        candidate = wire_candidate(context)
        for claim in candidate["claims"]:
            if claim["evidence_refs"] == ["accident:/事故认定原因"]:
                before = claim["quote"]
                claim["quote"] = "事故信息记载" + before
                candidate["report_markdown"] = candidate["report_markdown"].replace(before, claim["quote"])
        return candidate

    trace = asyncio.run(LiveEnvironment(
        ScriptedBackend(script), SyntheticRetrieval(),
        payload_guard=lambda payload: {"status": "单测不计token"},
    ).run(case))
    assert trace["status"] == "published"
    call = trace["calls"][2]
    validation = evaluate_response(trace, call)
    assert validation["training_eligible"], validation["hard_problems"]
    assert "fixture#chunk#0" in validation["access"]["knowledge"]
