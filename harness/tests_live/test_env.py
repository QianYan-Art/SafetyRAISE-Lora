"""本地完整闭环及冻结协议回归，所有模型响应均为显式夹具。"""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy

import pytest

from app.report_harness.business_roles import BusinessTransportRoles
from app.report_harness.contracts import canonical_digest
from app.report_harness.controlled_tools import ControlledTools
from app.report_harness.runtime_profiles import ConfiguredAttemptClient, ModelCapacity
from app.report_harness.transport_roles import RoleModel
from app.schemas.report_run import CandidateReport
from sr_eval.live.backends import ScriptedBackend, openai_response
from sr_eval.live.env import LiveEnvironment, VENDOR, production_budget
from sr_eval.live.scripted import SyntheticRetrieval, passing_review, passing_script, wire_candidate


@pytest.fixture
def case():
    accident = json.loads((VENDOR / "config/input_accident_template.json").read_text(encoding="utf-8"))
    accident = {key: "" for key in accident}
    accident.update({"事故标题": "虚构街道路口碰擦事故", "天气": "晴", "事故认定原因": "材料记载未注意观察"})
    return {"case_id": "fixture-live-37", "kind": "synthetic", "split": "train",
            "accident_data": accident, "guidance": {"建议": "仅根据合成输入分析"}}


def run(case, script=passing_script, *, retrieval=None, policy=None, **kwargs):
    backend = ScriptedBackend(script)
    trace = asyncio.run(LiveEnvironment(
        backend, retrieval or SyntheticRetrieval(), budget=policy,
        payload_guard=kwargs.pop("payload_guard", lambda _: {"status": "单测不计token"}),
        **kwargs,
    ).run(case))
    return trace, backend


def context(payload):
    return json.loads(payload["messages"][-1]["content"])


def tool(name, arguments, call_id="fixture-call"):
    return {"tool_calls": [{"call_id": call_id, "name": name, "arguments": arguments}]}


def test_one_version_publish_and_reviewer_isolation(case):
    trace, backend = run(case)
    assert trace["status"] == "published", trace.get("reason")
    assert trace["budget"]["tool_calls"] == 1
    assert len(trace["snapshot"]["fact_obligations"]) == 37
    assert len(trace["rounds"]) == 1
    assert len(backend.requests[0]["payload"]["messages"]) == 3
    reviewer = context(backend.requests[1]["payload"])
    assert not ({"prepared", "previous_candidate", "review_feedback",
                 "initial_knowledge_snippets"} & reviewer.keys())
    assert len(backend.requests[1]["payload"]["messages"]) == 2
    assert all(c["payload"]["response_format"] == {"type": "json_object"} for c in trace["calls"])
    assert all("tools" not in c["payload"] for c in trace["calls"])


def test_review_fail_revision_then_publish(case):
    def script(role, payload, index):
        ctx = context(payload)
        if role == "generator":
            return wire_candidate(ctx)
        review = passing_review(ctx)
        if index == 1:
            ref = ctx["snapshot"]["fact_obligations"][0]["source_refs"][0]
            review["completed_checks"][0]["passed"] = False
            review["issues"] = [{
                "issue_id": "model-temporary-id", "category": "facts", "severity": "minor",
                "target": "正文", "explanation": "合成审查认为需修订。",
                "source_refs": [ref], "closure_condition": "完成修订并由独立审查核对。",
                "status": "open",
            }]
        return review
    trace, _ = run(case, script)
    assert trace["status"] == "published"
    assert len(trace["rounds"]) == 2
    assert trace["rounds"][0]["review_enforced"]["issues"][0]["severity"] == "major"
    assert trace["rounds"][-1]["review_enforced"]["issues"][0]["status"] == "resolved"
    assert trace["rounds"][1]["candidate"]["issue_responses"]


@pytest.mark.parametrize("error", ["bad_json", "missing_obligation", "wrong_version", "quote_not_unique"])
def test_protocol_repair(case, error):
    def script(role, payload, index):
        ctx = context(payload)
        if role == "reviewer":
            return passing_review(ctx)
        candidate = wire_candidate(ctx)
        if index == 0:
            if error == "bad_json":
                return "{bad-json"
            if error == "missing_obligation":
                candidate["obligation_resolutions"].pop()
            elif error == "wrong_version":
                candidate["version"] = 9
            else:
                candidate["report_markdown"] += "\n" + candidate["claims"][0]["quote"]
        return candidate
    trace, _ = run(case, script)
    assert trace["status"] == "published"
    assert trace["protocol_repairs"] == 1
    feedback = context(trace["calls"][1]["payload"])["protocol_feedback"]
    assert feedback["repairs"][0]["errors"]


def test_unread_source_repair_requires_actual_read(case):
    def script(role, payload, index):
        ctx = context(payload)
        if role == "generator":
            if index == 1:
                return tool("read_knowledge", {"chunk_ids": ["fixture#chunk#4"]})
            candidate = wire_candidate(ctx)
            candidate["claims"][0]["knowledge_refs"] = ["fixture#chunk#4"]
            return candidate
        if not ctx["tool_results"]:
            return tool("read_knowledge", {"chunk_ids": ["fixture#chunk#4"]})
        return passing_review(ctx)
    trace, _ = run(case, script)
    assert trace["status"] == "published", trace["reason"]
    assert trace["protocol_repairs"] == 2
    feedback = context(trace["calls"][1]["payload"])["protocol_feedback"]
    assert any(e["type"] == "source_not_read" for e in feedback["repairs"][0]["errors"])
    assert any(item["name"] == "read_knowledge" for item in trace["tools"])


def test_rule_excerpt_forces_citation_failure_not_silent_protocol_relaxation(case):
    def script(role, payload, index):
        ctx = context(payload)
        if role == "generator":
            candidate = wire_candidate(ctx)
            candidate["claims"][0]["knowledge_refs"] = ["fixture#chunk#2"]
            return candidate
        if not ctx["tool_results"]:
            return tool("read_knowledge", {"chunk_ids": ["fixture#chunk#2"]})
        return passing_review(ctx)
    policy = production_budget().model_copy(update={"max_revision_rounds": 0})
    trace, _ = run(case, script, policy=policy)
    assert trace["status"] == "needs_review"
    assert trace["reason"] == "revision_rounds_exhausted"
    review = trace["rounds"][0]["review_enforced"]
    assert any(not c["passed"] for c in review["completed_checks"] if c["category"] == "citations")
    assert review["issues"][0]["severity"] == "major"
    assert "report" not in trace


@pytest.mark.parametrize("denials,expected", [(2, "published"), (3, "needs_review")])
def test_retrieval_policy_two_repair_rounds(case, denials, expected):
    def script(role, payload, index):
        if role == "generator" and index < denials:
            return tool("search_knowledge", {"query": "测试", "top_k": 4})
        return passing_script(role, payload, index)
    trace, _ = run(case, script)
    assert trace["status"] == expected
    assert trace["retrieval_denials"] == denials
    if denials == 3:
        assert trace["reason"] == "invalid_role_response"
    else:
        ctx = context(trace["calls"][2]["payload"])
        assert ctx["tool_results"][0]["result"]["error"]["code"] == "retrieval_policy_exceeded"


def test_protocol_repairs_exhausted(case):
    trace, _ = run(case, lambda *_: "bad json")
    assert trace["status"] == "needs_review"
    assert trace["reason"] == "invalid_role_response"
    assert len(trace["calls"]) == 3


@pytest.mark.parametrize("updates", [
    {"max_tool_calls": 0},
    {"max_physical_requests": 0, "max_retrieval_requests": 0},
    {"max_active_seconds": 0},
    {"max_retrieval_requests": 0},
    {"max_total_tokens": 0},
])
def test_budget_exhaustion(case, updates):
    policy = production_budget().model_copy(update=updates)
    trace, _ = run(case, policy=policy)
    assert trace["status"] == "needs_review"
    assert trace["reason"] == "budget_exhausted"
    assert "report" not in trace


def test_token_usage_exceeded_and_usage_unknown(case):
    for usage, reason, status in [
        ({"total_tokens": 100}, "usage_exceeded", "needs_review"),
        ({}, "usage_unknown", "suspended"),
    ]:
        def script(role, payload, index):
            response = openai_response(passing_script(role, payload, index))
            response["usage"] = usage
            return response
        trace, _ = run(case, script, policy=production_budget().model_copy(update={"max_total_tokens": 50}))
        assert (trace["status"], trace["reason"]) == (status, reason)


def test_first_retrieval_truncated_fails_before_model(case):
    class OneLargeSource(SyntheticRetrieval):
        def initial(self, query, top_k):
            return deepcopy(self.knowledge_source[:1])
    trace, backend = run(case, retrieval=OneLargeSource(text_chars=50000))
    assert trace["status"] == "failed"
    assert trace["reason"] == "initial_knowledge_incomplete"
    assert not backend.requests


def test_same_input_replay_identical_trace(case):
    first, _ = run(case)
    second, _ = run(case)
    assert first == second
    assert canonical_digest(first) == canonical_digest(second)


def test_reasoning_retained_only_in_private_trace(case):
    def script(role, payload, index):
        return openai_response(passing_script(role, payload, index), reasoning="合成学生思考。")
    trace, _ = run(case, script)
    assert trace["calls"][0]["reasoning_content"] == "合成学生思考。"
    clean = trace["calls"][0]["sanitized_response"]["choices"][0]["message"]
    assert "reasoning_content" not in clean
    assert "合成学生思考" not in context(trace["calls"][1]["payload"]).__str__()


def test_payload_golden_direct_frozen_roles(case):
    trace, backend = run(case)
    profiles = {role: RoleModel("tencent/hy4-preview", json_object_mode=True)
                for role in ("generator", "reviewer", "expert")}
    class Transport:
        output_limit = 64000
    from sr_eval.live.env import production_workflow
    roles = BusinessTransportRoles(Transport(), profiles,
                                   business_prompts=production_workflow().prompts())
    for call in trace["calls"]:
        role, ctx = call["role"], context(call["payload"])
        # payload中schema已是quote；还原角色调用前的内部schema与指令。
        ctx["response_schema"] = (CandidateReport.model_json_schema() if role == "generator"
                                  else __import__("app.schemas.report_run", fromlist=["ReviewResult"])
                                  .ReviewResult.model_json_schema())
        ctx["instructions"] = (VENDOR / f"config/report_harness/{role}.md").read_text(encoding="utf-8")
        ctx["snapshot"] = deepcopy(trace["snapshot"])
        if role == "generator":
            ctx["prepared"] = deepcopy(trace["prepared"])
            ctx["initial_knowledge_snippets"] = deepcopy(trace["initial_retrieval"]["items"])
        _, expected = roles._payload(role, ctx)
        expected["reasoning"] = {"effort": "high", "exclude": True}
        from sr_eval.live.env import _MANIFEST
        from app.report_harness.runtime_profiles import capacity_from_metadata, openrouter_price_filter
        metadata = _MANIFEST["metadata"][role]
        capacity = capacity_from_metadata(metadata, model=metadata["id"], effort="high")
        expected["provider"] = openrouter_price_filter(metadata, capacity)
        for actual_message, expected_message in zip(call["payload"]["messages"], expected["messages"]):
            left, right = actual_message["content"], expected_message["content"]
            mismatch = next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), min(len(left), len(right)))
            assert left == right, (role, mismatch, left[mismatch:mismatch + 100], right[mismatch:mismatch + 100])
        assert call["payload"] == expected


def test_knowledge_pagination_no_credit_until_full_read(case):
    from app.report_harness.evidence import freeze_snapshot
    retrieval = SyntheticRetrieval(text_chars=50000)
    snapshot = freeze_snapshot(case["accident_data"], [], 0, retrieval.manifest_digest)
    tools = ControlledTools(snapshot, retrieval.knowledge_source, compact_registry=True)
    ids = ["fixture#chunk#4"]
    page = tools.execute("generator", "read_knowledge", {"chunk_ids": ids})
    assert page["truncated"]
    assert not tools.accessed_knowledge("generator")
    assert page["next_cursor"]
    while page["next_cursor"]:
        page = tools.execute("generator", "read_knowledge",
                             {"chunk_ids": ids, "cursor": page["next_cursor"]})
    assert tools.accessed_knowledge("generator") == set(ids)


def test_real_case_rejected_before_any_model(case):
    case["kind"] = "real"
    with pytest.raises(ValueError, match="合成"):
        run(case)


def test_model_turn_budget_exhaustion(case):
    def script(*_):
        return tool("list_evidence", {})
    trace, _ = run(case, script)
    assert trace["status"] == "needs_review"
    assert trace["reason"] == "budget_exhausted"
    assert trace["budget"]["model_turns"] == 24


def test_non_harness_backend_exception_has_failed_trace(case):
    trace, _ = run(case, [RuntimeError("合成后端故障")])
    assert trace["status"] == "failed"
    assert trace["reason"] == "execution_error"
    assert trace["calls"][0]["error_type"] == "RuntimeError"


def test_all_three_versions_exhausted_no_report(case):
    def script(role, payload, index):
        ctx = context(payload)
        if role == "generator":
            return wire_candidate(ctx)
        review = passing_review(ctx)
        review["completed_checks"][-1]["passed"] = False
        return review
    trace, _ = run(case, script)
    assert trace["reason"] == "revision_rounds_exhausted"
    assert len(trace["rounds"]) == 3
    assert "report" not in trace


def test_length_finish_is_rejected_not_published(case):
    def script(role, payload, index):
        return openai_response(passing_script(role, payload, index), finish_reason="length")
    trace, _ = run(case, script)
    assert trace["reason"] == "invalid_role_response"
    assert trace["status"] == "needs_review"


def test_training_window_prevents_call_and_discards_trace(case):
    from sr_eval.live.window import TrainingWindow
    counter = lambda *args, **kwargs: {
        "exact": True, "prompt_tokens": 16000, "response_tokens": 1, "token_count": 16001,
    }
    trace, backend = run(case, payload_guard=TrainingWindow(counter=counter))
    assert trace["reason"] == "training_window_exceeded"
    assert trace["calls"][0]["not_sent"]
    assert backend.requests == []
    assert not trace["training_eligible"]


def test_replay_with_ledger_issues_identical(case):
    def script(role, payload, index):
        ctx = context(payload)
        if role == "generator":
            return wire_candidate(ctx)
        review = passing_review(ctx)
        if index == 1:
            review["issues"] = [{
                "issue_id": "temporary", "category": "wording", "severity": "major",
                "target": "合成正文", "explanation": "待修订。",
                "source_refs": ctx["snapshot"]["fact_obligations"][0]["source_refs"],
                "closure_condition": "独立审查确认修订。", "status": "open",
            }]
        return review
    first, _ = run(case, script)
    second, _ = run(case, script)
    assert first["status"] == "published"
    assert first == second
