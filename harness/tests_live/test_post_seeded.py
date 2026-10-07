"""R2-B 种子错误对测试；夹具全部由本地协议合成，不使用模型响应。"""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from harness.tests_live.test_post_revise import fixture_trace
from sr_eval.live.revise import evaluate_response
from sr_eval.live.seeded import ERROR_CLASSES, explain_seedability, seed_errors


def _tool_content(name: str = "search_knowledge", arguments: dict | None = None) -> str:
    return json.dumps({"tool_calls": [{
        "call_id": "seed-tool", "name": name,
        "arguments": arguments or {"query": "明确合成测试", "top_k": 3},
    }]}, ensure_ascii=False)


def test_seeded_final_pairs_have_real_single_failure_and_exact_length_ratio() -> None:
    trace = fixture_trace()
    call = trace["calls"][0]
    pairs = seed_errors(trace, call, call["content"], max_pairs=12)
    assert pairs
    for pair in pairs:
        validation = pair["validation"]
        assert validation["baseline"]["training_eligible"] is True
        assert validation["rejected"]["training_eligible"] is False
        assert validation["only_error_class"] is True
        assert validation["unique_fail_signals"] == (
            validation["rejected"].get("fail_signals", [])
            + validation["rejected"].get("seed_oracles", [])
        )
        assert 0.85 <= validation["length_ratio"] <= 1.15
        assert pair["synthetic"] is True


def test_seeded_accepts_tool_parameter_overlimit_only_from_valid_tool_baseline() -> None:
    trace = fixture_trace()
    call = trace["calls"][0]
    chosen = _tool_content()
    baseline = evaluate_response(trace, call, chosen)
    assert baseline["kind"] == "tool"
    assert baseline["training_eligible"] is True, baseline["fail_signals"]
    pairs = seed_errors(trace, call, chosen,
                        error_classes=["tool_parameter_overlimit"], max_pairs=1)
    assert len(pairs) == 1
    assert pairs[0]["error_class"] == "tool_parameter_overlimit"
    assert pairs[0]["validation"]["rejected"]["fail_signals"] == [
        "tool:retrieval_policy_exceeded"
    ]


def test_seeded_optional_tool_oracles_are_explicit() -> None:
    trace = fixture_trace()
    trace["seed_oracles"] = {"tool_necessary": False}
    call = trace["calls"][0]
    chosen = _tool_content()
    pairs = seed_errors(trace, call, chosen,
                        error_classes=["unnecessary_retrieval"], max_pairs=1)
    assert len(pairs) == 1
    assert pairs[0]["validation"]["rejected"]["fail_signals"] == []
    assert pairs[0]["validation"]["rejected"]["seed_oracles"] == [
        "seed_oracle:unnecessary_retrieval"
    ]


def test_seeded_rejects_nonpassing_input_and_does_not_fill_expected_signals() -> None:
    trace = fixture_trace()
    call = deepcopy(trace["calls"][0])
    bad = json.loads(call["content"])
    bad["version"] = 9
    chosen = json.dumps(bad, ensure_ascii=False)
    assert evaluate_response(trace, call, chosen)["training_eligible"] is False
    assert seed_errors(trace, call, chosen, error_classes=ERROR_CLASSES, max_pairs=12) == []


@pytest.mark.parametrize("error_class", ERROR_CLASSES)
def test_each_r2b_class_is_probed_or_safely_dropped(error_class: str) -> None:
    """十二类均经过同一安全门；缺少冲突、未读来源或必要性 oracle 时只能丢弃。"""
    trace = fixture_trace()
    call = trace["calls"][0]
    diagnostic = explain_seedability(trace, call, call["content"],
                                     error_classes=[error_class], max_pairs=1)
    assert diagnostic["requested"] == [error_class]
    assert set(diagnostic["produced"]) | set(diagnostic["dropped"]) == {error_class}
    if diagnostic["produced"]:
        pair = seed_errors(trace, call, call["content"],
                           error_classes=[error_class], max_pairs=1)[0]
        assert pair["validation"]["only_error_class"] is True


def test_final_json_length_ratio_ignores_shared_long_think() -> None:
    trace = fixture_trace()
    call = deepcopy(trace["calls"][0])
    call["reasoning_content"] = "合成思考。" * 3000
    pair = seed_errors(trace, call, call["content"],
                       error_classes=["omit_key_fact"], max_pairs=1)[0]
    assert 0.85 <= pair["validation"]["length_ratio"] <= 1.15
    assert pair["validation"]["length_tokens"]["chosen"] > 0
