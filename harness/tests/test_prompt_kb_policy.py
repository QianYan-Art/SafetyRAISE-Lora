import pytest

from sr_eval import prompt as P
from sr_eval.kb import build_initial_query, synthetic_kb, truncate_context
from sr_eval.policy import OutboundDenied, OutboundPolicy


def test_render_replaces_each_placeholder_once(case):
    tpl = P.load_asset("report_prompt.md")
    kb = synthetic_kb()
    _q, initial = kb.retrieve(case["accident_data"], 3)
    out = P.render_report_prompt(tpl, case["accident_data"], case["guidance"], initial, [], [])
    for ph in (P.REPORT_GUIDANCE_PLACEHOLDER, P.REPORT_ACCIDENT_PLACEHOLDER, P.REPORT_ACCIDENT_ANCHOR_PLACEHOLDER,
               P.REPORT_INITIAL_SNIPPETS_PLACEHOLDER, P.REPORT_ADDITIONAL_SNIPPETS_PLACEHOLDER, P.REPORT_AGENTIC_HISTORY_PLACEHOLDER):
        assert ph not in out
    assert "夜间路口两车碰撞" in out and initial[0]["id"] in out


def test_render_requires_all_placeholders(case):
    with pytest.raises(ValueError):
        P.render_report_prompt("没有占位符", case["accident_data"], case["guidance"], [], [], [])


def test_anchor_summary_falls_back_and_strips_internal():
    assert "暂无可提取" in P.build_anchor_summary({})
    assert "_secret" not in str(P.strip_internal_fields({"_secret": 1, "a": {"_b": 2, "c": 3}}))


def test_exec_prompt_variants_differ():
    prod = P.build_exec_prompt(False, "production")
    concise = P.build_exec_prompt(False, "concise")
    assert "尽可能充分展开" in prod and "尽可能充分展开" not in concise and "分析项不要漏" in concise
    assert "充分展开核心分析" in P.build_exec_prompt(True, "production")
    with pytest.raises(ValueError):
        P.build_exec_prompt(False, "x")


def test_sanitize_strips_fence_and_reasoning():
    raw = "<think>内部</think>```markdown\n# 标题\n正文\n```"
    assert P.sanitize_markdown_output(raw) == "# 标题\n\n正文"   # 冻结的生产清洗会在标题后补空行
    assert P.sanitize_markdown_output("只有正文").startswith("# 交通事故分析报告")


def test_extract_json_action():
    assert P.extract_json_action('说明 {"action":"retrieve","query":"q"}')["query"] == "q"
    assert P.extract_json_action("没有 json") is None


def test_tool_schema_matches_production_limits():
    fn = P.retrieve_tool_schema()[0]["function"]
    assert fn["name"] == "retrieve_knowledge" and fn["parameters"]["properties"]["top_k"]["maximum"] == 3
    assert fn["parameters"]["required"] == ["query"]


def test_initial_query_uses_priority_fields_and_dedupes():
    q = build_initial_query({"事故标题": "A", "事故类型": "A", "事故形态": "B", "无关": "Z"})
    assert q == "A\nB"


def test_kb_search_finds_relevant_rule():
    kb = synthetic_kb()
    hits = kb.search("机动车倒车 观察车后情况", 3)
    assert hits and "倒车" in hits[0]["title"] and hits[0]["record_type"] == "chunk"
    assert hits[0]["id"].startswith("synth_kb_")


def test_truncate_context_limits_total_chars():
    recs = [{"id": str(i), "content": "字" * 3000} for i in range(3)]
    out = truncate_context(recs, 5200)
    assert sum(len(r["content"]) for r in out) <= 5200 + 2 and out[-1].get("content_truncated")


def test_outbound_policy_gates():
    pol = OutboundPolicy(synthetic_cases=True, public_statutes=False, real_cases=False)
    pol.check(kb_class="synthetic", case_kind="synthetic")
    with pytest.raises(OutboundDenied):
        pol.check(kb_class="public_statutes", case_kind="synthetic")
    with pytest.raises(OutboundDenied):
        pol.check(kb_class="synthetic", case_kind="real")
    assert OutboundPolicy.load().real_cases is False  # 仓内配置:真实案件永不外发
