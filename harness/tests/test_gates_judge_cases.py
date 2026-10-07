import json

from sr_eval import gates as G
from sr_eval import judge as J
from sr_eval.cases import FIELDS, guidance_skeleton, scenario_for, validate_accident, validate_guidance

SNIP = [{"id": "kb_1", "source": "law_a", "title": "道路交通安全法实施条例", "authority": "国务院",
         "content": "第四十五条 夜间行驶应当降低速度。第五十一条 机动车行经人行横道应当减速。"}]
GOOD = (
    "# 交通事故分析报告\n\n## 一、事故概要\n事故发生于夜间路口,时间 21:40。\n"
    "## 二、事实与证据边界\n信息不足:未提供行驶速度。\n## 三、事故经过与关键时序重构\n两车侧面碰撞。\n"
    "## 四、致因分析\n夜间应降低速度[依据: law_a#kb_1]。\n## 五、责任分析\n可能存在注意义务,待核实。\n"
    "## 六、依据与待补证据\n《道路交通安全法实施条例》第四十五条[依据: kb_1]。\n"
)


def trace(report=GOOD, **over):
    t = {"status": "ok", "final_markdown": report, "visible_snippets": SNIP, "tool_call_issues": [],
         "totals": {"any_truncated": False, "retrieval_rounds": 1}}
    t.update(over)
    return t


def codes(res, level="hard"):
    return {x["code"] for x in res[level]}


def test_clean_report_passes(case):
    res = G.evaluate(trace(), case)
    assert not res["hard_fail"], res["hard"]


def test_citation_must_be_visible(case):
    res = G.evaluate(trace(GOOD + "补充[依据: law_z#nope]"), case)
    assert "citation_not_visible" in codes(res)


def test_article_not_in_visible_text(case):
    bad = GOOD + "\n另依据《中华人民共和国道路交通安全法》第九十九条。"
    res = G.evaluate(trace(bad), case)
    assert "article_not_in_visible_text" in codes(res)
    ok_ = GOOD + "\n《道路交通安全法实施条例》第五十一条。"
    assert "article_not_in_visible_text" not in codes(G.evaluate(trace(ok_), case))


def test_cn_numerals():
    assert G.cn2int("四十五") == 45 and G.cn2int("一百零二") == 102 and G.cn2int("十") == 10 and G.cn2int("71") == 71


def test_chinese_numerals_in_input_support_arabic_in_report(case):
    assert {"9", "12", "50", "2026"} <= G.cn_numbers("上午九时十二分,能见度约五十米,二零二六年")
    case = {**case, "accident_data": {"事故发生时间": "上午九时十二分", "能见度": "约五十米"}}
    res = G.evaluate(trace(GOOD.replace(",时间 21:40", "") + "\n事发于 9 时 12 分,能见度约 50 米。"), case)
    assert "fabricated_quantity" not in codes(res) and "unsupported_number" not in codes(res, "warn")


def test_fabricated_speed_and_clock(case):
    res = G.evaluate(trace(GOOD + "\n碰撞时车速约 87 公里/小时,事发于 23 时 10 分。"), case)
    assert codes(res) >= {"fabricated_quantity"}
    assert "21:40" in json.dumps(case["accident_data"], ensure_ascii=False)
    assert "fabricated_quantity" not in codes(G.evaluate(trace(GOOD + "\n事发于 21 时 40 分。"), case))


def test_sentencing_and_privacy_hard(case):
    res = G.evaluate(trace(GOOD + "\n建议判处有期徒刑。联系电话 13812345678。"), case)
    assert {"sentencing_or_crime_language", "privacy_leak"} <= codes(res)


def test_liability_assertion_is_warning_not_hard(case):
    res = G.evaluate(trace(GOOD + "\n驾驶人应负主要责任。"), case)
    assert "liability_assertion_unhedged" in codes(res, "warn") and not res["hard_fail"]


def test_truncated_and_missing_report(case):
    assert "truncated_output" in codes(G.evaluate(trace(totals={"any_truncated": True, "retrieval_rounds": 1}), case))
    assert "no_report" in codes(G.evaluate({"status": "error", "error": "x", "totals": {}}, case))


def test_tool_markup_and_issues(case):
    res = G.evaluate(trace(GOOD + '\n{"action":"retrieve","query":"x"}', tool_call_issues=[{"round": 0, "issue": "arguments_not_json"}]), case)
    assert {"tool_markup_in_report", "tool_call_arguments_not_json"} <= codes(res)


def test_metrics_detect_duplication():
    m = G.report_metrics(GOOD + GOOD)
    assert m["dup_12gram_ratio"] > 0.4 and m["sections"] >= 12


# ---- 评审解析与播种错误 ----
def judgement(scores=(4, 4, 3, 4, 4), quote="夜间应降低速度"):
    dims = {d: {"score": s, "reason": "r"} for d, s in zip(J.DIMENSIONS, scores)}
    return json.dumps({"dimensions": dims, "concision": {"score": 4, "reason": "r"},
                       "defects": [{"dimension": J.DIMENSIONS[2], "type": "unsupported_cause", "quote": quote, "explanation": "e"}],
                       "overall_comment": "c"}, ensure_ascii=False)


def test_parse_judgement_and_quote_verification():
    out = J.parse_judgement("```json\n" + judgement() + "\n```", GOOD)
    assert out["valid"] and out["total"] == 19 and out["unverified_defects"] == 0
    bad = J.parse_judgement(judgement(quote="报告里根本没有的话"), GOOD)
    assert bad["unverified_defects"] == 1 and bad["defects"][0]["quote_verified"] is False


def test_parse_judgement_rejects_bad_scores():
    assert not J.parse_judgement(judgement(scores=(4, 4, 7, 4, 4)), GOOD)["valid"]
    assert not J.parse_judgement("不是 json", GOOD)["valid"]


def test_mutations_change_reports_and_plan_is_shuffled_but_complete():
    plan = J.calibration_plan([("r1", GOOD)])
    assert {p["mutation"] for p in plan} == {"orig", *J.MUTATIONS}
    by = {p["mutation"]: p["text"] for p in plan}
    assert "致因分析" not in by["drop_causation"] and "[依据" not in by["strip_citations"]
    assert "判处" in by["overclaim_liability"] and "第九十九条" in by["fake_law"]
    # 硬门应当抓到确定性的播种错误
    c = {"accident_data": {}, "guidance": {}}
    assert G.evaluate(trace(by["fake_law"]), c)["hard_fail"] and G.evaluate(trace(by["overclaim_liability"]), c)["hard_fail"]


def test_calibration_summary_detects_drop():
    items = [{"report_id": "r1", "mutation": "orig", "target": "", "valid": True, "total": 21, "scores": {d: 4 for d in J.DIMENSIONS} | {J.DIMENSIONS[0]: 5}},
             {"report_id": "r1", "mutation": "fake_speed", "target": J.DIMENSIONS[0], "valid": True, "total": 17,
              "scores": {d: 4 for d in J.DIMENSIONS} | {J.DIMENSIONS[0]: 1}}]
    s = J.summarize_calibration(items)
    assert s["fake_speed"] == {"n": 1, "mean_total_drop": 4.0, "target_dim_detect_rate": 1.0} and s["_orig_full_marks_rate"] == 0.0


# ---- 案件形态 ----
def test_scenario_is_deterministic_and_varied():
    assert scenario_for(3) == scenario_for(3)
    assert len({scenario_for(i)["type"] for i in range(20)}) == 20


def test_validate_accident_enforces_template_keys():
    ok, why = validate_accident({k: "x" for k in FIELDS[:12]})
    assert why == "ok" and set(ok) == set(FIELDS)
    assert validate_accident({"乱加的键": "x"})[1].startswith("unknown_keys")
    assert validate_accident({FIELDS[0]: "x"})[1].startswith("too_sparse")


def test_validate_guidance_shape_and_prefix():
    sk = guidance_skeleton()

    def fill(o):
        return {k: fill(v) for k, v in o.items()} if isinstance(o, dict) else "需要关注：核对该项"

    assert validate_guidance(fill(sk)) == "ok"
    bad = fill(sk)
    bad["元信息"]["可信度说明"] = "随便写"
    assert validate_guidance(bad) == "bad_value"
    assert validate_guidance({"元信息": {}}) == "keys_mismatch"


def test_ratio_claim_ignores_negations_but_catches_real_ratios(case):
    assert not G.RATIO_RE.search("现有材料不能判定责任比例,责任比例待证据核定,亦不作量化划分。")
    for pos in ("双方责任比例为7:3", "甲方承担 70% 责任", "按 6:4 划分"):
        assert G.RATIO_RE.search(pos), pos


def test_resolve_run_prefers_exact_tag(tmp_path, monkeypatch):
    from sr_eval import cli
    for n in ("20260101_000000_lunatrain", "20260101_000001_lunatrain2"):
        (tmp_path / n).mkdir()
    monkeypatch.setattr(cli, "RUNS_DIR", tmp_path)
    assert cli.resolve_run("lunatrain").name.endswith("_lunatrain")
    assert cli.resolve_run("lunatrain2").name.endswith("_lunatrain2")


def test_fact_coverage_counts_fields_reflected_in_report():
    acc = {"事故标题": "夜间路口两车碰撞", "天气": "大雨", "路表情况": "路面积水湿滑", "号牌号码": "已脱敏", "年龄": "", "伤害程度": "轻微擦伤"}
    full = "夜间路口两车碰撞,大雨天气,路面积水湿滑,当事人轻微擦伤。"
    cov = G.fact_coverage(acc, full)
    assert cov["fields"] == 4 and cov["covered"] == 4 and cov["ratio"] == 1.0   # 号牌号码/空字段不计
    part = G.fact_coverage(acc, "夜间路口两车碰撞,大雨天气。")
    assert part["ratio"] == 0.5 and set(part["missing"]) == {"路表情况", "伤害程度"}


def test_reasoning_metrics_and_loop_warning(case):
    loop = "我需要再核对一遍事故信息中的时间地点与车辆类型,然后继续检查证据。" * 80
    tr = trace(calls=[{"reasoning": loop}])
    res = G.evaluate(tr, case)
    assert res["metrics"]["reasoning_chars"] == len(loop) and res["metrics"]["reasoning_loop_12gram"] > 0.9
    assert "reasoning_loop" in codes(res, "warn")
    ok_ = G.evaluate(trace(calls=[{"reasoning": "先判断资料是否足够,再决定检索。"}]), case)
    assert ok_["metrics"]["reasoning_loop_12gram"] == 0.0 and "reasoning_loop" not in codes(ok_, "warn")
