from __future__ import annotations

from sr_eval.live.metrics import EVALUATION_FIELDS, FIELD_NAMES, evaluate_candidate


def _candidate(report: str, claims: list[dict], treatments: list[dict] | None = None) -> dict:
    return {
        "version": 1,
        "report_markdown": report,
        "claims": claims,
        "obligation_resolutions": treatments or [],
    }


def test_field_table_has_exactly_37_fields() -> None:
    assert len(FIELD_NAMES) == 37
    assert len(set(FIELD_NAMES)) == 37
    assert set(EVALUATION_FIELDS) == {"事故认定原因", "事故初查原因", "致死原因", "主要违法行为", "事故责任"}


def test_evaluation_field_requires_attribution_marker() -> None:
    case = {"accident_data": {"事故认定原因": "未按规定让行"}}
    report = "事故认定原因：未按规定让行。"
    claims = [{"claim_id": "c1", "type": "fact", "text_span": {"start": 0, "end": len(report)},
               "evidence_refs": ["accident:/事故认定原因"], "knowledge_refs": []}]
    result = evaluate_candidate(_candidate(report, claims), case)
    assert result["checks"]["evaluation_attribution"]["status"] == "fail"

    report = "事故信息记载未按规定让行，仍需结合现场材料核实。"
    claims[0]["text_span"] = {"start": 0, "end": len(report)}
    result = evaluate_candidate(_candidate(report, claims), case)
    assert result["checks"]["evaluation_attribution"]["status"] == "pass"


def test_empty_field_reference_is_not_accepted_as_fact() -> None:
    case = {"accident_data": {"年龄": ""}}
    report = "驾驶人年龄为30岁。"
    claims = [{"claim_id": "c1", "type": "fact", "text_span": {"start": 0, "end": len(report)},
               "evidence_refs": ["accident:/年龄"], "knowledge_refs": []}]
    result = evaluate_candidate(_candidate(report, claims), case)
    assert result["checks"]["empty_field_fabrication"]["status"] == "fail"
    assert result["strict_pass"] is False


def test_evidence_ref_alignment_rejects_wrong_field_value() -> None:
    case = {"accident_data": {"天气": "小雨"}}
    report = "天气：晴。"
    claims = [{"claim_id": "c1", "type": "fact", "text_span": {"start": 0, "end": len(report)},
               "evidence_refs": ["accident:/天气"], "knowledge_refs": []}]
    result = evaluate_candidate(_candidate(report, claims), case)
    assert result["checks"]["evidence_ref_alignment"]["status"] == "fail"


def test_evidence_ref_alignment_does_not_borrow_value_from_later_sentence() -> None:
    case = {"accident_data": {"天气": "小雨"}}
    report = "天气：晴。其他记录为小雨。"
    claims = [{"claim_id": "c1", "type": "fact", "text_span": {"start": 0, "end": len(report)},
               "evidence_refs": ["accident:/天气"], "knowledge_refs": []}]
    result = evaluate_candidate(_candidate(report, claims), case)
    assert result["checks"]["evidence_ref_alignment"]["status"] == "fail"


def test_evaluation_attribution_marker_must_be_in_target_sentence() -> None:
    case = {"accident_data": {"事故认定原因": "未按规定让行"}}
    report = "事故认定原因：未按规定让行。材料显示天气为小雨。"
    claims = [{"claim_id": "c1", "type": "fact", "text_span": {"start": 0, "end": len(report)},
               "evidence_refs": ["accident:/事故认定原因"], "knowledge_refs": []}]
    result = evaluate_candidate(_candidate(report, claims), case)
    assert result["checks"]["evaluation_attribution"]["status"] == "fail"

    report = "未按规定让行。材料显示事故认定原因仍需结合记录核实。"
    claims[0]["text_span"] = {"start": 0, "end": len(report)}
    result = evaluate_candidate(_candidate(report, claims), case)
    assert result["checks"]["evaluation_attribution"]["status"] == "unknown"


def test_conflict_is_unknown_without_explicit_case_metadata() -> None:
    case = {"accident_data": {"事故形态": "侧面碰撞", "事故类型": "追尾"}}
    report = "材料显示事故形态仍待核实。"
    result = evaluate_candidate(_candidate(report, [], []), case)
    assert result["checks"]["conflict_disclosure"]["status"] == "unknown"


def test_explicit_conflict_requires_disclosure_marker() -> None:
    case = {"accident_data": {"事故形态": "侧面碰撞"}, "conflicts": [{"fields": ["事故形态"]}]}
    no_marker = evaluate_candidate(_candidate("事故形态为侧面碰撞。", [], []), case)
    assert no_marker["checks"]["conflict_disclosure"]["status"] == "fail"
    with_marker = evaluate_candidate(_candidate("事故形态存在两种记载，需核对。", [], []), case)
    assert with_marker["checks"]["conflict_disclosure"]["status"] == "pass"


def test_production_field_conflict_pointer_is_supported() -> None:
    case = {"accident_data": {"天气": "小雨"},
            "snapshot": {"supplemental_records": [{"field_conflicts": [{"accident_field": "/天气", "explanation": "两条记录不同"}]}]}}
    report = "天气字段存在冲突，需核对两条记录。"
    result = evaluate_candidate(_candidate(report, [], []), case)
    assert result["checks"]["conflict_disclosure"]["status"] == "pass"


def test_obligation_status_is_conservative_when_mapping_missing() -> None:
    case = {"accident_data": {"事故标题": "测试事故"}}
    result = evaluate_candidate(_candidate("测试事故。", [], []), case)
    assert result["checks"]["obligation_consistency"]["status"] == "unknown"
    assert result["strict_pass"] is False


def test_obligation_duplicates_and_unknown_ids_are_hard_fail() -> None:
    case = {"accident_data": {"天气": "小雨"}}
    report = "天气：小雨。"
    claims = [{"claim_id": "c1", "type": "fact", "text_span": {"start": 0, "end": len(report)},
               "evidence_refs": ["accident:/天气"], "knowledge_refs": []}]
    treatments = [
        {"obligation_id": "accident:/天气", "treatment": "covered", "resolution": "正文已覆盖。"},
        {"obligation_id": "fact:天气", "treatment": "uncertain", "resolution": "信息不足，不能补造。"},
        {"obligation_id": "unknown:/天气", "treatment": "covered", "resolution": "错误映射。"},
    ]
    result = evaluate_candidate(_candidate(report, claims, treatments), case)
    check = result["checks"]["obligation_consistency"]
    assert check["status"] == "fail"
    assert check["fields"]["天气"]["status"] == "fail"
    assert check["fields"]["__unknown_ids__"]["status"] == "fail"


def _oblig(field: str, treatment: str, resolution: str) -> dict:
    return {"obligation_id": f"fact:/{field}", "treatment": treatment, "resolution": resolution}


def _field_result(case: dict, item: dict, field: str) -> dict:
    result = evaluate_candidate(_candidate("正文。", [], [item]), case)
    return result["checks"]["obligation_consistency"]["fields"][field]


def test_present_field_declared_missing_is_fail() -> None:
    # Codex 抽检发现的系统性错误:字段有值却写"该字段未提供"
    case = {"accident_data": {"交通方式": "机动车通行"}}
    bad = _field_result(case, _oblig("交通方式", "not_relevant", "该字段未提供，且不影响本案分析。"), "交通方式")
    assert bad["status"] == "fail" and "非空字段" in bad["reason"]
    # 说明里明确引用字段值(承认它有值)则不触发
    ok = _field_result(case, _oblig("交通方式", "not_relevant", "机动车通行已记载，但不影响本案分析；缺少车速信息。"), "交通方式")
    assert ok["status"] != "fail"


def test_key_outcome_field_empty_cannot_be_not_relevant() -> None:
    case = {"accident_data": {"事故责任": ""}}
    bad = _field_result(case, _oblig("事故责任", "not_relevant", "字段为空，不影响分析。"), "事故责任")
    assert bad["status"] == "fail" and "关键结论字段" in bad["reason"]
    good = _field_result(case, _oblig("事故责任", "uncertain", "事故责任字段为空，报告不作责任结论。"), "事故责任")
    assert good["status"] == "pass"
    other = {"accident_data": {"年龄": ""}}
    still_ok = _field_result(other, _oblig("年龄", "not_relevant", "字段为空，不影响判断。"), "年龄")
    assert still_ok["status"] == "pass"
