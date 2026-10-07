"""线上 CandidateReport 的可解释解读质量指标。

本模块只做确定性、证据保守的检查。无法从结构化案件和候选报告中确认的
语义不会被猜成通过，而是返回 ``unknown``。这样指标可用于筛选和复核，
不能替代独立审查者或人工核验。
"""

from __future__ import annotations

import re
import hashlib
from typing import Any, Iterable


FIELD_SPECS: tuple[dict[str, Any], ...] = (
    {"name": "事故标题", "kind": "fact", "group": "identity", "sensitive": False, "handling": "可直接引用；不得把标题扩写成未给出的事实。"},
    {"name": "事故形态", "kind": "fact", "group": "event", "sensitive": False, "handling": "按原字段描述事故形态；与其他字段冲突时标明待核实。"},
    {"name": "单车事故", "kind": "fact", "group": "participants", "sensitive": False, "handling": "仅写明输入中是否为单车事故。"},
    {"name": "天气", "kind": "fact", "group": "environment", "sensitive": False, "handling": "作为环境事实；不得据此臆造能见度、附着条件或责任。"},
    {"name": "车辆类型", "kind": "fact", "group": "vehicle", "sensitive": False, "handling": "按车辆类型描述，不补造品牌、型号或性能。"},
    {"name": "交通方式", "kind": "fact", "group": "participants", "sensitive": False, "handling": "按材料写明机动车、非机动车、行人等。"},
    {"name": "现场形态", "kind": "fact", "group": "scene", "sensitive": False, "handling": "作为现场观察事实；几何关系不足时保留缺口。"},
    {"name": "路表情况", "kind": "fact", "group": "road", "sensitive": False, "handling": "按原值写明路面状况，不从天气推导未记载状态。"},
    {"name": "照明条件", "kind": "fact", "group": "environment", "sensitive": False, "handling": "作为光照事实，不推导实际视距。"},
    {"name": "性别", "kind": "fact", "group": "person", "sensitive": True, "handling": "仅在必要时引用；不得由称谓或角色猜测。"},
    {"name": "信号灯", "kind": "fact", "group": "traffic_control", "sensitive": False, "handling": "按材料描述信号状态；相位未知时标待核实。"},
    {"name": "车辆间事故", "kind": "fact", "group": "participants", "sensitive": False, "handling": "按原字段描述接触关系，不把单车字段强行改写。"},
    {"name": "道路线型", "kind": "fact", "group": "road", "sensitive": False, "handling": "只引用已给线形，缺少测量时不补半径、坡度等。"},
    {"name": "是否载运危险品", "kind": "fact", "group": "hazmat", "sensitive": False, "handling": "只说明载运状态；危险品后果另按字段处理。"},
    {"name": "事故发生时间", "kind": "fact", "group": "time_place", "sensitive": False, "handling": "按原时间粒度引用，不能补出精确时分。"},
    {"name": "能见度", "kind": "fact", "group": "environment", "sensitive": False, "handling": "仅写明记录值；不要从雾雨直接生成数值。"},
    {"name": "号牌号码", "kind": "fact", "group": "identity", "sensitive": True, "handling": "默认脱敏或不引用；不得创造真实号牌。"},
    {"name": "路口路段类型", "kind": "fact", "group": "road", "sensitive": False, "handling": "按材料写明路口或路段类型。"},
    {"name": "事故类型", "kind": "fact", "group": "event", "sensitive": False, "handling": "作为输入分类使用，不扩大为未证实的碰撞过程。"},
    {"name": "交通信号方式", "kind": "fact", "group": "traffic_control", "sensitive": False, "handling": "只引用已有信号方式，未知时保留未知。"},
    {"name": "道路类型", "kind": "fact", "group": "road", "sensitive": False, "handling": "按原字段描述道路，不补道路等级或限速。"},
    {"name": "事故认定原因", "kind": "evaluation", "group": "assessment", "sensitive": False, "handling": "必须归因于事故信息记载、认定材料或初查记录，不写成独立查明事实。"},
    {"name": "地点（包括路名，路号）", "kind": "fact", "group": "time_place", "sensitive": True, "handling": "使用输入地点；合成案地点不得替换成真实地址。"},
    {"name": "财产损失", "kind": "fact", "group": "consequence", "sensitive": False, "handling": "按记录写明损失，不补金额或鉴定结论。"},
    {"name": "年龄", "kind": "fact", "group": "person", "sensitive": True, "handling": "仅引用明确年龄；不得从外观、称谓推断。"},
    {"name": "驾龄", "kind": "fact", "group": "person", "sensitive": True, "handling": "仅引用明确驾龄，不把驾龄直接等同于过错。"},
    {"name": "安全带/头盔使用情况", "kind": "fact", "group": "person", "sensitive": False, "handling": "按记载引用，未知时写待核实。"},
    {"name": "逃逸事故是否侦破", "kind": "fact", "group": "procedure", "sensitive": False, "handling": "只描述程序状态，不据此推断责任。"},
    {"name": "事故初查原因", "kind": "evaluation", "group": "assessment", "sensitive": False, "handling": "明确标为初查或材料记载，不能当作最终原因。"},
    {"name": "受伤部位", "kind": "fact", "group": "injury", "sensitive": True, "handling": "仅引用已记载部位，不补医学判断。"},
    {"name": "伤害程度", "kind": "fact", "group": "injury", "sensitive": True, "handling": "按输入程度描述，不自行升级或减轻。"},
    {"name": "致死原因", "kind": "evaluation", "group": "injury", "sensitive": True, "handling": "仅在材料明确时引用，并归因于记录或鉴定。"},
    {"name": "主要违法行为", "kind": "evaluation", "group": "assessment", "sensitive": False, "handling": "作为待核实或材料记载的评价，不独立定性。"},
    {"name": "事故责任", "kind": "evaluation", "group": "assessment", "sensitive": False, "handling": "不得输出最终定责；只能写责任分析边界或材料状态。"},
    {"name": "参与者数目", "kind": "fact", "group": "participants", "sensitive": False, "handling": "按输入计数，不由事故类型推算。"},
    {"name": "事故人数目", "kind": "fact", "group": "participants", "sensitive": False, "handling": "按输入计数；与参与者数目不一致时应标冲突。"},
    {"name": "危险品事故后果", "kind": "fact", "group": "hazmat", "sensitive": False, "handling": "只引用已记载后果，不从是否载运危险品推导泄漏或燃烧。"},
)

FIELD_NAMES = tuple(spec["name"] for spec in FIELD_SPECS)
FIELD_SPEC_BY_NAME = {spec["name"]: spec for spec in FIELD_SPECS}
FACT_FIELDS = tuple(spec["name"] for spec in FIELD_SPECS if spec["kind"] == "fact")
EVALUATION_FIELDS = tuple(spec["name"] for spec in FIELD_SPECS if spec["kind"] == "evaluation")
SENSITIVE_FIELDS = tuple(spec["name"] for spec in FIELD_SPECS if spec["sensitive"])

_ATTRIBUTION_MARKERS = (
    "事故信息记载", "材料显示", "初查记录", "调查记录", "认定材料", "记录显示",
    "已提供材料", "输入记载", "鉴定记录", "报告记载", "据记载", "待核实",
)
_MISSING_MARKERS = ("信息不足", "未提供", "未记载", "缺少", "待核实", "无法确认", "未知", "空缺", "输入为空",
                    "为空", "空白", "无记载", "没有记载", "未见记载", "未填写", "未给出", "缺失")  # 后 8 项:模型常写"该字段为空"等,同属对缺失的明示
_KEY_OUTCOME_FIELDS = ("事故认定原因", "事故初查原因", "主要违法行为", "事故责任")
_CONFLICT_MARKERS = ("冲突", "矛盾", "不一致", "两种记载", "存在差异", "需核对", "待核对")
_OBLIGATION_PREFIXES = ("fact_obligation:", "fact-obligation:", "obligation:", "fact:")


def _dump(value: Any) -> Any:
    if isinstance(value, dict):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    return value


def accident_data(case: dict[str, Any] | Any) -> dict[str, str]:
    """从 case/snapshot 的常见包装中取得 37 字段；不改变输入。"""
    obj = _dump(case) or {}
    for key in ("accident_data", "accident", "snapshot", "data"):
        nested = obj.get(key) if isinstance(obj, dict) else None
        if isinstance(nested, dict):
            if isinstance(nested.get("accident_data"), dict):
                nested = nested["accident_data"]
            if any(k in nested for k in FIELD_NAMES):
                obj = nested
                break
    return {name: str(obj.get(name, "") or "") for name in FIELD_NAMES} if isinstance(obj, dict) else {name: "" for name in FIELD_NAMES}


def _claims(candidate: dict[str, Any] | Any) -> list[dict[str, Any]]:
    obj = _dump(candidate) or {}
    return [_dump(item) for item in (obj.get("claims") or []) if isinstance(_dump(item), dict)]


def _report(candidate: dict[str, Any] | Any) -> str:
    obj = _dump(candidate) or {}
    return str(obj.get("report_markdown", "") or "")


def _field_from_ref(ref: Any) -> str | None:
    if not isinstance(ref, str):
        return None
    text = ref.strip()
    for prefix in ("accident:/", "evidence:/", "snapshot:/", "/"):
        if text.startswith(prefix):
            raw = text[len(prefix):]
            direct = raw.replace("~1", "/").replace("~0", "~")
            if direct in FIELD_SPEC_BY_NAME:
                return direct
            token = raw.split("/", 1)[0].replace("~1", "/").replace("~0", "~")
            return token if token in FIELD_SPEC_BY_NAME else None
    if text in FIELD_SPEC_BY_NAME:
        return text
    for name in FIELD_NAMES:
        if text.endswith("/" + name) or text.endswith(":" + name):
            return name
    return None


def _claim_text(claim: dict[str, Any], report: str) -> str:
    span = claim.get("text_span") or {}
    try:
        start, end = int(span.get("start", 0)), int(span.get("end", 0))
    except (TypeError, ValueError):
        return ""
    if 0 <= start < end <= len(report):
        return report[start:end]
    return ""


def _sentence_parts(text: str) -> list[str]:
    """按可见标点切成局部句，避免把后文证据借给前文断言。"""
    return [part for part in re.split(r"(?<=[。！？!?；;\n])", text) if part.strip()]


def _target_sentence(text: str, field: str, value: str) -> str | None:
    """定位唯一的字段/值局部句；目标不唯一时返回 unknown 所需的 None。"""
    parts = _sentence_parts(text)
    field_hits = [part for part in parts if field in part]
    if field_hits:
        return field_hits[0] if len(field_hits) == 1 and (not value or value in field_hits[0]) else None
    value_hits = [part for part in parts if value and value in part]
    if len(value_hits) == 1:
        return value_hits[0]
    return None


def _explicit_field_value(text: str, field: str) -> str | None:
    """读取“字段：值”的第一段值；没有结构化标签时保留 unknown。"""
    match = re.search(
        rf"{re.escape(field)}\s*[：:]\s*([^。！？!?；;\n]+)",
        text,
    )
    if not match:
        return None
    return match.group(1).strip().strip("，, ")


def _evidence_index(case: dict[str, Any] | Any) -> dict[str, dict[str, Any]]:
    obj = _dump(case) or {}
    values: list[Any] = []
    for key in ("evidence", "evidence_records", "evidence_store"):
        value = obj.get(key) if isinstance(obj, dict) else None
        if isinstance(value, list):
            values.extend(value)
        elif isinstance(value, dict):
            values.extend(value.values())
    snap = obj.get("snapshot") if isinstance(obj, dict) else None
    if isinstance(snap, dict):
        values.extend(snap.get("evidence") or snap.get("evidence_records") or [])
    out: dict[str, dict[str, Any]] = {}
    for item in values:
        item = _dump(item)
        if isinstance(item, dict):
            ident = item.get("id") or item.get("evidence_id") or item.get("ref")
            if ident is not None:
                out[str(ident)] = item
    return out


def _explicit_conflicts(case: dict[str, Any] | Any) -> list[dict[str, Any]]:
    obj = _dump(case) or {}
    values: list[Any] = []
    for holder in (obj, obj.get("snapshot") if isinstance(obj, dict) else None):
        if not isinstance(holder, dict):
            continue
        values.extend(holder.get("conflicts") or holder.get("field_conflicts") or [])
        records = holder.get("supplemental_records") or []
        if isinstance(records, list):
            for record in records:
                record = _dump(record)
                if isinstance(record, dict):
                    values.extend(record.get("field_conflicts") or [])
    result = []
    for item in values:
        if isinstance(item, str):
            result.append({"fields": [item]})
            continue
        item = _dump(item)
        if isinstance(item, dict):
            fields = item.get("fields") or item.get("field_names") or item.get("accident_fields") or []
            if not fields and item.get("accident_field"):
                fields = [item.get("accident_field")]
            if isinstance(fields, str):
                fields = [fields]
            normalized_fields = []
            for field in fields:
                field_text = str(field)
                normalized_fields.append(_field_from_ref(field_text) or field_text)
            result.append({**item, "fields": normalized_fields})
    return result


def _status(statuses: Iterable[str]) -> str:
    values = set(statuses)
    if "fail" in values:
        return "fail"
    if "unknown" in values:
        return "unknown"
    return "pass"


def _result(status: str, **extra: Any) -> dict[str, Any]:
    return {"status": status, **extra}


def _used_fields(claims: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {name: [] for name in FIELD_NAMES}
    for claim in claims:
        refs = (claim.get("evidence_refs") or []) + (claim.get("knowledge_refs") or [])
        for ref in refs:
            field = _field_from_ref(ref)
            if field:
                out[field].append(claim)
    return out


def _alignment_for_claim(field: str, value: str, claim: dict[str, Any], report: str,
                         evidence: dict[str, dict[str, Any]]) -> str:
    """判断引用是否实际支撑字段值；语义改写保留 unknown。"""
    text = _claim_text(claim, report)
    if not text.strip():
        return "fail"
    explicit = _explicit_field_value(text, field)
    if explicit is not None:
        # 只接受字段标签后局部值；全文后文再次出现真值不能覆盖前面的冲突值。
        return "pass" if value and value in explicit else "fail"
    local = _target_sentence(text, field, value)
    if local is not None and value and value in local:
        return "pass"
    if field in text:
        # 字段被明确提及但值未能在同一局部句确认时，不能靠证据全文命中冒充语义对齐。
        return "fail" if any(ch in text for ch in "：:") else "unknown"
    # 直接证据记录可支持不逐字复述的 claim；只有记录明确包含值才判 pass。
    for ref in (claim.get("evidence_refs") or []):
        item = evidence.get(str(ref))
        if isinstance(item, dict):
            evidence_text = str(item.get("text") or item.get("content") or "")
            if value and value in evidence_text:
                return "pass"
    return "unknown"


def _obligation_field(obligation_id: Any) -> str | None:
    """将义务 ID 映射到唯一字段；未知 ID 不得静默落到首个匹配。"""
    norm = str(obligation_id or "")
    for prefix in _OBLIGATION_PREFIXES:
        if norm.startswith(prefix):
            norm = norm[len(prefix):]
            break
    if norm.startswith("accident:/"):
        norm = norm.split("/", 1)[1]
    norm = norm.replace("~1", "/").replace("~0", "~")
    norm = norm.lstrip("/")
    return norm if norm in FIELD_SPEC_BY_NAME else None


def _obligation_matches(items: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    return [item for item in items if _obligation_field(item.get("obligation_id")) == field]


def _obligation_for(items: list[dict[str, Any]], field: str) -> dict[str, Any] | None:
    matches = _obligation_matches(items, field)
    return matches[0] if matches else None


def _state_for(field: str, value: str, conflicts: list[dict[str, Any]]) -> str:
    if any(field in item.get("fields", []) for item in conflicts):
        return "conflict"
    return "present" if value.strip() else "missing"


def evaluate_candidate(candidate: dict[str, Any] | Any, case: dict[str, Any] | Any) -> dict[str, Any]:
    """计算 CandidateReport + 案件的保守指标。

    返回值只使用 ``pass``、``fail``、``unknown`` 三态；``overall`` 只有在
    所有适用检查均为 pass 且没有 unknown 时才是 pass。
    """
    data = accident_data(case)
    report = _report(candidate)
    claims = _claims(candidate)
    used = _used_fields(claims)
    conflicts = _explicit_conflicts(case)
    candidate_obj = _dump(candidate) or {}

    coverage_details: dict[str, dict[str, Any]] = {}
    coverage_states: list[str] = []
    evidence = _evidence_index(case)
    alignment_details: dict[str, dict[str, Any]] = {}
    alignment_states: list[str] = []
    for spec in FIELD_SPECS:
        field, value = spec["name"], data[spec["name"]]
        if spec["kind"] != "fact" or not value.strip():
            continue
        refs = used[field]
        if refs:
            claim_states = [_alignment_for_claim(field, value, claim, report, evidence) for claim in refs]
            state = _status(claim_states)
            for claim, claim_state in zip(refs, claim_states):
                alignment_states.append(claim_state)
            alignment_details[field] = {"status": state, "claim_ids": [str(c.get("claim_id", "")) for c in refs],
                                        "claim_statuses": claim_states}
        else:
            # 语义改写可能不复述原值，不能把缺少直接引用冒充漏写或通过。
            state = "unknown"
            alignment_states.append(state)
            alignment_details[field] = {"status": state, "claim_ids": []}
        coverage_states.append(state)
        coverage_details[field] = {"status": state, "claim_ids": [str(c.get("claim_id", "")) for c in refs]}
    coverage = _result(_status(coverage_states) if coverage_states else "unknown",
                       applicable=len(coverage_states),
                       passed=sum(s == "pass" for s in coverage_states),
                       failed=sum(s == "fail" for s in coverage_states),
                       unknown=sum(s == "unknown" for s in coverage_states),
                       fields=coverage_details)
    evidence_alignment = _result(_status(alignment_states) if alignment_states else "unknown",
                                 applicable=len(alignment_states),
                                 passed=sum(s == "pass" for s in alignment_states),
                                 failed=sum(s == "fail" for s in alignment_states),
                                 unknown=sum(s == "unknown" for s in alignment_states),
                                 fields=alignment_details,
                                 note="只有字段值或已登记证据文本能直接支撑时才判 pass；语义改写保留 unknown。")

    hidden_values: dict[str, Any] = {}
    obj = _dump(case) or {}
    for key in ("hidden_values", "latent_fields", "forbidden_values"):
        value = obj.get(key) if isinstance(obj, dict) else None
        if isinstance(value, dict):
            hidden_values.update(value)
    fabrication_states: list[str] = []
    fabrication_details: dict[str, dict[str, Any]] = {}
    for spec in FIELD_SPECS:
        field = spec["name"]
        if data[field].strip():
            continue
        refs = used[field]
        state = "unknown"
        if refs:
            state = "fail" if any(not any(m in _claim_text(c, report) for m in _MISSING_MARKERS) for c in refs) else "pass"
        latent = hidden_values.get(field)
        if isinstance(latent, str) and latent.strip() and latent in report:
            state = "fail"
        elif isinstance(latent, (list, tuple)) and any(str(v).strip() and str(v) in report for v in latent):
            state = "fail"
        fabrication_states.append(state)
        fabrication_details[field] = {"status": state, "claim_ids": [str(c.get("claim_id", "")) for c in refs]}
    fabrication = _result(_status(fabrication_states) if fabrication_states else "unknown",
                          applicable=len(fabrication_states),
                          failed=sum(s == "fail" for s in fabrication_states),
                          unknown=sum(s == "unknown" for s in fabrication_states),
                          fields=fabrication_details,
                          note="空字段没有隐含真值时，无法从字符串安全判定所有编造，故保留 unknown。")

    attribution_states: list[str] = []
    attribution_details: dict[str, dict[str, Any]] = {}
    for field in EVALUATION_FIELDS:
        value = data[field]
        if not value.strip():
            continue
        refs = used[field]
        if not refs:
            state = "unknown"
        else:
            claim_states: list[str] = []
            for claim in refs:
                text = _claim_text(claim, report)
                if not text.strip():
                    claim_states.append("fail")
                    continue
                local = _target_sentence(text, field, value)
                if local is None:
                    claim_states.append("unknown")
                elif any(marker in local for marker in _ATTRIBUTION_MARKERS):
                    claim_states.append("pass")
                else:
                    claim_states.append("fail")
            state = _status(claim_states)
        attribution_states.append(state)
        attribution_details[field] = {"status": state, "claim_ids": [str(c.get("claim_id", "")) for c in refs]}
    attribution = _result(_status(attribution_states) if attribution_states else "unknown",
                          applicable=len(attribution_states),
                          passed=sum(s == "pass" for s in attribution_states),
                          failed=sum(s == "fail" for s in attribution_states),
                          unknown=sum(s == "unknown" for s in attribution_states),
                          fields=attribution_details)

    obligations = [item for item in (candidate_obj.get("obligation_resolutions") or []) if isinstance(item, dict)]
    obligations_by_field = {field: _obligation_matches(obligations, field) for field in FIELD_NAMES}
    unknown_obligations = [item for item in obligations if _obligation_field(item.get("obligation_id")) is None]
    obligation_states: list[str] = []
    obligation_details: dict[str, dict[str, Any]] = {}
    for spec in FIELD_SPECS:
        field, value = spec["name"], data[spec["name"]]
        matches = obligations_by_field[field]
        state = _state_for(field, value, conflicts)
        if not matches:
            result = "unknown"
            detail = {"status": result, "state": state, "reason": "未找到可映射的义务处置。"}
        elif len(matches) > 1:
            result = "fail"
            detail = {"status": result, "state": state, "reason": "同一字段存在重复义务处置。",
                      "obligation_ids": [str(item.get("obligation_id", "")) for item in matches]}
        else:
            item = matches[0]
            treatment = str(item.get("treatment", ""))
            resolution = str(item.get("resolution", ""))
            # 2026-10-06 按线上 generator.md 语义放宽(原规则要求空字段必须 uncertain+关键词,会把线上明确允许的
            # "对不影响判断的空缺字段给出不相关理由"误判为失败,导致约半数教师修订被拒):
            #  - 非空且被断言引用:covered,或(评价/敏感类字段)uncertain 且有说明 → 通过;
            #  - 空字段:uncertain(带缺口说明)或 not_relevant(带理由)→ 通过;
            #  - 冲突字段:仍必须 uncertain 且明确指出冲突;
            #  - 其余(例如把已被引用的事实字段标成 not_relevant、空字段标成 covered)仍判失败,种子错误对依赖这一点。
            has_text = bool(resolution.strip())
            detail_reason = None
            hint = re.sub(r"[\s，。,.;；:：、（）()]", "", value)[:6]
            if (state == "present" and treatment in ("not_relevant", "uncertain") and has_text
                    and any(m in resolution for m in _MISSING_MARKERS) and not (hint and hint in resolution)):
                # 2026-10-07 Codex 抽检发现的系统性错误:字段有值(如交通方式=机动车通行)却写"该字段未提供"。
                result = "fail"
                detail_reason = "非空字段被说成缺失/未提供。"
            elif state == "missing" and treatment == "not_relevant" and field in _KEY_OUTCOME_FIELDS:
                # 事故认定原因/初查原因/主要违法行为/事故责任为空时属于影响判断的缺口,应写 uncertain 而不是"不相关"。
                result = "fail"
                detail_reason = "关键结论字段为空应标不确定,不可标不相关。"
            elif state == "present" and used[field] and treatment == "covered" and has_text:
                result = "pass"
            elif (state == "present" and used[field] and treatment == "uncertain" and has_text
                  and FIELD_SPEC_BY_NAME[field]["kind"] != "fact"):
                result = "pass"
            elif state == "missing" and has_text and (
                    treatment == "not_relevant"
                    or (treatment == "uncertain" and any(m in resolution for m in _MISSING_MARKERS + _CONFLICT_MARKERS))):
                result = "pass"
            elif state == "conflict" and treatment == "uncertain" and any(m in resolution for m in _MISSING_MARKERS + _CONFLICT_MARKERS):
                result = "pass"
            elif state == "present" and not used[field]:
                result = "unknown"
            else:
                result = "fail"
            detail = {"status": result, "state": state, "treatment": treatment, "resolution": resolution}
            if result == "fail" and detail_reason:
                detail["reason"] = detail_reason
        obligation_states.append(result)
        obligation_details[field] = detail
    if unknown_obligations:
        obligation_states.append("fail")
        obligation_details["__unknown_ids__"] = {
            "status": "fail",
            "reason": "存在无法映射到37字段的义务 ID。",
            "obligation_ids": [str(item.get("obligation_id", "")) for item in unknown_obligations],
        }
    obligations_result = _result(_status(obligation_states) if obligation_states else "unknown",
                                  applicable=len(obligation_states),
                                  passed=sum(s == "pass" for s in obligation_states),
                                  failed=sum(s == "fail" for s in obligation_states),
                                  unknown=sum(s == "unknown" for s in obligation_states),
                                  fields=obligation_details)

    conflict_details: list[dict[str, Any]] = []
    conflict_states: list[str] = []
    for conflict in conflicts:
        fields = [f for f in conflict.get("fields", []) if f in FIELD_SPEC_BY_NAME]
        if not fields:
            conflict_states.append("unknown")
            conflict_details.append({"status": "unknown", "fields": fields})
            continue
        local_texts: list[str] = []
        for field in fields:
            local_texts.extend(_claim_text(claim, report) for claim in used[field])
            local_texts.extend(
                str(item.get("resolution", "")) for item in obligations
                if _obligation_for([item], field) is not None
            )
            # 没有结构化 claim 时，只查看字段标签附近窗口，不接受全文任意 marker。
            pos = report.find(field)
            if pos >= 0:
                local_texts.append(report[max(0, pos - 80):pos + len(field) + 160])
        if any(marker in text for text in local_texts for marker in _CONFLICT_MARKERS):
            state = "pass"
        else:
            state = "fail"
        conflict_states.append(state)
        conflict_details.append({"status": state, "fields": fields})
    conflict_result = _result(_status(conflict_states) if conflict_states else "unknown",
                              applicable=len(conflict_states), fields=conflict_details,
                              note="只有案件显式提供 conflicts/field_conflicts 时才判定冲突；语义冲突未被猜测。")

    # 隐私门只在案件提供隐藏值/禁止值时有可验证真值；否则不制造“无泄漏”硬通过。
    privacy_values: list[str] = []
    obj = _dump(case) or {}
    if isinstance(obj, dict):
        for key in ("hidden_values", "private_values", "forbidden_values"):
            value = obj.get(key)
            if isinstance(value, dict):
                privacy_values.extend(str(item) for item in value.values() if str(item).strip())
            elif isinstance(value, (list, tuple)):
                privacy_values.extend(str(item) for item in value if str(item).strip())
    privacy_states: list[str] = []
    privacy_details: list[dict[str, Any]] = []
    for value in privacy_values:
        state = "fail" if value in report else "pass"
        privacy_states.append(state)
        privacy_details.append({"status": state, "value_digest": hashlib.sha256(value.encode("utf-8")).hexdigest()})
    # 合成号牌等明显模式只作为发现信号，不能代替真实隐藏值；发现时直接隔离。
    if re.search(r"(?<![A-Za-z0-9])[京沪津渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼][A-Z][A-Z0-9]{5}(?![A-Za-z0-9])", report):
        privacy_states.append("fail")
        privacy_details.append({"status": "fail", "reason": "检测到未脱敏号牌形态"})
    privacy_result = _result(_status(privacy_states) if privacy_states else "unknown",
                             applicable=len(privacy_states), fields=privacy_details,
                             note="没有隐藏值 oracle 时不伪造隐私通过；显式泄漏或明显未脱敏号牌直接失败。")

    checks = {
        "nonempty_fact_coverage": coverage,
        "evidence_ref_alignment": evidence_alignment,
        "empty_field_fabrication": fabrication,
        "evaluation_attribution": attribution,
        "obligation_consistency": obligations_result,
        "conflict_disclosure": conflict_result,
        "privacy_safety": privacy_result,
    }
    statuses = [item["status"] for item in checks.values() if item["applicable"] > 0]
    overall = _status(statuses) if statuses else "unknown"
    return {
        "schema_version": 1,
        "overall": overall,
        "strict_pass": overall == "pass",
        "checks": checks,
        "field_count": len(FIELD_NAMES),
        "nonempty_fields": sum(bool(value.strip()) for value in data.values()),
        "unknown_is_not_pass": True,
    }


compute_quality_metrics = evaluate_candidate
compute_metrics = evaluate_candidate


__all__ = [
    "FIELD_SPECS", "FIELD_NAMES", "FACT_FIELDS", "EVALUATION_FIELDS", "SENSITIVE_FIELDS",
    "accident_data", "evaluate_candidate", "compute_quality_metrics", "compute_metrics",
]
