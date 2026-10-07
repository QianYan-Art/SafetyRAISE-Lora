"""从已通过的 live 候选构造单类、可验证的合成错误对。

本模块只处理明确标记为合成且已经通过 ``revise.evaluate_response`` 的输入。
每个 rejected 都会重新走同一个评估入口；无法证明只有目标错误、无法读取真实
token 数，或长度比例不在窗口内时，宁可丢弃该种子，不以字符数或猜测替代证据。
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Iterable

from . import metrics
from .revise import evaluate_response
from .samples import render_call_sample


FINAL_ERROR_CLASSES = (
    "omit_key_fact",
    "empty_field_fabrication",
    "evaluation_as_fact",
    "wrong_evidence_ref",
    "obligation_mismatch",
    "swallow_conflict",
    "unread_source",
    "rule_excerpt_basis",
    "quote_not_unique",
)
TOOL_ERROR_CLASSES = (
    "tool_parameter_overlimit",
    "unnecessary_retrieval",
    "missing_required_retrieval",
)
ERROR_CLASSES = FINAL_ERROR_CLASSES + TOOL_ERROR_CLASSES

_ALIASES = {
    "missing_key_fact": "omit_key_fact",
    "key_fact_omission": "omit_key_fact",
    "omit_fact": "omit_key_fact",
    "fabricate_empty_field": "empty_field_fabrication",
    "empty_fabrication": "empty_field_fabrication",
    "evaluation_field_as_fact": "evaluation_as_fact",
    "wrong_ref": "wrong_evidence_ref",
    "obligation_treatment_mismatch": "obligation_mismatch",
    "obligation_resolution_mismatch": "obligation_mismatch",
    "swallowed_conflict": "swallow_conflict",
    "conflict_swallowed": "swallow_conflict",
    "rule_excerpt_as_evidence": "rule_excerpt_basis",
    "rule_excerpt": "rule_excerpt_basis",
    "quote_non_unique": "quote_not_unique",
    "tool_params_overlimit": "tool_parameter_overlimit",
    "tool_parameter_violation": "tool_parameter_overlimit",
    "unnecessary_search": "unnecessary_retrieval",
    "missing_read": "missing_required_retrieval",
    "required_retrieval_missing": "missing_required_retrieval",
}

_ATTRIBUTION_MARKERS = (
    "事故信息记载", "材料显示", "初查记录", "调查记录", "认定材料", "记录显示",
    "已提供材料", "输入记载", "鉴定记录", "报告记载", "据记载", "待核实",
)
_CONFLICT_MARKERS = ("冲突", "矛盾", "不一致", "两种记载", "存在差异", "需核对", "待核对")
_MISSING_MARKERS = ("信息不足", "未提供", "未记载", "缺少", "待核实", "无法确认", "未知", "空缺", "输入为空",
                    "为空", "空白", "无记载", "没有记载", "未见记载", "未填写", "未给出", "缺失")  # 后 8 项:模型常写"该字段为空"等,同属对缺失的明示


def _json_text(value: Any) -> str:
    # 沿用线上普通 ``json.dumps`` 的空格风格，避免把 JSON 压缩本身误作偏好信号。
    return json.dumps(value, ensure_ascii=False)


def _dump(value: Any) -> Any:
    if isinstance(value, dict):
        return value
    method = getattr(value, "model_dump", None)
    return method(mode="json") if callable(method) else value


def _canonical_class(value: str) -> str | None:
    text = str(value).strip()
    if text in ERROR_CLASSES:
        return text
    return _ALIASES.get(text)


def _trace_case(trace: dict[str, Any]) -> dict[str, Any]:
    case = trace.get("case") if isinstance(trace, dict) else None
    return case if isinstance(case, dict) else {}


def _wire_from_baseline(result: dict[str, Any]) -> dict[str, Any] | None:
    parsed = result.get("parsed")
    if isinstance(parsed, dict) and "tool_calls" not in parsed:
        return deepcopy(parsed)
    candidate = result.get("candidate")
    if not isinstance(candidate, dict) or "report_markdown" not in candidate:
        return None
    wire = deepcopy(candidate)
    claims = []
    report = str(wire.get("report_markdown") or "")
    for claim in wire.get("claims") or []:
        claim = deepcopy(claim)
        span = claim.pop("text_span", None) or {}
        try:
            quote = report[int(span["start"]):int(span["end"])]
        except (KeyError, TypeError, ValueError):
            return None
        if not quote.strip():
            return None
        claim["quote"] = quote
        claims.append(claim)
    wire["claims"] = claims
    return wire


def _field_from_ref(ref: Any) -> str | None:
    if not isinstance(ref, str):
        return None
    text = ref.strip()
    for prefix in ("accident:/", "evidence:/", "snapshot:/", "/"):
        if text.startswith(prefix):
            raw = text[len(prefix):]
            direct = raw.replace("~1", "/").replace("~0", "~")
            if direct in metrics.FIELD_NAMES:
                return direct
            token = raw.split("/", 1)[0].replace("~1", "/").replace("~0", "~")
            return token if token in metrics.FIELD_NAMES else None
    if text in metrics.FIELD_NAMES:
        return text
    for field in metrics.FIELD_NAMES:
        if text.endswith("/" + field) or text.endswith(":" + field):
            return field
    return None


def _accident_data(trace: dict[str, Any]) -> dict[str, str]:
    case = _trace_case(trace)
    return metrics.accident_data(case)


def _conflicts(trace: dict[str, Any]) -> list[dict[str, Any]]:
    case = _trace_case(trace)
    values: list[Any] = []
    for holder in (case, trace.get("snapshot") if isinstance(trace, dict) else None):
        if not isinstance(holder, dict):
            continue
        values.extend(holder.get("conflicts") or holder.get("field_conflicts") or [])
        for record in holder.get("supplemental_records") or []:
            if isinstance(record, dict):
                values.extend(record.get("field_conflicts") or [])
    out: list[dict[str, Any]] = []
    for item in values:
        if isinstance(item, str):
            out.append({"fields": [item]})
        elif isinstance(item, dict):
            fields = item.get("fields") or item.get("field_names") or item.get("accident_fields") or []
            if isinstance(fields, str):
                fields = [fields]
            out.append({**item, "fields": [(_field_from_ref(field) or str(field)) for field in fields]})
    return out


def _claim_quote(claim: dict[str, Any]) -> str:
    return str(claim.get("quote") or "")


def _replace_quote(report: str, old: str, new: str) -> str | None:
    if not old or report.count(old) != 1:
        return None
    return report.replace(old, new, 1)


def _candidate_claim_fields(claim: dict[str, Any]) -> set[str]:
    return {
        field for ref in (claim.get("evidence_refs") or []) + (claim.get("knowledge_refs") or [])
        if (field := _field_from_ref(ref)) is not None
    }


def _obligation_field(item: dict[str, Any]) -> str | None:
    value = str(item.get("obligation_id") or "")
    for prefix in ("fact_obligation:", "fact-obligation:", "obligation:", "fact:"):
        if value.startswith(prefix):
            value = value[len(prefix):]
            break
    if value.startswith("accident:/"):
        value = value.split("/", 1)[1]
    value = value.replace("~1", "/").replace("~0", "~").lstrip("/")
    return value if value in metrics.FIELD_NAMES else None


def _obligation_for(wire: dict[str, Any], field: str) -> dict[str, Any] | None:
    matches = [item for item in wire.get("obligation_resolutions") or []
               if isinstance(item, dict) and _obligation_field(item) == field]
    return matches[0] if len(matches) == 1 else None


def _first_claim_for_field(wire: dict[str, Any], field: str) -> tuple[int, dict[str, Any]] | None:
    for index, claim in enumerate(wire.get("claims") or []):
        if isinstance(claim, dict) and field in _candidate_claim_fields(claim):
            return index, claim
    return None


def _present_fact(trace: dict[str, Any], wire: dict[str, Any]) -> tuple[str, int, dict[str, Any]] | None:
    data = _accident_data(trace)
    fact_names = {spec["name"] for spec in metrics.FIELD_SPECS if spec["kind"] == "fact"}
    for field in metrics.FIELD_NAMES:
        if field not in fact_names or not data.get(field, "").strip():
            continue
        found = _first_claim_for_field(wire, field)
        if found:
            return field, found[0], found[1]
    return None


def _empty_fact_without_claim(trace: dict[str, Any], wire: dict[str, Any]) -> str | None:
    data = _accident_data(trace)
    used = set().union(*(_candidate_claim_fields(claim) for claim in wire.get("claims") or [] if isinstance(claim, dict)))
    for spec in metrics.FIELD_SPECS:
        field = spec["name"]
        if spec["kind"] == "fact" and not data.get(field, "").strip() and field not in used:
            if _obligation_for(wire, field) is not None:
                return field
    return None


def _evaluation_claim(trace: dict[str, Any], wire: dict[str, Any]) -> tuple[str, int, dict[str, Any]] | None:
    data = _accident_data(trace)
    for field in metrics.EVALUATION_FIELDS:
        if not data.get(field, "").strip():
            continue
        found = _first_claim_for_field(wire, field)
        if not found:
            continue
        index, claim = found
        quote = _claim_quote(claim)
        if any(marker in quote for marker in _ATTRIBUTION_MARKERS):
            return field, index, claim
    return None


def _conflict_claim(trace: dict[str, Any], wire: dict[str, Any]) -> tuple[str, int, dict[str, Any]] | None:
    conflicts = _conflicts(trace)
    if not conflicts:
        return None
    for conflict in conflicts:
        for field in conflict.get("fields", []):
            found = _first_claim_for_field(wire, field)
            if found and any(marker in _claim_quote(found[1]) for marker in _CONFLICT_MARKERS):
                return field, found[0], found[1]
            obligation = _obligation_for(wire, field)
            if obligation and any(marker in str(obligation.get("resolution") or "") for marker in _CONFLICT_MARKERS):
                return field, found[0] if found else -1, found[1] if found else {}
    return None


def _available_unread_source(trace: dict[str, Any], validation: dict[str, Any]) -> str | None:
    registry = [item for item in trace.get("knowledge_registry") or [] if isinstance(item, dict)]
    accessed = set((validation.get("access") or {}).get("knowledge") or [])
    for item in registry:
        ident = item.get("id")
        if (item.get("source_kind") in {"source_chunk", None}
                and isinstance(ident, str) and ident not in accessed):
            return ident
    return None


def _read_rule_excerpt(trace: dict[str, Any], validation: dict[str, Any]) -> str | None:
    accessed = set((validation.get("access") or {}).get("knowledge") or [])
    for item in trace.get("knowledge_registry") or []:
        if (isinstance(item, dict) and item.get("source_kind") == "rule_excerpt"
                and item.get("id") in accessed):
            return str(item["id"])
    return None


def _tool_oracle(trace: dict[str, Any], call: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
    """只接受明确写入 trace/call/context 的工具必要性 oracle；不从模型文本猜测。"""
    out: dict[str, Any] = {}
    holders = [trace, call, context or {}]
    for holder in holders:
        if not isinstance(holder, dict):
            continue
        for key in ("tool_necessity", "seed_oracles", "seed_expectations"):
            value = holder.get(key)
            if isinstance(value, dict):
                out.update(value)
        for key in ("tool_necessary", "retrieval_necessary", "needs_retrieval",
                    "requires_retrieval", "required_retrieval", "required_tool"):
            if key in holder:
                out[key] = holder[key]
    return out


def _tool_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    parsed = result.get("parsed")
    return deepcopy(parsed) if isinstance(parsed, dict) and "tool_calls" in parsed else None


def _tool_call_mutation(result: dict[str, Any], mutator) -> dict[str, Any] | None:
    payload = _tool_payload(result)
    if not payload or not isinstance(payload.get("tool_calls"), list) or not payload["tool_calls"]:
        return None
    mutated = deepcopy(payload)
    first = mutated["tool_calls"][0]
    if not isinstance(first, dict):
        return None
    if not isinstance(first.get("arguments"), dict):
        return None
    if not mutator(first):
        return None
    return mutated


def _mutate_omit_fact(trace: dict[str, Any], wire: dict[str, Any]) -> tuple[dict[str, Any], list[str]] | None:
    chosen = _present_fact(trace, wire)
    if not chosen:
        return None
    field, index, claim = chosen
    quote = _claim_quote(claim)
    report = _replace_quote(str(wire.get("report_markdown") or ""), quote, "")
    if report is None:
        return None
    rejected = deepcopy(wire)
    rejected["report_markdown"] = "\n".join(line for line in report.splitlines() if line.strip())
    rejected["claims"].pop(index)
    obligation = _obligation_for(rejected, field)
    if obligation is None:
        return None
    # 按种子协议将被删字段标成 not_relevant；这会联动义务一致性，
    # 两个信号都属于“漏关键事实”这一错误族，不把联动误报成第二类错误。
    obligation["treatment"] = "not_relevant"
    obligation["resolution"] = "该字段未在候选正文中覆盖，当前不适用。"
    return rejected, [f"seed_oracle:missing_key_fact:{field}"]


def _mutate_empty_fabrication(trace: dict[str, Any], wire: dict[str, Any]) -> tuple[dict[str, Any], list[str]] | None:
    field = _empty_fact_without_claim(trace, wire)
    if not field:
        return None
    rejected = deepcopy(wire)
    value = "合成占位值"
    line = f"{field}：{value}。"
    rejected["report_markdown"] = (str(rejected.get("report_markdown") or "").rstrip() + "\n" + line).strip()
    rejected.setdefault("claims", []).append({
        "claim_id": f"seed-fabrication-{len(rejected['claims'])}", "type": "fact", "quote": line,
        "evidence_refs": [f"accident:/{field}"], "knowledge_refs": [],
    })
    obligation = _obligation_for(rejected, field)
    if obligation is None:
        return None
    obligation["treatment"] = "uncertain"
    obligation["resolution"] = "输入为空，无法确认。"
    return rejected, [f"metric:empty_field_fabrication:{field}"]


def _mutate_evaluation_as_fact(trace: dict[str, Any], wire: dict[str, Any]) -> tuple[dict[str, Any], list[str]] | None:
    chosen = _evaluation_claim(trace, wire)
    if not chosen:
        return None
    field, index, claim = chosen
    old = _claim_quote(claim)
    new = old
    for marker in _ATTRIBUTION_MARKERS:
        if marker in new:
            new = new.replace(marker, "", 1)
            break
    if new == old:
        return None
    rejected = deepcopy(wire)
    report = _replace_quote(str(rejected.get("report_markdown") or ""), old, new)
    if report is None:
        return None
    rejected["report_markdown"] = report
    rejected["claims"][index]["quote"] = new
    return rejected, [f"metric:evaluation_attribution:{field}"]


def _mutate_wrong_ref(trace: dict[str, Any], wire: dict[str, Any]) -> tuple[dict[str, Any], list[str]] | None:
    data = _accident_data(trace)
    for index, claim in enumerate(wire.get("claims") or []):
        if not isinstance(claim, dict) or not claim.get("evidence_refs"):
            continue
        source_field = _field_from_ref(claim["evidence_refs"][0])
        if source_field is None:
            continue
        target = next((field for field, value in data.items() if field != source_field and value.strip()), None)
        if target is None:
            continue
        rejected = deepcopy(wire)
        rejected["claims"][index]["evidence_refs"][0] = f"accident:/{target}"
        return rejected, [f"seed_oracle:evidence_ref_mismatch:claims:{index}"]
    return None


def _mutate_obligation(trace: dict[str, Any], wire: dict[str, Any]) -> tuple[dict[str, Any], list[str]] | None:
    data = _accident_data(trace)
    for item in wire.get("obligation_resolutions") or []:
        if not isinstance(item, dict):
            continue
        field = _obligation_field(item)
        if field is None:
            continue
        if data.get(field, "").strip() and _first_claim_for_field(wire, field):
            if item.get("treatment") != "covered":
                continue
            rejected = deepcopy(wire)
            target = _obligation_for(rejected, field)
            target["treatment"] = "uncertain"
            target["resolution"] = "该字段处置状态待核实。"
            return rejected, [f"metric:obligation_consistency:{field}"]
        if not data.get(field, "").strip() and item.get("treatment") == "uncertain":
            rejected = deepcopy(wire)
            target = _obligation_for(rejected, field)
            target["treatment"] = "covered"
            target["resolution"] = "已在正文覆盖。"
            return rejected, [f"metric:obligation_consistency:{field}"]
    return None


def _mutate_conflict(trace: dict[str, Any], wire: dict[str, Any]) -> tuple[dict[str, Any], list[str]] | None:
    chosen = _conflict_claim(trace, wire)
    if not chosen:
        return None
    field, index, claim = chosen
    rejected = deepcopy(wire)
    if index >= 0:
        old = _claim_quote(claim)
        new = old
        for marker in _CONFLICT_MARKERS:
            if marker in new:
                new = new.replace(marker, "", 1)
                break
        if new != old:
            report = _replace_quote(str(rejected.get("report_markdown") or ""), old, new)
            if report is None:
                return None
            rejected["report_markdown"] = report
            rejected["claims"][index]["quote"] = new
    obligation = _obligation_for(rejected, field)
    if obligation is not None:
        resolution = str(obligation.get("resolution") or "")
        for marker in _CONFLICT_MARKERS:
            resolution = resolution.replace(marker, "", 1)
        obligation["resolution"] = resolution or "输入记载待确认。"
    return rejected, [f"metric:conflict_disclosure:{field}"]


def _mutate_unread(trace: dict[str, Any], wire: dict[str, Any], validation: dict[str, Any]) -> tuple[dict[str, Any], list[str]] | None:
    source = _available_unread_source(trace, validation)
    if source is None:
        return None
    if not wire.get("claims"):
        return None
    rejected = deepcopy(wire)
    claim = rejected["claims"][0]
    refs = claim.setdefault("knowledge_refs", [])
    if source in refs:
        return None
    refs.append(source)
    return rejected, [f"source_not_read:claims:0:knowledge_refs"]


def _mutate_rule_excerpt(trace: dict[str, Any], wire: dict[str, Any], validation: dict[str, Any]) -> tuple[dict[str, Any], list[str]] | None:
    source = _read_rule_excerpt(trace, validation)
    if source is None or not wire.get("claims"):
        return None
    rejected = deepcopy(wire)
    refs = rejected["claims"][0].setdefault("knowledge_refs", [])
    if source in refs:
        return None
    refs.append(source)
    return rejected, ["rule_excerpt_as_evidence:claims:0"]


def _mutate_quote_nonunique(trace: dict[str, Any], wire: dict[str, Any]) -> tuple[dict[str, Any], list[str]] | None:
    for claim in wire.get("claims") or []:
        if not isinstance(claim, dict):
            continue
        quote = _claim_quote(claim)
        if not quote:
            continue
        rejected = deepcopy(wire)
        rejected["report_markdown"] = str(rejected.get("report_markdown") or "").rstrip() + "\n" + quote
        return rejected, ["quote_invalid"]
    return None


def _mutate_tool_overlimit(result: dict[str, Any]) -> tuple[dict[str, Any], list[str]] | None:
    def mutate(call: dict[str, Any]) -> bool:
        name = call.get("name")
        args = call.get("arguments")
        if not isinstance(args, dict):
            return False
        if name == "search_knowledge":
            args["top_k"] = 11
            return True
        if name in {"read_knowledge", "read_evidence"}:
            key = "chunk_ids" if name == "read_knowledge" else "evidence_ids"
            values = args.get(key)
            if not isinstance(values, list):
                return False
            args[key] = list(values)[:10] + [f"seed-overlimit-{i}" for i in range(max(1, 11 - min(10, len(values))))]
            return True
        return False

    payload = _tool_call_mutation(result, mutate)
    return (payload, ["tool:retrieval_policy_exceeded"]) if payload is not None else None


def _mutate_tool_necessity(result: dict[str, Any], *, missing: bool, oracle: dict[str, Any], trace: dict[str, Any]) -> tuple[dict[str, Any], list[str]] | None:
    tool = _tool_payload(result)
    if not tool:
        return None
    calls = tool.get("tool_calls") or []
    if not calls or not isinstance(calls[0], dict) or not isinstance(calls[0].get("arguments"), dict):
        return None
    call = calls[0]
    args = call["arguments"]
    if missing:
        required = oracle.get("required_tool") or oracle.get("required_retrieval")
        required_ids = oracle.get("required_chunk_ids") or oracle.get("required_ids") or []
        if call.get("name") != "read_knowledge" or not required_ids:
            return None
        registry_ids = [item.get("id") for item in trace.get("knowledge_registry") or [] if isinstance(item, dict)]
        alternative = next((ident for ident in registry_ids if ident not in set(required_ids)), None)
        if alternative is None:
            return None
        mutated = deepcopy(tool)
        mutated["tool_calls"][0]["arguments"]["chunk_ids"] = [alternative]
        return mutated, ["seed_oracle:missing_required_retrieval"]
    necessary = oracle.get("tool_necessary", oracle.get("retrieval_necessary", oracle.get("needs_retrieval")))
    if necessary is not False:
        return None
    mutated = deepcopy(tool)
    if mutated["tool_calls"][0].get("name") == "search_knowledge":
        query = str(mutated["tool_calls"][0]["arguments"].get("query") or "检索")
        mutated["tool_calls"][0]["arguments"]["query"] = query + " 无需检索"
    elif mutated["tool_calls"][0].get("name") == "read_knowledge":
        ids = list(mutated["tool_calls"][0]["arguments"].get("chunk_ids") or [])
        if not ids:
            return None
        mutated["tool_calls"][0]["arguments"]["chunk_ids"] = ids
        mutated["tool_calls"][0]["arguments"]["cursor"] = "seed-unnecessary"
    else:
        return None
    return mutated, ["seed_oracle:unnecessary_retrieval"]


def _mutate(kind: str, trace: dict[str, Any], result: dict[str, Any], wire: dict[str, Any] | None,
            validation: dict[str, Any]) -> tuple[str, list[str], list[str]] | None:
    """返回 rejected 文本、预期信号、影响字段；不在这里调用评估。"""
    if kind == "omit_key_fact" and wire is not None:
        out = _mutate_omit_fact(trace, wire)
        if out:
            rejected, signals = out
            return _json_text(rejected), signals, [signals[0].rsplit(":", 1)[-1]]
    if kind == "empty_field_fabrication" and wire is not None:
        out = _mutate_empty_fabrication(trace, wire)
        if out:
            rejected, signals = out
            return _json_text(rejected), signals, [signals[0].rsplit(":", 1)[-1]]
    if kind == "evaluation_as_fact" and wire is not None:
        out = _mutate_evaluation_as_fact(trace, wire)
        if out:
            rejected, signals = out
            return _json_text(rejected), signals, [signals[0].rsplit(":", 1)[-1]]
    if kind == "wrong_evidence_ref" and wire is not None:
        out = _mutate_wrong_ref(trace, wire)
        if out:
            rejected, signals = out
            return _json_text(rejected), signals, [f"claims[{signals[0].split(':')[-1]}]"]
    if kind == "obligation_mismatch" and wire is not None:
        out = _mutate_obligation(trace, wire)
        if out:
            rejected, signals = out
            return _json_text(rejected), signals, [signals[0].rsplit(":", 1)[-1]]
    if kind == "swallow_conflict" and wire is not None:
        out = _mutate_conflict(trace, wire)
        if out:
            rejected, signals = out
            return _json_text(rejected), signals, [signals[0].rsplit(":", 1)[-1]]
    if kind == "unread_source" and wire is not None:
        out = _mutate_unread(trace, wire, validation)
        if out:
            rejected, signals = out
            return _json_text(rejected), signals, ["claims[0].knowledge_refs"]
    if kind == "rule_excerpt_basis" and wire is not None:
        out = _mutate_rule_excerpt(trace, wire, validation)
        if out:
            rejected, signals = out
            return _json_text(rejected), signals, ["claims[0].knowledge_refs"]
    if kind == "quote_not_unique" and wire is not None:
        out = _mutate_quote_nonunique(trace, wire)
        if out:
            rejected, signals = out
            return _json_text(rejected), signals, ["report_markdown"]
    if kind == "tool_parameter_overlimit":
        out = _mutate_tool_overlimit(result)
        if out:
            rejected, signals = out
            return _json_text(rejected), signals, ["tool_calls[0].arguments"]
    if kind in {"unnecessary_retrieval", "missing_required_retrieval"}:
        context = {}
        try:
            context = json.loads((call := trace.get("_seed_call_payload", {})).get("messages", [{}])[-1].get("content", "{}"))
        except (AttributeError, IndexError, TypeError, ValueError):
            context = {}
        oracle = _tool_oracle(trace, {}, context)
        out = _mutate_tool_necessity(result, missing=kind == "missing_required_retrieval", oracle=oracle, trace=trace)
        if out:
            rejected, signals = out
            return _json_text(rejected), signals, ["tool_calls[0]"]
    return None


def _measure(trace: dict[str, Any], call: dict[str, Any], content: str, template: Any) -> dict[str, Any] | None:
    clone = deepcopy(call)
    clone["content"] = content
    try:
        parsed = json.loads(content)
    except (TypeError, ValueError):
        parsed = None
    if isinstance(parsed, dict) and isinstance(parsed.get("tool_calls"), list):
        clone["tool_calls"] = deepcopy(parsed["tool_calls"])
    try:
        sample = render_call_sample(trace, clone, template=template, reasoning_effort="compact")
    except Exception as exc:  # 真实计数失败时由调用方丢弃该种子
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}", "response_tokens": None}
    if sample.get("status") not in {"exact", "exact_bridge"} or not isinstance(sample.get("response_tokens"), int):
        return {"status": sample.get("status"), "error": sample.get("error") or "真实tokenizer不可用", "response_tokens": None}
    content_start = sample.get("c_start")
    content_tokens = (sample["response_tokens"] - content_start
                      if isinstance(content_start, int) else None)
    return {"status": sample["status"], "response_tokens": sample["response_tokens"],
            "content_tokens": content_tokens,
            "prompt_tokens": sample.get("prompt_tokens"), "total_tokens": (sample.get("prompt_tokens") or 0) + sample["response_tokens"]}


def _annotate(result: dict[str, Any], signals: Iterable[str]) -> dict[str, Any]:
    out = deepcopy(result)
    seed_signals = sorted(set(str(item) for item in signals))
    # seed oracle 是本模块独立证明，不冒充 revise.evaluate_response 的 metric/hard 信号。
    out["seed_oracles"] = seed_signals
    out["training_eligible"] = (out.get("kind") != "invalid"
                                 and not out.get("fail_signals") and not seed_signals)
    return out


def _field_from_signal(signal: str) -> str | None:
    return signal.rsplit(":", 1)[-1] if ":" in signal else None


def _seed_oracle_signals(kind: str, baseline_wire: dict[str, Any] | None,
                         rejected_wire: dict[str, Any] | None, trace: dict[str, Any],
                         rejected_raw: dict[str, Any], affected: list[str],
                         expected: list[str]) -> list[str]:
    """只返回由种子自身结构证明的补充信号，不把预期值直接当作观测值。"""
    if kind == "omit_key_fact" and baseline_wire and rejected_wire:
        field = _field_from_signal(expected[0]) if expected else None
        if (field and _accident_data(trace).get(field, "").strip()
                and _first_claim_for_field(baseline_wire, field)
                and not _first_claim_for_field(rejected_wire, field)):
            return [f"seed_oracle:missing_key_fact:{field}"]
    if kind == "wrong_evidence_ref" and baseline_wire and rejected_wire:
        index = None
        for item in affected:
            if item.startswith("claims[") and item.endswith("]"):
                try:
                    index = int(item[7:-1])
                except ValueError:
                    pass
        if index is not None:
            before = (baseline_wire.get("claims") or [])[index]
            after = (rejected_wire.get("claims") or [])[index]
            before_refs = before.get("evidence_refs") or []
            after_refs = after.get("evidence_refs") or []
            if before_refs and after_refs and before_refs[0] != after_refs[0]:
                return [f"seed_oracle:evidence_ref_mismatch:claims:{index}"]
    if kind in {"unnecessary_retrieval", "missing_required_retrieval"}:
        # 两类工具必要性只允许在显式 oracle 下生成；mutator 已执行参数层变更，
        # 这里再次确认返回仍是合法 tool 回合，避免把坏 JSON 当语义错误。
        parsed = rejected_raw.get("parsed")
        if not isinstance(parsed, dict) or rejected_raw.get("kind") != "tool":
            return []
        calls = parsed.get("tool_calls") or []
        if not calls or not isinstance(calls[0], dict):
            return []
        if kind == "unnecessary_retrieval":
            oracle = _tool_oracle(trace, {})
            necessary = oracle.get("tool_necessary", oracle.get("retrieval_necessary"))
            if necessary is False and calls[0].get("name") in {"search_knowledge", "read_knowledge"}:
                return ["seed_oracle:unnecessary_retrieval"]
        else:
            oracle = _tool_oracle(trace, {})
            required_ids = set(oracle.get("required_chunk_ids") or oracle.get("required_ids") or [])
            args = calls[0].get("arguments") or {}
            if required_ids and calls[0].get("name") == "read_knowledge" \
                    and required_ids.isdisjoint(set(args.get("chunk_ids") or [])):
                return ["seed_oracle:missing_required_retrieval"]
    return []


_TARGET_CHECKS = {
    "omit_key_fact": {"nonempty_fact_coverage", "evidence_ref_alignment", "obligation_consistency"},
    "empty_field_fabrication": {"empty_field_fabrication", "obligation_consistency"},
    "evaluation_as_fact": {"evaluation_attribution"},
    "wrong_evidence_ref": {"evidence_ref_alignment", "nonempty_fact_coverage", "obligation_consistency"},
    "obligation_mismatch": {"obligation_consistency"},
    "swallow_conflict": {"conflict_disclosure", "obligation_consistency"},
}


def _metric_statuses(metrics_result: dict[str, Any]) -> dict[str, Any]:
    return metrics_result.get("checks", {}) if isinstance(metrics_result, dict) else {}


def _metrics_only_target_changed(kind: str, baseline: dict[str, Any], rejected: dict[str, Any],
                                 affected: list[str]) -> bool:
    """除目标检查外，检查状态和字段状态必须保持不变。"""
    base_checks, rej_checks = _metric_statuses(baseline), _metric_statuses(rejected)
    if not base_checks or not rej_checks:
        return True
    target = _TARGET_CHECKS.get(kind, set())
    affected_field = affected[0] if affected and "[" not in affected[0] else None
    for name in set(base_checks) | set(rej_checks):
        if name in target:
            continue
        before, after = base_checks.get(name), rej_checks.get(name)
        if not isinstance(before, dict) or not isinstance(after, dict):
            return False
        if before.get("status") != after.get("status"):
            return False
        bf, af = before.get("fields"), after.get("fields")
        if isinstance(bf, dict) and isinstance(af, dict):
            for key in set(bf) | set(af):
                bitem, aitem = bf.get(key), af.get(key)
                if not isinstance(bitem, dict) or not isinstance(aitem, dict):
                    return False
                if bitem.get("status") != aitem.get("status"):
                    return False
    return True


def _length_ratio(chosen: dict[str, Any], rejected: dict[str, Any]) -> float | None:
    a, b = chosen.get("content_tokens"), rejected.get("content_tokens")
    if type(a) is not int or type(b) is not int or a <= 0:
        return None
    return round(b / a, 6)


def _attempt(trace: dict[str, Any], call: dict[str, Any], chosen_content: str, kind: str,
            baseline: dict[str, Any], template: Any) -> dict[str, Any] | None:
    wire = _wire_from_baseline(baseline) if baseline.get("kind") == "final" else None
    trace_for_seed = deepcopy(trace)
    trace_for_seed["_seed_call_payload"] = deepcopy((call.get("payload") or {}))
    mutated = _mutate(kind, trace_for_seed, baseline, wire, baseline)
    if mutated is None:
        return None
    rejected_content, expected, affected = mutated
    rejected_raw = evaluate_response(trace, call, rejected_content)
    # 评估器对路径问题使用 JSON 编码，工具额度错误也可能由冻结的累计额度
    # 先于参数上限触发；只接受同一错误族中唯一、可观测的实际信号。
    observed = set(rejected_raw.get("fail_signals") or [])
    if kind == "unread_source":
        family = sorted(item for item in observed if item.startswith("source_not_read:"))
        if len(family) == 1:
            expected = family
    elif kind == "tool_parameter_overlimit":
        family = sorted(item for item in observed if item.startswith("tool:"))
        if len(family) == 1:
            expected = family
    elif kind == "wrong_evidence_ref":
        family = sorted(item for item in observed if item.startswith("metric:evidence_ref_alignment:"))
        if len(family) == 1:
            expected = sorted(set(expected) | set(family))
    seed_signals = _seed_oracle_signals(kind, wire, _wire_from_baseline(rejected_raw), trace,
                                        rejected_raw, affected, expected)
    observed = set(rejected_raw.get("fail_signals") or [])
    if kind in {"unread_source", "tool_parameter_overlimit"}:
        family = "source_not_read:" if kind == "unread_source" else "tool:"
        target_observed = {item for item in observed if item.startswith(family)}
    elif kind == "rule_excerpt_basis":
        target_observed = {item for item in observed if item.startswith("rule_excerpt_as_evidence:")}
    elif kind == "quote_not_unique":
        target_observed = {item for item in observed if item == "quote_invalid"}
    elif kind == "omit_key_fact":
        target_observed = {
            item for item in observed
            if item.startswith("metric:nonempty_fact_coverage:")
            or item.startswith("metric:obligation_consistency:")
        }
    elif kind == "empty_field_fabrication":
        target_observed = {item for item in observed if item.startswith("metric:empty_field_fabrication:")}
    elif kind == "evaluation_as_fact":
        target_observed = {item for item in observed if item.startswith("metric:evaluation_attribution:")}
    elif kind == "obligation_mismatch":
        target_observed = {item for item in observed if item.startswith("metric:obligation_consistency:")}
    elif kind == "swallow_conflict":
        target_observed = {item for item in observed if item.startswith("metric:conflict_disclosure:")}
    elif kind == "wrong_evidence_ref":
        target_observed = {item for item in observed if item.startswith("metric:evidence_ref_alignment:")}
    else:
        target_observed = set()
    effective = sorted(set(seed_signals) | target_observed)
    allowed = set(effective)
    unexpected = sorted(observed - allowed)
    if not effective or unexpected:
        return None
    rejected = _annotate(rejected_raw, seed_signals)
    if not _metrics_only_target_changed(kind, baseline.get("metrics", {}), rejected.get("metrics", {}), affected):
        return None
    missing = sorted(set(expected) - set(effective)) if kind in {"unread_source", "tool_parameter_overlimit", "rule_excerpt_basis", "quote_not_unique", "empty_field_fabrication", "evaluation_as_fact", "obligation_mismatch", "swallow_conflict"} else []
    chosen_tokens = _measure(trace, call, chosen_content, template)
    rejected_call = deepcopy(call)
    rejected_call["content"] = rejected_content
    if rejected.get("kind") == "tool" and isinstance(rejected.get("parsed"), dict):
        rejected_call["tool_calls"] = deepcopy(rejected["parsed"].get("tool_calls") or [])
    rejected_tokens = _measure(trace, rejected_call, rejected_content, template)
    ratio = _length_ratio(chosen_tokens or {}, rejected_tokens or {})
    if ratio is None or not 0.85 <= ratio <= 1.15 or unexpected or missing or rejected.get("training_eligible"):
        return None
    validation = {
        "baseline": {**baseline, "token_count": chosen_tokens},
        "rejected": {**rejected, "token_count": rejected_tokens},
        "unique_failure_class": kind,
        "unique_fail_signals": effective,
        "only_error_class": True,
        "length_ratio": ratio,
        "length_tokens": {"chosen": chosen_tokens.get("content_tokens"), "rejected": rejected_tokens.get("content_tokens")},
    }
    return {
        "chosen_content": chosen_content,
        "rejected_content": rejected_content,
        "error_class": kind,
        "affected_fields": affected,
        "validation": validation,
        "synthetic": True,
    }


def seed_errors(trace: dict[str, Any], call: dict[str, Any], chosen_content: str, *,
                error_classes: Iterable[str] | None = None, template: Any = None,
                max_pairs: int = 4) -> list[dict[str, Any]]:
    """从一条已通过调用产生最多 ``max_pairs`` 个单类种子错误对。"""
    if not isinstance(chosen_content, str):
        raise TypeError("chosen_content 必须是字符串。")
    if type(max_pairs) is not int or max_pairs < 0:
        raise ValueError("max_pairs 必须是非负整数。")
    if max_pairs == 0:
        return []
    baseline = evaluate_response(trace, call, chosen_content)
    if not baseline.get("training_eligible"):
        return []
    requested = ERROR_CLASSES if error_classes is None else tuple(
        canonical for value in error_classes if (canonical := _canonical_class(str(value))) is not None
    )
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for kind in requested:
        if kind in seen or len(result) >= max_pairs:
            continue
        seen.add(kind)
        pair = _attempt(trace, call, chosen_content, kind, baseline, template)
        if pair is not None:
            result.append(pair)
    return result


def explain_seedability(trace: dict[str, Any], call: dict[str, Any], chosen_content: str, *,
                        error_classes: Iterable[str] | None = None, template: Any = None,
                        max_pairs: int = 12) -> dict[str, Any]:
    """返回不改变数据的诊断，便于记录安全丢弃原因。"""
    baseline = evaluate_response(trace, call, chosen_content)
    pairs = seed_errors(trace, call, chosen_content, error_classes=error_classes,
                        template=template, max_pairs=max_pairs) if baseline.get("training_eligible") else []
    produced = {item["error_class"] for item in pairs}
    requested = ERROR_CLASSES if error_classes is None else tuple(
        canonical for value in error_classes if (canonical := _canonical_class(str(value))) is not None
    )
    return {"baseline": baseline, "requested": list(requested), "produced": sorted(produced),
            "dropped": [kind for kind in requested if kind not in produced]}


__all__ = ["FINAL_ERROR_CLASSES", "TOOL_ERROR_CLASSES", "ERROR_CLASSES", "seed_errors", "explain_seedability"]
