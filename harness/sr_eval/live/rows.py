"""构建 R2 后训练行。

本模块只负责把一条 ``derived`` 记录转换成训练脚本可直接读取的行。
模板、分词器和思考/报告边界统一复用 ``live.samples``；任何无法得到真实
token 的情况都闭锁，不用字符数估算，也不静默截断超窗样本。
"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from typing import Any

from .samples import render_call_sample


MAX_LEN_DEFAULT = 28000
HARD_WINDOW = 32768  # 部署每槽位上下文;仓库所有者 2026-10-07 方案B:训练窗口可放宽到此值(默认仍 28000)
REASONING_EFFORT = "compact"   # 2026-10-06 仓库所有者定:后训练与部署用专用 compact 档位(模板 profiles/assets/qwen3.8-compact-v1;构建时需 SR_QWEN_PROFILE_DIR 指向它)
DERIVATIONS = {"student_original", "minimax_revision", "seeded"}
ANCHOR_KINDS = {"anchor_final", "anchor_tool"}


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: Any) -> str:
    if isinstance(value, bytes):
        raw = value
    elif isinstance(value, str):
        raw = value.encode("utf-8")
    else:
        raw = _canonical(value).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _safe_text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _looks_like_json_object(text: str) -> bool:
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        return False
    return isinstance(value, dict)


def _looks_like_tool_json(text: str, record: dict[str, Any]) -> bool:
    if isinstance(record.get("tool_calls"), list) and record.get("tool_calls"):
        return True
    for evidence_key in ("validation", "chosen_validation"):
        evidence = record.get(evidence_key)
        if isinstance(evidence, dict) and str(evidence.get("kind") or "").lower() == "tool":
            return True
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        return False
    if not isinstance(value, dict):
        return False
    calls = value.get("tool_calls")
    if isinstance(calls, list) and bool(calls):
        return True
    # RoleLoop.normalize_tool_response 也接受精确单工具对象，而不是 wrapper。
    return (
        isinstance(value.get("call_id"), str)
        and bool(value.get("call_id"))
        and isinstance(value.get("name"), str)
        and bool(value.get("name"))
        and isinstance(value.get("arguments"), dict)
    )


def _validation_failed(value: Any) -> bool:
    """只拦截明确失败证据；unknown 不被伪装为通过，也不在此处擅自升级。"""

    if value is None:
        return False
    if value is False:
        return True
    if not isinstance(value, dict):
        return False
    if value.get("passed") is False or value.get("strict_pass") is False:
        return True
    if value.get("training_eligible") is False:
        return True
    for key in ("overall", "status", "result", "state"):
        status = value.get(key)
        if isinstance(status, str) and status.strip().lower() in {
            "fail",
            "failed",
            "invalid",
            "reject",
            "rejected",
        }:
            return True
    return False


def _validation_for(record: dict[str, Any], side: str) -> Any:
    if side == "chosen":
        for key in ("chosen_validation", "validation", "chosen_evidence"):
            if key in record:
                return record[key]
    else:
        for key in ("rejected_validation", "rejected_evidence"):
            if key in record:
                return record[key]
    return None


def _normalize_error_class(value: Any) -> str:
    text = str(value or "").strip()
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", text).strip("._-")


def _kind_for(record: dict[str, Any], chosen: str, rejected: str | None) -> tuple[str | None, str | None]:
    derivation = record.get("derivation")
    explicit = record.get("kind")
    error_class = _normalize_error_class(record.get("error_class"))
    if derivation not in DERIVATIONS:
        return None, "invalid_derivation"

    if explicit is not None:
        kind = str(explicit)
        if kind in ANCHOR_KINDS:
            if rejected is not None:
                return None, "anchor_has_rejected_content"
            if kind == "anchor_tool" and not _looks_like_tool_json(chosen, record):
                return None, "anchor_tool_without_tool_calls"
            if kind == "anchor_final" and _looks_like_tool_json(chosen, record):
                return None, "tool_calls_need_anchor_tool"
            return kind, None
        if kind == "pair_revised":
            if derivation != "minimax_revision" or rejected is None:
                return None, "invalid_pair_revised"
            return kind, None
        if kind.startswith("pair_seeded_"):
            suffix = kind[len("pair_seeded_") :]
            if derivation != "seeded" or rejected is None or not suffix:
                return None, "invalid_pair_seeded"
            if error_class and suffix != error_class:
                return None, "error_class_kind_mismatch"
            return kind, None
        return None, "invalid_kind"

    if rejected is not None:
        if derivation == "minimax_revision":
            return "pair_revised", None
        if derivation == "seeded":
            if not error_class:
                return None, "error_class_missing"
            return f"pair_seeded_{error_class}", None
        return None, "student_original_pair_unsupported"
    if _looks_like_tool_json(chosen, record):
        return "anchor_tool", None
    return "anchor_final", None


def _call(payload: dict[str, Any], content: str, reasoning: str, record: dict[str, Any]) -> dict[str, Any]:
    """拼出 ``render_call_sample`` 所需的最小调用投影，不改动原 payload。"""

    call: dict[str, Any] = {
        "payload": deepcopy(payload),
        "content": content,
        "reasoning_content": reasoning,
        "reasoning": reasoning,
        "finish_reason": str(record.get("finish_reason") or "stop"),
        "tool_calls": deepcopy(record.get("tool_calls") or []),
    }
    for key in ("with_tools",):
        if key in record:
            call[key] = deepcopy(record[key])
    return call


def _valid_ids(sample: dict[str, Any]) -> bool:
    for name in ("prompt_ids", "response_ids"):
        value = sample.get(name)
        if not isinstance(value, list) or not value:
            return False
        if any(not isinstance(token, int) or isinstance(token, bool) for token in value):
            return False
    start = sample.get("c_start")
    return isinstance(start, int) and not isinstance(start, bool) and 0 <= start < len(sample["response_ids"])


def _pair_id(record: dict[str, Any], kind: str, chosen: str, rejected: str | None) -> str:
    raw = "|".join(
        (
            str(record.get("case_id") or ""),
            str(record.get("source_call_sha256") or ""),
            kind,
            _sha256(chosen),
            _sha256(rejected or ""),
        )
    )
    return "r2-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _metadata(
    record: dict[str, Any],
    kind: str,
    pair_id: str,
    chosen: str,
    rejected: str | None,
    chosen_sample: dict[str, Any],
    rejected_sample: dict[str, Any] | None,
    tokens: int | None,
) -> dict[str, Any]:
    payload = record.get("payload")
    return {
        "pair_id": pair_id,
        "kind": kind,
        "case_id": record.get("case_id"),
        "derivation": record.get("derivation"),
        "error_class": record.get("error_class"),
        "source_call_sha256": record.get("source_call_sha256"),
        "synthetic": record.get("synthetic"),
        "training_eligible": record.get("training_eligible"),
        "reasoning_effort": REASONING_EFFORT,
        "payload_sha256": _sha256(payload),
        "prompt_sha256": _sha256(chosen_sample.get("prompt_text") or ""),
        "reasoning_sha256": _sha256(record.get("reasoning_content", "")),
        "chosen_content_sha256": _sha256(chosen),
        "rejected_content_sha256": _sha256(rejected or "") if rejected is not None else None,
        "chosen_content": chosen,
        "rejected_content": rejected,
        "tokens": tokens,
        "prompt_tokens": chosen_sample.get("prompt_tokens"),
        "chosen_tokens": chosen_sample.get("response_tokens"),
        "rejected_tokens": rejected_sample.get("response_tokens") if rejected_sample else None,
        "c_start": chosen_sample.get("c_start"),
        "r_start": rejected_sample.get("c_start") if rejected_sample else None,
        "template_sha256": (chosen_sample.get("tokenizer") or {}).get("template_sha256"),
        "tokenizer_sha256": (chosen_sample.get("tokenizer") or {}).get("tokenizer_sha256"),
        "validation": record.get("validation"),
        "chosen_validation": _validation_for(record, "chosen"),
        "rejected_validation": _validation_for(record, "rejected"),
    }


def build_row(record: dict[str, Any], *, template: Any = None, max_len: int = MAX_LEN_DEFAULT, kind: str | None = None) -> dict[str, Any]:
    """构建一条锚点或偏好训练行。

    返回值固定为 ``row/reason/tokens/metadata`` 四个键。``row`` 只包含训练
    脚本字段；审查所需的文本和哈希全部留在 ``metadata`` 中。
    """

    result: dict[str, Any] = {"row": None, "reason": None, "tokens": None, "metadata": {}}
    if not isinstance(record, dict):
        result["reason"] = "record_not_mapping"
        return result
    if not isinstance(max_len, int) or isinstance(max_len, bool) or not 0 < max_len <= HARD_WINDOW:
        raise ValueError(f"max_len 必须在 1..{HARD_WINDOW} 范围内")
    if record.get("training_eligible") is not True:
        result["reason"] = "training_ineligible"
        return result
    if record.get("synthetic") is not True:
        result["reason"] = "non_synthetic"
        return result
    if not str(record.get("case_id") or "").strip():
        result["reason"] = "case_id_missing"
        return result
    if not str(record.get("source_call_sha256") or "").strip():
        result["reason"] = "source_call_sha256_missing"
        return result
    payload = record.get("payload")
    if not isinstance(payload, dict):
        result["reason"] = "payload_missing"
        return result
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        result["reason"] = "payload_messages_missing"
        return result
    chosen = record.get("chosen_content")
    rejected = record.get("rejected_content")
    reasoning = record.get("reasoning_content", "")
    if not isinstance(chosen, str) or not chosen.strip():
        result["reason"] = "chosen_content_missing"
        return result
    if rejected is not None and (not isinstance(rejected, str) or not rejected.strip()):
        result["reason"] = "rejected_content_empty"
        return result
    if not isinstance(reasoning, str):
        result["reason"] = "reasoning_content_invalid"
        return result
    if str(record.get("finish_reason") or "stop") == "length":
        result["reason"] = "truncated_finish_reason"
        return result
    if not _looks_like_json_object(chosen):
        result["reason"] = "chosen_content_not_json_object"
        return result
    # 修订偏好对的 rejected 是学生失败答复；坏 JSON 正是应学习拒绝的核心错误，
    # 因而只对 seeded 变体要求两边都保持合法 JSON。
    if rejected is not None and not _looks_like_json_object(rejected) and record.get("derivation") != "minimax_revision":
        result["reason"] = "rejected_content_not_json_object"
        return result
    if _validation_failed(_validation_for(record, "chosen")):
        result["reason"] = "chosen_validation_failed"
        return result
    selected_kind, kind_reason = _kind_for({**record, **({"kind": kind} if kind is not None else {})}, chosen, rejected)
    if selected_kind is None:
        result["reason"] = kind_reason or "kind_invalid"
        return result
    if rejected is not None and chosen == rejected:
        result["reason"] = "identical_responses"
        return result

    chosen_call = _call(payload, chosen, reasoning, record)
    chosen_sample = render_call_sample({}, chosen_call, template=template, reasoning_effort=REASONING_EFFORT)
    rejected_sample = None
    if rejected is not None:
        rejected_sample = render_call_sample({}, _call(payload, rejected, reasoning, record), template=template, reasoning_effort=REASONING_EFFORT)

    samples = [chosen_sample] + ([rejected_sample] if rejected_sample is not None else [])
    if any(sample.get("status") not in {"exact", "exact_bridge"} for sample in samples):
        result["reason"] = "tokenizer_unavailable" if any(sample.get("status") == "unavailable" for sample in samples) else "template_render_error"
        result["metadata"] = {"kind": selected_kind, "tokenizer": [sample.get("tokenizer") for sample in samples]}
        return result
    if not _valid_ids(chosen_sample) or (rejected_sample is not None and not _valid_ids(rejected_sample)):
        result["reason"] = "token_boundary_invalid"
        return result
    if rejected_sample is not None:
        if chosen_sample["prompt_ids"] != rejected_sample["prompt_ids"]:
            result["reason"] = "prompt_tokens_mismatch"
            return result
        if chosen_sample["response_ids"][: chosen_sample["c_start"]] != rejected_sample["response_ids"][: rejected_sample["c_start"]]:
            result["reason"] = "reasoning_prefix_mismatch"
            return result
        if chosen_sample["response_ids"] == rejected_sample["response_ids"]:
            result["reason"] = "identical_response_tokens"
            return result
        for key in ("template_sha256", "tokenizer_sha256"):
            if (chosen_sample.get("tokenizer") or {}).get(key) != (rejected_sample.get("tokenizer") or {}).get(key):
                result["reason"] = "tokenizer_mismatch"
                return result

    token_count = len(chosen_sample["prompt_ids"]) + max(
        len(chosen_sample["response_ids"]), len(rejected_sample["response_ids"]) if rejected_sample else 0
    )
    pair_id = _pair_id(record, selected_kind, chosen, rejected)
    metadata = _metadata(record, selected_kind, pair_id, chosen, rejected, chosen_sample, rejected_sample, token_count)
    result["metadata"] = metadata
    result["tokens"] = token_count
    if token_count > max_len:
        result["reason"] = "over_max_len"
        return result

    row = {
        "pair_id": pair_id,
        "kind": selected_kind,
        "prompt_ids": chosen_sample["prompt_ids"],
        "chosen_ids": chosen_sample["response_ids"],
        "rejected_ids": rejected_sample["response_ids"] if rejected_sample else [],
        "c_start": chosen_sample["c_start"],
    }
    if rejected_sample is not None:
        row["r_start"] = rejected_sample["c_start"]
    result["row"] = row
    return result


__all__ = ["MAX_LEN_DEFAULT", "REASONING_EFFORT", "build_row"]
