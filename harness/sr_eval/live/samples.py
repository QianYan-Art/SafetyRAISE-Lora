"""把 live trace 的每次模型调用渲染为真实 Qwen 模板 token 样本。

默认复用 ``sr_eval.sft.QwenTemplate``。该模板依赖 ``tokenizers``；运行环境
没有依赖时，本模块返回 ``status=unavailable``，不做字符数或比例估算。
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable

try:  # live 最小环境可能没有 tokenizers；读取 trace/构建 manifest 仍应可运行。
    from .. import sft as _sft
except Exception:  # pragma: no cover - 依赖存在时由真实模板路径覆盖
    _sft = None


EOS = "<|im_end|>"


def _json_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _dump(value: Any) -> Any:
    if isinstance(value, dict):
        return value
    method = getattr(value, "model_dump", None)
    if callable(method):
        return method(mode="json")
    return value


def _response_message(response: Any) -> dict[str, Any]:
    response = _dump(response) or {}
    if not isinstance(response, dict):
        return {}
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        choice = _dump(choices[0]) or {}
        if isinstance(choice, dict) and isinstance(choice.get("message"), dict):
            return dict(choice["message"])
    if isinstance(response.get("message"), dict):
        return dict(response["message"])
    return response


def normalize_call(trace: dict[str, Any], call: dict[str, Any], call_index: int = 0) -> dict[str, Any]:
    """将新旧 trace 调用字段归一为 ``sft.build_sample`` 可接受的字典。"""
    call = _dump(call) or {}
    payload = _dump(call.get("payload") or call.get("provider_payload") or call.get("request") or {})
    if not isinstance(payload, dict):
        payload = {}
    messages = payload.get("messages") or call.get("messages") or []
    if not isinstance(messages, list):
        messages = []
    tools = payload.get("tools") if "tools" in payload else call.get("tools")
    response = _dump(call.get("response") or call.get("raw_response") or {})
    message = _response_message(response)
    content = call.get("content")
    if content is None:
        content = message.get("content")
    content = _json_text(content)
    reasoning = call.get("reasoning_content")
    if reasoning is None:
        reasoning = call.get("reasoning")
    if reasoning is None:
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    reasoning = _json_text(reasoning)
    tool_calls = call.get("tool_calls")
    if tool_calls is None:
        tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        tool_calls = []
    # 线上 JSON 工具协议通常把 tool_calls 放在 content 中，不要把它误转成旧 XML 工具目标。
    with_tools = bool(call.get("with_tools", bool(tools)))
    role = str(call.get("role") or payload.get("role") or "generator")
    return {
        "call_index": int(call_index),
        "role": role,
        "messages": messages,
        "tools": tools,
        "with_tools": with_tools,
        "tool_calls": tool_calls,
        "content": content,
        "reasoning": reasoning,
        "finish_reason": call.get("finish_reason") or (message.get("finish_reason") if isinstance(message, dict) else None),
        "payload": payload,
        "response": response,
        "context_digest": call.get("context_digest"),
        "request_sha256": call.get("request_sha256"),
    }


def _template(template: Any = None, profile_dir: Path | None = None) -> tuple[Any | None, str | None]:
    if template is not None:
        return template, None
    try:
        if _sft is None:
            from .. import sft as loaded_sft
        else:
            loaded_sft = _sft
        return loaded_sft.QwenTemplate(profile_dir or loaded_sft.PROFILE_DIR), None
    except Exception as exc:  # 缺 tokenizers/jinja2 或损坏 tokenizer 均如实暴露为不可用
        return None, f"真实 tokenizer 不可用: {type(exc).__name__}: {exc}"


def _legacy_python(python_exe: str | Path | None = None) -> Path | None:
    """定位已有的完整 Python 环境；不下载、不安装依赖。"""
    if python_exe:
        path = Path(python_exe)
        return path if path.exists() else None
    root = Path(__file__).resolve().parents[3]
    candidate = root / ".venv" / "Scripts" / "python.exe"
    return candidate if candidate.exists() else None


_BRIDGE_SCRIPT = r'''
import hashlib, json, sys
from sr_eval.sft import QwenTemplate, assistant_message
req = json.load(sys.stdin)
tpl = QwenTemplate()
messages = req.get("messages") or []
tools = req.get("tools") if req.get("with_tools") else None
call = {"content": req.get("content") or "", "reasoning": req.get("reasoning") or "", "reasoning_content": req.get("reasoning") or "", "tool_calls": req.get("tool_calls") or []}
call["tool_calls"] = call["tool_calls"] if isinstance(call["tool_calls"], list) else []
prompt = tpl.render(messages, tools, add_generation_prompt=True, reasoning_effort=req.get("reasoning_effort", "xhigh"))
full = tpl.render(messages + [assistant_message(call, call["reasoning"])], tools, add_generation_prompt=False, reasoning_effort=req.get("reasoning_effort", "xhigh"))
if not full.startswith(prompt):
    raise ValueError("完整对话不是生成提示的前缀")
target = full[len(prompt):]
if not target.endswith("<|im_end|>\n"):
    raise ValueError("目标未以 <|im_end|> 结尾")
target = target[:-1]
content = str(req.get("content") or "")
after = target.find("</think>")
start = after + len("</think>") if after >= 0 else 0
char_start = target.find(content.strip(), start) if content.strip() else start
if char_start < 0:
    char_start = start
pids, rids = tpl.encode(prompt), tpl.encode(target)
result = {"prompt_text": prompt, "target_text": target, "prompt_ids": pids, "response_ids": rids,
          "prompt_tokens": len(pids), "response_tokens": len(rids),
          "c_start": len(tpl.encode(target[:char_start])), "r_start": len(tpl.encode(target[:char_start])),
          "template_sha256": tpl.template_sha256, "tokenizer_sha256": tpl.tokenizer_sha256,
          "target_sha256": hashlib.sha256(target.encode("utf-8")).hexdigest()}
json.dump(result, sys.stdout, ensure_ascii=False, separators=(",", ":"))
'''


def _legacy_bridge_sample(normalized: dict[str, Any], reasoning_effort: str, python_exe: str | Path | None = None) -> dict[str, Any] | None:
    """在 .venv-live 缺 tokenizers 时调用已有 .venv 做精确渲染；返回 None 表示不可用。"""
    exe = _legacy_python(python_exe)
    if exe is None:
        return None
    root = Path(__file__).resolve().parents[3]
    request = {
        "messages": normalized.get("messages") or [], "tools": normalized.get("tools"),
        "with_tools": normalized.get("with_tools", False), "tool_calls": normalized.get("tool_calls") or [],
        "content": normalized.get("content", ""), "reasoning": normalized.get("reasoning", ""),
        "reasoning_effort": reasoning_effort,
    }
    try:
        proc = subprocess.run([str(exe), "-B", "-c", _BRIDGE_SCRIPT], cwd=str(root),
                              env={**os.environ, "PYTHONPATH": "harness", "PYTHONIOENCODING": "utf-8"},
                              input=json.dumps(request, ensure_ascii=False), text=True,
                              encoding="utf-8", capture_output=True, check=False, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    try:
        obj = json.loads(proc.stdout)
    except (TypeError, ValueError):
        return None
    return obj if isinstance(obj, dict) and isinstance(obj.get("prompt_ids"), list) else None


def _render(template: Any, messages: list[dict[str, Any]], tools: Any, *, generation: bool, effort: str) -> str:
    return template.render(messages, tools, add_generation_prompt=generation, reasoning_effort=effort)


def _assistant_message(call: dict[str, Any]) -> dict[str, Any]:
    # 复用旧模块对原生 tool_calls 的精确渲染；JSON 工具协议没有 tool_calls 时只保留 content。
    if _sft is not None:
        return _sft.assistant_message(call, call.get("reasoning", ""))
    # 仅在无法导入旧模块时使用；有 tokenizer 的运行会走上面的官方实现。
    message = {"role": "assistant", "content": call.get("content", "").strip(),
               "reasoning_content": call.get("reasoning", "")}
    if call.get("tool_calls"):
        message["tool_calls"] = call["tool_calls"]
    return message


def _content_start(target: str, content: str, reasoning: str) -> int | None:
    if content.strip():
        marker = content.strip()
        # 思考文本中可能重复 JSON；优先在结束思考标记后查找。
        after_think = target.find("</think>")
        start = after_think + len("</think>") if after_think >= 0 else 0
        found = target.find(marker, start)
        if found >= 0:
            return found
        found = target.find(content, start)
        if found >= 0:
            return found
    close = target.find("</think>")
    if close >= 0:
        return close + len("</think>")
    return 0 if target else None


def render_call_sample(
    trace: dict[str, Any],
    call: dict[str, Any],
    *,
    call_index: int = 0,
    template: Any = None,
    profile_dir: Path | None = None,
    reasoning_effort: str = "xhigh",
    python_exe: str | Path | None = None,
) -> dict[str, Any]:
    """渲染一次调用，返回 prompt/response token 与报告段起点。

    ``c_start``/``r_start`` 都是响应 token 内的偏移；对 chosen/rejected 使用
    同一 prompt 时可直接写入 ``train_simpo.py --report_only``。
    """
    normalized = normalize_call(trace, call, call_index)
    tpl, error = _template(template, profile_dir)
    base = {
        "call_index": normalized["call_index"],
        "role": normalized["role"],
        "reasoning_effort": reasoning_effort,
        "with_tools": normalized["with_tools"],
        "context_digest": normalized.get("context_digest"),
        "request_sha256": normalized.get("request_sha256"),
        "content_chars": len(normalized["content"]),
        "reasoning_chars": len(normalized["reasoning"]),
        "tokenizer": {
            "status": "unavailable" if error else "exact",
            "error": error,
            "template_sha256": getattr(tpl, "template_sha256", None),
            "tokenizer_sha256": getattr(tpl, "tokenizer_sha256", None),
        },
    }
    if tpl is None:
        bridged = _legacy_bridge_sample(normalized, reasoning_effort, python_exe)
        if bridged is not None:
            final_start = bridged.get("c_start")
            return {
                **base,
                "status": "exact",
                "tokenizer": {"status": "exact_bridge", "error": error,
                              "template_sha256": bridged.get("template_sha256"),
                              "tokenizer_sha256": bridged.get("tokenizer_sha256")},
                "prompt_text": bridged.get("prompt_text"), "target_text": bridged.get("target_text"),
                "prompt_ids": bridged.get("prompt_ids", []), "response_ids": bridged.get("response_ids", []),
                "prompt_tokens": bridged.get("prompt_tokens"), "response_tokens": bridged.get("response_tokens"),
                "c_start": final_start, "r_start": bridged.get("r_start", final_start),
                "segments": {"thinking_start": 0, "content_start": final_start,
                             "response_end": bridged.get("response_tokens")},
                "target_sha256": bridged.get("target_sha256"),
            }
        return {**base, "status": "unavailable", "prompt_ids": [], "response_ids": [], "c_start": None, "r_start": None,
                "prompt_tokens": None, "response_tokens": None, "target_text": None, "segments": {}}
    messages = normalized["messages"]
    tools = normalized["tools"] if normalized["with_tools"] else None
    try:
        prompt_text = _render(tpl, messages, tools, generation=True, effort=reasoning_effort)
        assistant = _assistant_message(normalized)
        full_text = _render(tpl, messages + [assistant], tools, generation=False, effort=reasoning_effort)
        if not full_text.startswith(prompt_text):
            raise ValueError("完整对话不是生成提示的前缀")
        target_text = full_text[len(prompt_text):]
        if not target_text.endswith(EOS + "\n"):
            raise ValueError("目标未以 <|im_end|> 结尾")
        target_text = target_text[:-1]
        prompt_ids = tpl.encode(prompt_text)
        response_ids = tpl.encode(target_text)
        char_start = _content_start(target_text, normalized["content"], normalized["reasoning"])
        final_start = len(tpl.encode(target_text[:char_start])) if char_start is not None else None
        think_end = target_text.find("</think>")
        think_start = len(tpl.encode(target_text[:think_end])) if think_end >= 0 else 0
        result = {
            **base,
            "status": "exact",
            "prompt_text": prompt_text,
            "target_text": target_text,
            "prompt_ids": prompt_ids,
            "response_ids": response_ids,
            "prompt_tokens": len(prompt_ids),
            "response_tokens": len(response_ids),
            "c_start": final_start,
            "r_start": final_start,
            "segments": {
                "thinking_start": 0,
                "thinking_end": think_start,
                "content_start": final_start,
                "content_char_start": char_start,
                "response_end": len(response_ids),
            },
            "target_sha256": hashlib.sha256(target_text.encode("utf-8")).hexdigest(),
        }
        return result
    except Exception as exc:
        return {**base, "status": "error", "error": f"模板渲染失败: {type(exc).__name__}: {exc}",
                "prompt_ids": [], "response_ids": [], "c_start": None, "r_start": None,
                "prompt_tokens": None, "response_tokens": None, "target_text": None, "segments": {}}


def sample_trace_calls(
    trace: dict[str, Any],
    *,
    template: Any = None,
    profile_dir: Path | None = None,
    reasoning_effort: str = "xhigh",
    roles: Iterable[str] | None = None,
    python_exe: str | Path | None = None,
) -> list[dict[str, Any]]:
    """将 trace.calls 全部转换为逐调用 token 样本，保留 0-based call_index。"""
    allowed = set(roles) if roles is not None else None
    out = []
    for index, call in enumerate(trace.get("calls") or []):
        role = str(call.get("role", "")) if isinstance(call, dict) else ""
        if allowed is not None and role not in allowed:
            continue
        out.append(render_call_sample(trace, call, call_index=index, template=template,
                                       profile_dir=profile_dir, reasoning_effort=reasoning_effort,
                                       python_exe=python_exe))
    return out


def count_payload(
    payload: dict[str, Any],
    tpl: Any = None,
    *,
    reasoning_effort: str = "xhigh",
    response: dict[str, Any] | None = None,
    reasoning_content: str | None = None,
    content: Any = None,
    python_exe: str | Path | None = None,
) -> dict[str, Any]:
    """计算一次线上 payload 的真实模板 token 数。

    没有可用 tokenizer 时返回 ``token_count=None`` 与 ``exact=False``；调用方
    必须据此统计未计数比例，不能把字符数当 token 数。
    """
    payload = _dump(payload) or {}
    trace = {"calls": []}
    call = {"payload": payload, "response": response or {}, "reasoning_content": reasoning_content, "content": content}
    sample = render_call_sample(trace, call, template=tpl, reasoning_effort=reasoning_effort, python_exe=python_exe)
    if sample["status"] != "exact":
        return {"status": sample["status"], "exact": False, "token_count": None,
                "prompt_tokens": sample.get("prompt_tokens"), "response_tokens": sample.get("response_tokens"),
                "error": sample.get("error") or sample.get("tokenizer", {}).get("error")}
    return {"status": "exact", "exact": True, "token_count": sample["prompt_tokens"] + sample["response_tokens"],
            "prompt_tokens": sample["prompt_tokens"], "response_tokens": sample["response_tokens"],
            "c_start": sample["c_start"], "r_start": sample["r_start"],
            "template_sha256": sample["tokenizer"].get("template_sha256"),
            "tokenizer_sha256": sample["tokenizer"].get("tokenizer_sha256")}


def window_guard(sample: dict[str, Any], *, max_len: int = 32768, stable_limit: int = 24000) -> dict[str, Any]:
    """训练窗口预检：24K 是稳定线，32768 是硬窗口；未知 token 绝不判通过。"""
    prompt_tokens, response_tokens = sample.get("prompt_tokens"), sample.get("response_tokens")
    if sample.get("status") not in {"exact", "exact_bridge"} or not isinstance(prompt_tokens, int) or not isinstance(response_tokens, int):
        return {"status": "unknown", "exact": False, "total_tokens": None, "stable_ok": None, "hard_ok": None,
                "max_len": max_len, "stable_limit": stable_limit, "reason": "没有真实 tokenizer 计数"}
    total = prompt_tokens + response_tokens
    return {"status": "pass" if total <= stable_limit else ("warn" if total <= max_len else "fail"),
            "exact": True, "total_tokens": total, "prompt_tokens": prompt_tokens, "response_tokens": response_tokens,
            "stable_ok": total <= stable_limit, "hard_ok": total <= max_len,
            "max_len": max_len, "stable_limit": stable_limit}


preflight_window = window_guard


__all__ = ["normalize_call", "render_call_sample", "sample_trace_calls", "count_payload", "window_guard", "preflight_window"]
