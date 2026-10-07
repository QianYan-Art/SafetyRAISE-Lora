"""可插拔模型后端；真实发送沿用旧客户端的账本及外发闸门。"""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from typing import Any, Callable, Protocol

from sr_eval.llm import LLMClient, LLMError, load_models
from sr_eval.policy import OutboundPolicy


class ChatBackend(Protocol):
    deterministic: bool

    def model_for(self, role: str) -> str: ...

    def effort_for(self, role: str) -> str: ...

    async def chat(self, role: str, payload: dict[str, Any]) -> dict[str, Any]: ...


class RoleBackends:
    """生成者和独立审查者可使用不同后端，仍共享整次运行预算。"""

    def __init__(self, backends: dict[str, ChatBackend]):
        if not {"generator", "reviewer"} <= backends.keys():
            raise ValueError("必须分别登记生成者与审查者后端。")
        self.backends = dict(backends)
        self.deterministic = all(item.deterministic for item in backends.values())
        self.last_payload = None
        self.before_attempt = None

    def model_for(self, role):
        return self.backends.get(role, self.backends["generator"]).model_for(role)

    def effort_for(self, role):
        return self.backends.get(role, self.backends["generator"]).effort_for(role)

    def output_limit_for(self, role):
        return getattr(self.backends[role], "max_tokens", 64000)

    def bind_before_attempt(self, callback):
        self.before_attempt = callback
        for backend in self.backends.values():
            bind = getattr(backend, "bind_before_attempt", None)
            if bind is not None:
                bind(callback)

    async def chat(self, role, payload):
        backend = self.backends[role]
        if getattr(backend, "bind_before_attempt", None) is None and self.before_attempt is not None:
            self.before_attempt()
        response = await backend.chat(role, payload)
        self.last_payload = deepcopy(getattr(backend, "last_payload", None) or payload)
        return response


def openai_response(content: Any, *, reasoning: str = "", finish_reason: str = "stop",
                    prompt_tokens: int = 0, completion_tokens: int = 0,
                    model: str = "scripted") -> dict:
    """脚本响应也显式提供用量，不把未知用量伪装成已结算。"""
    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    return {
        "model": model,
        "choices": [{"index": 0, "finish_reason": finish_reason,
                     "message": {"role": "assistant", "content": text,
                                 "reasoning_content": reasoning}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens},
    }


class ScriptedBackend:
    """列表或回调驱动的无网络后端；回调接收角色、实际请求及调用序号。"""

    deterministic = True

    def __init__(self, script: list | Callable, *, model="tencent/hy4-preview"):
        self.script = script
        self.model = model
        self.index = 0
        self.requests: list[dict] = []
        self.last_payload: dict | None = None

    def model_for(self, role):
        return self.model

    def effort_for(self, role):
        return "high"

    async def chat(self, role, payload):
        self.last_payload = deepcopy(payload)
        self.requests.append({"role": role, "payload": deepcopy(payload)})
        index = self.index
        self.index += 1
        if callable(self.script):
            result = self.script(role, deepcopy(payload), index)
            if asyncio.iscoroutine(result):
                result = await result
        else:
            if index >= len(self.script):
                raise RuntimeError("脚本响应已耗尽，不能生成假成功响应。")
            result = self.script[index]
            if isinstance(result, tuple):
                expected_role, result = result
                if expected_role != role:
                    raise AssertionError(f"脚本角色不符：{expected_role} != {role}")
            if callable(result):
                result = result(role, deepcopy(payload), index)
        if isinstance(result, Exception):
            raise result
        return deepcopy(result if isinstance(result, dict) and "choices" in result
                        else openai_response(result, model=self.model))


class _PayloadClient(LLMClient):
    """局部扩展请求体，不修改旧客户端或绕过其发送、结算路径。"""

    wire_payload: dict | None = None
    effective_payload: dict | None = None

    def _body(self, spec, messages, tools, max_tokens, reasoning, temperature):
        body = super()._body(spec, messages, None, max_tokens, reasoning, temperature)
        wire = self.wire_payload or {}
        if "response_format" in wire:
            body["response_format"] = deepcopy(wire["response_format"])
        if "verbosity" in wire:
            body["verbosity"] = wire["verbosity"]
        self.effective_payload = deepcopy(body)
        return body


class _RecordingTransport:
    """透明保留原始服务响应；物理重试也必须逐次经过内存预算。"""

    def __init__(self, delegate):
        self.delegate = delegate
        self.before_attempt = None
        self.response = None
        self.attempts = []

    def route_for(self, host):
        return getattr(self.delegate, "route_for", lambda _: "direct")(host)

    def request(self, host, method, path, headers, body, timeout):
        if self.before_attempt is not None:
            self.before_attempt()
        result = self.delegate.request(host, method, path, headers, body, timeout)
        status, _, data = result
        self.attempts.append({"http_status": status, "method": method})
        if status == 200:
            try:
                self.response = json.loads(data)
            except (TypeError, ValueError):
                self.response = None
        return result


class _LedgerBackend:
    deterministic = False
    model_key: str
    provider: str

    def __init__(self, *, client: LLMClient | None = None, roles=None,
                 max_tokens=None, timeout=1800, allow_network=False,
                 case_kind="synthetic", kb_class="synthetic", policy_path=None):
        self.roles = frozenset(roles or ("generator", "reviewer", "reviser"))
        if not self.roles <= {"generator", "reviewer", "reviser"}:
            raise ValueError("仅允许登记生成、审查和改写角色。")
        source = client or LLMClient(load_models(), scope="live-env")
        self.recording_transport = _RecordingTransport(source.transport)
        self.client = _PayloadClient(
            models=source.models, ledger=source.ledger, transport=self.recording_transport,
            cap_usd=source.cap_usd, scope=source.scope, sleep=source.sleep,
        )
        spec = self.client.models[self.model_key]
        if spec.provider != self.provider:
            raise ValueError("模型提供者与后端不符，禁止路由至其他提供者。")
        if max_tokens is None:
            max_tokens = min(spec.max_output_limit, 64000)
        if type(max_tokens) is not int or not 0 < max_tokens <= spec.max_output_limit:
            raise ValueError("输出预留必须在登记的模型容量内。")
        self.max_tokens, self.timeout = max_tokens, timeout
        self.allow_network = allow_network
        self.case_kind, self.kb_class = case_kind, kb_class
        self.policy_path = policy_path
        self.last_payload: dict | None = None
        self.last_result: dict | None = None
        self._lock = asyncio.Lock()

    def bind_before_attempt(self, callback):
        self.recording_transport.before_attempt = callback

    def model_for(self, role):
        return self.client.models[self.model_key].slug

    async def chat(self, role, payload):
        if role not in self.roles:
            raise LLMError("role_not_authorized")
        if not self.allow_network:
            raise LLMError("live_network_disabled", detail="本轮禁止外部模型调用。")
        OutboundPolicy.load(self.policy_path).check(
            kb_class=self.kb_class, case_kind=self.case_kind,
        )
        if self.case_kind != "synthetic":
            raise LLMError("live_real_case_forbidden")
        async with self._lock:
            self.recording_transport.response = None
            self.recording_transport.attempts = []
            self.client.wire_payload = deepcopy(payload)
            result = await asyncio.to_thread(
                self.client.chat, self.model_key, payload["messages"],
                tools=None, max_tokens=self.max_tokens,
                reasoning={"effort": self.effort_for(role)},
                temperature=payload.get("temperature"), timeout=self.timeout,
                purpose=f"live:{role}",
                meta={"case_kind": self.case_kind, "kb_class": self.kb_class,
                      "live_wire_model": payload["model"]},
            )
            self.last_payload = deepcopy(self.client.effective_payload)
            self.last_result = result.to_dict()
        response = deepcopy(self.recording_transport.response)
        if not isinstance(response, dict):
            raise LLMError("invalid_provider_response", billing_state="unknown")
        response["live_ledger"] = {
            "call_id": result.call_id, "priced": result.priced, "cost_usd": result.cost_usd,
            "attempts": deepcopy(self.recording_transport.attempts),
        }
        return response


class MiniMaxBackend(_LedgerBackend):
    """MiniMax-M3.1-Flash-Preview，固定 max；默认禁止发送。"""

    model_key, provider = "minimax", "minimax"

    def effort_for(self, role):
        return "max"


class LocalOpenAIBackend(_LedgerBackend):
    """沿用 local27 和 SR_LOCAL_HOST；不探测、不启动、不修改学生服务。"""

    model_key, provider = "local27", "local"

    def effort_for(self, role):
        # 本地学生的思考档位:环境变量 SR_LIVE_EFFORT(xhigh|medium|low;Qwen3.8 模板不认 high),默认 xhigh
        import os
        return os.environ.get("SR_LIVE_EFFORT", "xhigh")
