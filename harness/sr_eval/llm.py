"""OpenAI 兼容聊天客户端 + 无密钥账本 + 预算硬门。

- 通道按主机路由(config/network.json):OpenRouter 经环境代理(仓库所有者 2026-10-02 指示),MiniMax 直连;
- 密钥只在进程内读取(环境变量或补充.txt 的指定字段),不进入账本、日志、异常文本;
- 每次请求先按最坏情况预留费用,超过预算上限就不发送;预留与结算都写入账本;
- 不自动重试可能已生成结果的请求(网络中断 → billing_state=unknown,由人对账)。
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import ssl
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

HARNESS_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = HARNESS_DIR.parent
CONFIG_DIR = HARNESS_DIR / "config"
LEDGER_PATH = HARNESS_DIR / "ledger" / "api-calls.jsonl"
SECRETS_FILE = DATASET_DIR / "补充.txt"

PROVIDERS = {
    "openrouter": {"host": "openrouter.ai", "path": "/api/v1/chat/completions", "key_env": ("OPENROUTER_API_KEY",),
                   "key_fields": ("openrouter_key", "openrouter_api_key")},
    # Orin 本地评测服务:8000 = harness/orin/serve_local.py(HF,慢),8080 = llama-server(GGUF);用环境变量 SR_LOCAL_HOST 切换
    "local": {"host": os.environ.get("SR_LOCAL_HOST", "192.168.55.1:8000"), "path": "/v1/chat/completions", "key_env": (), "key_fields": ()},
    "minimax": {"host": "api.minimaxi.com", "path": "/v1/chat/completions", "key_env": ("MINIMAX_API_KEY",),
                "key_fields": ("key",)},
}


class LLMError(RuntimeError):
    def __init__(self, code: str, *, billing_state: str = "not_sent", http_status: int | None = None, detail: str = "") -> None:
        super().__init__(f"{code} (http={http_status}, billing={billing_state}) {detail}".strip())
        self.code = code
        self.billing_state = billing_state
        self.http_status = http_status
        self.detail = detail


class BudgetExceeded(LLMError):
    def __init__(self, detail: str) -> None:
        super().__init__("budget_exceeded", billing_state="not_sent", detail=detail)


@dataclass(frozen=True)
class ModelSpec:
    name: str                      # 评测台内部别名
    provider: str                  # openrouter | minimax
    slug: str                      # 供应商侧模型 ID
    price_in: float                # USD / token(minimax 未定价记 0,priced=False)
    price_out: float
    priced: bool = True
    supports_tools: bool = True
    reasoning: dict[str, Any] | None = None   # openrouter: {"effort": "high"};minimax 见 _body
    max_output_limit: int = 64000  # 供应商文档上限(预留费用不得超过该上限)
    role: str = "candidate"        # candidate | judge | both
    price_date: str = ""


def load_models(path: Path | None = None) -> dict[str, ModelSpec]:
    data = json.loads((path or CONFIG_DIR / "models.json").read_text(encoding="utf-8"))
    return {m["name"]: ModelSpec(**m) for m in data["models"]}


def read_secret(env_names: tuple[str, ...], file_fields: tuple[str, ...], secrets_file: Path | None = None) -> str | None:
    for name in env_names:
        v = os.environ.get(name)
        if v and v.strip():
            return v.strip()
    path = secrets_file or SECRETS_FILE
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        m = re.match(r"^\s*([^:=：\s]{1,40})\s*[:=：]\s*(.+?)\s*$", line)
        if m and m.group(1).strip().lower() in file_fields:
            return m.group(2).strip()
    return None


def _scrub(text: str, secret: str | None) -> str:
    return text.replace(secret, "***") if secret else text


def _sha(obj: Any) -> str:
    raw = obj if isinstance(obj, bytes) else json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class Transport(Protocol):
    def request(self, host: str, method: str, path: str, headers: dict[str, str], body: bytes | None,
                timeout: float) -> tuple[int, dict[str, str], bytes]: ...


class DirectTransport:
    """HTTPS 直连(http.client 本身不使用任何代理环境变量)。"""

    def request(self, host, method, path, headers, body, timeout):
        conn = http.client.HTTPSConnection(host, timeout=timeout, context=ssl.create_default_context())
        try:
            conn.request(method, path, body=body, headers=headers)
            resp = conn.getresponse()
            data = resp.read(32 * 1024 * 1024)
            return resp.status, {k.lower(): v for k, v in resp.getheaders()}, data
        finally:
            conn.close()


def env_proxy_url(environ: dict[str, str] | None = None) -> tuple[str, int]:
    """从进程环境读 HTTP(S) 代理(只支持 http:// 代理做 CONNECT 隧道);读不到则抛出发送前错误。"""
    from urllib.parse import urlparse
    env = os.environ if environ is None else environ
    for name in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        val = env.get(name)
        if val:
            u = urlparse(val if "//" in val else "http://" + val)
            if u.scheme == "http" and u.hostname and u.port:
                return u.hostname, u.port
    raise LLMError("proxy_unavailable", billing_state="not_sent", detail="按网络配置此主机必须走代理,但环境里没有可用的 http:// 代理(HTTPS_PROXY)")


class ProxyTransport:
    """经 HTTP 代理的 CONNECT 隧道访问 HTTPS 站点;代理地址每次请求时从环境读取,不落盘。"""

    def __init__(self, environ: dict[str, str] | None = None, connection_factory=http.client.HTTPSConnection) -> None:
        self._environ = environ
        self._factory = connection_factory

    def request(self, host, method, path, headers, body, timeout):
        ph, pp = env_proxy_url(self._environ)
        conn = self._factory(ph, pp, timeout=timeout, context=ssl.create_default_context())
        conn.set_tunnel(host, 443)
        try:
            conn.request(method, path, body=body, headers=headers)
            resp = conn.getresponse()
            data = resp.read(32 * 1024 * 1024)
            return resp.status, {k.lower(): v for k, v in resp.getheaders()}, data
        finally:
            conn.close()


class HttpTransport:
    """明文 HTTP(只用于局域网内的 Orin 本地评测服务,host 形如 192.168.55.1:8000)。"""

    def request(self, host, method, path, headers, body, timeout):
        conn = http.client.HTTPConnection(host, timeout=timeout)
        try:
            conn.request(method, path, body=body, headers=headers)
            resp = conn.getresponse()
            data = resp.read(32 * 1024 * 1024)
            return resp.status, {k.lower(): v for k, v in resp.getheaders()}, data
        finally:
            conn.close()


class RoutingTransport:
    """按主机选择通道:config/network.json 的 routes[host] = "env_proxy" | "direct"(默认 direct)。"""

    def __init__(self, routes: dict[str, str] | None = None, direct: Transport | None = None, proxy: Transport | None = None) -> None:
        if routes is None:
            cfg = CONFIG_DIR / "network.json"
            routes = json.loads(cfg.read_text(encoding="utf-8"))["routes"] if cfg.exists() else {}
        self.routes = routes
        self._direct = direct or DirectTransport()
        self._proxy = proxy or ProxyTransport()
        self._http = HttpTransport()

    def route_for(self, host: str) -> str:
        return self.routes.get(host, "direct")

    def request(self, host, method, path, headers, body, timeout):
        route = self.route_for(host)
        t = self._proxy if route == "env_proxy" else self._http if route == "http" else self._direct
        return t.request(host, method, path, headers, body, timeout)


# ---------------- 账本与预算 ----------------

class Ledger:
    """追加式 JSONL 账本;预算 = 已结算实际费用 + 未结算(含 unknown)预留。线程安全。"""

    def __init__(self, path: Path = LEDGER_PATH) -> None:
        self.path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def _events(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def committed_usd(self, scope: str | None = None) -> float:
        reserved: dict[str, float] = {}
        actual: dict[str, float] = {}
        for ev in self._events():
            if scope and ev.get("scope") != scope:
                continue
            cid = ev["call_id"]
            if ev["event"] == "reserved":
                reserved[cid] = ev["reserved_usd"]
            elif ev["event"] in {"completed", "failed_known_zero"}:
                actual[cid] = ev.get("actual_usd") or 0.0
        total = sum(actual.values())
        total += sum(v for cid, v in reserved.items() if cid not in actual)  # 未结算/unknown 按预留计
        return total

    def append(self, event: dict[str, Any]) -> None:
        event = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **event}
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")

    def reserve(self, call_id: str, cap_usd: float, amount: float, scope: str, meta: dict[str, Any]) -> None:
        with self._lock:
            committed = self.committed_usd(scope)
            if committed + amount > cap_usd + 1e-12:
                raise BudgetExceeded(f"已用/预留 {committed:.4f} + 本次最坏 {amount:.4f} > 上限 {cap_usd:.2f} USD")
            event = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "event": "reserved",
                     "call_id": call_id, "scope": scope, "reserved_usd": round(amount, 6), "cap_usd": cap_usd, **meta}
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, ensure_ascii=False) + "\n")


@dataclass
class ChatResult:
    content: str
    reasoning: str
    tool_calls: list[dict[str, Any]]
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    cost_usd: float
    priced: bool
    response_model: str
    request_id: str | None
    latency_s: float
    call_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    calls = []
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function") or {}
        raw = fn.get("arguments", "")
        try:
            args = json.loads(raw) if isinstance(raw, str) and raw.strip() else (raw if isinstance(raw, dict) else {})
            parse_ok = isinstance(args, dict)
        except ValueError:
            args, parse_ok = {}, False
        calls.append({"id": tc.get("id"), "name": fn.get("name"), "arguments": args if parse_ok else {},
                      "arguments_raw": raw, "arguments_valid": parse_ok})
    return calls


@dataclass
class LLMClient:
    models: dict[str, ModelSpec]
    ledger: Ledger = field(default_factory=Ledger)
    transport: Transport = field(default_factory=RoutingTransport)
    cap_usd: float = 0.0
    scope: str = "default"
    sleep: Callable[[float], None] = time.sleep

    # ---- 密钥/上限核验(无费用的 GET /key) ----
    def openrouter_key_status(self) -> dict[str, Any]:
        key = read_secret(*[PROVIDERS["openrouter"][k] for k in ("key_env", "key_fields")])
        if not key:
            raise LLMError("credential_unavailable", detail="未找到 OPENROUTER_API_KEY 或 补充.txt 的 openrouter_key 字段")
        status, _h, body = self.transport.request("openrouter.ai", "GET", "/api/v1/key",
                                                  {"Authorization": f"Bearer {key}", "Accept": "application/json"}, None, 30)
        if status != 200:
            raise LLMError("key_status_failed", http_status=status)
        d = json.loads(body).get("data", {})
        return {"limit": d.get("limit"), "limit_remaining": d.get("limit_remaining"), "usage": d.get("usage"),
                "is_free_tier": d.get("is_free_tier")}

    def openrouter_credits(self) -> dict[str, Any]:
        """账户额度(无费用的只读 GET /credits):total_credits - total_usage = 剩余预付额度。"""
        key = read_secret(*[PROVIDERS["openrouter"][k] for k in ("key_env", "key_fields")])
        if not key:
            raise LLMError("credential_unavailable", detail="未找到 OpenRouter 密钥")
        status, _h, body = self.transport.request("openrouter.ai", "GET", "/api/v1/credits",
                                                  {"Authorization": f"Bearer {key}", "Accept": "application/json"}, None, 30)
        if status != 200:
            raise LLMError("credits_status_failed", http_status=status)
        d = json.loads(body).get("data", {})
        total, used = d.get("total_credits"), d.get("total_usage")
        return {"total_credits": total, "total_usage": used,
                "remaining": round(total - used, 4) if isinstance(total, (int, float)) and isinstance(used, (int, float)) else None}

    # ---- 请求体 ----
    def _body(self, spec: ModelSpec, messages, tools, max_tokens, reasoning, temperature) -> dict[str, Any]:
        body: dict[str, Any] = {"model": spec.slug, "messages": messages, "stream": False}
        if spec.provider == "local":
            body.update({"max_tokens": max_tokens,
                         "chat_template_kwargs": {"reasoning_effort": (reasoning or spec.reasoning or {}).get("effort", "low")}})
        elif spec.provider == "minimax":
            effort = (reasoning or spec.reasoning or {}).get("effort", "max")
            if effort not in {"xhigh", "max"}:  # 适配器文档只登记了这两档
                effort = "xhigh"
            body.update({"thinking": {"type": "adaptive"}, "reasoning_effort": effort,
                         "reasoning_split": True, "max_completion_tokens": max_tokens})
        else:
            body["max_tokens"] = max_tokens
            eff = reasoning if reasoning is not None else spec.reasoning
            if eff:
                body["reasoning"] = eff
        if temperature is not None:
            body["temperature"] = temperature
        if tools and spec.supports_tools:
            body["tools"] = tools
            if spec.provider in {"openrouter", "local"}:   # 与生产 openai_report._build_payload 一致
                body["tool_choice"] = "auto"
                body["parallel_tool_calls"] = False
        return body

    def chat(self, model: str, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = None,
             max_tokens: int, reasoning: dict[str, Any] | None = None, temperature: float | None = None,
             timeout: float = 1800, purpose: str = "", meta: dict[str, Any] | None = None) -> ChatResult:
        spec = self.models[model]
        provider = PROVIDERS[spec.provider]
        if max_tokens > spec.max_output_limit:
            raise LLMError("max_tokens_above_limit", detail=f"{max_tokens} > {spec.max_output_limit}")
        key = "local" if spec.provider == "local" else read_secret(provider["key_env"], provider["key_fields"])
        if not key:
            raise LLMError("credential_unavailable", detail=f"{spec.provider} 密钥不可用")
        body = self._body(spec, messages, tools, max_tokens, reasoning, temperature)
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        # 最坏费用:输入按 1 token/字符的保守上界,输出按 max_tokens(含思考)
        est_in = len(json.dumps(messages, ensure_ascii=False)) + (len(json.dumps(tools, ensure_ascii=False)) if tools else 0)
        worst = est_in * spec.price_in + max_tokens * spec.price_out
        call_id = "call_" + uuid.uuid4().hex[:16]
        route = getattr(self.transport, "route_for", lambda _h: "direct")(provider["host"])
        base = {"call_id": call_id, "scope": self.scope, "model": model, "slug": spec.slug, "provider": spec.provider, "route": route,
                "purpose": purpose, "payload_sha256": _sha(payload), "input_sha256": _sha(messages), **(meta or {})}
        if spec.priced:
            self.ledger.reserve(call_id, self.cap_usd, worst, self.scope, base)
        else:  # MiniMax 套餐额度未定价:不计美元,仍记录请求
            self.ledger.append({"event": "reserved", "reserved_usd": 0.0, "unpriced": True, **base})

        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json", "Accept": "application/json"}
        t0 = time.time()
        status = None
        # 明确"未生成"的限流/过载(429 限流、503 不可用、529 服务过载):指数退避重试最多 5 次(5/15/45/90/150 秒,加抖动);
        # 2026-10-06 起加入 529——此前 529 不在重试集合里,一次过载就让整批评审终止(并发 16 路 max 档时发生)。
        waits = (5, 15, 45, 90, 150)
        for attempt in range(len(waits) + 1):
            try:
                status, hdrs, data = self.transport.request(provider["host"], "POST", provider["path"], headers, payload, timeout)
            except (OSError, http.client.HTTPException) as exc:
                self.ledger.append({"event": "unknown", "call_id": call_id, "scope": self.scope,
                                    "error": type(exc).__name__})
                raise LLMError("network_outcome_unknown", billing_state="unknown", detail=_scrub(str(exc), key)) from exc
            if status in {429, 503, 529} and attempt < len(waits):
                # MiniMax 的 429 里"已达到 Token Plan 用量上限(2056)"是窗口型限额(2026-10-06 实测约 40 分钟内恢复,
                # 套餐总额度并未用完):遇到它改用长等待(5/10/15/20/25 分钟),不要几十秒内空转重试。
                quota_window = status == 429 and (b"2056" in data or "用量上限".encode("utf-8") in data)
                wait = (300 * (attempt + 1)) if quota_window else waits[attempt]
                self.sleep(wait * (0.8 + 0.4 * (hash(call_id) % 100) / 100))
                continue
            break
        latency = time.time() - t0
        if status != 200:
            snippet = _scrub(data[:300].decode("utf-8", "replace"), key)
            self.ledger.append({"event": "failed_known_zero" if status in {400, 401, 402, 403, 404, 422, 429, 503, 529} else "unknown",
                                "call_id": call_id, "scope": self.scope, "http_status": status, "actual_usd": 0.0 if status in {400, 401, 402, 403, 404, 422, 429, 503, 529} else None})
            raise LLMError("provider_http_error", billing_state="known_zero" if status in {400, 401, 402, 403, 404, 422, 429, 503, 529} else "unknown",
                           http_status=status, detail=snippet)
        try:
            obj = json.loads(data)
            choice = obj["choices"][0]
            msg = choice.get("message") or {}
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            head = _scrub(data[:400].decode("utf-8", "replace"), key)
            self.ledger.append({"event": "unknown", "call_id": call_id, "scope": self.scope, "error": "invalid_response",
                                "response_head": head, "latency_s": round(latency, 1)})
            raise LLMError("invalid_provider_response", billing_state="unknown", http_status=status, detail=head[:300]) from exc
        usage = obj.get("usage") or {}
        p_tok = int(usage.get("prompt_tokens") or 0)
        c_tok = int(usage.get("completion_tokens") or 0)
        r_tok = int(((usage.get("completion_tokens_details") or {}).get("reasoning_tokens")) or 0)
        if spec.priced:
            reported = usage.get("cost")
            cost = float(reported) if isinstance(reported, (int, float)) else p_tok * spec.price_in + c_tok * spec.price_out
        else:
            cost = 0.0
        reasoning_text = msg.get("reasoning_content") or msg.get("reasoning") or ""
        if not isinstance(reasoning_text, str):
            reasoning_text = json.dumps(reasoning_text, ensure_ascii=False)
        result = ChatResult(
            content=msg.get("content") or "", reasoning=reasoning_text, tool_calls=_parse_tool_calls(msg),
            finish_reason=choice.get("finish_reason") or "", prompt_tokens=p_tok, completion_tokens=c_tok,
            reasoning_tokens=r_tok, cost_usd=cost, priced=spec.priced, response_model=obj.get("model", ""),
            request_id=obj.get("id") or hdrs.get("x-request-id"), latency_s=round(latency, 2), call_id=call_id,
        )
        self.ledger.append({"event": "completed", "call_id": call_id, "scope": self.scope, "actual_usd": round(cost, 6),
                            "prompt_tokens": p_tok, "completion_tokens": c_tok, "reasoning_tokens": r_tok,
                            "finish_reason": result.finish_reason, "response_model": result.response_model,
                            "provider_request_id": result.request_id, "response_sha256": _sha(data),
                            "latency_s": result.latency_s})
        return result
