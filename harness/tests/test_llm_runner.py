import json

import pytest

from conftest import ok
from sr_eval import prompt as P
from sr_eval.kb import synthetic_kb
from sr_eval.llm import BudgetExceeded, LLMError
from sr_eval.policy import OutboundDenied, OutboundPolicy
from sr_eval.runner import RunConfig, run_case

MSG = [{"role": "user", "content": "你好"}]


def test_chat_records_cost_and_never_logs_secret(make_client, tmp_path):
    c, t = make_client([ok("答复", cost=0.0123)])
    r = c.chat("luna", MSG, max_tokens=1000)
    assert r.cost_usd == pytest.approx(0.0123) and r.reasoning_tokens == 100
    raw = (tmp_path / "ledger.jsonl").read_text(encoding="utf-8")
    assert "sk-or-test-secret" not in raw and '"event": "completed"' in raw
    assert t.requests[0]["body"]["reasoning"] == {"effort": "max"} and t.requests[0]["body"]["max_tokens"] == 1000
    assert t.requests[0]["headers"]["Authorization"].startswith("Bearer ")


def test_budget_blocks_before_sending(make_client):
    c, t = make_client([ok("x")], cap=0.0001)
    with pytest.raises(BudgetExceeded):
        c.chat("hy4", MSG, max_tokens=40000)  # 最坏费用 > 上限
    assert t.requests == []


def test_budget_counts_unsettled_reservations(make_client):
    c, _t = make_client([ok("x", cost=0.0001), ok("y")], cap=0.07)
    c.chat("hy4", MSG, max_tokens=20000)           # 最坏约 0.05,结算为 0.0001
    c.ledger.append({"event": "reserved", "call_id": "ghost", "scope": "openrouter", "reserved_usd": 0.06})  # 未结算的 unknown
    with pytest.raises(BudgetExceeded):
        c.chat("hy4", MSG, max_tokens=20000)


def test_retry_once_on_429_then_success(make_client):
    c, t = make_client([(429, {"error": "rate"}), ok("好")])
    assert c.chat("luna", MSG, max_tokens=500).content == "好" and len(t.requests) == 2


def test_network_error_is_unknown_and_not_retried(make_client, tmp_path):
    c, t = make_client([(0, OSError("reset")), ok("不应被使用")])
    with pytest.raises(LLMError) as e:
        c.chat("luna", MSG, max_tokens=500)
    assert e.value.billing_state == "unknown" and len(t.requests) == 1
    assert '"event": "unknown"' in (tmp_path / "ledger.jsonl").read_text(encoding="utf-8")


def test_max_tokens_above_model_limit_rejected(make_client):
    c, t = make_client([])
    with pytest.raises(LLMError):
        c.chat("hy4", MSG, max_tokens=200000)


def test_minimax_body_uses_recipe_and_is_unpriced(make_client):
    c, t = make_client([ok("好", cost=0)])
    c.chat("minimax", MSG, max_tokens=5000)
    b = t.requests[0]["body"]
    assert t.requests[0]["host"] == "api.minimaxi.com" and b["thinking"] == {"type": "adaptive"} and b["reasoning_split"] is True
    assert "max_completion_tokens" in b and "max_tokens" not in b


def test_tool_call_arguments_parsed(make_client):
    c, _t = make_client([ok("", tool_calls=[("retrieve_knowledge", {"query": "倒车", "top_k": 2}), ("x", "{坏json")])])
    r = c.chat("luna", MSG, max_tokens=500, tools=P.retrieve_tool_schema())
    assert r.tool_calls[0]["arguments"]["top_k"] == 2 and r.tool_calls[1]["arguments_valid"] is False


REPORT = "# 交通事故分析报告\n\n## 一、事故概要\n夜间无信号灯路口两车碰撞。[依据: synthetic_traffic_rules#synth_kb_003]\n"


def test_run_case_react_loop_rerenders_system_prompt(make_client, case):
    script = [
        ok("", tool_calls=[("retrieve_knowledge", {"query": "夜间 低能见度 行驶", "reason": "核对速度义务", "top_k": 2})]),
        ok(REPORT),
    ]
    c, t = make_client(script)
    tr = run_case(c, "luna", case, synthetic_kb(), RunConfig(), policy=OutboundPolicy.load(), run_id="t")
    assert tr["status"] == "ok" and tr["totals"]["retrieval_rounds"] == 1 and tr["totals"]["calls"] == 2
    second = t.requests[1]["body"]["messages"]
    assert [m["role"] for m in second] == ["system", "user"]            # 生产旧路径:没有 tool 消息
    assert "夜间 低能见度 行驶" in second[0]["content"]                   # 检索历史被重新渲染进 system
    assert len(tr["visible_snippets"]) <= 9 and tr["final_markdown"].startswith("# 交通事故分析报告")
    assert all(s["id"] in {x["id"] for x in tr["visible_snippets"]} for s in tr["rounds"][0]["snippets"])


def test_run_case_forces_finalize_after_max_rounds(make_client, case):
    tool = ("retrieve_knowledge", {"query": "q", "top_k": 1})
    script = [ok("", tool_calls=[tool]) for _ in range(3)] + [ok(REPORT)]
    c, t = make_client(script)
    tr = run_case(c, "luna", case, synthetic_kb(), RunConfig(exec_variant="production"), policy=OutboundPolicy.load())
    assert tr["totals"]["retrieval_rounds"] == 2 and tr.get("forced_finalize") and tr["status"] == "ok"
    assert "tools" not in t.requests[3]["body"]                           # 强制收束不带工具
    assert "充分展开核心分析" in t.requests[3]["body"]["messages"][1]["content"]


def test_run_case_unknown_tool_is_error(make_client, case):
    c, _t = make_client([ok("", tool_calls=[("search_web", {"query": "x"})])])
    tr = run_case(c, "luna", case, synthetic_kb(), RunConfig(), policy=OutboundPolicy.load())
    assert tr["status"] == "error" and any(i["issue"] == "unknown_tool" for i in tr["tool_call_issues"])


def test_run_case_blocks_public_kb_before_any_request(make_client, case):
    kb = synthetic_kb()
    kb.outbound_class = "public_statutes"
    c, t = make_client([ok(REPORT)])
    with pytest.raises(OutboundDenied):
        run_case(c, "luna", case, kb, RunConfig(), policy=OutboundPolicy(synthetic_cases=True, public_statutes=False, real_cases=False))
    assert t.requests == []


def test_run_case_truncation_recorded(make_client, case):
    c, _t = make_client([ok(REPORT, finish="length")])
    tr = run_case(c, "luna", case, synthetic_kb(), RunConfig(), policy=OutboundPolicy.load())
    assert tr["totals"]["any_truncated"] is True


def test_openrouter_key_status_requires_limit(make_client):
    c, _t = make_client([(200, {"data": {"limit": None, "limit_remaining": None, "usage": 1.2}})])
    from sr_eval.cli import preflight_openrouter
    with pytest.raises(LLMError) as e:
        preflight_openrouter(c, ["luna"])
    assert e.value.code == "key_has_no_limit"


# ---- 通道路由:OpenRouter 经环境代理,MiniMax 直连 ----
def test_env_proxy_url_parsing_and_missing():
    from sr_eval.llm import env_proxy_url
    assert env_proxy_url({"HTTPS_PROXY": "http://127.0.0.1:7897"}) == ("127.0.0.1", 7897)
    assert env_proxy_url({"http_proxy": "127.0.0.1:8080"}) == ("127.0.0.1", 8080)
    with pytest.raises(LLMError) as e:
        env_proxy_url({"ALL_PROXY": "socks5://127.0.0.1:7897"})  # socks 不支持 → 发送前报错,而不是悄悄直连
    assert e.value.code == "proxy_unavailable" and e.value.billing_state == "not_sent"


def test_routing_transport_picks_channel_by_host():
    from sr_eval.llm import RoutingTransport

    class Tag:
        def __init__(self, name): self.name = name
        def request(self, *a, **k): return 200, {}, self.name.encode()

    rt = RoutingTransport({"openrouter.ai": "env_proxy", "api.minimaxi.com": "direct"}, direct=Tag("direct"), proxy=Tag("proxy"))
    assert rt.request("openrouter.ai", "GET", "/", {}, None, 1)[2] == b"proxy"
    assert rt.request("api.minimaxi.com", "GET", "/", {}, None, 1)[2] == b"direct"
    assert rt.request("elsewhere.example", "GET", "/", {}, None, 1)[2] == b"direct"
    import json as _j
    from sr_eval.llm import CONFIG_DIR
    routes = _j.loads((CONFIG_DIR / "network.json").read_text(encoding="utf-8"))["routes"]
    assert routes["openrouter.ai"] == "env_proxy" and routes["api.minimaxi.com"] == "direct"
    assert routes["192.168.55.1:8000"] == "http"   # Orin 本地评测服务,局域网明文 HTTP


def test_proxy_transport_uses_connect_tunnel():
    from sr_eval.llm import ProxyTransport
    seen = {}

    class Resp:
        status = 200
        def read(self, n): return b"{}"
        def getheaders(self): return [("X-Test", "1")]

    class Conn:
        def __init__(self, host, port, timeout=None, context=None): seen["proxy"] = (host, port)
        def set_tunnel(self, host, port): seen["tunnel"] = (host, port)
        def request(self, method, path, body=None, headers=None): seen["req"] = (method, path)
        def getresponse(self): return Resp()
        def close(self): seen["closed"] = True

    pt = ProxyTransport({"HTTPS_PROXY": "http://127.0.0.1:7897"}, connection_factory=Conn)
    assert pt.request("openrouter.ai", "GET", "/api/v1/key", {}, None, 5)[0] == 200
    assert seen == {"proxy": ("127.0.0.1", 7897), "tunnel": ("openrouter.ai", 443), "req": ("GET", "/api/v1/key"), "closed": True}


def test_ledger_records_route(make_client, tmp_path):
    c, _t = make_client([ok("好")])
    c.chat("luna", MSG, max_tokens=500)
    assert '"route": "direct"' in (tmp_path / "ledger.jsonl").read_text(encoding="utf-8")  # FakeTransport 无路由信息 → 默认 direct


def test_invalid_response_head_is_captured(make_client, tmp_path):
    c, _t = make_client([(200, {"error": {"message": "upstream timeout", "code": 524}})])
    with pytest.raises(LLMError) as e:
        c.chat("hy4", MSG, max_tokens=500)
    assert e.value.code == "invalid_provider_response" and "upstream timeout" in e.value.detail
    assert "upstream timeout" in (tmp_path / "ledger.jsonl").read_text(encoding="utf-8")


def test_run_case_provider_finish_error_is_reported(make_client, case):
    c, _t = make_client([ok("", finish="error")])
    tr = run_case(c, "luna", case, synthetic_kb(), RunConfig(), policy=OutboundPolicy.load())
    assert tr["status"] == "error" and "provider_finish_error" in tr["error"]


def test_tool_requests_match_production_payload(make_client):
    c, t = make_client([ok("好")])
    c.chat("luna", MSG, max_tokens=500, tools=P.retrieve_tool_schema())
    b = t.requests[0]["body"]
    assert b["tool_choice"] == "auto" and b["parallel_tool_calls"] is False and "temperature" not in b
