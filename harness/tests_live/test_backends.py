"""假传输验证真实后端的请求体、账本和闸门，绝不联网。"""

import asyncio
import json

import pytest

from sr_eval.llm import LLMClient, LLMError, Ledger, load_models
from sr_eval.live.backends import LocalOpenAIBackend, MiniMaxBackend, RoleBackends


class FakeTransport:
    def __init__(self):
        self.calls = []

    def route_for(self, host):
        return "direct"

    def request(self, host, method, path, headers, body, timeout):
        self.calls.append((host, json.loads(body)))
        return 200, {}, json.dumps({
            "id": "fixture-response", "model": json.loads(body)["model"],
            "choices": [{"message": {"content": "{}", "reasoning_content": "夹具思考"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }).encode()


@pytest.mark.parametrize("backend_cls", [MiniMaxBackend, LocalOpenAIBackend])
def test_effective_payload_ledger_and_json_mode(tmp_path, monkeypatch, backend_cls):
    monkeypatch.setenv("MINIMAX_API_KEY", "not-a-real-key")
    transport = FakeTransport()
    client = LLMClient(load_models(), ledger=Ledger(tmp_path / "ledger.jsonl"),
                       transport=transport, scope="live-test")
    backend = backend_cls(client=client, allow_network=True)
    payload = {"model": backend.model_for("generator"),
               "messages": [{"role": "system", "content": "合成协议测试"}],
               "response_format": {"type": "json_object"},
               "reasoning": {"effort": backend.effort_for("generator"), "exclude": True}}
    response = asyncio.run(backend.chat("generator", payload))
    host, sent = transport.calls[0]
    assert "openrouter" not in host
    assert sent["response_format"] == payload["response_format"]
    assert "tools" not in sent
    if backend_cls is MiniMaxBackend:
        assert sent["reasoning_effort"] == "max"
        assert sent["thinking"] == {"type": "adaptive"}
        assert sent["reasoning_split"] is True
    else:
        assert sent["chat_template_kwargs"] == {"reasoning_effort": "xhigh"}
    assert response["choices"][0]["message"]["reasoning_content"] == "夹具思考"
    events = [json.loads(line) for line in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    assert [event["event"] for event in events] == ["reserved", "completed"]


def test_default_no_network():
    backend = MiniMaxBackend()
    with pytest.raises(LLMError, match="live_network_disabled"):
        asyncio.run(backend.chat("generator", {"messages": []}))


def test_outbound_denied_before_transport(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "not-a-real-key")
    policy = tmp_path / "outbound.json"
    policy.write_text(json.dumps({"synthetic_cases_external_send": False}), encoding="utf-8")
    transport = FakeTransport()
    client = LLMClient(load_models(), ledger=Ledger(tmp_path / "ledger.jsonl"), transport=transport)
    backend = MiniMaxBackend(client=client, allow_network=True, policy_path=policy)
    with pytest.raises(LLMError, match="outbound_denied"):
        asyncio.run(backend.chat("generator", {"messages": []}))
    assert transport.calls == []
    assert not (tmp_path / "ledger.jsonl").exists()


def test_local_generator_minimax_reviewer_full_environment(tmp_path, monkeypatch):
    from sr_eval.live.env import LiveEnvironment, VENDOR
    from sr_eval.live.scripted import SyntheticRetrieval, passing_script
    monkeypatch.setenv("MINIMAX_API_KEY", "not-a-real-key")

    class FullTransport(FakeTransport):
        def request(self, host, method, path, headers, body, timeout):
            payload = json.loads(body)
            ctx = json.loads(payload["messages"][-1]["content"])
            role = "generator" if "candidate_version" in ctx else "reviewer"
            self.calls.append((host, payload))
            return 200, {}, json.dumps({
                "model": payload["model"],
                "choices": [{"finish_reason": "stop", "message": {
                    "content": json.dumps(passing_script(role, payload, len(self.calls)), ensure_ascii=False),
                    "reasoning_content": "本地假传输思考",
                }}],
                "usage": {"total_tokens": 15, "prompt_tokens": 10, "completion_tokens": 5},
            }, ensure_ascii=False).encode()

    fake = FullTransport()
    client = LLMClient(load_models(), ledger=Ledger(tmp_path / "ledger.jsonl"), transport=fake)
    local = LocalOpenAIBackend(client=client, allow_network=True, roles=("generator",))
    minimax = MiniMaxBackend(client=client, allow_network=True, roles=("reviewer",))
    accident = {key: "" for key in json.loads(
        (VENDOR / "config/input_accident_template.json").read_text(encoding="utf-8"),
    )}
    accident["事故标题"] = "虚构混合后端协议案件"
    case = {"case_id": "mixed-fixture", "kind": "synthetic", "split": "train",
            "accident_data": accident, "guidance": {}}
    trace = asyncio.run(LiveEnvironment(
        RoleBackends({"generator": local, "reviewer": minimax}), SyntheticRetrieval(),
        payload_guard=lambda _: {"status": "假传输定向测试"},
    ).run(case))
    assert trace["status"] == "published", trace["reason"]
    assert trace["budget"]["physical_requests"] == 2
    assert fake.calls[0][1]["chat_template_kwargs"] == {"reasoning_effort": "xhigh"}
    assert fake.calls[1][1]["reasoning_effort"] == "max"
    assert all(item[1]["response_format"] == {"type": "json_object"} for item in fake.calls)
    assert trace["calls"][0]["response"]["choices"][0]["message"]["reasoning_content"]
    assert not trace["training_eligible"]


def test_physical_retry_is_counted_and_stopped(tmp_path, monkeypatch):
    from sr_eval.live.env import MemoryBudget, production_budget
    from app.report_harness.errors import HarnessError
    monkeypatch.setenv("MINIMAX_API_KEY", "not-a-real-key")

    class RateLimit(FakeTransport):
        def request(self, host, method, path, headers, body, timeout):
            self.calls.append((host, {}))
            return 429, {}, b'{"error":"fixture"}'

    fake = RateLimit()
    client = LLMClient(load_models(), ledger=Ledger(tmp_path / "ledger.jsonl"),
                       transport=fake, sleep=lambda _: None)
    backend = MiniMaxBackend(client=client, allow_network=True)
    budget = MemoryBudget(production_budget().model_copy(update={"max_physical_requests": 1}),
                          deterministic=True)
    backend.bind_before_attempt(budget.reserve_request)
    with pytest.raises(HarnessError, match="physical_request_budget_exhausted"):
        asyncio.run(backend.chat("generator", {"model": backend.model_for("generator"),
                                               "messages": [{"role": "user", "content": "合成"}]}))
    assert len(fake.calls) == 1
    assert budget.requests == 1
