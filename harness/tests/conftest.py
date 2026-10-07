import json
import sys
from pathlib import Path

import pytest

HARNESS = Path(__file__).resolve().parent.parent
if str(HARNESS) not in sys.path:
    sys.path.insert(0, str(HARNESS))

from sr_eval.llm import Ledger, LLMClient, load_models  # noqa: E402


class FakeTransport:
    """按脚本依次返回响应;记录每次请求体,供断言。"""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []

    def request(self, host, method, path, headers, body, timeout):
        self.requests.append({"host": host, "method": method, "path": path, "body": json.loads(body) if body else None,
                              "headers": headers})
        status, obj = self.script.pop(0)
        if isinstance(obj, Exception):
            raise obj
        return status, {"x-request-id": "req_test"}, json.dumps(obj).encode("utf-8")


def ok(content="", tool_calls=None, cost=0.001, finish=None, prompt=1000, completion=500, reasoning=100):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = [
            {"id": f"tc{i}", "type": "function", "function": {"name": n, "arguments": a if isinstance(a, str) else json.dumps(a, ensure_ascii=False)}}
            for i, (n, a) in enumerate(tool_calls)
        ]
    return 200, {"id": "gen-1", "model": "x/y", "choices": [{"message": msg, "finish_reason": finish or ("tool_calls" if tool_calls else "stop")}],
                 "usage": {"prompt_tokens": prompt, "completion_tokens": completion, "cost": cost,
                           "completion_tokens_details": {"reasoning_tokens": reasoning}}}


@pytest.fixture(autouse=True)
def _secrets(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-secret")
    monkeypatch.setenv("MINIMAX_API_KEY", "mm-test-secret")


@pytest.fixture
def make_client(tmp_path):
    def _make(script, cap=5.0):
        t = FakeTransport(script)
        c = LLMClient(models=load_models(), ledger=Ledger(tmp_path / "ledger.jsonl"), transport=t, cap_usd=cap, scope="openrouter",
                      sleep=lambda s: None)
        return c, t
    return _make


@pytest.fixture
def case():
    return {
        "case_id": "syn_dev_000", "kind": "synthetic",
        "accident_data": {"事故标题": "夜间路口两车碰撞", "事故类型": "碰撞", "事故形态": "侧面碰撞", "事故发生时间": "21:40",
                          "车辆类型": "小型客车;电动自行车", "路口路段类型": "无信号灯十字路口", "天气": "晴"},
        "guidance": {"元信息": {"事故标题": "需要关注：核对事故标题"}, "指导意见清单": {"责任判定提示": "需要关注：责任研判维度需证据支撑"}},
    }
