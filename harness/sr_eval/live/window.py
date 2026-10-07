"""当前单调用训练窗口；不切分、不删减线上上下文、不静默截断。"""

from __future__ import annotations

from collections import OrderedDict

from app.report_harness.contracts import canonical_digest
from app.report_harness.errors import HarnessError

from .samples import count_payload

_COUNTS: OrderedDict[str, dict] = OrderedDict()


class TrainingWindow:
    def __init__(self, *, stable_limit=24000, max_len=32768, output_reserve=8192,
                 reasoning_effort="xhigh", counter=count_payload):
        if not 0 < output_reserve < stable_limit <= max_len <= 32768:
            raise ValueError("当前训练窗口参数不合法，不能超过已批准的 32768。")
        self.stable_limit, self.max_len = stable_limit, max_len
        self.output_reserve, self.reasoning_effort = output_reserve, reasoning_effort
        self.counter = counter

    def __call__(self, payload):
        key = canonical_digest({"payload": payload, "effort": self.reasoning_effort})
        if key not in _COUNTS or self.counter is not count_payload:
            counted = self.counter(payload, reasoning_effort=self.reasoning_effort)
            if self.counter is count_payload:
                _COUNTS[key] = counted
                if len(_COUNTS) > 128:
                    _COUNTS.popitem(last=False)
        else:
            counted = _COUNTS[key]
        if not counted.get("exact"):
            raise HarnessError("tokenizer_unavailable")
        total = counted["prompt_tokens"] + self.output_reserve
        if total > min(self.stable_limit, self.max_len):
            raise HarnessError("training_window_exceeded", details={
                **counted, "output_reserve": self.output_reserve,
                "reserved_sequence_tokens": total, "stable_limit": self.stable_limit,
            })
        return {**counted, "output_reserve": self.output_reserve,
                "reserved_sequence_tokens": total, "stable_limit": self.stable_limit,
                "max_len": self.max_len, "stage": "请求前精确计数；不限制模型输出"}

    def check_response(self, payload, response):
        message = response["choices"][0]["message"]
        counted = self.counter(
            payload, response=response, content=message.get("content") or "",
            reasoning_content=message.get("reasoning_content") or message.get("reasoning") or "",
            reasoning_effort=self.reasoning_effort,
        )
        if not counted.get("exact"):
            raise HarnessError("tokenizer_unavailable")
        if counted["token_count"] > min(self.stable_limit, self.max_len):
            raise HarnessError("training_window_exceeded", details=counted)
        return {**counted, "stage": "完整prompt+思考+最终JSON精确计数"}
