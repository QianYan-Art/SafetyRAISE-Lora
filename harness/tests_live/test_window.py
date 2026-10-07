"""窗口硬门边界，精确 tokenizer 的真实链路在离线冒烟中验证。"""

import pytest

from app.report_harness.errors import HarnessError
from sr_eval.live.window import TrainingWindow


def counter(total):
    return lambda *args, **kwargs: {
        "exact": True, "prompt_tokens": total, "response_tokens": 4,
        "token_count": total + 4,
    }


def test_preflight_window_includes_output_reservation():
    guard = TrainingWindow(counter=counter(15808))
    assert guard({})["reserved_sequence_tokens"] == 24000
    with pytest.raises(HarnessError, match="training_window_exceeded"):
        TrainingWindow(counter=counter(15809))({})


def test_unknown_tokenizer_never_passes():
    guard = TrainingWindow(counter=lambda *a, **k: {"exact": False})
    with pytest.raises(HarnessError, match="tokenizer_unavailable"):
        guard({})


def test_full_response_window_is_not_silently_truncated():
    guard = TrainingWindow(counter=counter(24000))
    response = {"choices": [{"message": {"content": "{}", "reasoning_content": "合成思考"}}]}
    with pytest.raises(HarnessError, match="training_window_exceeded"):
        guard.check_response({}, response)
