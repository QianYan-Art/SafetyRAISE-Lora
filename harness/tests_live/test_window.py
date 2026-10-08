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


def test_window_cap_defaults_to_trained_window(monkeypatch):
    monkeypatch.delenv("SR_LIVE_MAX_WINDOW", raising=False)
    with pytest.raises(ValueError, match="SR_LIVE_MAX_WINDOW"):
        TrainingWindow(stable_limit=40960, max_len=40960, output_reserve=1, counter=counter(1000))


def test_delivery_window_can_be_enabled_for_evaluation(monkeypatch):
    # 交付窗口是每槽位 40960；只有显式设环境变量才允许评测守卫放宽到它
    monkeypatch.setenv("SR_LIVE_MAX_WINDOW", "40960")
    guard = TrainingWindow(stable_limit=40960, max_len=40960, output_reserve=1, counter=counter(40000))
    assert guard({})["reserved_sequence_tokens"] == 40001
    with pytest.raises(HarnessError, match="training_window_exceeded"):
        TrainingWindow(stable_limit=40960, max_len=40960, output_reserve=1, counter=counter(40960))({})
    with pytest.raises(ValueError):
        TrainingWindow(stable_limit=40961, max_len=40961, output_reserve=1, counter=counter(1000))
