from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from sr_eval.live.rows import build_row


def _record(**overrides):
    record = {
        "case_id": "syn_post_001",
        "source_call_sha256": "a" * 64,
        "payload": {
            "messages": [
                {"role": "system", "content": "系统提示"},
                {"role": "user", "content": "请返回 JSON"},
            ],
            "response_format": {"type": "json_object"},
        },
        "reasoning_content": "先检查材料，再输出 JSON。",
        "chosen_content": json.dumps({"answer": "通过"}, ensure_ascii=False),
        "derivation": "student_original",
        "training_eligible": True,
        "synthetic": True,
    }
    record.update(overrides)
    return record


def test_anchor_row_is_training_script_shape_and_exact_compact_tokens() -> None:
    result = build_row(_record())
    assert result["reason"] is None
    assert isinstance(result["tokens"], int)
    row = result["row"]
    assert row is not None
    assert set(row) == {"pair_id", "kind", "prompt_ids", "chosen_ids", "rejected_ids", "c_start"}
    assert row["kind"] == "anchor_final"
    assert row["rejected_ids"] == []
    assert row["prompt_ids"] and row["chosen_ids"]
    assert 0 <= row["c_start"] < len(row["chosen_ids"])
    assert result["metadata"]["reasoning_effort"] == "compact"
    assert result["metadata"]["chosen_content"] == _record()["chosen_content"]
    assert result["tokens"] == len(row["prompt_ids"]) + len(row["chosen_ids"])


def test_row_is_fieldwise_identical_to_pairs_response_ids_and_json_boundary() -> None:
    record = _record(reasoning_content="  两端空白思考  ")
    result = build_row(record)
    assert result["reason"] is None
    row = result["row"]
    assert row is not None
    root = Path(__file__).resolve().parents[2]
    legacy_python = root / ".venv" / "Scripts" / "python.exe"
    script = r'''
import json, sys
from sr_eval.sft import QwenTemplate
from sr_eval.pairs import response_ids
from sr_eval.sft import assistant_message
req = json.load(sys.stdin)
call = {"messages": req["messages"], "with_tools": False,
        "content": req["content"], "reasoning": req["reasoning"],
        "tool_calls": [], "finish_reason": "stop"}
tpl = QwenTemplate()
prompt_ids, response_ids_ = response_ids(tpl, call, "compact")
prompt = tpl.render(call["messages"], None, add_generation_prompt=True, reasoning_effort="compact")
full = tpl.render(call["messages"] + [assistant_message(call, call["reasoning"].strip())], None,
                  add_generation_prompt=False, reasoning_effort="compact")
target = full[len(prompt):]
assert target.endswith("<|im_end|>\n")
target = target[:-1]
after_think = target.find("</think>") + len("</think>")
char_start = target.find(call["content"].strip(), after_think)
c_start = len(tpl.encode(target[:char_start]))
print(json.dumps({
    "prompt_ids": prompt_ids,
    "response_ids": response_ids_,
    "c_start": c_start,
    "eos_id": tpl.tokenizer.token_to_id("<|im_end|>"),
    "prefix": tpl.tokenizer.decode(response_ids_[:c_start], skip_special_tokens=False),
    "suffix": tpl.tokenizer.decode(response_ids_[c_start:], skip_special_tokens=False),
}, ensure_ascii=False))
'''
    env = {**os.environ, "PYTHONPATH": "harness", "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run(
        [str(legacy_python), "-B", "-c", script],
        cwd=str(root),
        input=json.dumps({"messages": record["payload"]["messages"], "content": record["chosen_content"], "reasoning": record["reasoning_content"]}, ensure_ascii=False),
        text=True,
        encoding="utf-8",
        env=env,
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    reference = json.loads(proc.stdout)
    assert row["prompt_ids"] == reference["prompt_ids"]
    assert row["chosen_ids"] == reference["response_ids"]
    assert row["c_start"] == reference["c_start"]
    assert row["chosen_ids"][-1] == reference["eos_id"]
    assert reference["prefix"].endswith("</think>\n\n")
    assert reference["suffix"].startswith(record["chosen_content"])
    assert reference["suffix"].endswith("<|im_end|>")


def test_revised_pair_shares_prompt_and_thinking_prefix() -> None:
    result = build_row(
        _record(
            derivation="minimax_revision",
            rejected_content=json.dumps({"answer": "不合格"}, ensure_ascii=False),
        )
    )
    assert result["reason"] is None
    row = result["row"]
    assert row is not None
    assert row["kind"] == "pair_revised"
    assert row["prompt_ids"]
    assert row["rejected_ids"]
    assert row["chosen_ids"][: row["c_start"]] == row["rejected_ids"][: row["r_start"]]
    assert row["chosen_ids"] != row["rejected_ids"]
    assert result["tokens"] == len(row["prompt_ids"]) + max(len(row["chosen_ids"]), len(row["rejected_ids"]))


def test_revised_pair_keeps_malformed_student_rejection() -> None:
    result = build_row(
        _record(
            derivation="minimax_revision",
            rejected_content="{学生原答复未闭合",
        )
    )
    assert result["reason"] is None
    assert result["row"]["kind"] == "pair_revised"


def test_seeded_pair_still_requires_json_rejection() -> None:
    result = build_row(
        _record(
            derivation="seeded",
            error_class="missing_fact",
            rejected_content="坏 seeded 内容",
        )
    )
    assert result["row"] is None
    assert result["reason"] == "rejected_content_not_json_object"


def test_seeded_kind_and_tool_anchor_are_deterministic() -> None:
    tool = json.dumps({"tool_calls": [{"name": "search_knowledge", "arguments": {"query": "天气", "top_k": 1}}]}, ensure_ascii=False)
    result = build_row(_record(chosen_content=tool))
    assert result["reason"] is None
    assert result["row"]["kind"] == "anchor_tool"

    seeded = build_row(
        _record(
            derivation="seeded",
            error_class="missing_fact",
            rejected_content=json.dumps({"answer": "漏事实"}, ensure_ascii=False),
        )
    )
    assert seeded["reason"] is None
    assert seeded["row"]["kind"] == "pair_seeded_missing_fact"
    assert seeded["row"]["pair_id"] == build_row(
        _record(
            derivation="seeded",
            error_class="missing_fact",
            rejected_content=json.dumps({"answer": "漏事实"}, ensure_ascii=False),
        )
    )["row"]["pair_id"]


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"training_eligible": False}, "training_ineligible"),
        ({"synthetic": False}, "non_synthetic"),
        ({"chosen_content": "不是 JSON"}, "chosen_content_not_json_object"),
        ({"finish_reason": "length"}, "truncated_finish_reason"),
    ],
)
def test_hard_gates_never_emit_a_row(changes, reason) -> None:
    result = build_row(_record(**changes))
    assert result["row"] is None
    assert result["reason"] == reason


def test_window_overflow_is_discarded_without_truncating() -> None:
    result = build_row(_record(), max_len=1)
    assert result["row"] is None
    assert result["reason"] == "over_max_len"
    assert result["metadata"]["chosen_content"] == _record()["chosen_content"]


def test_direct_api_cannot_raise_r2_window_above_32768() -> None:
    with pytest.raises(ValueError, match="1..32768"):
        build_row(_record(), max_len=32769)


def test_tokenizer_unavailable_is_a_closed_gate(monkeypatch) -> None:
    import sr_eval.live.rows as rows

    def unavailable(*args, **kwargs):
        return {"status": "unavailable", "tokenizer": {}}

    monkeypatch.setattr(rows, "render_call_sample", unavailable)
    result = rows.build_row(_record())
    assert result["row"] is None
    assert result["reason"] == "tokenizer_unavailable"
