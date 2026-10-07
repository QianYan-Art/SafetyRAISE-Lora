from __future__ import annotations

import json
from pathlib import Path

import pytest

from sr_eval.live.audit import build_audit


def _rows_and_records(count: int = 20):
    rows = []
    records = {}
    for index in range(count):
        kind = "anchor_final" if index % 2 == 0 else "pair_revised"
        pair_id = f"r2-test-{index:03d}"
        chosen = json.dumps({"answer": f"通过-{index}"}, ensure_ascii=False)
        rejected = json.dumps({"answer": f"失败-{index}"}, ensure_ascii=False) if kind.startswith("pair_") else None
        row = {
            "pair_id": pair_id,
            "kind": kind,
            "prompt_ids": [1, 2, 3],
            "chosen_ids": [4, 5, index + 10],
            "rejected_ids": [4, 5, index + 11] if rejected else [],
            "c_start": 1,
        }
        if rejected:
            row["r_start"] = 1
        rows.append(row)
        records[pair_id] = {
            "case_id": f"case-{index}",
            "payload": {"messages": [{"role": "user", "content": f"任务-{index}"}]},
            "reasoning_content": "不应出现在盲评材料中",
            "chosen_content": chosen,
            "rejected_content": rejected,
            "metadata": {"pair_id": pair_id, "kind": kind},
        }
    return rows, records


def test_stratified_sampling_minimum_and_blind_mapping(tmp_path: Path) -> None:
    rows, records = _rows_and_records()
    out = tmp_path / "audit"
    result = build_audit(rows, records, out, fraction=0.2, minimum=15, seed=7)
    assert result["total_rows"] == 20
    assert result["sampled_count"] == 15
    assert result["selected_rows"] == 15
    assert result["shortfall"] == 0
    assert result["sampled_pair_ids"]
    assert result["selected_kind_counts"]["anchor_final"] >= 2
    assert result["selected_kind_counts"]["pair_revised"] >= 2
    assert Path(result["answers_path"]).parent == out.parent
    assert Path(result["answers_path"]) != out / "audit-answers.json" or not Path(result["answers_path"]).is_relative_to(out)
    assert (out / "TASK.md").exists()
    assert (out / "RUBRIC.md").exists()
    materials = sorted(out.glob("pair_*.md"))
    assert len(materials) == 15
    text = materials[0].read_text(encoding="utf-8")
    assert "## 任务输入" in text
    assert "思考内容" not in text
    assert "样本类型" not in text
    answers = json.loads(Path(result["answers_path"]).read_text(encoding="utf-8"))
    assert answers
    assert all(set(value) >= {"pair_id", "kind", "A"} for value in answers.values())
    for stem, mapping in answers.items():
        material = (out / f"{stem}.md").read_text(encoding="utf-8")
        if mapping["kind"].startswith("pair_"):
            assert "## 回答 A" in material and "## 回答 B" in material
            assert "chosen_content" not in material and "rejected_content" not in material
            section_a, section_b = material.split("## 回答 A", 1)[1].split("## 回答 B", 1)
            rec = records[mapping["pair_id"]]
            assert (rec["chosen_content"] in section_a) is (mapping["A"] == "chosen")
            assert (rec["chosen_content"] in section_b) is (mapping["B"] == "chosen")


def test_small_batch_is_fully_sampled_and_records_shortfall(tmp_path: Path) -> None:
    rows, records = _rows_and_records(4)
    result = build_audit(rows, records, tmp_path / "small", minimum=15, seed=7)
    assert result["sampled_count"] == 4
    assert result["insufficient"] is True
    assert result["shortfall"] == 11
    assert result["minimum_satisfied"] is False


def test_output_path_must_stay_inside_project(tmp_path: Path) -> None:
    rows, records = _rows_and_records(1)
    with pytest.raises(ValueError, match="项目目录"):
        build_audit(rows, records, Path("C:/tmp/outside-audit"))


def test_missing_records_are_counted_and_not_materialized(tmp_path: Path) -> None:
    rows, records = _rows_and_records(2)
    rows.append({"pair_id": "missing", "kind": "anchor_final"})
    result = build_audit(rows, records, tmp_path / "missing")
    assert result["total_rows"] == 3
    assert result["missing_records"] == 1
    assert result["sampled_count"] == 2
