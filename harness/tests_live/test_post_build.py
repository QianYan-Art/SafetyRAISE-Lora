"""第2轮离线构建管线回归，输入只用本地合成协议夹具。"""

import asyncio
import json
import subprocess
import sys
from copy import deepcopy

from app.report_harness.contracts import canonical_digest
from sr_eval.live.backends import ScriptedBackend, openai_response
from sr_eval.live.build_post import _audit_status, _checked_revision, build_post
from sr_eval.live.revise import revise_call
from test_post_revise import fixture_trace


def trace_dir(tmp_path, *, split="train"):
    directory = tmp_path / "traces"
    directory.mkdir()
    trace = fixture_trace()
    trace["case"]["split"] = split
    (directory / "fixture.json").write_text(json.dumps(trace, ensure_ascii=False), encoding="utf-8")
    return directory, trace


def test_full_offline_build_anchor_manifest_and_audit(tmp_path):
    source, trace = trace_dir(tmp_path)
    out = tmp_path / "post"
    manifest = build_post(source, out, seed_errors_enabled=False)
    rows = [json.loads(line) for line in (out / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["kind"] == "anchor_final"
    assert rows[0]["rejected_ids"] == []
    assert len(rows[0]["prompt_ids"]) + len(rows[0]["chosen_ids"]) <= 28000
    assert manifest["release_eligible"] is False
    assert manifest["independent_review"] == "sampled"
    assert manifest["training_forbidden"]
    assert manifest["unknown_metrics"]
    assert manifest["window_distribution"]["max"] > 0
    assert manifest["sources"][0]["trace_digest"] == canonical_digest(trace)
    assert manifest["audit"]["sampled_count"] == 1
    assert (out / "audit").is_dir()


def test_default_pipeline_includes_single_class_seeded_rows(tmp_path):
    source, _ = trace_dir(tmp_path)
    out = tmp_path / "seeded-post"
    manifest = build_post(source, out)
    rows = [json.loads(line) for line in (out / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
    assert manifest["counts"]["anchors"] == 1
    assert 1 <= manifest["counts"]["pairs"] <= 4
    assert any(row["kind"].startswith("pair_seeded_") for row in rows)
    assert all(len(row["prompt_ids"]) + max(len(row["chosen_ids"]), len(row["rejected_ids"])) <= 28000
               for row in rows)
    provenance = [json.loads(line) for line in (out / "provenance.jsonl").read_text(encoding="utf-8").splitlines()]
    seeded = [row for row in provenance if row["derivation"] == "seeded"]
    assert seeded and all(row["seed_evidence"]["only_error_class"] for row in seeded)
    assert manifest["training_forbidden"]


def test_offline_revision_pair_revalidated_and_hash_tampering_rejected(tmp_path):
    source, trace = trace_dir(tmp_path)
    call = trace["calls"][0]
    original = call["content"]
    bad = json.loads(original)
    bad["version"] = 8
    call["content"] = json.dumps(bad, ensure_ascii=False)
    call["response"] = openai_response(call["content"], reasoning=call["reasoning_content"])
    (source / "fixture.json").write_text(json.dumps(trace, ensure_ascii=False), encoding="utf-8")
    result = asyncio.run(revise_call(trace, call, ScriptedBackend([openai_response(original)])))
    result["trace_sha256"] = canonical_digest(trace)
    revision = tmp_path / "revisions"
    revision.mkdir()
    path = revision / "derived.json"
    path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    manifest = build_post(source, tmp_path / "valid", revise_results=revision, seed_errors_enabled=False)
    assert manifest["counts"]["anchors"] == 1 and manifest["counts"]["pairs"] == 1
    result["teacher_calls"][0]["response_sha256"] = "0" * 64
    path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    manifest = build_post(source, tmp_path / "tampered", revise_results=revision, seed_errors_enabled=False)
    assert manifest["counts"]["written"] == 0
    assert manifest["discard_reasons"]["teacher_digest_invalid"] == 1


def test_holdout_not_training_dryrun_marked_and_window_drop(tmp_path):
    source, _ = trace_dir(tmp_path, split="test")
    manifest = build_post(source, tmp_path / "holdout", seed_errors_enabled=False)
    assert manifest["counts"]["written"] == 0
    assert manifest["discard_reasons"]["split_not_train"] == 1
    manifest = build_post(source, tmp_path / "dry", dry_run=True, seed_errors_enabled=False)
    assert manifest["counts"]["written"] == 1 and manifest["training_forbidden"]
    manifest = build_post(source, tmp_path / "short", dry_run=True, max_len=20,
                          seed_errors_enabled=False)
    assert manifest["counts"]["written"] == 0
    assert sum(manifest["discard_reasons"].values()) >= 1


def test_audit_failure_rolls_back_entire_batch(tmp_path):
    source, _ = trace_dir(tmp_path)
    first = build_post(source, tmp_path / "first", seed_errors_enabled=False)
    ident = first["audit"]["sampled_pair_ids"][0]
    report = {"rows_sha256": first["pre_rollback_rows_sha256"], "reviews": [
        {"pair_id": ident, "passed": False, "reviewer_kind": "codex", "reviewer_id": "离线测试评审",
         "offline": True, "blind": True}]}
    path = tmp_path / "audit-result.json"
    path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "rollback"
    manifest = build_post(source, out, seed_errors_enabled=False, audit_results=path)
    assert manifest["independent_review_status"]["rollback"]
    assert (out / "rows.jsonl").read_text(encoding="utf-8") == ""
    assert (out / "rows.isolated.jsonl").read_text(encoding="utf-8")
    assert manifest["counts"]["written"] == 0


def test_audit_accept_requires_all_samples_and_digest(tmp_path):
    audit = {"sampled_pair_ids": ["id"], "shortfall": 0}
    path = tmp_path / "review.json"
    report = {"rows_sha256": "bound", "reviews": [
        {"pair_id": "id", "passed": True, "reviewer_kind": "kimi", "reviewer_id": "离线测试",
         "offline": True, "blind": True}]}
    path.write_text(json.dumps(report), encoding="utf-8")
    assert _audit_status(path, audit, "bound")["batch_training_eligible"]
    assert not _audit_status(path, audit, "wrong")["batch_training_eligible"]
    report["reviews"] = []
    path.write_text(json.dumps(report), encoding="utf-8")
    assert not _audit_status(path, audit, "bound")["batch_training_eligible"]


def test_audit_pair_winner_must_confirm_chosen(tmp_path):
    answers = tmp_path / "answers.json"
    answers.write_text(json.dumps({"pair_01": {
        "pair_id": "id", "kind": "pair_revised", "A": "rejected", "B": "chosen"}}), encoding="utf-8")
    audit = {"sampled_pair_ids": ["id"], "shortfall": 0, "answers_path": str(answers)}
    path = tmp_path / "review.json"
    report = {"rows_sha256": "bound", "reviews": [
        {"pair": "pair_01", "passed": True, "winner": "A",
         "reviewer_kind": "kimi", "reviewer_id": "离线测试",
         "offline": True, "blind": True}]}
    path.write_text(json.dumps(report), encoding="utf-8")
    assert _audit_status(path, audit, "bound")["rollback"]
    report["reviews"][0]["winner"] = "B"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert _audit_status(path, audit, "bound")["batch_training_eligible"]


def test_teacher_assistant_binding_is_checked_even_with_consistent_hash():
    trace = fixture_trace()
    call = trace["calls"][0]
    original = call["content"]
    bad = json.loads(original)
    bad["version"] = 8
    call["content"] = json.dumps(bad, ensure_ascii=False)
    call["response"] = openai_response(call["content"], reasoning=call["reasoning_content"])
    result = asyncio.run(revise_call(trace, call, ScriptedBackend([openai_response(original)])))
    result["trace_sha256"] = canonical_digest(trace)
    teacher = result["teacher_calls"][0]
    teacher["payload"]["messages"][3]["content"] = "不同学生上下文"
    teacher["request_sha256"] = canonical_digest(teacher["payload"])
    assert _checked_revision(result, trace, call)[1] == "teacher_messages_binding_invalid"


def test_cli_full_path_without_seeded_network(tmp_path):
    source, _ = trace_dir(tmp_path)
    out = tmp_path / "cli"
    process = subprocess.run(
        [sys.executable, "-B", "-m", "sr_eval.live.build_post", "--traces", str(source),
         "--out", str(out), "--dry-run", "--no-seeded"],
        cwd=str(__import__("sr_eval.live.revise", fromlist=["ROOT"]).ROOT),
        capture_output=True, text=True, encoding="utf-8", check=False,
    )
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout)["counts"]["written"] == 1
