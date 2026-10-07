from sr_eval import pairs as PR


def cand(name, judge, cov=0.7, reasons=(), chars=1000, ok=True):
    return {"name": name, "trace": {}, "call": {}, "reasons": list(reasons), "judge_ok": ok, "judge": judge if ok else None,
            "coverage": cov, "reasoning_chars": chars}


def test_chosen_is_best_eligible_and_gate_failure_never_chosen():
    cands = [cand("a", 24, reasons=["hard_gate"]), cand("b", 22), cand("c", 21.5, cov=0.9)]
    ch, rj, kind = PR.pick(cands, 21, 2.0)
    assert ch["name"] in {"b", "c"} and rj["name"] == "a" and kind == "gate"     # 硬门失败者即便分高也只能当 rejected


def test_terminal_failures_take_priority_as_rejected():
    cands = [cand("good", 23), cand("trunc", None, ok=False, reasons=["truncated"], chars=90000), cand("gate", 20, reasons=["hard_gate"])]
    ch, rj, kind = PR.pick(cands, 21, 2.0)
    assert ch["name"] == "good" and rj["name"] == "trunc" and kind == "terminal"


def test_quality_pair_requires_margin_and_no_pair_otherwise():
    ch, rj, kind = PR.pick([cand("x", 23, cov=0.8), cand("y", 22, cov=0.75)], 21, 2.0)
    assert ch["name"] == "x" and rj is None and kind == ""
    ch, rj, kind = PR.pick([cand("x", 24, cov=0.9), cand("y", 20, cov=0.5)], 21, 2.0)
    assert rj["name"] == "y" and kind == "quality"


def test_no_eligible_candidate_gives_no_chosen():
    assert PR.pick([cand("a", 18), cand("b", 19)], 21, 2.0) == (None, None, "")


def test_ties_prefer_shorter_reasoning():
    ch, _, _ = PR.pick([cand("long", 23, cov=0.7, chars=50000), cand("short", 23, cov=0.7, chars=8000)], 21, 2.0)
    assert ch["name"] == "short"


def cand_nj(name, cov, chars=3000, rc=30000, cit=3, reasons=()):
    return {"name": name, "trace": {}, "call": {}, "reasons": list(reasons), "judge_ok": False, "judge": None, "coverage": cov,
            "reasoning_chars": rc, "chars": chars, "citations": cit}


def test_no_judge_mode_uses_deterministic_scores(monkeypatch):
    monkeypatch.setitem(PR.MODE, "judge", False)
    ch, rj, kind = PR.pick([cand_nj("hi", 0.95), cand_nj("lo", 0.70), cand_nj("gate", 0.9, reasons=["hard_gate"])], 0, 0.08)
    assert ch["name"] == "hi" and rj["name"] == "gate" and kind == "gate"
    ch, rj, kind = PR.pick([cand_nj("hi", 0.95), cand_nj("lo", 0.70)], 0, 0.08)
    assert rj["name"] == "lo" and kind == "quality"
    assert PR.pick([cand_nj("nocite", 0.99, cit=0)], 0, 0.08) == (None, None, "")      # 无引用不可当 chosen
    ch, _, _ = PR.pick([cand_nj("long", 0.9, chars=6000, rc=90000), cand_nj("short", 0.9, chars=2500, rc=20000)], 0, 0.08)
    assert ch["name"] == "short"


def test_rejected_must_share_prompt_signature(monkeypatch):
    monkeypatch.setitem(PR.MODE, "judge", False)
    good = dict(cand_nj("good", 0.9), sig="A")
    bad_other_prompt = dict(cand_nj("other", 0.5, reasons=["truncated"]), sig="B")
    ch, rj, kind = PR.pick([good, bad_other_prompt], 0, 0.08)
    assert ch["name"] == "good" and rj is None and kind == ""      # 提示不同:不能配对
    bad_same = dict(cand_nj("same", 0.5, reasons=["loop"]), sig="A")
    ch, rj, kind = PR.pick([good, bad_other_prompt, bad_same], 0, 0.08)
    assert rj["name"] == "same" and kind == "terminal"


def test_terminal_rejected_prefers_similar_reasoning_length(monkeypatch):
    monkeypatch.setitem(PR.MODE, "judge", False)
    good = dict(cand_nj("good", 0.9, rc=30000), sig="A")
    far = dict(cand_nj("far", 0.5, rc=120000, reasons=["loop"]), sig="A")
    near = dict(cand_nj("near", 0.5, rc=36000, reasons=["truncated"]), sig="A")
    _, rj, _ = PR.pick([good, far, near], 0, 0.08)
    assert rj["name"] == "near"                                    # 不让"被拒的总是更长"成为捷径


def test_pick_all_falls_back_to_next_eligible(monkeypatch):
    monkeypatch.setitem(PR.MODE, "judge", False)
    a = dict(cand_nj("best", 0.95), sig="A"); b = dict(cand_nj("second", 0.85), sig="A")
    names = [c[0]["name"] for c in PR.pick_all([a, b], 0, 0.08)]
    assert names == ["best", "second"]


def test_judges_agree_requires_two_judges_and_both_higher():
    a = {"judges": {"luna": 18, "minimax": 17}}; b = {"judges": {"luna": 16, "minimax": 16}}
    assert PR.judges_agree(a, b)
    assert not PR.judges_agree({"judges": {"luna": 18}}, {"judges": {"luna": 10}})                   # 只有一个评审:不算一致
    assert not PR.judges_agree({"judges": {"luna": 18, "minimax": 15}}, {"judges": {"luna": 16, "minimax": 16}})   # MiniMax 不认可
