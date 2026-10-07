"""时间受限训练集的选择策略(每调用一行、种子对/原样锚点限额、长行保底)。"""
from sr_eval.live.build_post import _select_unique


def cand(call, kind, tokens, derivation="minimax_revision"):
    return {"row": {"pair_id": f"{call}-{kind}", "kind": kind}, "tokens": tokens,
            "record": {"source_call_sha256": call, "derivation": derivation}, "metadata": {}}


def pool():
    out = []
    for i in range(12):
        c = f"call{i:02d}"
        long = 29000 if i % 3 == 0 else 24000
        out += [cand(c, "anchor_final", long), cand(c, "pair_revised", long),
                cand(c, "pair_seeded_quote_not_unique", long)]
    for i in range(12, 20):
        out.append(cand(f"call{i:02d}", "anchor_final", 23000, "student_original"))
    return out


def test_each_call_at_most_once_and_caps_respected():
    sel, drops = _select_unique(pool(), anchors=10, pairs=6, seeded_cap=2, original_anchor_cap=3,
                                min_long=0, long_tokens=28000, seed=7)
    calls = [c["record"]["source_call_sha256"] for c in sel]
    assert len(calls) == len(set(calls))
    pairs = [c for c in sel if not c["row"]["kind"].startswith("anchor_")]
    anchors = [c for c in sel if c["row"]["kind"].startswith("anchor_")]
    assert len(pairs) <= 6 and len(anchors) <= 10
    assert sum(c["row"]["kind"].startswith("pair_seeded_") for c in sel) <= 2
    assert sum(c["record"]["derivation"] == "student_original" for c in anchors) <= 3
    assert drops


def test_revised_pairs_come_first_and_long_floor_met():
    sel, _ = _select_unique(pool(), anchors=4, pairs=5, seeded_cap=1, original_anchor_cap=1,
                            min_long=3, long_tokens=28000, seed=7)
    assert sum(c["tokens"] > 28000 for c in sel) >= 3
    kinds = [c["row"]["kind"] for c in sel]
    assert kinds.count("pair_revised") >= 4


def test_deterministic_for_same_seed():
    a, _ = _select_unique(pool(), anchors=5, pairs=5, seeded_cap=2, original_anchor_cap=2, min_long=2, long_tokens=28000, seed=3)
    b, _ = _select_unique(pool(), anchors=5, pairs=5, seeded_cap=2, original_anchor_cap=2, min_long=2, long_tokens=28000, seed=3)
    assert [c["row"]["pair_id"] for c in a] == [c["row"]["pair_id"] for c in b]
