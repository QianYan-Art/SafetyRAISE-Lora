"""抽检:确定性规则选出的"质量对"(chosen vs rejected)是否也被 luna(max) 评审认可。

  keys   <pairs.jsonl> [--kind quality]        → 每行 "run目录名<TAB>case__model,case__model"(供 cli judge --only)
  report <pairs.jsonl> [--kind quality]        → chosen 评审分 − rejected 评审分:胜率、均值差、自助法 95% 区间
  filter <pairs.jsonl> --out X [--min-diff 1]  → 终止/硬门类全留;质量类只留评审有效且 chosen−rejected ≥ min-diff 的
"""
import argparse, collections, json, random, statistics
from pathlib import Path

RUNS = Path(__file__).resolve().parent.parent / "runs"


def split(name: str, case_id: str) -> tuple[str, str]:
    run, model = name.split(":", 1)
    return run, f"{case_id}__{model}"


def load(path: str, kind: str):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    return [r for r in rows if r["kind"] == kind]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["keys", "report", "filter"]); ap.add_argument("pairs"); ap.add_argument("--kind", default="quality")
    ap.add_argument("--out"); ap.add_argument("--min-diff", type=float, default=1.0)
    a = ap.parse_args()
    if a.mode == "filter":
        allrows = [json.loads(l) for l in open(a.pairs, encoding="utf-8")]
        cache, kept, dropped = {}, [], collections.Counter()
        for r in allrows:
            if r["kind"] != "quality":
                kept.append(r); continue
            sc = []
            for n in (r["chosen"], r["rejected"]):
                run, key = split(n, r["pair_id"])
                if run not in cache:
                    pj = RUNS / run / "judgements.json"
                    cache[run] = json.loads(pj.read_text(encoding="utf-8")) if pj.exists() else {}
                j = (cache[run].get(key) or {}).get("luna")
                sc.append(j["total"] if j and j.get("valid") else None)
            if None in sc: dropped["评审缺失/无效"] += 1
            elif sc[0] - sc[1] < a.min_diff: dropped["评审不认可"] += 1
            else: kept.append(r)
        Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + chr(10) for r in kept), encoding="utf-8")
        print(f"保留 {len(kept)}/{len(allrows)}(质量类剔除:{dict(dropped)});按类:{dict(collections.Counter(r['kind'] for r in kept))}")
        return
    rows = load(a.pairs, a.kind)
    if a.mode == "keys":
        by = collections.defaultdict(set)
        for r in rows:
            for n in (r["chosen"], r["rejected"]):
                run, key = split(n, r["pair_id"]); by[run].add(key)
        for run, ks in by.items():
            print(run + "\t" + ",".join(sorted(ks)))
        return
    cache, diffs, detail = {}, [], []
    for r in rows:
        sc = []
        for n in (r["chosen"], r["rejected"]):
            run, key = split(n, r["pair_id"])
            if run not in cache:
                p = RUNS / run / "judgements.json"
                cache[run] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
            j = (cache[run].get(key) or {}).get("luna")
            sc.append(j["total"] if j and j.get("valid") else None)
        if None in sc:
            detail.append((r["pair_id"], "评审无效/缺失", sc)); continue
        diffs.append(sc[0] - sc[1]); detail.append((r["pair_id"], sc[0] - sc[1], sc))
    for d in detail: print(d)
    if not diffs:
        print("无有效配对"); return
    rng = random.Random(0)
    boots = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(5000))
    wins = sum(d > 0 for d in diffs); ties = sum(d == 0 for d in diffs)
    print(f"有效对 {len(diffs)}/{len(rows)};chosen 更高 {wins}、持平 {ties}、更低 {len(diffs)-wins-ties};均值差 {statistics.mean(diffs):+.2f} "
          f"(95%区间 {boots[125]:+.2f}~{boots[4875]:+.2f});评审分制:总分,未经人工核实")


if __name__ == "__main__":
    main()
