"""把同一对系统的多份盲评(按案件范围分成多个目录)合并成一个结论。
用法: blind_agg.py <系统名X> <评审输出文件名> <盲评目录> [<盲评目录> ...]
例:   blind_agg.py v3f codex_blind.md harness/runs/blind/v3f_vs_teacher harness/runs/blind/v3f_vs_teacher_b2 ...
输出:X 对对手的 胜/平/负、平均分差(X−对手)与 bootstrap 95% 区间。目录里没有可解析输出的会单独列出。"""
import json, random, re, statistics as st, sys
from pathlib import Path


def last_json_array(text: str):
    dec, best = json.JSONDecoder(), None
    for m in re.finditer(r"\[", text):
        try:
            o, _ = dec.raw_decode(text[m.start():])
        except ValueError:
            continue
        if isinstance(o, list) and o and all(isinstance(x, dict) and "pair" in x for x in o):
            best = o
    return best


def main() -> None:
    x, fname, dirs = sys.argv[1], sys.argv[2], [Path(d) for d in sys.argv[3:]]
    diffs, w, t, l, missing = [], 0, 0, 0, []
    for d in dirs:
        ans_path = d.parent / f"answers_{d.name}.json"
        p = d / fname
        arr = last_json_array(p.read_text(encoding="utf-8", errors="replace")) if p.exists() else None
        if not arr or not ans_path.exists():
            missing.append(d.name)
            continue
        answers = json.loads(ans_path.read_text(encoding="utf-8"))
        for r in arr:
            a = answers.get(r["pair"])
            if not a:
                continue
            sc = {a["A"]: r["score_A"], a["B"]: r["score_B"]}
            other = [k for k in sc if k != x][0]
            diffs.append(sc[x] - sc[other])
            win = r.get("winner")
            if win == "tie":
                t += 1
            elif a[win] == x:
                w += 1
            else:
                l += 1
    if not diffs:
        print("没有可解析的结果;缺失:", missing)
        return
    rng = random.Random(1)
    boots = sorted(st.mean(rng.choices(diffs, k=len(diffs))) for _ in range(4000))
    print(f"{fname}: {x} 胜 {w}、平 {t}、负 {l}(共 {len(diffs)} 组);{x}−对手 平均分差 {st.mean(diffs):+.2f}(95% 区间 {boots[100]:+.2f} 到 {boots[3900]:+.2f})"
          + (f";缺失/无法解析:{missing}" if missing else ""))


main()
