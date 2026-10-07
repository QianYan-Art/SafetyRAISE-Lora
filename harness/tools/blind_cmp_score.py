"""两个系统盲评结果汇总。用法: blind_cmp_score.py <盲评目录> <系统名X> <评审输出文件> [...]
统计"系统 X 胜/平/负"与 X−对手的平均分差。解析最后一个"元素带 pair 字段"的 JSON 数组。"""
import json, re, statistics as st, sys
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
    d, x = Path(sys.argv[1]), sys.argv[2]
    answers = json.loads((d.parent / f"answers_{d.name}.json").read_text(encoding="utf-8"))
    for fname in sys.argv[3:]:
        p = d / fname
        arr = last_json_array(p.read_text(encoding="utf-8", errors="replace")) if p.exists() else None
        if not arr:
            print(f"{fname}: 无可解析结果"); continue
        w = t = l = 0
        diffs = []
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
        print(f"{d.name} / {fname}: {x} 胜 {w}、平 {t}、负 {l};{x}−对手 平均分差 {st.mean(diffs):+.2f}(n={len(diffs)})")


main()
