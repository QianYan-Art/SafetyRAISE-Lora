"""盲评结果汇总:解析各评审输出里最后一个 JSON 数组,对照答案映射,统计"修订稿胜出"比例与均分差。
用法: blind_ab_score.py <盲评目录> <评审输出文件名> [...]   例: blind_ab_score.py .../blind1 codex_blind.md kimi_blind.md"""
import json, re, statistics as st, sys
from pathlib import Path


def last_json_array(text: str):
    """取文本里最后一个"元素带 pair 字段"的 JSON 数组(跳过内层的缺陷数组)。"""
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
    d = Path(sys.argv[1])
    answers = json.loads((d.parent / f"answers_{d.name}.json").read_text(encoding="utf-8"))
    for fname in sys.argv[2:]:
        p = d / fname
        arr = last_json_array(p.read_text(encoding="utf-8", errors="replace")) if p.exists() else None
        if not arr:
            print(f"{fname}: 无可解析的 JSON 结果"); continue
        wins = ties = losses = 0
        diffs = []
        for r in arr:
            a = answers.get(r["pair"])
            if not a:
                continue
            s = {a["A"]: r["score_A"], a["B"]: r["score_B"]}
            diffs.append(s["rev"] - s["orig"])
            w = r.get("winner")
            if w == "tie":
                ties += 1
            elif (a[w] == "rev"):
                wins += 1
            else:
                losses += 1
        print(f"{fname}: 修订稿胜 {wins}、平 {ties}、负 {losses};修订−原稿 平均分差 {st.mean(diffs):+.2f}(n={len(diffs)})")


main()
