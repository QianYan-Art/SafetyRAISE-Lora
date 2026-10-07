"""旧教师基线(默认档,10-02/03)与新教师基线(luna max 档,10-06)的非评审指标对比,以及(若已有)MiniMax 评审的配对对比。
luna 不再参与评分:这里不读任何 luna 评审分。
用法: teacher_compare.py dev|test    输出到终端并写 harness/reports/teacher-compare-<split>.md
"""
import glob, json, random, statistics as st, sys
from pathlib import Path

H = Path(__file__).resolve().parent.parent
RUNS = H / "runs"
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "dev"
OLD = {"dev": ["lunadev", "lunadev2"], "test": ["lunatest"]}[SPLIT]
NEW = {"dev": ["lunamax_dev"], "test": ["lunamax_test"]}[SPLIT]


def load(tags):
    T, G, J = {}, {}, {}
    for tag in tags:
        g = glob.glob(str(RUNS / f"*_{tag}"))
        if not g:
            continue
        d = Path(g[0])
        gates = json.loads((d / "gates.json").read_text(encoding="utf-8")) if (d / "gates.json").exists() else {}
        jud = json.loads((d / "judgements.json").read_text(encoding="utf-8")) if (d / "judgements.json").exists() else {}
        for p in sorted((d / "traces").glob("*__luna.json")):
            case = p.name.split("__")[0]
            if case in T:
                continue
            T[case] = json.loads(p.read_text(encoding="utf-8"))
            G[case] = gates.get(f"{case}__luna") or {}
            m = (jud.get(f"{case}__luna") or {}).get("minimax")
            if m and m.get("valid"):
                J[case] = m
    return T, G, J


def med(xs):
    return st.median(xs) if xs else float("nan")


def boot(vals, B=4000, seed=1):
    r = random.Random(seed)
    m = sorted(st.mean(r.choices(vals, k=len(vals))) for _ in range(B))
    return st.mean(vals), m[int(.025 * B)], m[int(.975 * B)]


def row(T, G):
    cs = list(T)
    calls = lambda c: T[c].get("totals", {})
    metrics = lambda c: (G[c].get("metrics") or {})
    hard = [c for c in cs if G[c].get("hard_fail")]
    codes = {}
    for c in hard:
        for h in G[c].get("hard", []):
            codes[h["code"]] = codes.get(h["code"], 0) + 1
    return {
        "n": len(cs),
        "报告字数 中位": int(med([len(T[c].get("final_markdown") or "") for c in cs])),
        "报告字数 均值": int(st.mean([len(T[c].get("final_markdown") or "") for c in cs])),
        "引用条数 均值": round(st.mean([metrics(c).get("citations", 0) for c in cs]), 2),
        "待核实/信息不足 次数 均值": round(st.mean([metrics(c).get("hedge_marks", 0) for c in cs]), 1),
        "检索轮数 均值": round(st.mean([len(T[c].get("rounds", [])) for c in cs]), 2),
        "至少检索 1 轮 占比": round(sum(1 for c in cs if T[c].get("rounds")) / len(cs), 2),
        "调用次数 均值": round(st.mean([calls(c).get("calls", 0) for c in cs]), 2),
        "思考 token 中位": int(med([calls(c).get("reasoning_tokens", 0) for c in cs])),
        "输出 token 中位": int(med([calls(c).get("completion_tokens", 0) for c in cs])),
        "最终步思考 token 中位": int(med([T[c]["calls"][-1].get("reasoning_tokens", 0) if T[c].get("calls") else 0 for c in cs])),
        "单案成本 均值(美元)": round(st.mean([calls(c).get("cost_usd", 0) for c in cs]), 4),
        "单案耗时 中位(秒)": int(med([calls(c).get("latency_s", 0) for c in cs])),
        "硬门失败": f"{len(hard)}/{len(cs)}",
        "硬门类型": codes,
        "警告项 均值": round(st.mean([len(G[c].get("warn", [])) for c in cs]), 2),
    }


def main():
    To, Go, Jo = load(OLD)
    Tn, Gn, Jn = load(NEW)
    ro, rn = row(To, Go), row(Tn, Gn)
    lines = [f"# 教师基线对比({SPLIT}):旧(默认档)vs 新(luna max 档)\n", f"旧 = {'+'.join(OLD)};新 = {'+'.join(NEW)}。均为同一条生产注入流程(concise 变体)。luna 不参与评分,下表无 luna 评审分。\n",
             "| 指标 | 旧 | 新 |", "| --- | --- | --- |"]
    for k in ro:
        lines.append(f"| {k} | {ro[k]} | {rn[k]} |")
    common = sorted(set(Jo) & set(Jn))
    lines.append(f"\n## MiniMax(max)评审的配对对比(两边都有有效评审的案件 {len(common)} 个;满分 25;未经人工核实)\n")
    if len(common) >= 5:
        so = [Jo[c]["total"] for c in common]
        sn = [Jn[c]["total"] for c in common]
        d = [b - a for a, b in zip(so, sn)]
        m, lo, hi = boot(d)
        lines.append(f"- 旧均分 {st.mean(so):.2f};新均分 {st.mean(sn):.2f};新−旧 {m:+.2f}(95% 区间 {lo:+.2f} 到 {hi:+.2f});新胜/平/负 {sum(x > 0 for x in d)}/{sum(x == 0 for x in d)}/{sum(x < 0 for x in d)}")
        dims = sorted({k for c in common for k in (Jo[c].get("dimensions") or {})})
        for k in dims:
            a = [Jo[c]["dimensions"][k]["score"] for c in common if k in (Jo[c].get("dimensions") or {})]
            b = [Jn[c]["dimensions"][k]["score"] for c in common if k in (Jn[c].get("dimensions") or {})]
            if a and b:
                lines.append(f"  - {k}:旧 {st.mean(a):.2f} → 新 {st.mean(b):.2f}")
        lines.append(f"- 每份缺陷数:旧 {st.mean(len(Jo[c]['defects']) for c in common):.2f};新 {st.mean(len(Jn[c]['defects']) for c in common):.2f}")
    else:
        lines.append(f"(MiniMax 评审还没跑完:旧 {len(Jo)} 份有效,新 {len(Jn)} 份有效)")
    text = "\n".join(lines)
    print(text)
    (H / "reports" / f"teacher-compare-{SPLIT}.md").write_text(text + "\n", encoding="utf-8")


main()
