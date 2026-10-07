"""结果快照:把各模型在开发集上的对照数据固化成项目内文件,作为更新 KBase 记录文档的依据。
输出 harness/reports/results-latest.md 与 results-latest.json(含运行目录、产物位置、关键数字)。用法: snapshot_results.py
口径:luna(max) 严格评审总分(满分 25);未经人工核实。"""
import glob, json, random, statistics as st, time
from pathlib import Path

H = Path(__file__).resolve().parent.parent
RUNS = H / "runs"
CASES12 = json.loads((RUNS / "evalset12.json").read_text(encoding="utf-8"))["cases"]
CASES50 = [f"syn_dev_{i:03d}" for i in range(50)]
SYSTEMS = {  # 显示名: [(run 标签, 模型键)]
    "基座(云端xhigh)": [("baseapi_dev33", "qwen27")], "v2a": [("sft_dev50", "local27")], "teacher(luna)": [("lunadev", "luna"), ("lunadev2", "luna")],
    "v3b": [("v3b_dev12", "local27")], "v3c": [("v3c_dev12", "local27")], "v3d": [("v3d_dev12", "local27"), ("v3d_dev12to49", "local27")],
    "v3e(失败)": [("v3e_dev12", "local27")], "v3f": [("v3f_dev12", "local27"), ("v3f_dev12to49", "local27")]}


def run_dir(tag):
    g = glob.glob(str(RUNS / f"*_{tag}"))
    return Path(g[0]) if g else None


def load(name):
    J, G, T = {}, {}, {}
    for tag, model in SYSTEMS[name]:
        d = run_dir(tag)
        if not d:
            continue
        j = json.loads((d / "judgements.json").read_text(encoding="utf-8")) if (d / "judgements.json").exists() else {}
        g = json.loads((d / "gates.json").read_text(encoding="utf-8")) if (d / "gates.json").exists() else {}
        for c in CASES50:
            k = f"{c}__{model}"
            if c not in G and k in g:
                G[c] = g[k]
            x = (j.get(k) or {}).get("luna")
            if c not in J and x and x.get("valid"):
                J[c] = x
            p = d / "traces" / f"{k}.json"
            if c not in T and p.exists():
                T[c] = json.loads(p.read_text(encoding="utf-8"))
    return J, G, T


def ci(vals, B=4000, seed=1):
    r = random.Random(seed)
    m = sorted(st.mean(r.choices(vals, k=len(vals))) for _ in range(B))
    return st.mean(vals), m[int(.025 * B)], m[int(.975 * B)]


def main() -> None:
    data = {n: load(n) for n in SYSTEMS}
    out, js = [], {"generated": time.strftime("%Y-%m-%d %H:%M"), "note": "luna(max) 严格评审,满分 25;未经人工核实", "systems": {}}
    out.append(f"# 结果快照({js['generated']})\n\n口径:luna(max) 严格评审总分(满分 25),**未经人工核实**。评审、硬门、思考指标均由 `harness/` 评测台产出;原始轨迹/评审在各 run 目录。\n")
    out.append("## 开发集前 12 案(dev_000–011)\n\n| 系统 | 有效评审 | 均分 | 每份缺陷 | 硬门失败 | 报告字数中位 | 检索轮数 | 总思考字符中位 | 最终步思考字符中位 |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for n, (J, G, T) in data.items():
        cs = [c for c in CASES12 if c in T]
        if not cs:
            continue
        sc = [J[c]["total"] for c in cs if c in J]
        if not sc:
            continue
        defects = st.mean(len(J[c]["defects"]) for c in cs if c in J)
        hf = sum(1 for c in cs if c in G and G[c]["hard_fail"])
        chars = st.median(len(T[c].get("final_markdown") or "") for c in cs)
        rounds = st.mean(len(T[c].get("rounds", [])) for c in cs)
        tt = st.median(sum(len(x.get("reasoning") or "") for x in T[c]["calls"]) for c in cs)
        tl = st.median(len(T[c]["calls"][-1].get("reasoning") or "") for c in cs if T[c]["calls"])
        out.append(f"| {n} | {len(sc)}/{len(cs)} | {st.mean(sc):.2f} | {defects:.2f} | {hf}/{len(cs)} | {int(chars)} | {rounds:.2f} | {int(tt)} | {int(tl)} |")
        js["systems"].setdefault(n, {})["dev12"] = {"n_valid": len(sc), "luna_mean": round(st.mean(sc), 3), "defects": round(defects, 2), "hard_fail": hf,
                                                    "report_chars_median": int(chars), "rounds": round(rounds, 2), "think_total_median": int(tt), "think_last_median": int(tl)}
    out.append("\n## 完整 50 个开发案(dev_000–049;基座与 v2a 只覆盖 dev_000–032)\n\n| 系统 | 有效评审 | 均分(95% CI) | 硬门失败 | 配对差 vs teacher(95% CI) | 胜/平/负 |\n| --- | --- | --- | --- | --- | --- |")
    tj = data["teacher(luna)"][0]
    for n, (J, G, T) in data.items():
        if len(J) < 30 or n == "teacher(luna)":
            continue
        sc = [x["total"] for x in J.values()]
        m, lo, hi = ci(sc)
        d = [J[c]["total"] - tj[c]["total"] for c in J if c in tj]
        dm, dlo, dhi = ci(d) if len(d) > 5 else (float("nan"),) * 3
        hf = sum(1 for g in G.values() if g["hard_fail"])
        out.append(f"| {n} | {len(sc)} | {m:.2f} ({lo:.2f}~{hi:.2f}) | {hf}/{len(G)} | {dm:+.2f} ({dlo:+.2f}~{dhi:+.2f}) | {sum(x > 0 for x in d)}/{sum(x == 0 for x in d)}/{sum(x < 0 for x in d)} |")
        js["systems"].setdefault(n, {})["dev50"] = {"n": len(sc), "luna_mean": round(m, 3), "ci": [round(lo, 3), round(hi, 3)], "hard_fail": hf, "paired_vs_teacher": round(dm, 3), "paired_ci": [round(dlo, 3), round(dhi, 3)]}
    sc = [x["total"] for x in tj.values()]
    m, lo, hi = ci(sc)
    out.append(f"| teacher(luna) | {len(sc)} | {m:.2f} ({lo:.2f}~{hi:.2f}) | {sum(1 for g in data['teacher(luna)'][1].values() if g['hard_fail'])}/{len(data['teacher(luna)'][1])} | – | – |")
    out.append("\n## 运行目录与产物位置\n")
    for n, srcs in SYSTEMS.items():
        for tag, _ in srcs:
            d = run_dir(tag)
            if d:
                out.append(f"- {n} / `{tag}`:`harness/runs/{d.name}`(轨迹 {len(list((d / 'traces').glob('*.json')))} 份;`gates.json`、`judgements.json`)")
    out.append("\n- 训练数据:`harness/sft/{v3b,v3c,v3d,v3e_ret,v3f,v3e_gate,v3g_gate}/`(`manifest.json` 记录数量、来源、筛选统计)")
    out.append("- 盲评材料与 Codex/Kimi 输出:`harness/runs/blind/<名>/`(答案映射 `harness/runs/blind/answers_<名>.json`)")
    out.append("- 自动链日志:`harness/runs/{auto_eval_v3*.log,big_loop.log,expert_iter.log,eval_rest_*.log}`;Orin 训练日志在 Orin `~/work/runs/*_sft/log.jsonl`(已同步摘要见 `harness/reports/orin-train-logs/`)")
    (H / "reports").mkdir(exist_ok=True)
    (H / "reports" / "results-latest.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    (H / "reports" / "results-latest.json").write_text(json.dumps(js, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(out[:30]))


main()
