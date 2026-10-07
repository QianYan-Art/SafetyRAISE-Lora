"""思考与检索相关指标对比。用法: think_metrics.py <run标签:模型键:显示名> [...]   例: think_metrics.py v3d_dev12:local27:v3d v3e_dev12:local27:v3e
只统计 evalset12(dev_000–011);若 run 里没有这些案件则统计该 run 的全部轨迹。"""
import glob, json, re, statistics as st, sys
from pathlib import Path

H = Path(__file__).resolve().parent.parent
CHECK = re.compile(r"核对.{0,8}(引用|依据)|引用自检|先检索|凭记忆")
cases12 = json.loads((H / "runs" / "evalset12.json").read_text(encoding="utf-8"))["cases"]


def stats(tag: str, model: str):
    d = Path(glob.glob(str(H / "runs" / f"*_{tag}"))[0])
    G = json.loads((d / "gates.json").read_text(encoding="utf-8"))
    J = json.loads((d / "judgements.json").read_text(encoding="utf-8")) if (d / "judgements.json").exists() else {}
    rows = []
    for p in sorted((d / "traces").glob("*.json")):
        t = json.loads(p.read_text(encoding="utf-8"))
        if t["model"] != model:
            continue
        rows.append(t)
    sel = [t for t in rows if t["case_id"] in cases12] or rows
    out = {"n": len(sel)}
    out["rounds"] = st.mean(len(t.get("rounds", [])) for t in sel)
    out["first_call_retrieves"] = sum(1 for t in sel if t["calls"] and t["calls"][0]["tool_calls"]) / len(sel)
    out["think_total"] = st.median(sum(len(c.get("reasoning") or "") for c in t["calls"]) for t in sel)
    out["think_last"] = st.median(len(t["calls"][-1].get("reasoning") or "") for t in sel if t["calls"])
    keys = [f"{t['case_id']}__{t['model']}" for t in sel]
    gs = [G.get(k) for k in keys]
    out["hard_fail"] = sum(1 for g in gs if g and g["hard_fail"]) / len(sel)
    out["art_invisible"] = sum(1 for g in gs if g and any(h["code"] == "article_not_in_visible_text" for h in g["hard"])) / len(sel)
    out["no_report"] = sum(1 for t in sel if not t.get("final_markdown")) / len(sel)
    out["loops"] = sum(1 for g in gs if g and "reasoning_loop" in {w["code"] for w in g.get("warn", [])}) / len(sel)
    out["check_phrase"] = sum(1 for t in sel if t["calls"] and CHECK.search(t["calls"][-1].get("reasoning") or "")) / len(sel)
    sc = [J[k]["luna"]["total"] for k in keys if k in J and (J[k].get("luna") or {}).get("valid")]
    out["luna"] = st.mean(sc) if sc else float("nan")
    return out


def main() -> None:
    specs = [a.split(":") for a in sys.argv[1:]]
    res = {s[2]: stats(s[0], s[1]) for s in specs}
    labels = list(res)
    rows = [("案件数", "n", "{:.0f}"), ("检索轮数(均)", "rounds", "{:.2f}"), ("首个调用就检索的比例", "first_call_retrieves", "{:.0%}"),
            ("总思考字符(中位)", "think_total", "{:.0f}"), ("最终步思考字符(中位)", "think_last", "{:.0f}"),
            ("最终步思考里出现'核对引用/先检索'的比例", "check_phrase", "{:.0%}"), ("引用不可见法条的硬门失败", "art_invisible", "{:.0%}"),
            ("硬门失败总比例", "hard_fail", "{:.0%}"), ("没写出报告", "no_report", "{:.0%}"), ("思考循环警示", "loops", "{:.0%}"),
            ("luna 评审均分", "luna", "{:.2f}")]
    print(f"{'指标':36s}" + "".join(f"{l:>10s}" for l in labels))
    for name, key, fmt in rows:
        print(f"{name:36s}" + "".join(f"{fmt.format(res[l][key]):>10s}" for l in labels))


main()
