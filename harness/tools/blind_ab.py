"""盲评材料生成:把"原稿 vs 教师修订稿"随机排成 A/B,交给与训练评审无关的 agent(Codex/Kimi)独立评判。
用法: blind_ab.py <输出目录> <修订 run 名> [对数=8]    答案映射写在 <输出目录>/../answers_<名>.json(不放进被评审目录)。"""
import glob, json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sr_eval import judge as J  # noqa: E402

HARNESS = Path(__file__).resolve().parent.parent


def main(out_dir: str, rev_name: str, n: int = 8) -> None:
    rev = Path(glob.glob(str(HARNESS / "runs" / f"*_{rev_name}"))[0])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(7)
    keys = sorted(p.stem for p in (rev / "traces").glob("*.json"))
    # 每个案件只取一对,跨案件取前 n 个
    seen, picks = set(), []
    for k in keys:
        case = k.split("__")[0]
        t = json.loads((rev / "traces" / f"{k}.json").read_text(encoding="utf-8"))
        if case in seen or not t.get("final_markdown"):
            continue
        seen.add(case)
        picks.append((case, t))
        if len(picks) >= n:
            break
    answers = {}
    for i, (case, t) in enumerate(picks, 1):
        src = json.loads((HARNESS / "runs" / t["source"]["run"] / "traces" / f"{case}__{t['source']['model']}.json").read_text(encoding="utf-8"))
        case_obj = json.loads((HARNESS / "cases" / "train" / f"{case}.json").read_text(encoding="utf-8"))
        kb = [{"id": s.get("id"), "title": s.get("title"), "content": s.get("content"), "effect_level": s.get("effect_level", "")} for s in t["visible_snippets"]]
        reports = [("orig", src["final_markdown"]), ("rev", t["final_markdown"])]
        rng.shuffle(reports)
        answers[f"pair_{i:02d}"] = {"A": reports[0][0], "B": reports[1][0], "case": case, "src_model": t["source"]["model"]}
        body = ("# 盲评材料 pair_%02d\n\n## 事故信息\n```json\n%s\n```\n\n## 专家指导意见(仅作提醒层)\n```json\n%s\n```\n\n## 报告生成时可见的知识库片段\n```json\n%s\n```\n\n"
                "## 报告 A\n%s\n\n## 报告 B\n%s\n") % (i, json.dumps(case_obj["accident_data"], ensure_ascii=False, indent=1),
                                                          json.dumps(case_obj["guidance"], ensure_ascii=False, indent=1),
                                                          json.dumps(kb, ensure_ascii=False, indent=1), reports[0][1], reports[1][1])
        (out / f"pair_{i:02d}.md").write_text(body, encoding="utf-8")
    (out / "RUBRIC.md").write_text("# 评审标准(与项目严格评审同一把尺)\n\n" + J.SYSTEM_PROMPT + "\n", encoding="utf-8")
    (out.parent / f"answers_{out.name}.json").write_text(json.dumps(answers, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"生成 {len(picks)} 组盲评材料 → {out};答案映射 {out.parent / ('answers_' + out.name + '.json')}")


main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 8)
