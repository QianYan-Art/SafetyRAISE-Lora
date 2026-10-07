"""两个系统的盲评材料:同一批开发案件上,系统 X 与系统 Y 的报告随机排成 A/B,交给与训练评审无关的 agent 独立评判。
用法: blind_cmp.py <输出目录> <run标签X:模型X:显示名X> <run标签Y:模型Y:显示名Y>
例:   blind_cmp.py .../blind_v3c_vs_teacher v3c_dev12:local27:v3c lunadev:luna:teacher
可见知识库片段取两份报告各自可见片段的并集(去重)。答案映射写在 <输出目录>/../answers_<目录名>.json。"""
import glob, json, os, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sr_eval import judge as J  # noqa: E402

HARNESS = Path(__file__).resolve().parent.parent
SPLIT = os.environ.get("BLIND_SPLIT", "dev")   # dev|test:案件所在集合(环境变量,默认 dev)


def load(spec: str, case: str):
    tags, model, name = spec.split(":")
    for tag in tags.split("+"):                       # 一个系统可由多个 run 拼成(如 v3f_dev12+v3f_dev12to49)
        g = glob.glob(str(HARNESS / "runs" / f"*_{tag}"))
        if not g:
            continue
        p = Path(g[0]) / "traces" / f"{case}__{model}.json"
        if p.exists():
            t = json.loads(p.read_text(encoding="utf-8"))
            return (t if t.get("final_markdown") else None), name
    return None, name


TASK = """# 盲评任务(只读,不要修改任何文件)

本目录有 {n} 组材料 `pair_01.md` … `{last}`,每组是同一起交通事故的两份分析报告(报告 A、报告 B;顺序是随机的,你不知道是哪个系统写的)。
评审标准在 `RUBRIC.md`(五个维度各 1~5 分,另有 concision 与缺陷清单)。请对每一组:
1. 严格按 RUBRIC 的口径分别给 A、B 打分(五个维度的总分即可,并各列最多 3 条最重要的缺陷:类型+引文+一句话说明);
2. 判定哪份更好(A / B / 持平),并用一句话说明主要依据。
注意:只以材料里的事故信息、指导意见、知识库片段为依据;材料没有的事实或法条不能算作有依据。不要联网。

## 输出格式(严格)
最后输出一个 JSON 数组,每组一个对象:
[{{"pair":"pair_01","score_A":整数,"score_B":整数,"winner":"A|B|tie","reason":"一句话"}}, ...]
数组前可以有你的分析,但 JSON 数组必须是最后一个代码块。
"""


def main(out_dir: str, spec_x: str, spec_y: str, rng_arg: str = "") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if rng_arg:
        a, b = (int(x) for x in rng_arg.split(":"))
        cases = [f"syn_{SPLIT}_{i:03d}" for i in range(a, b)]
    else:
        cases = json.loads((HARNESS / "runs" / "evalset12.json").read_text(encoding="utf-8"))["cases"]
    rng = random.Random(11)
    answers, i = {}, 0
    for case in cases:
        tx, nx = load(spec_x, case)
        ty, ny = load(spec_y, case)
        if not tx or not ty:
            continue
        i += 1
        case_obj = json.loads((HARNESS / "cases" / SPLIT / f"{case}.json").read_text(encoding="utf-8"))
        snips = {}
        for t in (tx, ty):
            for s in t["visible_snippets"]:
                snips.setdefault(s.get("id"), {"id": s.get("id"), "title": s.get("title"), "content": s.get("content"), "effect_level": s.get("effect_level", "")})
        reports = [(nx, tx["final_markdown"]), (ny, ty["final_markdown"])]
        rng.shuffle(reports)
        answers[f"pair_{i:02d}"] = {"A": reports[0][0], "B": reports[1][0], "case": case}
        body = ("# 盲评材料 pair_%02d\n\n## 事故信息\n```json\n%s\n```\n\n## 专家指导意见(仅作提醒层)\n```json\n%s\n```\n\n## 报告生成时可见的知识库片段\n```json\n%s\n```\n\n"
                "## 报告 A\n%s\n\n## 报告 B\n%s\n") % (i, json.dumps(case_obj["accident_data"], ensure_ascii=False, indent=1),
                                                          json.dumps(case_obj["guidance"], ensure_ascii=False, indent=1),
                                                          json.dumps(list(snips.values()), ensure_ascii=False, indent=1), reports[0][1], reports[1][1])
        (out / f"pair_{i:02d}.md").write_text(body, encoding="utf-8")
    (out / "RUBRIC.md").write_text("# 评审标准(与项目严格评审同一把尺)\n\n" + J.SYSTEM_PROMPT + "\n", encoding="utf-8")
    task = TASK.format(n=i, last=f"pair_{i:02d}.md")
    (out / "TASK.md").write_text(task, encoding="utf-8")
    (out.parent / f"answers_{out.name}.json").write_text(json.dumps(answers, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"生成 {i} 组盲评材料({spec_x.split(':')[2]} vs {spec_y.split(':')[2]}) → {out}")


main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "")
