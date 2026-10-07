"""严格模型评审(五维量表,沿用 09 实验口径)+ 播种错误校准。

评审只能给出"未经人工核实"的相对评分:它不替代硬门,也不是正确真值。
为抵消宽松倾向:要求先逐条列缺陷并引用报告原文,缺陷引文必须是报告的子串(否则记为未核实缺陷)。
"""

from __future__ import annotations

import json
import random
import re
from typing import Any

from .llm import ChatResult, LLMClient

DIMENSIONS = ("事实完整性", "证据边界与谨慎性", "因果与责任分析合理性", "法规与知识引用相关性", "文书可用性")
DEFECT_TYPES = ("fabrication", "misattributed_fact", "unsupported_cause", "bad_citation", "uncited_law",
                "overclaim_liability", "missing_key_fact", "redundancy", "format")

SYSTEM_PROMPT = """你是严格、挑剔的交通事故分析报告评审员。你只评审给定报告本身,看不到也不关心它由谁生成。
评分对象是"报告能否支持执法办案前的分析工作",不是文笔。

【五个维度,每维 1–5 的整数】
① 事实完整性:是否覆盖事故信息里与判断有关的关键事实;有无遗漏、虚构(含编造的时间、速度、距离、人数、伤情、鉴定结论)。
② 证据边界与谨慎性:是否区分"事故信息记载/可观察事实""推断""待核实";不确定处是否具体标出缺的是什么证据,而不是每章重复泛泛免责。
③ 因果与责任分析合理性:因果链是否由已知事实推出、有无跳跃;责任部分是否只写注意义务/研判方向而不下最终定责或量刑结论。
④ 法规与知识引用相关性:引用的依据是否出现在提供的知识库片段中、是否真正支撑相邻论点、是否适用(效力层级、地域、时间);报告里出现法条/标准/司法观点但片段中没有的,按重大缺陷处理。
⑤ 文书可用性:结构完整、术语规范、重复少、可直接作为分析底稿;同一事实在多个章节原句重复要扣分。
另给一个"简洁性"1–5(不计入总分):用尽量少的话说清必要内容得高分;空泛重复、凑篇幅得低分;不得因为篇幅长而加分。

【打分锚点(必须执行)】
5 = 逐条核对后找不到任何缺陷(极少出现);4 = 只有 1 处不影响结论的轻微缺陷;3 = 有 1 处实质缺陷,或 2 处以上轻微缺陷;2 = 有多处实质缺陷;1 = 存在严重缺陷(虚构关键事实、定罪量刑、编造法条等)。
先找缺陷再打分。典型合格报告应落在 3–4 分。宽松给分会让评审失去区分度,这是不被接受的。

【缺陷登记】
defects 列表里每条必须含:dimension(上述五维之一)、type(fabrication/misattributed_fact/unsupported_cause/bad_citation/uncited_law/overclaim_liability/missing_key_fact/redundancy/format 之一)、quote(报告原文中的连续片段,≤40 字,必须逐字出现在报告里)、explanation(一句话)。
没有缺陷就给空列表,但请确认你确实逐条核对了数字、时间、引用与责任措辞。

【输出】
只输出一个 JSON 对象,不要代码块和其他文字:
{"dimensions":{"事实完整性":{"score":1-5,"reason":"一句话"},"证据边界与谨慎性":{...},"因果与责任分析合理性":{...},"法规与知识引用相关性":{...},"文书可用性":{...}},
 "concision":{"score":1-5,"reason":"一句话"},
 "defects":[{"dimension":"","type":"","quote":"","explanation":""}],
 "overall_comment":"两句话以内"}"""


def build_judge_messages(case: dict[str, Any], report: str, snippets: list[dict[str, Any]]) -> list[dict[str, str]]:
    kb = [{"id": s.get("id"), "title": s.get("title"), "content": s.get("content"),
           "effect_level": s.get("effect_level", ""), "jurisdiction": s.get("jurisdiction", "")} for s in snippets]
    user = (
        "<事故信息>\n" + json.dumps(case["accident_data"], ensure_ascii=False, indent=1) + "\n</事故信息>\n"
        "<专家指导意见(仅作提醒层)>\n" + json.dumps(case["guidance"], ensure_ascii=False, indent=1) + "\n</专家指导意见>\n"
        "<报告生成时可见的知识库片段>\n" + json.dumps(kb, ensure_ascii=False, indent=1) + "\n</报告生成时可见的知识库片段>\n"
        "<待评审报告>\n" + report + "\n</待评审报告>\n"
        "请按要求输出评审 JSON。"
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def _extract_obj(text: str) -> dict[str, Any] | None:
    """从模型输出里取评审 JSON:先剥掉 <think> 块;依次试整段、代码块、再从每个 "{" 处 raw_decode,取含 dimensions 的对象。"""
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    cands = [t, *re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", t, re.DOTALL)]
    for c in cands:
        try:
            o = json.loads(c)
            if isinstance(o, dict):
                return o
        except (ValueError, TypeError):
            pass
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", t):
        try:
            o, _ = dec.raw_decode(t[m.start():])
        except ValueError:
            continue
        if isinstance(o, dict) and "dimensions" in o:
            return o
    return None


def parse_judgement(text: str, report: str) -> dict[str, Any]:
    obj = _extract_obj(text)
    if not isinstance(obj, dict):
        return {"valid": False, "error": "not_json", "raw_head": text.strip()[:200], "raw_len": len(text)}
    dims = obj.get("dimensions") or {}
    scores: dict[str, int] = {}
    for d in DIMENSIONS:
        s = (dims.get(d) or {}).get("score")
        if not isinstance(s, int) or isinstance(s, bool) or not 1 <= s <= 5:
            return {"valid": False, "error": f"bad_score:{d}"}
        scores[d] = s
    norm = re.sub(r"\s+", "", report)
    defects, unverified = [], 0
    for item in obj.get("defects") or []:
        if not isinstance(item, dict):
            continue
        ok = bool(item.get("quote")) and re.sub(r"\s+", "", str(item["quote"])) in norm
        unverified += 0 if ok else 1
        defects.append({**item, "quote_verified": ok})
    conc = (obj.get("concision") or {}).get("score")
    return {"valid": True, "scores": scores, "total": sum(scores.values()),
            "concision": conc if isinstance(conc, int) and 1 <= conc <= 5 else None,
            "defects": defects, "unverified_defects": unverified, "comment": str(obj.get("overall_comment", ""))[:300]}


def judge_report(client: LLMClient, judge_model: str, case: dict[str, Any], report: str, snippets: list[dict[str, Any]],
                 *, max_tokens: int = 96000, effort: str = "max", meta: dict[str, Any] | None = None) -> dict[str, Any]:
    res: ChatResult = client.chat(judge_model, build_judge_messages(case, report, snippets), max_tokens=max_tokens,
                                  reasoning={"effort": effort}, temperature=0, purpose="judge", meta=meta)
    out = parse_judgement(res.content, report)
    out.update({"judge": judge_model, "cost_usd": res.cost_usd, "finish_reason": res.finish_reason,
                "completion_tokens": res.completion_tokens})
    return out


# ---------------- 播种错误(校准评审区分度) ----------------

def _section_bounds(report: str, title_part: str) -> tuple[int, int] | None:
    ms = list(re.finditer(r"^#{1,6}\s+.+$", report, re.MULTILINE))
    for i, m in enumerate(ms):
        if title_part in m.group(0):
            return m.start(), (ms[i + 1].start() if i + 1 < len(ms) else len(report))
    return None


def mutate_drop_section(report: str, title_part: str) -> str:
    b = _section_bounds(report, title_part)
    return report if b is None else report[: b[0]] + report[b[1]:]


def mutate_fake_speed(report: str) -> str:
    return report.rstrip() + "\n\n另据现场勘查,涉事机动车碰撞前时速约 87 公里/小时,制动距离约 42 米。\n"


def mutate_overclaim(report: str) -> str:
    return report.rstrip() + "\n\n综上,驾驶人应当承担事故的全部责任,对方当事人无责任,建议依法追究其刑事责任并判处相应刑罚。\n"


def mutate_fake_law(report: str) -> str:
    return report.rstrip() + "\n\n依据《中华人民共和国道路交通安全法》第九十九条第三款,上述行为应从重处罚。\n"


def mutate_strip_citations(report: str) -> str:
    return re.sub(r"\s*\[依据[:：][^\]]+\]", "", report)


def mutate_duplicate(report: str) -> str:
    ms = list(re.finditer(r"^#{1,6}\s+.+$", report, re.MULTILINE))
    if len(ms) < 3:
        return report + "\n\n" + report
    segs = [report[m.start(): (ms[i + 1].start() if i + 1 < len(ms) else len(report))] for i, m in enumerate(ms)]
    longest = max(segs, key=len)
    return report.rstrip() + "\n\n" + longest.replace("##", "##", 1).strip() + "\n\n" + longest.strip() + "\n"


MUTATIONS = {
    "drop_causation": (lambda r: mutate_drop_section(r, "致因分析"), "因果与责任分析合理性"),
    "drop_evidence_boundary": (lambda r: mutate_drop_section(r, "事实与证据边界"), "证据边界与谨慎性"),
    "fake_speed": (mutate_fake_speed, "事实完整性"),
    "overclaim_liability": (mutate_overclaim, "因果与责任分析合理性"),
    "fake_law": (mutate_fake_law, "法规与知识引用相关性"),
    "strip_citations": (mutate_strip_citations, "法规与知识引用相关性"),
    "duplicate_section": (mutate_duplicate, "文书可用性"),
}


def calibration_plan(reports: list[tuple[str, str]], seed: int = 7) -> list[dict[str, str]]:
    """reports=[(report_id, text)];对每份原报告生成原件 + 每种播种错误各一份。"""
    rnd = random.Random(seed)
    jobs = []
    for rid, text in reports:
        jobs.append({"item_id": f"{rid}::orig", "report_id": rid, "mutation": "orig", "target": "", "text": text})
        for name, (fn, target) in MUTATIONS.items():
            mutated = fn(text)
            if mutated != text:
                jobs.append({"item_id": f"{rid}::{name}", "report_id": rid, "mutation": name, "target": target, "text": mutated})
    rnd.shuffle(jobs)
    return jobs


def summarize_calibration(items: list[dict[str, Any]]) -> dict[str, Any]:
    """items 含 report_id/mutation/target 与评审结果(valid/scores/total)。输出每种错误的平均扣分与检出率。"""
    by_report: dict[str, dict[str, Any]] = {}
    for it in items:
        if it.get("valid"):
            by_report.setdefault(it["report_id"], {})[it["mutation"]] = it
    summary: dict[str, Any] = {}
    for name, (_fn, target) in MUTATIONS.items():
        drops, hits, n = [], 0, 0
        for rid, ms in by_report.items():
            if "orig" in ms and name in ms:
                n += 1
                drops.append(ms["orig"]["total"] - ms[name]["total"])
                hits += int(ms[name]["scores"][target] < ms["orig"]["scores"][target])
        if n:
            summary[name] = {"n": n, "mean_total_drop": round(sum(drops) / n, 2), "target_dim_detect_rate": round(hits / n, 2)}
    origs = [ms["orig"]["total"] for ms in by_report.values() if "orig" in ms]
    summary["_orig_mean_total"] = round(sum(origs) / len(origs), 2) if origs else None
    summary["_orig_full_marks_rate"] = round(sum(t == 25 for t in origs) / len(origs), 2) if origs else None
    return summary
