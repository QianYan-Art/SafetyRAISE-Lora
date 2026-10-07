"""为教师轨迹的每次调用补写"简短中文思考"(事后合理化蒸馏,不是教师原始思考)。

教师(luna)的原始思考只有英文摘要且常为空,MiniMax 的原始思考冗长且为英文;为让学生学到
"先判断再行动、思考简短"的形态,这里让 luna 依据该次调用的输入和教师实际动作,写一段前瞻式的短思考。
产出会标注 thinking_source="luna_rationale_zh";原始思考仍保存在 trace 里,不丢。
"""

from __future__ import annotations

import json
import re
from typing import Any

from .gates import ART_RE, STD_RE
from .llm import LLMClient

MAX_CHARS = {"tool": 150, "final": 280}

SYSTEM = """你在为"交通事故分析报告生成模型"整理训练数据中的【简短思考】。
给你:事故信息、该次调用时模型可见的知识库片段(标题与摘要)、此前已做过的检索、以及模型这一步实际采取的动作。
请写出模型在采取该动作前的内部思考,要求:
1. 前瞻式、第一人称、口语化的中文短段落,不要标题、列表或 Markdown;
2. 检索步骤(动作是调用 retrieve_knowledge):2–4 句,≤150 字:先判断现有片段够不够、还缺哪类依据(规则/义务/天气路况/责任规则),再说明为什么用这个检索词;
3. 最终报告步骤:4–7 句,≤280 字:交代已有事实与边界(哪些明确、哪些缺失)、可用的依据(只用片段标题称呼,不要写条号)、报告要覆盖的要点与要标注"待核实"的缺口,并提醒自己不重复、不编造、不下最终定责;
4. 不得引入输入里没有的事实、数字、条号;不得说"我已经写好了"或复述报告原文;不要提到"教师""训练数据""动作"等元信息;
5. 只输出思考正文。"""


def _snip_line(s: dict[str, Any]) -> str:
    return f"- [{s.get('id')}] {s.get('title', '')[:40]}:{str(s.get('content', ''))[:90].replace(chr(10), ' ')}"


def build_messages(case: dict[str, Any], trace: dict[str, Any], call_index: int) -> list[dict[str, str]]:
    call = trace["calls"][call_index]
    # 该次调用时可见的片段 = 首轮 + 此前各轮返回
    seen = list(trace["initial_snippets"])
    for r in trace["rounds"][:call_index]:
        seen += r["snippets"]
    prior = [r["query"] for r in trace["rounds"][:call_index]]
    if call["tool_calls"]:
        tc = call["tool_calls"][0]["arguments"]
        action = f"【动作】调用 retrieve_knowledge,query={tc.get('query')!r},reason={tc.get('reason')!r},top_k={tc.get('top_k')}"
        kind = "tool"
    else:
        action = "【动作】不再检索,直接输出最终报告。报告正文如下(仅供你了解要点,思考里不要复述):\n" + call["content"].strip()[:4000]
        kind = "final"
    user = (
        "【事故信息】\n" + json.dumps({k: v for k, v in case["accident_data"].items() if v}, ensure_ascii=False) + "\n\n"
        "【此时可见的知识库片段】\n" + "\n".join(_snip_line(s) for s in seen) + "\n\n"
        "【此前的检索词】" + ("、".join(prior) if prior else "无") + "\n\n" + action +
        f"\n\n请按要求写出{'检索' if kind == 'tool' else '最终报告'}前的思考(≤{MAX_CHARS[kind]}字)。"
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def validate(text: str, kind: str, allowed_text: str) -> str | None:
    """返回 None 表示通过,否则给出拒绝原因。"""
    t = text.strip()
    if not t:
        return "empty"
    if len(t) > MAX_CHARS[kind] * 1.25:
        return "too_long"
    if len(t) < 20:
        return "too_short"
    if re.search(r"^\s*(#|[-*]\s|\d+[.、])", t, re.MULTILINE):
        return "markdown"
    if re.search(r"教师|训练数据|动作|最终报告如下|retrieve_knowledge", t):
        return "meta"
    for m in ART_RE.findall(t):  # 条号只允许出现在可见文本里
        if f"第{m}条" not in allowed_text:
            return "unseen_article"
    if STD_RE.search(t):
        return "unseen_standard"
    return None


def make_thinking(client: LLMClient, model: str, case: dict[str, Any], trace: dict[str, Any], call_index: int,
                  *, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    call = trace["calls"][call_index]
    kind = "tool" if call["tool_calls"] else "final"
    allowed = json.dumps(case["accident_data"], ensure_ascii=False) + " ".join(
        str(s.get("title", "")) + str(s.get("content", "")) for s in trace["visible_snippets"])
    last_reason = None
    for attempt in (1, 2):
        res = client.chat(model, build_messages(case, trace, call_index), max_tokens=16000, reasoning={"effort": "max"},
                          temperature=None, purpose="rationale", meta={**(meta or {}), "call_index": call_index})
        text = res.content.strip()
        last_reason = validate(text, kind, allowed)
        if last_reason is None:
            return {"ok": True, "text": text, "kind": kind, "attempt": attempt, "cost_usd": res.cost_usd}
    return {"ok": False, "reason": last_reason, "kind": kind}
