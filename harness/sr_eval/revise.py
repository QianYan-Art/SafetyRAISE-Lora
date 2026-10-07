"""教师修订:让强模型(luna max)在"学生看到的同一份输入"下,依据严格评审指出的缺陷,最小改动地修订学生的报告初稿。

动机(2026-10-03 同口径评审):基座 xhigh 的扣分集中在内容不克制与引用不规范(无依据因果、夸大责任、补入材料外事实、
有法律陈述未引用、引用不当),硬门管不到;基座自己多份回答里挑最好的也只有 17.7 分(teacher 约 21.6),
所以仅靠基座同源样本的偏好无法补齐,需要教师信号。修订稿与学生的原生思考保持一致(只做外科手术式的改动),
训练时 目标 = 学生原生思考 + 修订后的报告,思考不被替换。

修订稿必须再过硬门,并由 luna 与 MiniMax 再评一次:任一评审变差就不采用。
"""

from __future__ import annotations

import json
from typing import Any

REVISE_INSTRUCTION = """上面是你写的报告初稿。独立审稿人指出了以下问题(引文均摘自初稿):
{critique}

请据此修订,要求:
1. 凡是材料没有支持的事实、因果判断和责任判断,删除,或改写成"待核实/材料未载明"的条件式表述;不要新增任何材料之外的事实(时间、数字、伤情、当事人行为、争议点)。
2. 涉及法规、规则、标准的陈述,只能引用上文知识库片段中实际可见的条款,并按要求的引用格式标注;没有可见依据的法律陈述一律删除,引用不当的改引可见且相关的条款。
3. 在遵守以上两点的前提下,保留初稿对输入事实的覆盖、原有结构和措辞,尽量少改动;不要因为想"更稳妥"而删掉有依据的内容。
4. 只输出修订后的完整报告正文,不要解释,不要出现"修订"字样。"""


def format_critique(defects: list[dict[str, Any]], limit: int = 14) -> str:
    rows = []
    for i, d in enumerate(defects[:limit], 1):
        quote = str(d.get("quote") or "").strip().replace("\n", " ")[:160]
        rows.append(f"{i}. [{d.get('dimension', '')}/{d.get('type', '')}] 引文:「{quote}」——{str(d.get('explanation') or '').strip()[:200]}")
    return "\n".join(rows)


def revise_messages(call_messages: list[dict[str, Any]], draft: str, defects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """原始输入消息 + 初稿(助手) + 审稿意见与修订要求(用户)。"""
    return [*call_messages, {"role": "assistant", "content": draft},
            {"role": "user", "content": REVISE_INSTRUCTION.format(critique=format_critique(defects))}]


def revised_trace(src: dict[str, Any], revised_md: str, raw: str, *, src_run: str, reviser: str, usage: dict[str, Any]) -> dict[str, Any]:
    """从原 trace 复制出修订 trace:保留学生的检索/思考,最后一次调用的正文换成修订稿。键名加 +rev 后缀。"""
    out = json.loads(json.dumps(src))
    out["model"] = f"{src['model']}+rev"
    out["mode"] = "revise"
    out["status"] = "ok" if revised_md.strip() else "error"
    out["error"] = None if revised_md.strip() else "workflow: 修订稿为空"
    out["source"] = {"run": src_run, "model": src["model"], "reviser": reviser, **usage}
    out["final_raw"] = raw
    out["final_markdown"] = revised_md
    last = out["calls"][-1]
    last["content"] = raw
    last["tool_calls"] = []
    last["finish_reason"] = "stop"
    return out


REPAIRABLE_GATES = {"article_not_in_visible_text", "citation_not_visible", "sentencing_or_crime_language", "fabricated_quantity", "tool_markup_in_report"}


def gate_defects(gate_entry: dict[str, Any]) -> list[dict[str, Any]]:
    """把硬门失败项转成"审稿意见"(与评审缺陷同格式),供教师精准修复。"""
    out = []
    for h in gate_entry.get("hard", []):
        code, items = h["code"], h.get("items") or []
        quote = "、".join(str(x) for x in items)[:160]
        if code == "article_not_in_visible_text":
            expl = "报告写到了上文知识库片段里并不存在的条文。不得出现片段中看不到的条文编号或条文内容;删除相应的法律陈述,或改为引用上文可见且相关的条款。"
        elif code == "citation_not_visible":
            expl = "报告的引用标注指向了上文不可见的依据。引用只能指向上文知识库片段中实际存在的条目;删除无法对应的引用,或改引可见且相关的条目。"
        elif code == "sentencing_or_crime_language":
            expl = "报告出现了定罪量刑类措辞。本报告只做事实与责任分析,不得出现定罪、量刑、刑期、罪名等表述,请改写或删除。"
        elif code == "fabricated_quantity":
            expl = "报告出现了材料里没有的数量或比例。删除或改为材料已有的数值。"
        else:
            expl = "该处违反输出规范,请改写。"
        out.append({"dimension": "硬门", "type": code, "quote": quote, "explanation": expl})
    return out
