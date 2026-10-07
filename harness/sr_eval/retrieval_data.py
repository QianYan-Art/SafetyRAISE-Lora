"""主动检索训练数据(2026-10-04):让模型学会"写报告前先核对引用,可见片段里没有的条文要先检索"。

数据来源
  T1 事后重标注:最终调用写出了"可见片段里没有"的条文(article_not_in_visible_text),而这个调用当时还带着检索工具、该条文在知识库里存在
     → 目标 = 学生原生思考 + 结尾一段"核对引用、发现缺这条、先检索"的自检 + retrieve_knowledge 调用(查询由多个写手模型生成,
     并用真实的 kb.search 验证目标条文确实能被检到)。
  T2 结果筛选:好结果轨迹(硬门通过、评审达标、检索到的片段被最终报告引用)里的检索调用,原样作为目标(思考 + 工具调用都是学生自己的)。
"""

from __future__ import annotations

import json
import re
from typing import Any

CN = "零一二三四五六七八九"
LAW_ALIAS = {"道交法": "道路交通安全法", "交通安全法": "道路交通安全法", "道路交通安全法实施条例": "道路交通安全法实施条例"}


def int_to_cn(n: int) -> str:
    if n < 10:
        return CN[n]
    if n < 20:
        return "十" + (CN[n % 10] if n % 10 else "")
    if n < 100:
        return CN[n // 10] + "十" + (CN[n % 10] if n % 10 else "")
    r = n % 100
    head = CN[n // 100] + "百"
    if r == 0:
        return head
    if r < 10:
        return head + "零" + CN[r]
    return head + (int_to_cn(r) if r >= 20 else "一十" + (CN[r % 10] if r % 10 else ""))


def cn_to_int(s: str) -> int | None:
    s = s.strip()
    if s.isdigit():
        return int(s)
    try:
        total = 0
        if "百" in s:
            h, s = s.split("百", 1)
            total += (CN.index(h) if h else 1) * 100
            s = s.lstrip("零")
        if "十" in s:
            t, s = s.split("十", 1)
            total += (CN.index(t) if t else 1) * 10
        if s:
            total += CN.index(s)
        return total
    except ValueError:
        return None


def parse_item(item: str) -> tuple[str | None, int | None]:
    """'《中华人民共和国道路交通安全法（2021修正）》第四十三条' → ('道路交通安全法', 43)。"""
    m = re.search(r"第([0-9]+|[零一二三四五六七八九十百]+)条", item)
    n = cn_to_int(m.group(1)) if m else None
    law = re.search(r"《(.+?)》", item)
    key = None
    if law:
        key = re.sub(r"[（(].*?[）)]", "", law.group(1)).replace("中华人民共和国", "").strip()
        key = LAW_ALIAS.get(key, key)
    return key, n


def record_text(rec: dict[str, Any]) -> str:
    return " ".join(str(v) for v in rec.values() if isinstance(v, str))


def find_targets(kb: Any, item: str) -> list[dict[str, Any]]:
    """知识库里能提供该条文原文的片段(分块或规则)。法名能匹配则按法名,否则只按条号(宽松,供人工看)。"""
    law, n = parse_item(item)
    if n is None:
        return []
    law = law or "道路交通安全法"       # 只写"第七十六条"而不带法名时,默认按道路交通安全法匹配(其余法规不会靠条号凭空命中)
    pats = [f"第{int_to_cn(n)}条", f"第{n}条"]
    out = []
    for rec in [*kb.chunks, *kb.rules]:
        text = record_text(rec)
        if not any(p in text for p in pats):
            continue
        if law and law not in text:
            continue
        rid = rec.get("chunk_id") or rec.get("rule_id")
        out.append({"id": rid, "title": rec.get("title") or rec.get("name") or "", "excerpt": (rec.get("content") or text)[:260]})

    def rank(t: dict[str, Any]) -> tuple[int, int]:
        rid = str(t["id"] or "")
        statute = 0 if any(k in rid for k in ("court_case", "gongbao", "guide", "manual")) else 1      # 法律法规原文优先于案例/公报/说明
        return (statute, 1 if law in (str(t["title"]) + rid) else 0)

    out.sort(key=rank, reverse=True)
    return out


def cited_sentences(report: str, item: str, limit: int = 2) -> list[str]:
    law, n = parse_item(item)
    keys = [item]
    if n is not None:
        keys += [f"第{int_to_cn(n)}条", f"第{n}条"]
    out = []
    for s in re.split(r"(?<=[。；\n])", report):
        if any(k in s for k in keys) and s.strip():
            out.append(s.strip()[:220])
        if len(out) >= limit:
            break
    return out


WRITER_INSTRUCTION = """你在为一个交通事故分析报告模型构造训练样本,内容是"写报告前先核对引用"。
情境:模型在写最终报告前,发现草稿准备引用某条法规,但当前可见的知识库片段里**没有**这一条的原文。正确做法不是凭记忆引用,而是先调用 retrieve_knowledge 检索,拿到原文后再写。
下面给出若干情境(每个有 id)。请对每个情境写出:
- "closing":第一人称中文自检段落,80~220 字,接在模型已有思考的末尾。要点:草稿里哪处论点想引用哪条;已可见的依据(点名 1~3 条片段标题)里没有这一条的原文;不能凭记忆引用;所以先检索再写。不要出现"训练""样本"等元信息,不要复述整份报告。
- "query":检索关键词串,不超过 100 个字符,含法名、条号和该条的核心语义关键词,使该条原文能被检索到;
- "reason":一句话说明为什么要检索,不超过 60 字;
- "top_k":2 或 3。
**只输出一个 JSON 数组**(不要代码块之外的文字),每个情境一个对象:{"id":"…","closing":"…","query":"…","reason":"…","top_k":2}。
"""


def extract_array(text: str) -> list[dict[str, Any]] | None:
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    dec, best = json.JSONDecoder(), None
    for m in re.finditer(r"\[", t):
        try:
            o, _ = dec.raw_decode(t[m.start():])
        except ValueError:
            continue
        if isinstance(o, list) and o and all(isinstance(x, dict) and "id" in x for x in o):
            best = o
    return best


def verify_writer_item(kb: Any, ctx: dict[str, Any], out: dict[str, Any]) -> tuple[bool, str]:
    """真实检索验证:查询在 top_k 内必须检到目标片段之一;自检段落长度与要点合格。"""
    q = str(out.get("query", "")).strip()[:120]
    if not q:
        return False, "空查询"
    try:
        top_k = int(out.get("top_k", 3))
    except (TypeError, ValueError):
        return False, "top_k 非法"
    if top_k not in (1, 2, 3):
        return False, "top_k 超范围"
    hits = {str(s.get("id")) for s in kb.search(q, top_k)}
    if not hits & {t["id"] for t in ctx["targets"]}:
        return False, "查询未检到目标条文"
    closing = str(out.get("closing", "")).strip()
    if not (60 <= len(closing) <= 320):
        return False, f"自检段落长度 {len(closing)} 不合格"
    if not re.search(r"第[0-9零一二三四五六七八九十百]+条", closing):
        return False, "自检段落没点名条号"
    if re.search(r"训练|样本|示例|标注", closing):
        return False, "自检段落含元信息"
    if len(str(out.get("reason", "")).strip()) > 120:
        return False, "reason 过长"
    return True, "ok"


_SRC_ID = re.compile(r"[A-Za-z][A-Za-z0-9_]*(?:#rule)?#\d+")
_NAMED_ART = re.compile(r"《([^》]{2,30})》第([0-9零一二三四五六七八九十百]+)条")


def grounding_paragraph(report: str, visible: list[dict[str, Any]], variant: int) -> str | None:
    """正例自检:报告拟引用的条文/依据都能在已见到的片段里找到。全部由报告里的引用与可见片段推出;推不出就不写(返回 None)。"""
    titles = {str(x.get("id")): str(x.get("title") or "") for x in visible}
    cited = []
    for sid in _SRC_ID.findall(report):
        t = titles.get(sid)
        if t and t not in cited:
            cited.append(t)
    if not cited:
        return None
    short = [t if len(t) <= 22 else t[:22] + "…" for t in cited[:3]]
    names = "、".join(f"《{t}》" for t in short)
    arts = []
    for law, num in _NAMED_ART.findall(report):
        label = f"《{law}》第{num}条"
        if label not in arts:
            arts.append(label)
    arts = arts[:3]
    if arts:
        what = "、".join(arts)
        forms = [
            f"写报告前先核对引用:拟引用的{what}都能在已见到的{names}等片段里找到原文,没有凭记忆补充的条文,可以引用。",
            f"核对一下依据:报告里要用到{what},对应的原文在已见到的{names}等片段中都能看到,没有超出可见范围的条文。",
            f"动笔前再核对引用范围:{what}均见于{names}等已见到的片段,不需要再检索,可以直接引用。",
            f"引用自检:{what}在{names}等可见片段里都有原文,不是凭记忆写的,可以引用。",
        ]
    else:
        forms = [
            f"写报告前先核对引用:拟引用的依据都来自已见到的{names}等片段,没有超出可见范围的条文。",
            f"核对一下依据:要引用的内容在已见到的{names}等片段里都能找到,没有凭记忆补充的条文。",
            f"动笔前再核对引用范围:依据均来自{names}等已见到的片段,不需要再检索。",
            f"引用自检:所引依据都在{names}等可见片段里,不是凭记忆写的。",
        ]
    return forms[variant % len(forms)]
