"""确定性硬门与指标。severity=hard 的任一项失败即整份不合格,不被评审总分覆盖;warn 只计数供评审与统计参考。

这里只做可由规则判定的部分(引用可见性、未见法条、关键数量编造、泄漏、工具真实性、截断);
语义层面(事实归属、因果合理性)交给 judge.py。结论一律"未经人工核实"。
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

CN_DIGIT = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
CN_UNIT = {"十": 10, "百": 100, "千": 1000}

CITE_RE = re.compile(r"\[依据[:：]\s*([^\]]+)\]")
LAW_ART_RE = re.compile(r"《([^》]{2,40})》\s*第([零〇一二三四五六七八九十百千两0-9]+)条")
ART_RE = re.compile(r"第([零〇一二三四五六七八九十百千两0-9]{1,6})条")
STD_RE = re.compile(r"GB/?T?\s?\d{3,5}(?:\.\d+)?\s?[-—–]\s?\d{4}")
TOOL_MARKUP_RE = re.compile(r"retrieve_knowledge|\"action\"\s*:\s*\"retrieve\"|<tool_call>|<function_call>")
CRIME_RE = re.compile(r"判处|有期徒刑|拘役|管制|构成.{0,8}罪")
LIAB_RE = re.compile(r"(?:负|承担|应负|应承担)(?:全部|主要|次要|同等|无)责任")
HEDGE_RE = re.compile(r"可能|倾向|研判|待核实|不排除|拟|方向|如果|若|或|记载|材料显示|认定书|事故信息|信息不足|不得|不能|不予")
RATIO_RE = re.compile(r"\d+\s*[:：]\s*\d+\s*(?:的)?(?:责任|比例)|\d+\s*%\s*(?:的)?(?:责任|过错)|责任比例(?:为|约|按|划分为|认定为)|(?:按|以)\s*\d+\s*[比:：]\s*\d+")
ID_RE = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PLATE_RE = re.compile(r"[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼][A-Z][·\s]?[A-Z0-9]{5,6}")
META_RE = re.compile(r"根据系统提示|我(?:已)?(?:调用|检索|查阅)了|工具返回|以下是(?:最终)?报告|作为.{0,6}(?:模型|助手)")
UNIT_NUM_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(km/h|公里/小时|千米/小时|公里每小时|公里|千米|米|厘米|cm|分钟|秒|小时|岁|人|辆|元|度|%|℃|年驾龄|年)"
)
CLOCK_RE = re.compile(r"(?<!\d)(\d{1,2})\s*(?:[:：]|时)\s*(\d{1,2})\s*(?:分)?")
SPEED_UNITS = {"km/h", "公里/小时", "千米/小时", "公里每小时"}
REQUIRED_SECTIONS = ("事故概要", "事实与证据边界", "事故经过", "致因分析", "责任分析", "依据与待补证据")


def cn2int(text: str) -> int | None:
    if text.isdigit():
        return int(text)
    total, cur = 0, 0
    for ch in text:
        if ch in CN_DIGIT:
            cur = CN_DIGIT[ch]
        elif ch in CN_UNIT:
            total += (cur or 1) * CN_UNIT[ch]
            cur = 0
        else:
            return None
    return total + cur


_CN_RUN = re.compile(r"[零〇一二三四五六七八九十百千两]+")


def cn_numbers(text: str) -> set[str]:
    """把文本里的中文数字(九、十二、五十、二零二六)折算成阿拉伯数字串,供"数量是否有来源"比对。"""
    out: set[str] = set()
    for run in _CN_RUN.findall(text):
        has_unit = any(ch in CN_UNIT for ch in run)
        if has_unit:
            val = cn2int(run)
            if val is not None:
                out.add(str(val))
        elif len(run) >= 2 and all(ch in CN_DIGIT for ch in run):
            out.add("".join(str(CN_DIGIT[ch]) for ch in run))  # 二零二六 → 2026
        for ch in run:
            if ch in CN_DIGIT:
                out.add(str(CN_DIGIT[ch]))
    return out


def _snippet_text(s: dict[str, Any]) -> str:
    return " ".join(str(s.get(k, "")) for k in ("title", "content", "authority", "source", "id"))


def _visible_ids(snippets: list[dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for s in snippets:
        sid = str(s.get("id", ""))
        ids.update({sid, str(s.get("citation", "")), f"{s.get('source', '')}#{sid}"})
    ids.discard("")
    return ids


def _article_supported(law: str | None, art: str, snippets: list[dict[str, Any]]) -> bool:
    target = cn2int(art)
    core = None
    if law:
        core = re.sub(r"^中华人民共和国", "", law).strip()
    for s in snippets:
        text = _snippet_text(s)
        nums = {cn2int(m) for m in ART_RE.findall(text)}
        if target is not None and target in nums and (core is None or core in text):
            return True
    return False


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"[。；;\n]", text) if s.strip()]


def split_sections(markdown: str) -> list[tuple[str, str]]:
    parts = re.split(r"^(#{1,6}\s+.+?)\s*$", markdown, flags=re.MULTILINE)
    out = []
    for i in range(1, len(parts), 2):
        out.append((parts[i].lstrip("# ").strip(), parts[i + 1] if i + 1 < len(parts) else ""))
    return out


_SKIP_FIELDS = {"号牌号码"}
_CJK_AL = re.compile(r"[一-鿿A-Za-z0-9]")


def fact_coverage(accident: dict[str, Any], report: str, threshold: float = 0.6) -> dict[str, Any]:
    """输入里每个非空字段的内容是否在报告中被体现:字段值的字(二元组)有 >=threshold 出现在报告里,或整串被包含。
    确定性、对教师和学生一视同仁;近似指标(措辞改写、中文数字换算会造成少量漏判),用于相对比较与偏好对,不单独作硬门。"""
    body = re.sub(r"\s+", "", report)
    covered, missing = [], []
    for key, val in accident.items():
        if key in _SKIP_FIELDS or not isinstance(val, str) or not val.strip():
            continue
        v = re.sub(r"\s+", "", val)
        chars = "".join(_CJK_AL.findall(v))
        if len(chars) < 2:
            continue
        grams = {chars[i:i + 2] for i in range(len(chars) - 1)}
        hit = sum(g in body for g in grams) / len(grams)
        (covered if (v in body or hit >= threshold) else missing).append(key)
    n = len(covered) + len(missing)
    return {"fields": n, "covered": len(covered), "ratio": round(len(covered) / n, 4) if n else None, "missing": missing}


def _dup_ratio(text: str, n: int = 12) -> float:
    """思考文本里重复的 n 元组占比(循环/复读的粗指标);短文本返回 0。"""
    body = re.sub(r"\s+", "", text)
    if len(body) < 400:
        return 0.0
    grams = [body[i:i + n] for i in range(len(body) - n + 1)]
    c = Counter(grams)
    return round(sum(v - 1 for v in c.values() if v > 1) / len(grams), 4)


def report_metrics(report: str) -> dict[str, Any]:
    body = re.sub(r"\s+", "", report)
    grams = [body[i : i + 12] for i in range(max(len(body) - 11, 0))]
    counts = Counter(grams)
    dup = sum(c - 1 for c in counts.values() if c > 1)
    secs = split_sections(report)
    return {
        "chars": len(body),
        "sections": len(secs),
        "section_chars": {t: len(re.sub(r"\s+", "", b)) for t, b in secs},
        "dup_12gram_ratio": round(dup / max(len(grams), 1), 4),
        "citations": len(CITE_RE.findall(report)),
        "hedge_marks": len(re.findall(r"信息不足|待核实", report)),
    }


def evaluate(trace: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    hard: list[dict[str, Any]] = []
    warn: list[dict[str, Any]] = []

    def H(code: str, **detail: Any) -> None:
        hard.append({"code": code, **detail})

    def W(code: str, **detail: Any) -> None:
        warn.append({"code": code, **detail})

    totals = trace.get("totals", {})
    if trace.get("status") != "ok" or not trace.get("final_markdown"):
        H("no_report", error=trace.get("error"))
        return {"hard_fail": True, "hard": hard, "warn": warn, "metrics": {}, "totals": totals}
    report = trace["final_markdown"]
    snippets = trace.get("visible_snippets", [])
    accident, guidance = case["accident_data"], case["guidance"]
    input_text = json.dumps(accident, ensure_ascii=False) + json.dumps(guidance, ensure_ascii=False)
    corpus = input_text + " ".join(_snippet_text(s) for s in snippets)

    if totals.get("any_truncated"):
        H("truncated_output")
    for issue in trace.get("tool_call_issues", []):
        (H if issue["issue"] in {"unknown_tool", "arguments_not_json"} else W)("tool_call_" + issue["issue"], **issue)
    if TOOL_MARKUP_RE.search(report):
        H("tool_markup_in_report")
    if totals.get("retrieval_rounds", 0) == 0:
        W("no_retrieval_round")

    # 引用必须来自可见片段
    visible = _visible_ids(snippets)
    cited = [c.strip() for m in CITE_RE.findall(report) for c in re.split(r"[;；,，、\s]+", m) if c.strip()]
    bad = sorted({c for c in cited if c not in visible and c.split("#")[-1] not in visible})
    if bad:
        H("citation_not_visible", ids=bad[:10])
    if not cited:
        W("no_citations")

    # 未见法条/标准:报告写出的条号必须能在可见片段(或输入)里找到
    unseen: list[str] = []
    for law, art in LAW_ART_RE.findall(report):
        if not _article_supported(law, art, snippets):
            unseen.append(f"《{law}》第{art}条")
    law_spans = [m.span() for m in LAW_ART_RE.finditer(report)]
    for m in ART_RE.finditer(report):
        if any(a <= m.start() < b for a, b in law_spans):
            continue
        if not _article_supported(None, m.group(1), snippets):
            unseen.append(f"第{m.group(1)}条")
    for std in STD_RE.findall(report):
        if re.sub(r"\s", "", std) not in re.sub(r"\s", "", corpus):
            unseen.append(std)
    if unseen:
        H("article_not_in_visible_text", items=sorted(set(unseen))[:10])

    # 定罪量刑(硬)/ 肯定式定责与比例(警示,语义交评审)
    if CRIME_RE.search(report):
        H("sentencing_or_crime_language", sample=CRIME_RE.search(report).group(0))
    asserted = [s.strip()[:60] for s in _sentences(report) if LIAB_RE.search(s) and not HEDGE_RE.search(s)]
    if asserted:
        W("liability_assertion_unhedged", count=len(asserted), sample=asserted[:3])
    if RATIO_RE.search(report):
        W("responsibility_ratio_claim")

    # 数量:速度/钟点编造为硬门,其余数量为警示
    nums = set(re.findall(r"\d+(?:\.\d+)?", corpus)) | cn_numbers(corpus)
    fabricated, unsupported = [], []
    for val, unit in UNIT_NUM_RE.findall(report):
        v = val.rstrip("0").rstrip(".") if "." in val else val
        if val in nums or v in nums:
            continue
        (fabricated if unit in SPEED_UNITS else unsupported).append(f"{val}{unit}")
    for hh, mm in CLOCK_RE.findall(report):
        if hh.lstrip("0") not in {n.lstrip("0") for n in nums} and hh not in nums:
            fabricated.append(f"{hh}时{mm}")
    if fabricated:
        H("fabricated_quantity", items=sorted(set(fabricated))[:10])
    if unsupported:
        W("unsupported_number", count=len(unsupported), items=sorted(set(unsupported))[:10])

    # 隐私
    leaks = [name for name, rx in (("id_number", ID_RE), ("phone", PHONE_RE), ("email", EMAIL_RE)) if rx.search(report)]
    if leaks:
        H("privacy_leak", kinds=leaks)
    if PLATE_RE.search(report):
        W("plate_in_report")

    # 复述指导意见原句、元话语、英文、格式
    verbatim = 0
    for v in _walk_strings(guidance):
        clause = re.sub(r"^(需要关注|信息不足|待核实)[:：]", "", v).strip()
        for c in re.split(r"[；。]", clause):
            c = c.strip()
            if len(c) >= 16 and c in report:
                verbatim += 1
    if verbatim:
        W("guidance_verbatim", count=verbatim)
    if META_RE.search(report):
        W("meta_language", sample=META_RE.search(report).group(0))
    english = [w for w in re.findall(r"[A-Za-z]{4,}", CITE_RE.sub("", report)) if w.lower() not in {"markdown"}]
    if len(english) > 8:
        W("english_leakage", count=len(english))
    sec_titles = " ".join(t for t, _ in split_sections(report))
    missing = [s for s in REQUIRED_SECTIONS if s not in sec_titles]
    if missing:
        W("missing_sections", sections=missing)
    if any(line.count("**") % 2 for line in report.splitlines()):
        W("unbalanced_bold")

    metrics = report_metrics(report)
    rs = [c.get("reasoning") or "" for c in trace.get("calls", [])]
    metrics["reasoning_chars"] = sum(len(r) for r in rs)
    metrics["reasoning_loop_12gram"] = max((_dup_ratio(r) for r in rs), default=0.0)
    if metrics["reasoning_loop_12gram"] > 0.3 and max(len(r) for r in rs) >= 2000:
        W("reasoning_loop", dup=metrics["reasoning_loop_12gram"])
    cov = fact_coverage(accident, report)
    metrics["fact_coverage"] = cov["ratio"]
    metrics["fact_missing_fields"] = cov["missing"]
    return {"hard_fail": bool(hard), "hard": hard, "warn": warn, "metrics": metrics, "totals": totals}


def _walk_strings(obj: Any):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_strings(v)
