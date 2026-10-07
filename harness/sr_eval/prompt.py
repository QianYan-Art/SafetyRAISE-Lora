"""生产报告提示的渲染与旧路径(retrieve_knowledge)执行提示。

渲染逻辑与 TS_analysis_report/backend/app/workflow/prompt_rendering.py、nodes.py 保持一致;
提示模板取自 assets/report_prompt.md(冻结副本,来源与哈希见 assets/ASSETS.json)。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ASSET_DIR = Path(__file__).resolve().parent.parent / "assets"

REPORT_GUIDANCE_PLACEHOLDER = "{在这里粘贴指导意见JSON}"
REPORT_ACCIDENT_PLACEHOLDER = "{在这里粘贴结构化事故信息JSON}"
REPORT_ACCIDENT_ANCHOR_PLACEHOLDER = "{在这里粘贴事故信息关键锚点摘要}"
REPORT_INITIAL_SNIPPETS_PLACEHOLDER = "{在这里粘贴首轮知识库片段JSON}"
REPORT_ADDITIONAL_SNIPPETS_PLACEHOLDER = "{在这里粘贴模型追加检索获得的新知识库片段JSON}"
REPORT_AGENTIC_HISTORY_PLACEHOLDER = "{在这里粘贴模型追加检索历史摘要}"

RETRIEVE_TOOL_NAME = "retrieve_knowledge"
# 与 workflow.yaml 的 retrieval.agentic 一致(2026-09-25 主线)
AGENTIC_MAX_ROUNDS = 2
AGENTIC_TOP_K_PER_ROUND = 3
AGENTIC_MAX_TOTAL_SNIPPETS = 9
AGENTIC_MAX_QUERY_CHARS = 120
INITIAL_TOP_K = 6


def load_asset(name: str) -> str:
    return (ASSET_DIR / name).read_text(encoding="utf-8")


def strip_internal_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: strip_internal_fields(v) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, list):
        return [strip_internal_fields(v) for v in value]
    return value


_ANCHOR_FIELDS = [
    ("事故标题", ["事故标题"]),
    ("事故类型", ["事故类型"]),
    ("事故形态", ["事故形态"]),
    ("事故发生时间", ["事故发生时间", "事故时间"]),
    ("地点（包括路名，路号）", ["地点（包括路名，路号）"]),
    ("路口路段类型", ["路口路段类型"]),
    ("主要违法行为", ["主要违法行为"]),
    ("事故认定原因", ["事故认定原因"]),
    ("车辆类型", ["车辆类型"]),
    ("伤害程度", ["伤害程度", "伤亡情况"]),
]


def build_anchor_summary(accident_data: dict[str, Any]) -> str:
    summary: dict[str, Any] = {}
    for summary_key, candidates in _ANCHOR_FIELDS:
        for key in candidates:
            value = accident_data.get(key)
            if isinstance(value, str) and value.strip():
                summary[summary_key] = value.strip()
                break
    if not summary:
        for key, value in accident_data.items():
            if isinstance(value, str) and value.strip():
                summary[key] = value.strip()
                if len(summary) >= 8:
                    break
    if not summary:
        summary = {"提示": "事故信息中暂无可提取的关键锚点"}
    return json.dumps(summary, ensure_ascii=False, indent=2)


def summarize_rounds(rounds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "round": r["round"],
            "query": r["query"],
            "reason": r["reason"],
            "requested_top_k": r["requested_top_k"],
            "returned_count": r["returned_count"],
            "snippet_ids": [s.get("id", "") for s in r["snippets"]],
        }
        for r in rounds
    ]


def render_report_prompt(
    template: str,
    accident_data: dict[str, Any],
    guidance: dict[str, Any],
    initial_snippets: list[dict[str, Any]],
    additional_snippets: list[dict[str, Any]],
    rounds: list[dict[str, Any]],
) -> str:
    """每个占位符只替换首次匹配,与生产实现相同。"""
    data = strip_internal_fields(accident_data)
    replacements = {
        REPORT_GUIDANCE_PLACEHOLDER: json.dumps(guidance, ensure_ascii=False, indent=2),
        REPORT_ACCIDENT_PLACEHOLDER: json.dumps(data, ensure_ascii=False, indent=2),
        REPORT_ACCIDENT_ANCHOR_PLACEHOLDER: build_anchor_summary(data),
        REPORT_INITIAL_SNIPPETS_PLACEHOLDER: json.dumps(initial_snippets, ensure_ascii=False, indent=2),
        REPORT_ADDITIONAL_SNIPPETS_PLACEHOLDER: json.dumps(additional_snippets, ensure_ascii=False, indent=2),
        REPORT_AGENTIC_HISTORY_PLACEHOLDER: json.dumps(summarize_rounds(rounds), ensure_ascii=False, indent=2),
    }
    rendered = template
    for placeholder, value in replacements.items():
        if placeholder not in rendered:
            raise ValueError(f"报告提示词模板缺少占位内容: {placeholder}")
        rendered = rendered.replace(placeholder, value, 1)
    return rendered


def build_exec_prompt(force_finalize: bool, variant: str = "production") -> str:
    """nodes.py::_build_report_execution_prompt 的复刻。

    variant="production":逐字沿用生产文案(其中"尽可能充分展开"与提示词里的简洁要求互相冲突);
    variant="concise":仓库所有者 2026-10-02 定义——"简洁"指整体描述简洁,"展开"指不要漏项;用于数据集/微调目标(默认)。
    """
    if variant not in {"production", "concise"}:
        raise ValueError(f"未知执行提示变体: {variant}")
    if force_finalize:
        if variant == "concise":
            return "不要再申请知识库检索，也不要调用 retrieve_knowledge 工具，直接输出最终 Markdown 报告正文；描述要简洁（同一事实只写一次，不为凑篇幅重复），但分析项不要漏：事故概要、事实与证据边界、事故经过、致因、责任、依据与待补证据都要写到，人、车、路、环境和已知处置资源中与判断有关的事实不得遗漏。"
        return "不要再申请知识库检索，也不要调用 retrieve_knowledge 工具，直接输出最终 Markdown 报告正文；请充分展开核心分析，不要写成摘要。"
    if variant == "concise":
        return (
            "请先判断现有材料是否足够；若不足且系统提供了 retrieve_knowledge 工具，请直接调用该工具补充依据；"
            "只有在当前环境不提供工具时，才按模板约定仅输出检索 JSON；若材料已经足够，直接输出最终 Markdown 报告正文，"
            "描述要简洁（同一事实只写一次，不为凑篇幅重复），但分析项不要漏：事故概要、事实与证据边界、事故经过、致因、责任、依据与待补证据都要写到，"
            "人、车、路、环境和已知处置资源中与判断有关的事实不得遗漏。"
        )
    return (
        "请先判断现有材料是否足够；若不足且系统提供了 retrieve_knowledge 工具，请直接调用该工具补充依据；"
        "只有在当前环境不提供工具时，才按模板约定仅输出检索 JSON；若材料已经足够，直接输出最终 Markdown 报告正文，"
        "并尽可能充分展开事故经过、致因分析与责任分析。"
    )


def retrieve_tool_schema(max_top_k: int = AGENTIC_TOP_K_PER_ROUND) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": RETRIEVE_TOOL_NAME,
                "description": (
                    "当首轮知识片段不足以支撑责任分析、违法依据、道路控制信息、特殊天气路况解释时，"
                    "检索本地交通事故知识库并返回补充片段。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "本轮最关键的检索词，尽量短而准。"},
                        "reason": {"type": "string", "description": "本轮检索要补充的依据点。"},
                        "top_k": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": max_top_k,
                            "description": f"本轮希望返回的片段数量，范围 1 到 {max_top_k}。",
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        }
    ]


# ---- 输出清洗:原样使用冻结的生产实现(sr_eval/prod_model_output.py) ----
from .prod_model_output import sanitize_markdown_output  # noqa: E402,F401


def extract_json_action(text: str) -> dict[str, Any] | None:
    """JSON 回退协议:{"action":"retrieve"|"final", ...};解析不到则返回 None。"""
    if not text:
        return None
    candidates = [text.strip()]
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if m:
        candidates.append(m.group(1))
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start : end + 1])
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict) and str(obj.get("action", "")).strip().lower() in {"retrieve", "final"}:
            return obj
    return None
