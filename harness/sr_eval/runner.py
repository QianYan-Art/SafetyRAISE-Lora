"""模式 L:复刻生产旧路径(/api/v1/reports/generate)的报告生成循环。

生产行为(nodes.py::_generate_report_with_native_tool_calling):
- system = 渲染后的 report_prompt.md(首轮 6 片段 + 追加片段 + 检索历史摘要),user = 执行提示;
- 每轮最多处理第一个 retrieve_knowledge 调用;**没有 tool 消息历史**,检索结果靠重新渲染 system 提示带入;
- 最多 2 轮追加检索(共 ≤3 次带工具调用),片段总数 ≤9;超限后不带工具强制收束。
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import prompt as P
from .kb import KnowledgeBase
from .llm import ChatResult, LLMClient, LLMError
from .policy import OutboundPolicy


@dataclass
class RunConfig:
    exec_variant: str = "concise"         # concise(默认,仓库所有者 10/2 定义) | production(生产逐字)
    max_tokens: int = 40000               # 单次调用输出(含思考)上限;仅用于预算预留,截断记为失败
    reasoning: dict[str, Any] | None = None  # None 则用模型默认
    temperature: float | None = None
    max_rounds: int = P.AGENTIC_MAX_ROUNDS
    initial_top_k: int = P.INITIAL_TOP_K
    max_total_snippets: int = P.AGENTIC_MAX_TOTAL_SNIPPETS
    max_query_chars: int = P.AGENTIC_MAX_QUERY_CHARS
    timeout: float = 1800


def _merge(current: list[dict], incoming: list[dict], max_total: int) -> list[dict]:
    merged, seen = [], set()
    for item in current + incoming:
        iid = str(item.get("id", ""))
        if not iid or iid in seen:
            continue
        seen.add(iid)
        merged.append(item)
        if len(merged) >= max_total:
            break
    return merged


def _coerce_top_k(value: Any, max_top_k: int = P.AGENTIC_TOP_K_PER_ROUND) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        n = max_top_k
    return min(max(n, 1), max_top_k)


def _resolve_final(raw: str) -> str:
    """nodes.py::_resolve_final_report:正文或 {"action":"final","report_markdown":...}。"""
    action = P.extract_json_action(raw)
    if action is None:
        content = P.sanitize_markdown_output(raw)
        if not content:
            raise ValueError("报告模型返回为空")
        return content
    if str(action.get("action", "")).lower() != "final":
        raise ValueError("报告模型未产出最终报告正文")
    md = P.sanitize_markdown_output(str(action.get("report_markdown") or action.get("content") or ""))
    if not md:
        raise ValueError("报告模型 final 响应缺少 report_markdown")
    return md


def _call_record(res: ChatResult, system: str, user: str, with_tools: bool) -> dict[str, Any]:
    rec = res.to_dict()
    rec["messages"] = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    rec["with_tools"] = with_tools
    return rec


def run_case(client: LLMClient, model: str, case: dict[str, Any], kb: KnowledgeBase, cfg: RunConfig,
             *, policy: OutboundPolicy | None = None, run_id: str = "") -> dict[str, Any]:
    spec = client.models[model]
    if policy is not None and spec.provider != "local":
        policy.check(kb_class=kb.outbound_class, case_kind=case.get("kind", "real"))
    template = P.load_asset("report_prompt.md")
    accident, guidance = case["accident_data"], case["guidance"]
    meta = {"run_id": run_id, "case_id": case["case_id"]}
    trace: dict[str, Any] = {
        "case_id": case["case_id"], "model": model, "slug": spec.slug, "kb": kb.name, "mode": "L",
        "exec_variant": cfg.exec_variant, "config": {k: v for k, v in cfg.__dict__.items()},
        "calls": [], "rounds": [], "tool_call_issues": [], "status": "ok", "error": None,
    }
    t0 = time.time()
    query, initial = kb.retrieve(accident, cfg.initial_top_k)
    trace["initial_query"], trace["initial_snippets"] = query, initial
    combined, additional, rounds = list(initial), [], trace["rounds"]
    tools = P.retrieve_tool_schema() if spec.supports_tools else None
    final_raw: str | None = None

    def render() -> str:
        return P.render_report_prompt(template, accident, guidance, initial, additional, rounds)

    def call(system: str, user: str, with_tools: bool) -> ChatResult:
        res = client.chat(model, [{"role": "system", "content": system}, {"role": "user", "content": user}],
                          tools=P.retrieve_tool_schema() if with_tools and tools else None, max_tokens=cfg.max_tokens,
                          reasoning=cfg.reasoning, temperature=cfg.temperature, timeout=cfg.timeout,
                          purpose="report_generation", meta=meta)
        trace["calls"].append(_call_record(res, system, user, with_tools))
        if res.finish_reason == "error":  # OpenRouter:上游在生成中途报错,内容不可用
            raise LLMError("provider_finish_error", billing_state="settled", detail="finish_reason=error")
        return res

    try:
        forced = False
        for round_index in range(cfg.max_rounds + 1):
            system, user = render(), P.build_exec_prompt(False, cfg.exec_variant)
            res = call(system, user, True)
            tc = res.tool_calls[0] if res.tool_calls else None
            if len(res.tool_calls) > 1:
                trace["tool_call_issues"].append({"round": round_index, "issue": "multiple_tool_calls", "count": len(res.tool_calls)})
            if tc is None and not spec.supports_tools:  # 不支持原生工具的模型走 JSON 回退协议(近似复刻)
                action = P.extract_json_action(res.content)
                if action and str(action.get("action", "")).lower() == "retrieve":
                    tc = {"name": P.RETRIEVE_TOOL_NAME, "arguments": {k: action.get(k) for k in ("query", "reason", "top_k")},
                          "arguments_valid": True}
            if tc is None:
                if not res.content.strip():
                    raise ValueError("报告模型既未返回工具调用，也未返回报告正文")
                final_raw = res.content
                break
            if tc["name"] != P.RETRIEVE_TOOL_NAME:
                trace["tool_call_issues"].append({"round": round_index, "issue": "unknown_tool", "name": tc["name"]})
                raise ValueError(f"报告模型调用了未知工具: {tc['name']}")
            if not tc.get("arguments_valid", True):
                trace["tool_call_issues"].append({"round": round_index, "issue": "arguments_not_json"})
            if round_index >= cfg.max_rounds:
                forced = True
                break
            args = tc.get("arguments") or {}
            q = str(args.get("query", "")).strip()
            if not q:
                trace["tool_call_issues"].append({"round": round_index, "issue": "empty_query"})
                forced = True
                break
            q = q[: cfg.max_query_chars]
            top_k = _coerce_top_k(args.get("top_k"))
            new = kb.search(q, top_k)
            combined = _merge(combined, new, cfg.max_total_snippets)
            init_ids = {str(s.get("id", "")) for s in initial}
            additional = [s for s in combined if str(s.get("id", "")) not in init_ids]
            rounds.append({"round": len(rounds) + 1, "query": q, "reason": str(args.get("reason", "")).strip(),
                           "requested_top_k": top_k, "returned_count": len(new), "snippets": new})
        if final_raw is None:
            res = call(render(), P.build_exec_prompt(True, cfg.exec_variant), False)
            final_raw = res.content
            trace["forced_finalize"] = True
        trace["final_raw"] = final_raw
        trace["final_markdown"] = _resolve_final(final_raw)
    except LLMError as exc:
        trace["status"], trace["error"] = "error", f"{exc.code}: {exc.detail[:300]}"
        trace["billing_state"] = exc.billing_state
    except ValueError as exc:
        trace["status"], trace["error"] = "error", f"workflow: {exc}"
    trace["visible_snippets"] = combined
    calls = trace["calls"]
    trace["totals"] = {
        "calls": len(calls), "retrieval_rounds": len(rounds),
        "prompt_tokens": sum(c["prompt_tokens"] for c in calls),
        "completion_tokens": sum(c["completion_tokens"] for c in calls),
        "reasoning_tokens": sum(c["reasoning_tokens"] for c in calls),
        "reasoning_chars": sum(len(c.get("reasoning") or "") for c in calls),
        "cost_usd": round(sum(c["cost_usd"] for c in calls), 6),
        "latency_s": round(time.time() - t0, 1),
        "last_finish_reason": calls[-1]["finish_reason"] if calls else None,
        "any_truncated": any(c["finish_reason"] == "length" for c in calls),
    }
    return trace


def run_many(client: LLMClient, jobs: list[tuple[str, dict[str, Any]]], kb: KnowledgeBase, cfg: RunConfig,
             out_dir: Path, *, policy: OutboundPolicy | None, run_id: str, workers: int = 1) -> list[dict[str, Any]]:
    """jobs = [(model, case)];结果逐个落盘 out_dir/traces/<case>__<model>.json;已存在则跳过(可续跑)。"""
    (out_dir / "traces").mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    def one(job: tuple[str, dict[str, Any]]) -> dict[str, Any]:
        model, case = job
        path = out_dir / "traces" / f"{case['case_id']}__{model}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        trace = run_case(client, model, case, kb, cfg, policy=policy, run_id=run_id)
        path.write_text(json.dumps(trace, ensure_ascii=False, indent=1), encoding="utf-8")
        return trace

    if workers <= 1:
        for job in jobs:
            results.append(one(job))
        return results
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, j) for j in jobs]
        for fut in as_completed(futures):
            results.append(fut.result())
    return results
