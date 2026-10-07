"""学生逐调用校验及最小修订；不自动启用网络，不接受真实事故输入。"""

from __future__ import annotations

import asyncio
import difflib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.report_harness.contracts import (
    canonical_digest, candidate_contract_problems, candidate_round_problems,
)
from app.report_harness.controlled_tools import ControlledTools
from app.report_harness.errors import HarnessError
from app.report_harness.quoted_roles import resolve_quotes
from app.report_harness.role_loop import ToolTurn, normalize_tool_response, role_context
from app.report_harness.source_access import generator_access_problems
from app.schemas.report_run import CandidateReport
from sr_eval.llm import LLMError

from .env import production_workflow
from .metrics import FIELD_NAMES, FIELD_SPECS, evaluate_candidate
from .samples import normalize_call

ROOT = Path(__file__).resolve().parents[3]


def workspace_path(path: str | Path, *, output: bool = False) -> Path:
    """解析工作区路径，拒绝符号链接、junction 以及越界目录。"""
    original = Path(path)
    original = original if original.is_absolute() else ROOT / original
    resolved = original.resolve()
    if resolved != ROOT and ROOT not in resolved.parents:
        raise ValueError("路径必须位于本任务工作区。")
    cursor = original
    while cursor != ROOT and cursor.parent != cursor:
        if cursor.is_symlink() or getattr(cursor, "is_junction", lambda: False)():
            raise ValueError("路径不得经过符号链接或 junction。")
        cursor = cursor.parent
    if output and resolved == ROOT:
        raise ValueError("输出不能覆盖工作区根目录。")
    return resolved


def call_hash(trace: dict, call: dict) -> str:
    normalized = normalize_call(trace, call)
    return canonical_digest({
        "payload": normalized["payload"], "content": normalized["content"],
        "reasoning_content": normalized["reasoning"], "response": normalized["response"],
    })


def load_traces(directory: str | Path) -> tuple[list[tuple[Path, dict]], list[dict]]:
    """只加载明确合成的 live trace；错误只返回路径和类型，不泄露正文。"""
    directory = workspace_path(directory)
    if not directory.is_dir():
        raise ValueError("学生 trace 目录不存在。")
    records, errors = [], []
    for path in sorted(directory.rglob("*.json")):
        workspace_path(path)
        try:
            trace = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            errors.append({"path": str(path.relative_to(ROOT)), "reason": type(exc).__name__})
            continue
        if not isinstance(trace, dict) or not isinstance(trace.get("calls"), list):
            errors.append({"path": str(path.relative_to(ROOT)), "reason": "not_live_trace"})
            continue
        if (trace.get("case", {}).get("kind") != "synthetic"
                or trace.get("provenance", {}).get("synthetic") is not True):
            errors.append({"path": str(path.relative_to(ROOT)), "reason": "synthetic_provenance_required"})
            continue
        records.append((path, trace))
    return records, errors


def _context(trace: dict, call: dict) -> dict:
    payload = call["payload"]
    messages = payload["messages"]
    if len(messages) != 3 or [m["role"] for m in messages] != ["system", "user", "user"]:
        raise ValueError("必须使用学生实际收到的三条消息。")
    context = json.loads(messages[-1]["content"])
    snapshot = trace["snapshot"]
    expected, _ = role_context(snapshot)
    if (trace.get("case", {}).get("kind") != "synthetic"
            or trace.get("provenance", {}).get("synthetic") is not True
            or set(snapshot["accident_data"]) != set(FIELD_NAMES)
            or snapshot["accident_data"] != trace["case"]["accident_data"]
            or canonical_digest(snapshot) != trace["snapshot_digest"]
            or context["snapshot"] != expected):
        raise ValueError("合成输入或快照绑定不成立。")
    if call.get("request_sha256", canonical_digest(payload)) != canonical_digest(payload):
        raise ValueError("学生调用请求摘要不匹配。")
    if call.get("context_digest", canonical_digest(context)) != canonical_digest(context):
        raise ValueError("学生上下文摘要不匹配。")
    return context


def _replay_tools(trace: dict, context: dict) -> ControlledTools:
    """重放可见上下文中的读操作，保留分页和累计检索额度。"""
    registry = trace["knowledge_registry"]
    by_id = {item["id"]: item for item in registry}
    initial = context.get("initial_knowledge_snippets", [])
    search_items: list[dict] = []
    workflow = production_workflow()
    tools = ControlledTools(
        trace["snapshot"], registry, compact_registry=True,
        retrieval_policy=workflow.retrieval_policy(),
        search=lambda query, top_k: deepcopy(search_items),
        initial_search=lambda query, top_k: deepcopy(initial),
    )
    initial_event = next(
        item for item in trace["retrieval"]["events"] if item.get("step") == "initial"
    )
    if (initial_event["approved_ids"] != [item["id"] for item in initial]
            or initial_event["approved_projection_digest"] != canonical_digest(initial)):
        raise ValueError("首检原文未绑定控制器事件。")
    tools.initial_retrieval(workflow.initial_query(trace["snapshot"]["accident_data"]),
                            workflow.initial_top_k)
    for visible in context.get("tool_results", []):
        events = [
            event for event in trace["tools"]
            if event.get("role") == "generator" and event.get("name") == visible["name"]
            and event.get("private", {}).get("call_id") == visible["call_id"]
        ]
        event = next(event for event in events if "arguments" in event.get("private", {}))
        args = event["private"]["arguments"]
        expected = visible["result"]
        if visible["name"] == "search_knowledge" and "error" not in expected:
            search_items[:] = [by_id[item["id"]] for item in expected["items"]]
            if not any(
                entry.get("step") == "additional"
                and entry.get("query") == " ".join(args["query"].split())
                and entry.get("approved_ids") == [item["id"] for item in search_items]
                and entry.get("approved_projection_digest") == canonical_digest(search_items)
                and 0 < entry.get("top_k", 0) <= args["top_k"]
                for entry in trace["retrieval"]["events"]
            ):
                raise ValueError("追加检索未绑定控制器批准记录。")
        try:
            replayed = tools.execute("generator", visible["name"], args)
        except HarnessError as exc:
            if (expected.get("error", {}).get("code") != exc.code
                    or expected.get("error", {}).get("constraints") != tools.retrieval_constraints("generator")
                    or not any(item.get("code") == exc.code for item in events)):
                raise ValueError("历史工具拒绝无法重放。") from exc
        else:
            if replayed != expected or not any(
                item.get("private", {}).get("result") == expected
                and item.get("result_digest") == canonical_digest(expected) for item in events
            ):
                raise ValueError("工具结果或摘要无法重放。")
    search_items.clear()
    return tools


def _metric_fail_signals(metrics: dict) -> list[str]:
    signals = []
    for name, check in metrics.get("checks", {}).items():
        if check.get("status") != "fail":
            continue
        fields = check.get("fields", {})
        items = fields.items() if isinstance(fields, dict) else enumerate(fields)
        failures = [str(key) for key, value in items if value.get("status") == "fail"]
        signals.extend(f"metric:{name}:{key}" for key in (failures or ["overall"]))
    return sorted(signals)


def evaluate_response(trace: dict, call: dict, content: str | None = None) -> dict:
    """同款生产硬门加解读指标；unknown 计数但不当作 fail。"""
    hard: list[str] = []
    metrics, parsed, candidate = {}, None, None
    kind = "invalid"
    normalized = normalize_call(trace, call)
    original = content is None
    text = normalized["content"] if original else content
    context, access = {}, {}
    try:
        context = _context(trace, call)
        tools = _replay_tools(trace, context)
        _, inline = role_context(trace["snapshot"])
        access = {"evidence": sorted(inline | tools.accessed_evidence("generator")),
                  "knowledge": sorted(tools.accessed_knowledge("generator"))}
    except (KeyError, TypeError, ValueError, StopIteration, HarnessError) as exc:
        hard.append("source_context_invalid:" + type(exc).__name__)
        tools = None
    finish = (normalized["response"].get("choices") or [{}])[0].get("finish_reason")
    if original and (call.get("not_sent") or not normalized["response"]):
        hard.append("student_response_missing")
    if original and finish == "length":
        hard.append("truncated_output")
    try:
        parsed = json.loads(text, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        if not isinstance(parsed, dict):
            raise ValueError("响应必须是对象。")
        if len(text.encode("utf-8")) > 256 * 1024:
            hard.append("role_response_too_large")
        parsed = normalize_tool_response(parsed)
    except (TypeError, ValueError, ValidationError):
        hard.append("invalid_json")
        parsed = None
    if isinstance(parsed, dict):
        if "tool_calls" in parsed:
            kind = "tool"
            try:
                turn = ToolTurn.model_validate(parsed)
                for item in turn.tool_calls:
                    if tools is None:
                        break
                    try:
                        if item.name == "search_knowledge":
                            # 只验证决策，不生成或伪造检索结果；空返回不会增加条目数。
                            schema = next(
                                entry for entry in context["tools"] if entry["name"] == item.name
                            )
                            properties = schema["parameters"]["properties"]
                            if (type(item.arguments.get("top_k")) is not int
                                    or not 1 <= item.arguments["top_k"] <= properties["top_k"]["maximum"]
                                    or not isinstance(item.arguments.get("query"), str)
                                    or not 1 <= len(item.arguments["query"].strip()) <= properties["query"]["maxLength"]):
                                raise HarnessError("retrieval_policy_exceeded")
                        tools.execute("generator", item.name, item.arguments)
                    except StopIteration:
                        hard.append("tool:retrieval_policy_exceeded")
                    except HarnessError as exc:
                        hard.append("tool:" + exc.code)
            except ValidationError as exc:
                hard.extend("tool_schema:" + error["type"] for error in
                            exc.errors(include_input=False, include_url=False))
        else:
            kind = "final"
            try:
                candidate = resolve_quotes(parsed)
                draft = CandidateReport.model_validate(candidate)
                problems = [
                    *candidate_round_problems(
                        draft, version=context["candidate_version"],
                        open_issue_ids=[item["issue_id"] for item in context.get("unresolved_issues", [])],
                    ),
                    *candidate_contract_problems(
                        draft, [item["obligation_id"] for item in trace["snapshot"]["fact_obligations"]],
                    ),
                    *generator_access_problems(
                        draft, set(access.get("evidence", [])), set(access.get("knowledge", [])),
                    ),
                ]
                hard.extend(f"{code}:{json.dumps(list(loc), ensure_ascii=False)}"
                            for code, loc, _ in problems)
                rule_ids = {item["id"] for item in trace["knowledge_registry"]
                            if item.get("source_kind") == "rule_excerpt"}
                for index, claim in enumerate(draft.claims):
                    if rule_ids.intersection(claim.knowledge_refs):
                        hard.append(f"rule_excerpt_as_evidence:claims:{index}")
                for index, issue in enumerate(draft.issue_responses):
                    if rule_ids.intersection(issue.source_refs):
                        hard.append(f"rule_excerpt_as_evidence:issue_responses:{index}")
                metrics = evaluate_candidate(candidate, {**trace["case"], "snapshot": trace["snapshot"]})
            except HarnessError:
                hard.append("quote_invalid")
            except ValidationError as exc:
                hard.extend("candidate_schema:" + error["type"] + ":" + json.dumps(list(error["loc"]))
                            for error in exc.errors(include_input=False, include_url=False))
            except (KeyError, TypeError, ValueError):
                hard.append("candidate_context_invalid")
    failures = sorted(set(hard + _metric_fail_signals(metrics)))
    unknown = {name: check.get("unknown", 0) for name, check in metrics.get("checks", {}).items()
               if check.get("status") == "unknown" or check.get("unknown", 0)}
    return {"kind": kind, "parsed": parsed, "candidate": candidate, "access": access,
            "hard_problems": sorted(set(hard)), "metrics": metrics, "fail_signals": failures,
            "unknown_metrics": unknown, "training_eligible": not failures and kind != "invalid"}


def revision_instruction(validation: dict) -> str:
    fields = "\n".join(f"{index + 1}. {spec['name']}（{spec['kind']}）：{spec['handling']}"
                       for index, spec in enumerate(FIELD_SPECS))
    return (
        "只修复下面明确失败项，只改必要处，保留学生的结构、措辞、字段顺序与候选版本号；"
        "不要新增材料外事实。返回完整合法JSON对象，不要解释或代码围栏。\n"
        "失败项：" + json.dumps(validation["fail_signals"], ensure_ascii=False) + "\n"
        "逐项指标：" + json.dumps(validation["metrics"], ensure_ascii=False) + "\n"
        "37字段解读清单：\n" + fields + "\n"
        "空字段不得编造；评价字段须归因于“事故信息记载”；显式冲突须披露且保留待核实。"
        "evidence_refs必须指向真正含该事实的字段；全部37条义务的treatment/resolution须与"
        "已用、空缺、冲突状态一致。每个quote须是正文中唯一的连续原文。"
        "不得引用规则摘录或学生本轮未读来源。工具回合仅使用原始上下文提供的工具、"
        "参数、分页游标和剩余检索额度；不得伪造工具结果。"
    )


def _diff(before: str, after: str) -> dict:
    matcher = difflib.SequenceMatcher(a=before, b=after, autojunk=False)
    changes = [item for item in matcher.get_opcodes() if item[0] != "equal"]
    return {"similarity": round(matcher.ratio(), 6), "changed_ranges": len(changes),
            "removed_chars": sum(a1 - a0 for _, a0, a1, _, _ in changes),
            "added_chars": sum(b1 - b0 for _, _, _, b0, b1 in changes),
            "before_sha256": canonical_digest(before), "after_sha256": canonical_digest(after)}


async def revise_call(trace: dict, call: dict, backend=None, *, max_revisions: int = 2,
                      retries: int = 0, retry_delay: float = 1.0) -> dict:
    """最多两次内容修订；仅明确未发送/拒绝的临时错误可重试。"""
    if not 0 <= max_revisions <= 2 or not 0 <= retries <= 2:
        raise ValueError("修订及额外重试次数必须在0至2之间。")
    normalized = normalize_call(trace, call)
    validation = evaluate_response(trace, call)
    record = {
        "schema_version": 2, "case_id": trace.get("case_id"),
        "source_call_sha256": call_hash(trace, call),
        "payload": deepcopy(normalized["payload"]), "reasoning_content": normalized["reasoning"],
        "student_content": normalized["content"], "synthetic": True,
        "call_index": call.get("call_index"), "split": trace.get("case", {}).get("split"),
        "original_validation": validation, "teacher_calls": [],
        "teacher_backend": type(backend).__name__ if backend is not None else None,
        "teacher_protocol_fixture": bool(getattr(backend, "deterministic", False)) if backend is not None else False,
        "derivation": "student_original", "training_eligible": False,
    }
    if any(item.startswith("source_context_invalid") or item == "student_response_missing"
           for item in validation["hard_problems"]):
        return {**record, "validation": validation, "discard_reason": "student_context_invalid"}
    current = normalized["content"]
    if validation["training_eligible"]:
        return {**record, "chosen_content": current, "validation": validation,
                "training_eligible": True, "diff": _diff(current, current)}
    if backend is None:
        return {**record, "validation": validation, "discard_reason": "revision_required"}
    if not getattr(backend, "deterministic", False) and backend.effort_for("reviser") != "max":
        raise ValueError("教师修订器必须使用max思考档位。")
    for attempt in range(max_revisions):
        messages = deepcopy(normalized["messages"]) + [
            {"role": "assistant", "content": normalized["content"],
             "reasoning_content": normalized["reasoning"]},
            {"role": "user", "content": revision_instruction(validation)
             + ("\n上次修订仍不合格：" + current if attempt else "")},
        ]
        payload = {"model": backend.model_for("reviser"), "messages": messages,
                   "reasoning": {"effort": "max", "exclude": False},
                   "response_format": {"type": "json_object"}}
        response = None
        for retry in range(retries + 1):
            try:
                response = await backend.chat("reviser", payload)
                break
            except LLMError as exc:
                record["teacher_calls"].append({
                    "request_sha256": canonical_digest(payload), "error_code": exc.code,
                    "billing_state": exc.billing_state, "http_status": exc.http_status,
                })
                if (retry >= retries or exc.billing_state not in {"not_sent", "rejected"}
                        or exc.http_status not in {429, 503, 529}):
                    return {**record, "validation": validation,
                            "discard_reason": "teacher_error:" + exc.code}
                await asyncio.sleep(retry_delay * 2 ** retry)
        if response is None:
            break
        teacher = {"payload": deepcopy(payload), "request_sha256": canonical_digest(payload),
                   "response_sha256": canonical_digest(response),
                   "ledger_id": response.get("live_ledger", {}).get("call_id"),
                   "usage": response.get("usage"), "response": response}
        record["teacher_calls"].append(teacher)
        try:
            choice = response["choices"][0]
            current = choice["message"]["content"]
            if not isinstance(current, str) or choice.get("finish_reason") == "length":
                raise ValueError("修订响应空缺或截断。")
        except (KeyError, IndexError, TypeError, ValueError):
            validation = {**validation, "training_eligible": False,
                          "fail_signals": ["teacher_invalid_or_truncated_response"]}
            continue
        validation = evaluate_response(trace, call, current)
        if validation["training_eligible"]:
            return {**record, "derivation": "minimax_revision", "chosen_content": current,
                    "rejected_content": normalized["content"], "validation": validation,
                    "training_eligible": True, "diff": _diff(normalized["content"], current)}
    return {**record, "validation": validation, "discard_reason": "revision_exhausted"}


def run_revisions(traces_dir, out_dir, backend, workers=1, *, retries=0) -> dict:
    """同步入口；backend可为实例或工厂，工厂可使最多4个客户端独立并发。"""
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("MiniMax并发必须为1至4。")
    out_dir = workspace_path(out_dir, output=True)
    traces, errors = load_traces(traces_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(path, trace, call) for path, trace in traces for call in trace["calls"]
            if call.get("role") == "generator" and not call.get("not_sent")]

    async def execute():
        semaphore = asyncio.Semaphore(workers)

        async def one(path, trace, call):
            digest = call_hash(trace, call)
            target = workspace_path(out_dir / (digest + ".json"), output=True)
            async with semaphore:
                if target.exists():
                    cached = json.loads(target.read_text(encoding="utf-8"))
                    if (cached.get("source_call_sha256") != digest
                            or cached.get("trace_sha256") != canonical_digest(trace)):
                        raise ValueError("已存在修订记录与学生输入不匹配，禁止覆盖。")
                    if cached.get("training_eligible") is True:
                        from .build_post import _checked_revision

                        verified, reason = _checked_revision(cached, trace, call)
                        if verified is None:
                            raise ValueError("已存在修订记录重验失败：" + str(reason))
                    return cached
                selected = backend() if callable(backend) else backend
                try:
                    result = await revise_call(trace, call, selected, retries=retries)
                except Exception as exc:
                    result = {
                        "source_call_sha256": digest, "training_eligible": False,
                        "discard_reason": "revision_exception:" + type(exc).__name__,
                        "synthetic": True,
                    }
                result["trace_path"] = str(path.relative_to(ROOT))
                result["trace_sha256"] = canonical_digest(trace)
                target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                return result

        return await asyncio.gather(*(one(*job) for job in jobs))

    results = asyncio.run(execute())
    summary = {"records": len(results), "accepted": sum(r["training_eligible"] for r in results),
               "discarded": sum(not r["training_eligible"] for r in results),
               "load_errors": errors, "workers": workers}
    workspace_path(out_dir / "summary.json", output=True).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
