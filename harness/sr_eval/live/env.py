"""线上冻结角色、工具和合同的内存装配；只接受合成案。"""

from __future__ import annotations

import asyncio
import json
import os
import time
from copy import deepcopy
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import httpx
from pydantic import ValidationError

from app.report_harness.business_roles import BusinessTransportRoles
from app.report_harness.business_workflow import BusinessWorkflow
from app.report_harness.contracts import (
    canonical_digest, candidate_contract_problems, candidate_round_problems,
    enforce_semantic_severity, enforce_source_authority,
    validate_publication, validate_review_structure,
)
from app.report_harness.controlled_tools import ControlledTools
from app.report_harness.errors import HarnessError
from app.report_harness.evidence import freeze_snapshot
from app.report_harness.review_ledger import IssueLedger
from app.report_harness.role_loop import ResponseRejected, RoleLoop, role_context
from app.report_harness.runtime_profiles import (
    ConfiguredAttemptClient, ModelCapacity, capacity_from_metadata, openrouter_price_filter,
)
from app.report_harness.source_access import generator_access_problems, reviewer_access_problems
from app.report_harness.transport_roles import RoleModel
from app.schemas.report_run import BudgetPolicy, CandidateReport, ReviewResult

from sr_eval.llm import LLMError
from .backends import ChatBackend
from .window import TrainingWindow

ROOT = Path(__file__).resolve().parents[3]
VENDOR = ROOT / "harness/vendor/safetyraise_7200e30"
_MANIFEST = json.loads((VENDOR / "deployed/runtime-manifest.json").read_text(encoding="utf-8"))


def production_workflow() -> BusinessWorkflow:
    return BusinessWorkflow(
        (VENDOR / "config/guidance_prompt.md").read_text(encoding="utf-8"),
        (VENDOR / "config/report_prompt.md").read_text(encoding="utf-8"),
    )


def production_budget() -> BudgetPolicy:
    return BudgetPolicy.model_validate(_MANIFEST["budget"])


class MemoryBudget:
    """不冒充持久账本；记录单次运行请求、真实 usage 和主动耗时。"""

    def __init__(self, policy, *, deterministic):
        self.policy, self.deterministic = policy, deterministic
        self.requests = self.tokens = self.retrieval_requests = 0
        self.started = time.monotonic()
        self.virtual_seconds = 0.0

    def elapsed(self):
        return self.virtual_seconds if self.deterministic else time.monotonic() - self.started

    def active(self):
        if self.elapsed() >= self.policy.max_active_seconds:
            raise HarnessError("budget_exhausted")

    def reserve_request(self):
        self.active()
        if self.requests >= self.policy.max_physical_requests:
            raise HarnessError("physical_request_budget_exhausted")
        if self.tokens >= self.policy.max_total_tokens:
            raise HarnessError("token_budget_exhausted")
        self.requests += 1

    def settle(self, response):
        usage = response.get("usage")
        total = usage.get("total_tokens") if isinstance(usage, dict) else None
        if type(total) is not int or total < 0:
            raise HarnessError("usage_unknown")
        self.tokens += total
        if self.tokens > self.policy.max_total_tokens:
            raise HarnessError("usage_exceeded")

    def retrieve(self, callback, query, top_k):
        self.active()
        if self.retrieval_requests >= self.policy.max_retrieval_requests:
            raise HarnessError("retrieval_request_budget_exhausted")
        self.retrieval_requests += 1
        return callback(query, top_k)

    def view(self, loop=None):
        return {
            "physical_requests": self.requests, "total_tokens": self.tokens,
            "retrieval_requests": self.retrieval_requests,
            "model_turns": loop.model_turns if loop else 0,
            "tool_calls": loop.tool_calls if loop else 0,
            "active_seconds": round(self.elapsed(), 6),
            "policy": self.policy.model_dump(mode="json"),
            "reservation_kind": "内存计数及响应用量；非线上持久容量预留",
        }


class BackendTransport:
    """HTTP mock 仅桥接后端；ConfiguredAttemptClient 的真实净化原样执行。"""

    output_limit = 64000

    def __init__(self, backend, trace, budget, *, payload_guard=None):
        self.backend, self.trace, self.budget = backend, trace, budget
        self.payload_guard = payload_guard
        roles = ("generator", "reviewer")
        capacities, options = {}, {}
        for role in roles:
            metadata = _MANIFEST["metadata"][role]
            if backend.model_for(role) == metadata["id"]:
                capacities[role] = capacity_from_metadata(
                    metadata, model=metadata["id"], effort=backend.effort_for(role),
                )
                options[role] = {"provider": openrouter_price_filter(metadata, capacities[role])}
            else:
                role_limit = getattr(backend, "output_limit_for", None)
                output_tokens = role_limit(role) if role_limit is not None else getattr(backend, "max_tokens", 64000)
                capacities[role] = ModelCapacity(
                    backend.model_for(role), 32768, output_tokens,
                    canonical_digest({"kind": "训练窗口；非服务容量证明", "role": role}),
                    backend.effort_for(role),
                )
                options[role] = {}
        binding = getattr(backend, "bind_before_attempt", None)
        self.physical_bound = binding is not None
        if binding is not None:
            binding(budget.reserve_request)
        trace["runtime_profiles"] = {
            role: {
                "model": item.model, "output_tokens": item.output_tokens,
                "context_tokens": item.context_tokens, "effort": item.effort,
                "source": "冻结线上容量" if item.model == _MANIFEST["metadata"][role]["id"]
                else "当前训练窗口与既有后端输出配置，服务容量未实测",
            } for role, item in capacities.items()
        }
        self.client = ConfiguredAttemptClient(
            {role: f"http://live.invalid/{role}" for role in roles},
            {}, capacities, request_options=options,
        )
        self.original_client = self.client._client
        self.client._client = httpx.AsyncClient(
            transport=httpx.MockTransport(self._dispatch), trust_env=False,
        )

    async def _dispatch(self, request):
        role = request.url.path.lstrip("/")
        payload = json.loads(request.content)
        index = len(self.trace["calls"])
        call = {
            "call_index": index, "role": role, "payload": deepcopy(payload),
            "request_sha256": canonical_digest(payload),
            "context_digest": canonical_digest(json.loads(payload["messages"][-1]["content"])),
            "payload_summary": {
                "message_roles": [m["role"] for m in payload["messages"]],
                "message_chars": [len(m["content"]) for m in payload["messages"]],
                "response_format": payload.get("response_format"),
                "reasoning": payload.get("reasoning"),
            },
        }
        self.trace["calls"].append(call)
        if self.payload_guard is not None:
            try:
                call["window"] = self.payload_guard(payload)
            except HarnessError as exc:
                call["not_sent"] = True
                call["error_code"] = exc.code
                call["window"] = deepcopy(exc.details)
                raise
        if not self.physical_bound:
            self.budget.reserve_request()
        started = time.monotonic()
        try:
            response = await self.backend.chat(role, payload)
        except Exception as exc:
            call["error_type"] = type(exc).__name__
            call["error_code"] = getattr(exc, "code", None)
            call["elapsed_seconds"] = 0.0 if self.backend.deterministic else time.monotonic() - started
            raise
        call["elapsed_seconds"] = (
            0.0 if self.backend.deterministic else round(time.monotonic() - started, 6)
        )
        call["response"] = deepcopy(response)
        choices = response.get("choices") or [{}]
        message = choices[0].get("message") or {}
        call["reasoning_content"] = message.get("reasoning_content") or message.get("reasoning") or ""
        call["content"] = message.get("content") or ""
        call["usage"] = deepcopy(response.get("usage"))
        call["provider_payload"] = deepcopy(getattr(self.backend, "last_payload", None) or payload)
        call["response_sha256"] = canonical_digest(response)
        self.budget.settle(response)
        check_response = getattr(self.payload_guard, "check_response", None)
        if check_response is not None:
            try:
                call["response_window"] = check_response(payload, response)
            except HarnessError as exc:
                call["discard_entire_trace"] = True
                call["error_code"] = exc.code
                call["response_window"] = deepcopy(exc.details)
                raise
        return httpx.Response(200, json=response)

    async def request(self, role, payload, *, output_limit_field=None):
        if output_limit_field is not None:
            raise ValueError("业务角色不得添加线上不存在的输出限制。")
        # 单次调用超时:默认 1800s(与线上一致);本地学生并发采样时队列等待+长提示会超过,用 SR_LIVE_ATTEMPT_TIMEOUT 放宽
        response = await self.client.attempt(role, payload, timeout=int(os.environ.get("SR_LIVE_ATTEMPT_TIMEOUT", "1800")))
        self.trace["calls"][-1]["sanitized_response"] = deepcopy(response)
        return response

    async def close(self):
        await self.client.close()
        await self.original_client.aclose()


class LiveEnvironment:
    def __init__(self, backend: ChatBackend, retrieval, *, budget=None, payload_guard=None):
        self.backend, self.retrieval = backend, retrieval
        self.policy = budget or production_budget()
        self.payload_guard = payload_guard if payload_guard is not None else TrainingWindow()
        self.workflow = production_workflow()
        self.role_prompts = {
            role: (VENDOR / f"config/report_harness/{role}.md").read_text(encoding="utf-8")
            for role in ("generator", "reviewer")
        }

    async def run(self, case: dict, *, trace_path: Path | None = None) -> dict:
        if case.get("kind") != "synthetic":
            raise ValueError("本轮环境仅允许明确标记的合成案。")
        if not isinstance(case.get("guidance"), dict):
            raise ValueError("合成案必须提供已有的 guidance JSON。")
        digest = self.retrieval.manifest_digest
        snapshot = freeze_snapshot(case["accident_data"], [], 0, digest)
        trace = {
            "schema_version": 1, "case_id": case["case_id"], "case": deepcopy(case),
            "snapshot": deepcopy(snapshot), "snapshot_digest": canonical_digest(snapshot),
            "retrieval": {"mode": self.retrieval.mode, "manifest": deepcopy(self.retrieval.manifest),
                          "events": []},
            "calls": [], "tools": [], "rounds": [], "issue_history": [],
            "status": "running", "reason": None,
            "provenance": {"synthetic": True, "human_verified": False,
                           "source": "冻结协议+确定性本地执行"},
        }
        budget = MemoryBudget(self.policy, deterministic=self.backend.deterministic)
        transport = BackendTransport(self.backend, trace, budget, payload_guard=self.payload_guard)
        profiles = {role: RoleModel(self.backend.model_for(role), json_object_mode=True)
                    for role in ("generator", "reviewer", "expert")}
        roles = BusinessTransportRoles(transport, profiles, business_prompts=self.workflow.prompts())

        def retrieve(callback, query, top_k, step):
            result = budget.retrieve(callback, query, top_k)
            retriever = getattr(self.retrieval, "retriever", None)
            metadata = getattr(retriever, "metadata", None)
            trace["retrieval"]["events"].append({
                "step": step, "query": query, "top_k": top_k, "mode": self.retrieval.mode,
                "metadata": deepcopy(metadata),
                "approved_ids": [item["id"] for item in result],
                "approved_projection_digest": canonical_digest(result),
            })
            return result

        tools = ControlledTools(
            snapshot, list(self.retrieval.knowledge_source),
            search=lambda query, top_k: retrieve(self.retrieval.search, query, top_k, "additional"),
            initial_search=lambda query, top_k: retrieve(self.retrieval.initial, query, top_k, "initial"),
            retrieval_policy=self.workflow.retrieval_policy(), compact_registry=True,
        )

        def checkpoint(public, private):
            public = deepcopy(public)
            # 生产内存路径的事件 UUID 随机；轨迹用顺序号，不改模型看到的上下文。
            if "id" in public:
                public["id"] = f"event-{len(trace['tools']):04d}"
            trace["tools"].append({**public, "private": deepcopy(private)})

        loop = RoleLoop(tools, checkpoint, before_call=budget.active,
                        max_tool_calls=self.policy.max_tool_calls)
        ledger = IssueLedger(namespace=str(uuid5(NAMESPACE_URL, canonical_digest(snapshot))))
        try:
            prepared = {"guidance": deepcopy(case["guidance"]), "knowledge": []}
            trace["prepared"] = deepcopy(prepared)
            initial = await loop.initial_retrieval(
                self.retrieval.initial_query(snapshot["accident_data"]),
                self.workflow.initial_top_k,
            )
            trace["initial_retrieval"] = deepcopy(initial)
            if initial.get("truncated"):
                raise HarnessError("initial_knowledge_incomplete", 422)
            previous = feedback = None
            max_version = min(2, self.policy.max_revision_rounds) + 1
            for version in range(1, max_version + 1):
                candidate, review, check_args = await self._candidate_round(
                    trace, snapshot, prepared, roles, tools, loop, ledger, version,
                    previous, feedback, initial["items"],
                )
                blocking = any(i.severity in {"major", "blocker"} and i.status != "resolved"
                               for i in review.issues)
                passed = all(i.passed for i in [*review.coverage_checks, *review.completed_checks])
                if passed and not blocking:
                    budget.active()
                    validate_publication(candidate, review, *check_args,
                                         resolved_issue_ids=frozenset(ledger.resolved_ids()))
                    trace["status"] = "published"
                    trace["report"] = candidate.report_markdown
                    trace["publication"] = {
                        "candidate_digest": canonical_digest(candidate),
                        "review_digest": canonical_digest(review),
                        "snapshot_digest": trace["snapshot_digest"],
                    }
                    break
                if version == max_version:
                    trace.update(status="needs_review", reason="revision_rounds_exhausted")
                    break
                previous, feedback = candidate.model_dump(mode="json"), review.model_dump(mode="json")
        except (ValidationError, ValueError, TimeoutError) as exc:
            trace.update(status="needs_review",
                         reason="budget_exhausted" if isinstance(exc, TimeoutError)
                         else "invalid_review_or_candidate", error_type=type(exc).__name__)
        except HarnessError as exc:
            suspended = {"usage_unknown", "completion_unknown", "authorization_stale",
                         "authorization_required", "token_bound_unverified",
                         "resource_pressure", "resource_probe_failed", "unknown_cost_ack_required"}
            known = {"role_response_too_large", "invalid_role_response", "usage_exceeded",
                     "money_guard_blocked", "training_window_exceeded"}
            if exc.code in suspended:
                trace.update(status="suspended", reason=exc.code)
            elif exc.code.endswith("budget_exhausted") or exc.code in known:
                trace.update(status="needs_review",
                             reason="budget_exhausted" if exc.code.endswith("budget_exhausted")
                             else exc.code)
            else:
                trace.update(status="failed", reason=exc.code)
        except LLMError as exc:
            if exc.billing_state == "unknown":
                trace.update(status="suspended", reason="completion_unknown",
                             backend_error_code=exc.code)
            elif exc.code == "budget_exceeded":
                trace.update(status="needs_review", reason="budget_exhausted",
                             backend_error_code=exc.code)
            else:
                trace.update(status="failed", reason=exc.code)
        except Exception as exc:
            trace.update(status="failed", reason="execution_error", error_type=type(exc).__name__)
        finally:
            trace["budget"] = budget.view(loop)
            trace["issue_history"] = ledger.history()
            trace["knowledge_registry"] = tools.registered_knowledge()
            await roles.close()
        trace["protocol_repairs"] = sum(
            bool(json.loads(c["payload"]["messages"][-1]["content"]).get("protocol_feedback"))
            for c in trace["calls"]
        )
        trace["retrieval_denials"] = sum(
            item.get("code") == "retrieval_policy_exceeded" for item in trace["tools"]
        )
        trace["training_window_eligible"] = (
            trace["status"] == "published"
            and not any(c.get("discard_entire_trace") or c.get("not_sent") for c in trace["calls"])
        )
        trace["training_eligible"] = False
        trace["training_acceptance"] = "尚需解读质量硬门、隐私泄漏门及独立盲评；协议发布不等于训练接受"
        if trace_path is not None:
            destination = Path(trace_path).resolve()
            allowed = [(ROOT / "harness/reports").resolve(), (ROOT / "harness/runs").resolve()]
            if not any(destination.is_relative_to(a) for a in allowed) or destination.is_symlink():
                raise ValueError("轨迹产物仅允许写入 harness/reports 或 harness/runs。")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(trace, ensure_ascii=False, indent=2, allow_nan=False),
                                   encoding="utf-8")
        return trace

    async def _candidate_round(self, trace, snapshot, prepared, roles, tools, loop, ledger,
                               version, previous, feedback, initial_snippets):
        context_snapshot, inline_evidence = role_context(snapshot)
        obligations = snapshot["fact_obligations"]
        tools.reset_access("generator")
        tools.include_knowledge("generator", initial_snippets)
        obligation_ids = [item["obligation_id"] for item in obligations]
        open_issue_ids = [item["issue_id"] for item in ledger.unresolved()]

        def accept_candidate(draft):
            problems = [
                *candidate_round_problems(draft, version=version, open_issue_ids=open_issue_ids),
                *candidate_contract_problems(draft, obligation_ids),
                *generator_access_problems(draft, inline_evidence | tools.accessed_evidence("generator"),
                                           tools.accessed_knowledge("generator")),
            ]
            if problems:
                raise ResponseRejected(problems)

        round_trace = {"version": version}
        trace["rounds"].append(round_trace)
        candidate = CandidateReport.model_validate(await loop.run("generator", roles.generate, {
            "instructions": self.role_prompts["generator"],
            "response_schema": CandidateReport.model_json_schema(),
            "snapshot": deepcopy(context_snapshot), "prepared": deepcopy(prepared),
            "candidate_version": version, "previous_candidate": deepcopy(previous),
            "unresolved_issues": ledger.unresolved(), "review_feedback": deepcopy(feedback),
            "initial_knowledge_snippets": deepcopy(initial_snippets),
        }, accept=accept_candidate))
        candidate_data = candidate.model_dump(mode="json")
        round_trace["candidate"] = deepcopy(candidate_data)
        round_trace["generator_access"] = {
            "evidence": sorted(inline_evidence | tools.accessed_evidence("generator")),
            "knowledge": sorted(tools.accessed_knowledge("generator")),
        }
        ledger.record_responses(candidate.issue_responses)
        tools.reset_access("reviewer")
        required_evidence = {ref for item in obligations for ref in item["source_refs"]}
        candidate_knowledge = {ref for claim in candidate.claims for ref in claim.knowledge_refs}

        def accept_review(_review):
            problems = reviewer_access_problems(
                required_evidence, candidate_knowledge,
                inline_evidence | tools.accessed_evidence("reviewer"), tools.accessed_knowledge("reviewer"),
            )
            if problems:
                raise ResponseRejected(problems)

        review = ReviewResult.model_validate(await loop.run("reviewer", roles.review, {
            "instructions": self.role_prompts["reviewer"],
            "response_schema": ReviewResult.model_json_schema(),
            "snapshot": deepcopy(context_snapshot), "snapshot_digest": canonical_digest(snapshot),
            "candidate": deepcopy(candidate_data), "candidate_digest": canonical_digest(candidate_data),
            "unresolved_issues": ledger.unresolved(),
        }, accept=accept_review))
        round_trace["review_raw"] = review.model_dump(mode="json")
        review = ledger.apply(enforce_semantic_severity(enforce_source_authority(
            candidate, review, self.retrieval.knowledge_source, ledger.history(),
        )))
        round_trace["review_enforced"] = review.model_dump(mode="json")
        round_trace["issues"] = ledger.history()
        round_trace["reviewer_access"] = {
            "evidence": sorted(inline_evidence | tools.accessed_evidence("reviewer")),
            "knowledge": sorted(tools.accessed_knowledge("reviewer")),
        }
        round_trace["resolved_issue_ids"] = sorted(ledger.resolved_ids())
        check_args = (
            canonical_digest(snapshot), set(obligation_ids),
            inline_evidence | tools.accessed_evidence("reviewer"), tools.accessed_knowledge("reviewer"),
        )
        validate_review_structure(candidate, review, *check_args,
                                  resolved_issue_ids=frozenset(ledger.resolved_ids()))
        return candidate, review, check_args
