"""仅用于协议验证的合成夹具，不是教师或正式训练数据。"""

from __future__ import annotations

from copy import deepcopy

from app.report_harness.contracts import canonical_digest
from app.report_harness.evidence import freeze_snapshot


class SyntheticRetrieval:
    mode = "synthetic_fixture"
    manifest_digest = canonical_digest({"source": "本地虚构知识夹具"})
    manifest = {"mode": mode, "source": "合成；未经人工核实", "network_calls": 0}

    def __init__(self, *, text_chars=80):
        self.knowledge_source = []
        for index in range(5):
            text = f"虚构协议测试条款{index}：" + "测试材料。" * (text_chars // 5)
            self.knowledge_source.append({
                "id": f"fixture#chunk#{index}", "document_id": "fixture",
                "version": "synthetic-v1", "text": text,
                "digest": canonical_digest(text),
                "manifest_digest": self.manifest_digest,
                "source_kind": "source_chunk" if index != 2 else "rule_excerpt",
            })

    @staticmethod
    def initial_query(accident):
        return accident["事故标题"]

    def initial(self, query, top_k):
        return deepcopy(self.knowledge_source[:top_k])

    def search(self, query, top_k):
        return deepcopy(self.knowledge_source[:top_k])


def wire_candidate(context):
    """覆盖合成字段的机械候选，仅证明协议可跑通，不证明写作质量。"""
    accident = context["snapshot"]["accident_data"]
    lines, claims, resolutions = [], [], []
    for index, obligation in enumerate(context["snapshot"]["fact_obligations"]):
        source = obligation["source_refs"][0]
        pointer = source.removeprefix("accident:/")
        key = pointer.replace("~1", "/").replace("~0", "~")
        value = accident.get(key)
        if value not in ("", None):
            line = f"{key}：{value}。"
            lines.append(line)
            claims.append({
                "claim_id": f"fact-{index}", "type": "fact", "quote": line,
                "evidence_refs": [source], "knowledge_refs": [],
            })
        resolutions.append({
            "obligation_id": obligation["obligation_id"],
            "treatment": "covered" if value not in ("", None) else "uncertain",
            "resolution": "在正文保留输入记载。" if value not in ("", None) else "输入为空，不能补造。",
        })
    return {
        "version": context["candidate_version"], "report_markdown": "\n".join(lines),
        "claims": claims, "obligation_resolutions": resolutions,
        "issue_responses": [{
            "issue_id": issue["issue_id"], "action": "revised",
            "explanation": "协议夹具已按反馈修订。", "source_refs": issue["source_refs"],
        } for issue in context["unresolved_issues"]],
    }


def passing_review(context):
    refs = [item["source_refs"][0] for item in context["snapshot"]["fact_obligations"]]
    return {
        "candidate_digest": context["candidate_digest"],
        "snapshot_digest": context["snapshot_digest"],
        "coverage_checks": [{
            "obligation_id": item["obligation_id"], "passed": True,
            "claim_ids": [], "evidence_refs": item["source_refs"], "knowledge_refs": [],
            "conclusion": "协议夹具已核对该输入字段；仅验证检查结构。",
        } for item in context["snapshot"]["fact_obligations"]],
        "completed_checks": [{
            "category": category, "passed": True, "evidence_refs": refs,
            "knowledge_refs": [], "conclusion": "协议夹具核对输入，不代表独立语义审查。",
        } for category in ("facts", "coverage", "reasoning", "citations", "conciseness")],
        "issues": [{
            **issue, "status": "resolved", "explanation": "夹具已核对修订处。",
        } for issue in context["unresolved_issues"]],
    }


def passing_script(role, payload, index):
    import json

    context = json.loads(payload["messages"][-1]["content"])
    return wire_candidate(context) if role == "generator" else passing_review(context)
