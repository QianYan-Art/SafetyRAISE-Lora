"""探针(只读、本地):用线上冻结代码为若干合成案渲染"生成者第 1 回合"的真实 payload,写出 messages,供量 token 和零样本试跑。
不调用任何外部模型。运行环境 .venv-live;PYTHONPATH=harness;harness\\vendor\\safetyraise_7200e30
用法: live_prompt_probe.py <split> <起始> <条数> <输出json>
"""
import asyncio, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "harness" / "vendor" / "safetyraise_7200e30"
sys.path.insert(0, str(VENDOR))
sys.dont_write_bytecode = True

from app.providers.retrieval.local_jsonl_retriever import LocalJsonlRetriever  # noqa: E402
from app.report_harness.business_roles import BusinessTransportRoles  # noqa: E402
from app.report_harness.evidence import freeze_snapshot  # noqa: E402
from app.report_harness.prompts import load_role_prompts  # noqa: E402
from app.report_harness.role_loop import role_context, tool_schemas  # noqa: E402
from app.report_harness.transport_roles import RoleModel  # noqa: E402
from app.schemas.report_run import CandidateReport  # noqa: E402

split, start, n, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
KB = Path(r"<SafetyRAISE-knowledge-repo>\kbase\data")
manifest = json.loads((VENDOR / "deployed" / "runtime-manifest.json").read_text(encoding="utf-8"))
kdigest = manifest["knowledge_content_digest"]
prompts = load_role_prompts()
expert_prompt = (VENDOR / "config" / "guidance_prompt.md").read_text(encoding="utf-8")
report_prompt = (VENDOR / "config" / "report_prompt.md").read_text(encoding="utf-8")
retr = LocalJsonlRetriever(manifest_path=KB / "manifest.json", chunks_path=KB / "kbase_chunks.jsonl", rules_path=KB / "liability_rules.jsonl",
                           search_index_path=KB / "search_index.json", min_score=0.2, top_k_chunks=8, top_k_rules=12, max_context_chars=5200,
                           prefer_enhanced_rules=True, enable_search_index=True, watch_manifest_changes=False)
roles = BusinessTransportRoles(None, {"expert": RoleModel("probe"), "generator": RoleModel("probe", json_object_mode=True),
                                      "reviewer": RoleModel("probe", json_object_mode=True)},
                               business_prompts={"expert": expert_prompt, "report": report_prompt})
cases = sorted((ROOT / "harness" / "cases" / split).glob("*.json"))[start:start + n]
res = []
for p in cases:
    case = json.loads(p.read_text(encoding="utf-8"))
    snap = freeze_snapshot(case["accident_data"], [], 0, kdigest)
    ctx_snap, _ = role_context(snap)
    initial = retr.retrieve(case["accident_data"], 3)
    context = {
        "instructions": prompts["generator"], "response_schema": CandidateReport.model_json_schema(), "snapshot": ctx_snap,
        "prepared": {"guidance": case["guidance"], "knowledge": []}, "candidate_version": 1, "previous_candidate": None,
        "unresolved_issues": [], "review_feedback": None, "initial_knowledge_snippets": initial,
        "tool_results": [], "tools": tool_schemas(None),
    }
    _profile, payload = roles._payload("generator", context)
    res.append({"case_id": case["case_id"], "n_obligations": len(snap["fact_obligations"]), "messages": payload["messages"],
                "response_format": payload.get("response_format"), "n_initial": len(initial)})
Path(out).write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
print("写出", len(res), "个案件 →", out, "| 消息数", len(res[0]["messages"]), "| 字符数", [sum(len(m["content"]) for m in r["messages"]) for r in res])
