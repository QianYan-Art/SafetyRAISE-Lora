"""注入一致性核对:评测台复刻的"报告模型真实输入"是否与生产源码逐字一致。

只读使用生产仓库(不写字节码、不改文件、不联网、不发模型请求):
  - 用生产仓库自己的 Python 环境运行(它装了 pydantic 等),把生产 backend 与评测台 harness 同时放进 sys.path;
  - 对同一批合成案件,比较 生产的 render_report_prompt / 执行提示 / 工具定义 / 请求载荷 / 输出清洗
    与 评测台 sr_eval 的对应实现;
  - 检查冻结的提示资产哈希是否仍等于生产当前文件;检查检索配置常量。
结果写 harness/reports/conformance-<日期>.json(无密钥,不含案件正文)。

运行(在 <repo>):
  $env:PYTHONDONTWRITEBYTECODE="1"
  <SafetyRAISE-system-repo>\\.venv\\Scripts\\python.exe -B harness\\tools\\check_conformance.py
"""
from __future__ import annotations

import glob
import hashlib
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
PROD = Path(r"<SafetyRAISE-system-repo>")
HARNESS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROD / "backend"))
sys.path.insert(0, str(HARNESS))

from sr_eval import prompt as OURS  # noqa: E402
from sr_eval.llm import LLMClient, load_models  # noqa: E402

results: list[dict] = []


def record(name: str, ok: bool, **detail) -> None:
    results.append({"check": name, "ok": ok, **detail})
    print(("PASS " if ok else "FAIL ") + name, {k: v for k, v in detail.items() if k != "diffs"})


# ---- 0. 冻结资产 vs 生产当前文件 ----
assets = json.loads((HARNESS / "assets" / "ASSETS.json").read_text(encoding="utf-8"))
for name, meta in assets["files"].items():
    live_path = Path(meta["source"]) if "source" in meta else PROD / "backend" / "config" / name
    live = hashlib.sha256(live_path.read_bytes()).hexdigest()
    record(f"asset_hash:{name}", live == meta["sha256"], frozen=meta["sha256"][:12], live=live[:12])

# ---- 1. 渲染函数 ----
from app.workflow.prompt_rendering import render_report_prompt as prod_render  # noqa: E402

template = (PROD / "backend" / "config" / "report_prompt.md").read_text(encoding="utf-8")
cases = [json.loads(Path(p).read_text(encoding="utf-8")) for p in sorted(glob.glob(str(HARNESS / "cases" / "*" / "*.json")))]
rnd = random.Random(7)
diffs = 0
for c in cases:
    snippets = [{"id": f"s{i}", "title": f"片段{i}", "content": f"内容{i}" * rnd.randint(1, 30), "source": "src", "score": 0.5} for i in range(rnd.randint(3, 9))]
    n_init = min(6, len(snippets))
    initial, extra = snippets[:n_init], snippets[n_init:]
    rounds = []
    if extra:
        rounds = [{"round": 1, "query": "查询", "reason": "原因", "requested_top_k": 3, "returned_count": len(extra), "snippets": extra}]
    a = prod_render(template, c["accident_data"], c["guidance"], initial, extra, rounds)
    b = OURS.render_report_prompt(template, c["accident_data"], c["guidance"], initial, extra, rounds)
    diffs += a != b
record("render_report_prompt", diffs == 0, cases=len(cases), mismatches=diffs)

# ---- 2. 执行提示 / 工具定义(nodes.py,用假 self 调用未绑定方法) ----
try:
    from app.workflow import nodes as prod_nodes  # noqa: E402

    cls = next(v for v in vars(prod_nodes).values() if isinstance(v, type) and hasattr(v, "_build_report_execution_prompt"))
    fake = SimpleNamespace(_agentic_rag_enabled=lambda: True,
                           settings=SimpleNamespace(retrieval=SimpleNamespace(agentic=SimpleNamespace(top_k_per_round=3))))
    for force in (False, True):
        record(f"exec_prompt[force_finalize={force}]",
               cls._build_report_execution_prompt(fake, force) == OURS.build_exec_prompt(force, "production"))
    record("retrieve_tool_schema", cls._build_report_retrieval_tools(fake) == OURS.retrieve_tool_schema())
except Exception as exc:  # noqa: BLE001
    record("nodes_import", False, error=f"{type(exc).__name__}: {str(exc)[:160]}")

# ---- 3. 请求载荷(openai_report._build_payload) ----
try:
    from app.providers.llm import openai_report as prod_provider  # noqa: E402

    pcls = next(v for v in vars(prod_provider).values() if isinstance(v, type) and hasattr(v, "_build_payload"))
    endpoint = SimpleNamespace(name="openrouter_primary", verbosity=None, extra_body=None, reasoning_effort=None,
                               reasoning=SimpleNamespace(model_dump=lambda exclude_none=True: {"effort": "high"}))
    fake_p = SimpleNamespace(_lmstudio_by_endpoint={"openrouter_primary": SimpleNamespace(enabled=False, ttl_seconds=None)})
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    payload = pcls._build_payload(fake_p, endpoint, "tencent/hy4-preview", "S", "U", OURS.retrieve_tool_schema())
    client = LLMClient(models=load_models())
    ours = client._body(client.models["hy4"], msgs, OURS.retrieve_tool_schema(), 40000, None, None)
    only_prod = {k: payload[k] for k in payload if k not in ours}
    only_ours = {k: ours[k] for k in ours if k not in payload and k != "messages"}
    differ = {k: [payload[k], ours[k]] for k in payload if k in ours and payload[k] != ours[k]}
    record("request_payload", not only_prod and not differ, only_in_production=list(only_prod), only_in_harness=list(only_ours),
           differing_values=list(differ), note="harness 额外带 stream=false/max_tokens(预算预留所需);生产不传 max_tokens/temperature")
except Exception as exc:  # noqa: BLE001
    record("provider_import", False, error=f"{type(exc).__name__}: {str(exc)[:160]}")

# ---- 4. 输出清洗(生产 sanitize_markdown_output vs 评测台简化复刻) ----
from app.core.model_output import sanitize_markdown_output as prod_sanitize  # noqa: E402

raws = []
for p in glob.glob(str(HARNESS / "runs" / "*" / "traces" / "*.json")):
    t = json.loads(Path(p).read_text(encoding="utf-8"))
    if t.get("final_raw"):
        raws.append(t["final_raw"])
raws = raws[:300]
bad = [i for i, r in enumerate(raws) if prod_sanitize(r) != OURS.sanitize_markdown_output(r)]
record("sanitize_markdown_output", not bad, samples=len(raws), mismatches=len(bad), note="不一致时以生产为准,评测台需移植 _repair_markdown_structure")

# ---- 5. 检索配置常量 ----
try:
    import yaml  # noqa: E402

    cfg = yaml.safe_load((PROD / "backend" / "config" / "workflow.yaml").read_text(encoding="utf-8"))
    ag = cfg["retrieval"]["agentic"]
    expect = {"max_rounds": OURS.AGENTIC_MAX_ROUNDS, "top_k_per_round": OURS.AGENTIC_TOP_K_PER_ROUND,
              "max_total_snippets": OURS.AGENTIC_MAX_TOTAL_SNIPPETS, "max_query_chars": OURS.AGENTIC_MAX_QUERY_CHARS}
    mism = {k: [ag.get(k), v] for k, v in expect.items() if ag.get(k) != v}
    record("agentic_retrieval_limits", not mism, mismatches=mism, initial_top_k_prod=cfg["retrieval"].get("top_k"), initial_top_k_harness=OURS.INITIAL_TOP_K)
except Exception as exc:  # noqa: BLE001
    record("workflow_yaml", False, error=f"{type(exc).__name__}: {str(exc)[:160]}")

out = HARNESS / "reports"
out.mkdir(exist_ok=True)
(out / "conformance-latest.json").write_text(json.dumps({"production_commit_note": "见 assets/ASSETS.json", "results": results}, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n{sum(r['ok'] for r in results)}/{len(results)} 项一致")
sys.exit(0 if all(r["ok"] for r in results) else 1)
