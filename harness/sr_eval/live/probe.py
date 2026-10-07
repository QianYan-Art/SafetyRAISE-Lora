"""真实角色装配的离线 token 分布探针；模型响应明确为协议夹具。"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from collections import Counter

from .backends import ScriptedBackend
from .env import LiveEnvironment, ROOT
from .retrieval import build_retrieval
from .scripted import SyntheticRetrieval, passing_review, wire_candidate
from .window import TrainingWindow


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--split", choices=("train", "dev", "test"), default="dev")
    args = parser.parse_args(argv)
    if not 1 <= args.n <= 50:
        parser.error("小规模本地探针限制为1至50案。")
    paths = sorted((ROOT / f"harness/cases/{args.split}").glob("*.json"))[:args.n]
    records, traces = [], []
    for mode in ("synthetic", "fallback", "sparse_half"):
        retrieval = SyntheticRetrieval() if mode == "synthetic" else build_retrieval(mode)
        selected_ids = [item["id"] for item in sorted(
            (s for s in retrieval.knowledge_source if s.get("source_kind") != "rule_excerpt"),
            key=lambda s: (len(s["text"]), s["id"]),
        )[:3]]
        for path in paths:
            case = json.loads(path.read_text(encoding="utf-8"))

            def script(role, payload, index):
                ctx = json.loads(payload["messages"][-1]["content"])
                if not ctx["tool_results"]:
                    return {"tool_calls": [{
                        "call_id": "probe-read", "name": "read_knowledge",
                        "arguments": {"chunk_ids": selected_ids},
                    }]}
                return wire_candidate(ctx) if role == "generator" else passing_review(ctx)

            trace = asyncio.run(LiveEnvironment(
                ScriptedBackend(script), retrieval, payload_guard=TrainingWindow(),
            ).run(case))
            windows = []
            for call in trace["calls"]:
                counted = call.get("response_window") or call.get("window") or {}
                windows.append({
                    "role": call["role"], "call_index": call["call_index"],
                    "prompt_tokens": counted.get("prompt_tokens"),
                    "sequence_tokens": counted.get("token_count"),
                    "reserved_sequence_tokens": (call.get("window") or {}).get("reserved_sequence_tokens"),
                    "not_sent": call.get("not_sent", False),
                    "error_code": call.get("error_code"),
                })
            records.append({"case_id": case["case_id"], "mode": mode,
                            "status": trace["status"], "reason": trace["reason"], "calls": windows})
            traces.append(trace)
    counts = Counter((r["mode"], r["reason"] or r["status"]) for r in records)
    tokens = sorted(c["prompt_tokens"] for r in records for c in r["calls"]
                    if type(c["prompt_tokens"]) is int)
    summary = {
        "date": "2026-10-06", "case_count": len(paths), "runs": len(records),
        "template_effort": "xhigh", "source": "真实模板/装配/稀疏KB，模型为合成协议夹具",
        "human_verified": False, "network_calls": 0,
        "observed_prompt_tokens": {
            "count": len(tokens), "min": min(tokens) if tokens else None,
            "median": statistics.median(tokens) if tokens else None,
            "p95": tokens[min(len(tokens) - 1, int(len(tokens) * .95))] if tokens else None,
            "max": max(tokens) if tokens else None,
        },
        "termination_counts": [{"mode": key[0], "reason": key[1], "count": value}
                               for key, value in sorted(counts.items())],
        "window_discard_rate": sum(r["reason"] == "training_window_exceeded" for r in records) / len(records),
        "representativeness": "小规模合成探针；不能外推线上失败率或推理长度分布。",
        "records": records,
    }
    destination = ROOT / "harness/reports/live-token-probe-r1.json"
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    trace_destination = ROOT / "harness/reports/live-r1-traces/probe-with-read.json"
    trace_destination.parent.mkdir(parents=True, exist_ok=True)
    sparse_traces = [t for t in traces if t["retrieval"]["mode"] == "sparse_half"]
    selected = max(sparse_traces, key=lambda t: len(t["calls"]), default=None)
    if selected is not None:
        trace_destination.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
