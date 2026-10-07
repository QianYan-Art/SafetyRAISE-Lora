"""第1轮本地运行入口；真实模型发送需要显式开关，本轮没有发送。"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .backends import LocalOpenAIBackend, MiniMaxBackend, RoleBackends, ScriptedBackend
from .env import LiveEnvironment, ROOT
from .retrieval import build_retrieval
from .scripted import SyntheticRetrieval, passing_script
from .window import TrainingWindow


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--backend", choices=("scripted", "minimax", "local"), default="scripted")
    parser.add_argument("--reviewer", choices=("scripted", "minimax", "local"))
    parser.add_argument("--retrieval", choices=("synthetic", "fallback", "sparse_half", "mixed"),
                        default="synthetic")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--output-reserve", type=int, default=8192)
    parser.add_argument("--stable-limit", type=int, default=24000)
    args = parser.parse_args(argv)
    source = args.case.resolve()
    if not source.is_relative_to((ROOT / "harness/cases").resolve()):
        parser.error("只允许读取 harness/cases 中已登记的合成案件。")
    case = json.loads(source.read_text(encoding="utf-8"))
    if case.get("kind") != "synthetic":
        parser.error("本轮只允许合成案件。")
    retrieval = (SyntheticRetrieval() if args.retrieval == "synthetic"
                 else build_retrieval(args.retrieval, seed=args.seed, case_id=case["case_id"]))
    if args.backend == "scripted":
        backend = ScriptedBackend(passing_script)
    else:
        backend_cls = MiniMaxBackend if args.backend == "minimax" else LocalOpenAIBackend
        backend = backend_cls(
            allow_network=args.allow_network,
            kb_class="synthetic" if args.retrieval == "synthetic" else "public_statutes",
        )
    if args.reviewer is not None:
        reviewer = (ScriptedBackend(passing_script) if args.reviewer == "scripted" else
                    (MiniMaxBackend if args.reviewer == "minimax" else LocalOpenAIBackend)(
                        allow_network=args.allow_network, roles=("reviewer",),
                        kb_class="synthetic" if args.retrieval == "synthetic" else "public_statutes",
                    ))
        backend = RoleBackends({"generator": backend, "reviewer": reviewer})
    guard = TrainingWindow(stable_limit=args.stable_limit, output_reserve=args.output_reserve)
    trace = asyncio.run(LiveEnvironment(backend, retrieval, payload_guard=guard).run(
        case, trace_path=args.trace,
    ))
    print(json.dumps({
        "case_id": trace["case_id"], "status": trace["status"], "reason": trace["reason"],
        "trace_path": str(args.trace), "calls": len(trace["calls"]),
        "retrieval_mode": trace["retrieval"]["mode"],
        "training_window_eligible": trace["training_window_eligible"],
        "training_eligible": trace["training_eligible"],
        "source": "协议夹具，不是训练样本" if args.backend == "scripted" else "模型真实响应",
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
