"""批量在线上真实环境模拟器里采样(学生或 MiniMax 作生成者)。每个案件一个 trace,已存在则跳过(可断点续跑)。
运行环境 .venv-live;PYTHONPATH=harness;harness\\vendor\\safetyraise_7200e30
用法: live_batch.py --tag T --cases-file cases.txt [--backend local|minimax] [--reviewer scripted|minimax]
        [--retrieval mixed|fallback|sparse_half] [--workers 4] [--stable-limit 28000] [--output-reserve 4000] [--max-tokens 12000]
思考档位:本地学生用环境变量 SR_LIVE_EFFORT(xhigh|medium|low)。结果目录 harness/runs/live_<tag>/;结束输出 EVENT 行。
"""
import argparse, asyncio, concurrent.futures as cf, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "harness"), str(ROOT / "harness" / "vendor" / "safetyraise_7200e30")]
sys.dont_write_bytecode = True

from sr_eval.live.backends import LocalOpenAIBackend, MiniMaxBackend, RoleBackends, ScriptedBackend  # noqa: E402
from sr_eval.live.env import LiveEnvironment  # noqa: E402
from sr_eval.live.retrieval import build_retrieval  # noqa: E402
from sr_eval.live.scripted import passing_script  # noqa: E402
from sr_eval.live.window import TrainingWindow  # noqa: E402
from app.report_harness.errors import HarnessError  # noqa: E402


class FirstCallWindow(TrainingWindow):
    """只放行第一次请求;第二次起一律拦下,trace 保留第 1 次调用的完整响应。"""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._n = 0

    def __call__(self, payload):
        self._n += 1
        if self._n > 1:
            raise HarnessError("first_call_only_stop")
        return super().__call__(payload)

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--cases-file", required=True)
ap.add_argument("--backend", default="local", choices=("local", "minimax"))
ap.add_argument("--reviewer", default="scripted", choices=("scripted", "minimax"))
ap.add_argument("--retrieval", default="mixed")
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--stable-limit", type=int, default=28000)
ap.add_argument("--output-reserve", type=int, default=4000)
ap.add_argument("--max-tokens", type=int, default=12000)
ap.add_argument("--first-call-only", action="store_true", help="只做第一次模型调用(后续调用被拦,不浪费时间);训练行以第 1 回合为主")
args = ap.parse_args()
out = ROOT / "harness" / "runs" / f"live_{args.tag}"
out.mkdir(parents=True, exist_ok=True)
cases = [l.strip() for l in open(args.cases_file, encoding="utf-8") if l.strip()]


def one(case_path: str) -> dict:
    p = Path(case_path)
    if not p.is_absolute():
        p = ROOT / p
    case = json.loads(p.read_text(encoding="utf-8"))
    trace_path = out / f"{case['case_id']}.json"
    if trace_path.exists():
        return {"case_id": case["case_id"], "skipped": True}
    t0 = time.time()
    kb = "public_statutes"
    gen_cls = LocalOpenAIBackend if args.backend == "local" else MiniMaxBackend
    gen = gen_cls(allow_network=True, kb_class=kb, max_tokens=args.max_tokens, timeout=7200 if args.backend == "local" else 1800)
    if args.reviewer == "scripted":
        rev = ScriptedBackend(passing_script)
    else:
        rev = MiniMaxBackend(allow_network=True, roles=("reviewer",), kb_class=kb)
    backend = RoleBackends({"generator": gen, "reviewer": rev})
    retrieval = build_retrieval(args.retrieval, seed=args.seed, case_id=case["case_id"])
    guard = (FirstCallWindow if args.first_call_only else TrainingWindow)(stable_limit=args.stable_limit, output_reserve=args.output_reserve)
    try:
        trace = asyncio.run(LiveEnvironment(backend, retrieval, payload_guard=guard).run(case, trace_path=trace_path))
        return {"case_id": case["case_id"], "status": trace["status"], "reason": trace["reason"], "calls": len(trace["calls"]),
                "mode": trace["retrieval"]["mode"], "sec": round(time.time() - t0)}
    except Exception as e:  # noqa: BLE001
        return {"case_id": case["case_id"], "error": f"{type(e).__name__}: {str(e)[:160]}", "sec": round(time.time() - t0)}


with cf.ThreadPoolExecutor(args.workers) as ex:
    for r in ex.map(one, cases):
        print(json.dumps(r, ensure_ascii=False), flush=True)
print("EVENT: 批量采样完成", args.tag, len(cases), "案")
