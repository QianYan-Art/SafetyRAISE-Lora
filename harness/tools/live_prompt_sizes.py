"""只读本地:对一批案件在线上环境模拟器里渲染"生成者第 1 回合"并精确计 token(不调用任何模型)。
用 sparse_half 检索(与采样一致);窗口门只记录提示 token 后拦下。结果写 JSON:{case_id: prompt_tokens}。
运行环境 .venv-live;用法: live_prompt_sizes.py <输出json> <案件文件...>(或 --glob harness/cases/dev/*.json)
"""
import asyncio, glob, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("SR_QWEN_PROFILE_DIR", str(ROOT / "profiles" / "assets" / "qwen3.8-compact-v1"))
sys.path[:0] = [str(ROOT / "harness"), str(ROOT / "harness" / "vendor" / "safetyraise_7200e30")]
sys.dont_write_bytecode = True

from sr_eval.live.backends import RoleBackends, ScriptedBackend  # noqa: E402
from sr_eval.live.env import LiveEnvironment  # noqa: E402
from sr_eval.live.retrieval import build_retrieval  # noqa: E402
from sr_eval.live.samples import count_payload  # noqa: E402
from sr_eval.live.scripted import passing_script  # noqa: E402
from app.report_harness.errors import HarnessError  # noqa: E402

out = Path(sys.argv[1])
files = [f for a in sys.argv[2:] for f in sorted(glob.glob(a))]
res = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
tmp = ROOT / "harness" / "runs" / "_sizes_tmp"
tmp.mkdir(parents=True, exist_ok=True)


class Probe:
    def __init__(self):
        self.tokens = None

    def __call__(self, payload):
        self.tokens = count_payload(payload, reasoning_effort="compact")["prompt_tokens"]
        raise HarnessError("sizes_probe_stop")


for f in files:
    case = json.loads(Path(f).read_text(encoding="utf-8"))
    cid = case["case_id"]
    if cid in res:
        continue
    probe = Probe()
    backend = RoleBackends({"generator": ScriptedBackend(passing_script), "reviewer": ScriptedBackend(passing_script)})
    try:
        asyncio.run(LiveEnvironment(backend, build_retrieval("sparse_half", seed=7, case_id=cid), payload_guard=probe).run(case, trace_path=tmp / f"{cid}.json"))
    except Exception as e:  # noqa: BLE001
        print(cid, "error", type(e).__name__, str(e)[:100], flush=True)
    res[cid] = probe.tokens
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(cid, probe.tokens, flush=True)
import shutil; shutil.rmtree(tmp, ignore_errors=True)
print("EVENT: sizes done", len(res))
