"""对比若干在线环境运行目录(live_batch 产出):状态分布、思考/答案长度、截断、第 1 回合确定性合格率与失败信号、37 项解读指标。
用法: live_compare.py <名称=目录> [<名称=目录> ...]   运行环境 .venv-live
"""
import json, os, statistics as st, sys, collections
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("SR_QWEN_PROFILE_DIR", str(ROOT / "profiles" / "assets" / "qwen3.8-compact-v1"))
sys.path[:0] = [str(ROOT / "harness"), str(ROOT / "harness" / "vendor" / "safetyraise_7200e30")]
sys.dont_write_bytecode = True
from sr_eval.live.revise import evaluate_response, load_traces  # noqa: E402


def med(x):
    return int(st.median(x)) if x else None


def summarize(name, d):
    traces, errs = load_traces(str(d))
    status = collections.Counter(); reasons = collections.Counter()
    think, ans, finish = [], [], collections.Counter()
    first_ok = first_n = 0; fails = collections.Counter(); unknown = collections.Counter(); metric_fail_fields = collections.Counter()
    sent_calls = 0
    for _, t in traces:
        status[t["status"]] += 1
        if t.get("reason"):
            reasons[t["reason"]] += 1
        gens = [c for c in t["calls"] if c.get("role") == "generator" and not c.get("not_sent")]
        for i, c in enumerate(gens):
            sent_calls += 1
            rw = c.get("response_window") or {}
            fin = ((c.get("response") or {}).get("choices") or [{}])[0].get("finish_reason")
            finish[fin] += 1
            if rw.get("c_start"):
                think.append(rw["c_start"])
                if rw.get("response_tokens"):
                    ans.append(rw["response_tokens"] - rw["c_start"])
            if i == 0 and c.get("response"):
                first_n += 1
                try:
                    v = evaluate_response(t, c)
                except Exception as e:  # noqa: BLE001
                    fails["eval_exception:" + type(e).__name__] += 1
                    continue
                first_ok += bool(v["training_eligible"])
                for s in v.get("fail_signals", []) + v.get("hard_problems", []):
                    fails[str(s).split(":")[0] + ":" + str(s).split(":")[1] if str(s).startswith("metric") and ":" in str(s) else str(s)[:60]] += 1
                for k, n in (v.get("unknown_metrics") or {}).items():
                    unknown[k] += n
    return {"name": name, "traces": len(traces), "status": dict(status), "reasons": dict(reasons), "sent_calls": sent_calls,
            "think_tokens_median": med(think), "answer_tokens_median": med(ans), "answer_tokens_max": max(ans) if ans else None,
            "finish": dict(finish), "first_call_n": first_n, "first_call_deterministic_eligible": first_ok,
            "top_fail_signals": fails.most_common(8), "unknown_metrics": dict(unknown)}


for a in sys.argv[1:]:
    n, p = a.split("=", 1)
    print(json.dumps(summarize(n, ROOT / p), ensure_ascii=False, indent=1))
