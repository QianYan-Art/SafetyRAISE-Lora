"""S2–S4 自动推进:等学生采样 trace 够数 → MiniMax 最小改动修订(并发 ≤2,幂等复用已有结果)→ 构建训练行与抽检包。
运行环境 .venv-live;PYTHONPATH=harness;harness\\vendor\\safetyraise_7200e30;SR_QWEN_PROFILE_DIR 指向 compact 模板(本脚本自动设置)。
用法: post_pipeline.py --tag post1 [--min-traces 40] [--workers 2] [--final]
  --final:采样已结束(或已决定截止),用全部 trace 做最后一次构建。
结果:harness/reports/revisions_<tag>/(修订记录)、harness/reports/post_<tag>_<时间戳>/(rows.jsonl、manifest.json、audit/)。结束输出 EVENT。
"""
import argparse, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("SR_QWEN_PROFILE_DIR", str(ROOT / "profiles" / "assets" / "qwen3.8-compact-v1"))
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path[:0] = [str(ROOT / "harness"), str(ROOT / "harness" / "vendor" / "safetyraise_7200e30")]
sys.dont_write_bytecode = True

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--min-traces", type=int, default=40)
ap.add_argument("--workers", type=int, default=2)
ap.add_argument("--anchor-quota", type=int, default=70)
ap.add_argument("--pair-quota", type=int, default=30)
ap.add_argument("--max-len", type=int, default=32768, help="训练行窗口(方案B:32768=部署槽位上下文)")
ap.add_argument("--unique-calls", action="store_true", help="每个学生调用至多一行(时间受限训练集)")
ap.add_argument("--seeded-cap", type=int, default=4)
ap.add_argument("--original-anchor-cap", type=int, default=6)
ap.add_argument("--min-long", type=int, default=0)
ap.add_argument("--audit-fraction", type=float, default=.2)
ap.add_argument("--anchor-allowlist")
ap.add_argument("--exclude-calls")
ap.add_argument("--final", action="store_true")
ap.add_argument("--skip-revise", action="store_true", help="只用已有修订结果构建(MiniMax 不可用时)")
args = ap.parse_args()

traces = ROOT / "harness" / "runs" / f"live_{args.tag}"
rev_out = ROOT / "harness" / "reports" / f"revisions_{args.tag}"
n = lambda: len(list(traces.glob("*.json")))
while n() < args.min_traces:
    print(time.strftime("%H:%M:%S"), "等待 trace:", n(), "/", args.min_traces, flush=True)
    time.sleep(180)
print(time.strftime("%H:%M:%S"), "trace 数", n(), flush=True)

# 上次因教师暂时不可用(限流/过载)而丢弃的记录不应被永久缓存:清掉再重试
import json as _json
if rev_out.exists():
    for f in rev_out.glob("*.json"):
        if f.name == "summary.json":
            continue
        try:
            d = _json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(d.get("discard_reason", "")).startswith(("teacher_error", "revision_exception")):
            f.unlink()

# 规则/指标升级后,旧"合格"修订记录与现行校验(含修订指令绑定)可能不再吻合:归档到 *_stale,让它在现行规则下重做
def archive_stale():
    from sr_eval.live.build_post import _checked_revision
    from sr_eval.live.revise import call_hash, load_traces
    if not rev_out.exists():
        return
    idx = {}
    for _, t in load_traces(str(traces))[0]:
        for c in t["calls"]:
            if c.get("role") == "generator" and not c.get("not_sent"):
                idx[call_hash(t, c)] = (t, c)
    stale = rev_out.parent / (rev_out.name + "_stale")
    for f in rev_out.glob("*.json"):
        if f.name == "summary.json" or f.stem not in idx:
            continue
        d = _json.loads(f.read_text(encoding="utf-8"))
        if d.get("training_eligible") is True and _checked_revision(d, *idx[f.stem])[0] is None:
            stale.mkdir(exist_ok=True)
            f.replace(stale / f.name)
            print("归档过期修订记录:", f.name[:12], flush=True)


archive_stale()

if not args.skip_revise:
    from sr_eval.live.backends import MiniMaxBackend
    from sr_eval.live.revise import run_revisions

    def backend():
        return MiniMaxBackend(allow_network=True, roles=("reviser",), kb_class="public_statutes")

    summary = run_revisions(str(traces), str(rev_out), backend, workers=args.workers, retries=2)
    print("修订汇总:", {k: v for k, v in (summary or {}).items() if not isinstance(v, (list, dict))}, flush=True)

stamp = time.strftime("%Y%m%d_%H%M%S")
out = ROOT / "harness" / "reports" / f"post_{args.tag}_{stamp}"
cmd = [sys.executable, "-B", "-m", "sr_eval.live.build_post", "--traces", str(traces), "--out", str(out),
       "--anchor-quota", str(args.anchor_quota), "--pair-quota", str(args.pair_quota), "--max-len", str(args.max_len)]
cmd += ["--audit-fraction", str(args.audit_fraction)]
if args.exclude_calls:
    cmd += ["--exclude-calls", args.exclude_calls]
if args.anchor_allowlist:
    cmd += ["--anchor-allowlist", args.anchor_allowlist]
if args.unique_calls:
    cmd += ["--unique-calls", "--seeded-cap", str(args.seeded_cap), "--original-anchor-cap", str(args.original_anchor_cap), "--min-long", str(args.min_long)]
if rev_out.exists():
    cmd += ["--revise-results", str(rev_out)]
r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", env={**os.environ, "PYTHONPATH": f"{ROOT/'harness'};{ROOT/'harness'/'vendor'/'safetyraise_7200e30'}"})
print(r.stdout[-2500:], r.stderr[-1500:], flush=True)
print("EVENT:", time.strftime("%H:%M:%S"), "S2–S4 完成", "final" if args.final else "增量", str(out), flush=True)
