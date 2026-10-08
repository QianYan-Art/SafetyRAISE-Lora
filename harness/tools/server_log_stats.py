"""汇总 llama-server 日志里每次调用的计时:预填充 token 数与耗时、生成 token 数与速度、草稿接受率;并给出总量与按 token 加权的平均速度。
用法: server_log_stats.py <llama-server 日志> [--json out.json]
日志需是 print_timing 块(服务默认就有)。同一 task 的几行(prompt eval / eval / draft acceptance)按 task 编号合并。
"""
import collections, json, re, statistics, sys

path = sys.argv[1]
out = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
tasks = collections.OrderedDict()
pe = re.compile(r"print_timing: id\s+(\d+) \| task (\d+) \| prompt eval time =\s+([\d.]+) ms /\s+(\d+) tokens")
ev = re.compile(r"print_timing: id\s+(\d+) \| task (\d+) \|\s+eval time =\s+([\d.]+) ms /\s+(\d+) tokens")
dr = re.compile(r"print_timing: id\s+(\d+) \| task (\d+) \| draft acceptance = ([\d.]+) \(\s*(\d+) accepted /\s*(\d+) generated\), mean len =\s+([\d.]+)")
rel = re.compile(r"stop processing: n_tokens = (\d+)")
for line in open(path, encoding="utf-8", errors="replace"):
    m = pe.search(line)
    if m:
        t = tasks.setdefault(int(m[2]), {"slot": int(m[1])}); t["prompt_ms"] = float(m[3]); t["prompt_n"] = int(m[4]); continue
    m = ev.search(line)
    if m:
        t = tasks.setdefault(int(m[2]), {"slot": int(m[1])}); t["eval_ms"] = float(m[3]); t["eval_n"] = int(m[4]); continue
    m = dr.search(line)
    if m:
        t = tasks.setdefault(int(m[2]), {"slot": int(m[1])}); t["acc"] = int(m[4]); t["gen"] = int(m[5]); t["mean_len"] = float(m[6]); continue

rows = [dict(task=k, **v) for k, v in tasks.items() if "eval_n" in v]
tot_eval_n = sum(r["eval_n"] for r in rows)
tot_eval_ms = sum(r["eval_ms"] for r in rows)
tot_prompt_n = sum(r.get("prompt_n", 0) for r in rows)
tot_prompt_ms = sum(r.get("prompt_ms", 0) for r in rows)
acc = sum(r.get("acc", 0) for r in rows)
gen = sum(r.get("gen", 0) for r in rows)
rates = [r["eval_n"] / (r["eval_ms"] / 1000) for r in rows if r["eval_ms"] > 0 and r["eval_n"] >= 200]
summary = {
    "calls": len(rows),
    "prompt_tokens_evaluated": tot_prompt_n,
    "prompt_seconds": round(tot_prompt_ms / 1000, 1),
    "prompt_tok_s": round(tot_prompt_n / (tot_prompt_ms / 1000), 1) if tot_prompt_ms else None,
    "generated_tokens": tot_eval_n,
    "generated_seconds_sum_over_calls": round(tot_eval_ms / 1000, 1),
    "per_call_decode_tok_s_token_weighted": round(tot_eval_n / (tot_eval_ms / 1000), 2) if tot_eval_ms else None,
    "per_call_decode_tok_s_median": round(statistics.median(rates), 2) if rates else None,
    "draft_accept_rate": round(acc / gen, 3) if gen else None,
    "mean_len_median_over_calls": round(statistics.median([r["mean_len"] for r in rows if "mean_len" in r]), 2) if any("mean_len" in r for r in rows) else None,
}
print(json.dumps(summary, ensure_ascii=False, indent=1))
if out:
    json.dump({"summary": summary, "calls": rows}, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
