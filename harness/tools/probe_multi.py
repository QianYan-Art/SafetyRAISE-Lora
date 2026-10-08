"""多槽位并发解码吞吐探针:同一条提示、N 个并发请求(各自不同随机数种子),先用很短的生成预热(让每个槽位都缓存好提示,
排除预填充),再同时发起正式生成,汇总每路解码速度、合计吞吐和草稿接受率。
用法: probe_multi.py <trace.json> [--slots 6] [--n 500] [--warm-n 6] [--trim-user-chars 4000] [--temp T] [--host H] [--out out.json]
"""
import argparse, http.client, json, os, threading, time

ap = argparse.ArgumentParser()
ap.add_argument("trace")
ap.add_argument("--slots", type=int, default=6)
ap.add_argument("--n", type=int, default=500)
ap.add_argument("--warm-n", type=int, default=6)
ap.add_argument("--trim-user-chars", type=int, default=0)
ap.add_argument("--temp", type=float, default=None)
ap.add_argument("--host", default=os.environ.get("SR_LOCAL_HOST", "192.168.55.1:8080"))
ap.add_argument("--out", default=None)
a = ap.parse_args()

t = json.load(open(a.trace, encoding="utf-8"))
call = next(c for c in t["calls"] if c.get("role") == "generator" and not c.get("not_sent"))
msgs = json.loads(json.dumps(call["payload"]["messages"]))
if a.trim_user_chars > 0:
    for m in msgs:
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            m["content"] = m["content"][: a.trim_user_chars]


def one(seed, n, out, idx):
    body = {"model": "x", "messages": msgs, "max_tokens": n, "stream": False, "seed": seed}
    if a.temp is not None:
        body["temperature"] = a.temp
    c = http.client.HTTPConnection(a.host, timeout=7200)
    t0 = time.time()
    c.request("POST", "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    d = json.loads(c.getresponse().read())
    tm = d.get("timings", {})
    out[idx] = {"wall_s": round(time.time() - t0, 1), "prompt_n": tm.get("prompt_n"), "cache_n": tm.get("cache_n"),
                "predicted_n": tm.get("predicted_n"), "decode_tok_s": round(tm.get("predicted_per_second", 0), 2),
                "draft_n": tm.get("draft_n"), "draft_accepted": tm.get("draft_n_accepted")}


def burst(n, base_seed):
    res = [None] * a.slots
    th = [threading.Thread(target=one, args=(base_seed + i, n, res, i)) for i in range(a.slots)]
    t0 = time.time()
    [x.start() for x in th]
    [x.join() for x in th]
    return res, time.time() - t0


warm, tw = burst(a.warm_n, 100)
print(json.dumps({"phase": "warm", "wall_s": round(tw, 1), "cache_n": [w["cache_n"] for w in warm], "prompt_n": [w["prompt_n"] for w in warm]}, ensure_ascii=False), flush=True)
res, tm_ = burst(a.n, 200)
tot_tok = sum(r["predicted_n"] or 0 for r in res)
acc = [r["draft_accepted"] / r["draft_n"] for r in res if r.get("draft_n")]
summary = {"phase": "measure", "slots": a.slots, "wall_s": round(tm_, 1), "total_tokens": tot_tok, "aggregate_tok_s_wall": round(tot_tok / tm_, 2),
           "per_stream_tok_s": [r["decode_tok_s"] for r in res], "mean_per_stream": round(sum(r["decode_tok_s"] for r in res) / len(res), 2),
           "sum_per_stream": round(sum(r["decode_tok_s"] for r in res), 2), "accept_mean": round(sum(acc) / len(acc), 3) if acc else None,
           "prompt_n": [r["prompt_n"] for r in res]}
print(json.dumps(summary, ensure_ascii=False), flush=True)
if a.out:
    json.dump({"args": vars(a), "warm": warm, "measure": res, "summary": summary}, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
