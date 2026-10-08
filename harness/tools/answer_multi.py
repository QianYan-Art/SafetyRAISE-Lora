"""多槽位并发的答案区间吞吐探针:N 路同时在“完整提示 + 学生思考文本”之后续写答案段,汇总每路速度、合计吞吐和草稿接受率。
(常规 probe_multi.py 只测思考开头;交付形态约一半的生成 token 在答案段,草稿接受率约 95%,最优草稿长度与思考段不同,所以单独测。)
先用很短的生成让 N 个槽位各自缓存好同一份提示(预填充 N 次,约 N×85 秒),再同时发起正式生成。
用法: answer_multi.py <answer_trace.json> [--slots 6] [--n 800] [--host 127.0.0.1:8081] [--out out.json]
"""
import argparse, http.client, json, os, threading, time

ap = argparse.ArgumentParser()
ap.add_argument("trace")
ap.add_argument("--slots", type=int, default=6)
ap.add_argument("--n", type=int, default=800)
ap.add_argument("--warm-n", type=int, default=4)
ap.add_argument("--host", default=os.environ.get("SR_LOCAL_HOST", "127.0.0.1:8081"))
ap.add_argument("--out", default=None)
a = ap.parse_args()
t = json.load(open(a.trace, encoding="utf-8"))


def post(path, body):
    c = http.client.HTTPConnection(a.host, timeout=7200)
    c.request("POST", path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(c.getresponse().read())


prompt = post("/apply-template", {"messages": t["messages"]})["prompt"]
if not prompt.rstrip().endswith("<think>"):
    prompt += "<think>\n"
text = prompt + t["reasoning"].strip("\n") + "\n</think>\n\n"


def one(seed, n, out, idx):
    t0 = time.time()
    d = post("/completion", {"prompt": text, "n_predict": n, "cache_prompt": True, "stream": False, "seed": seed})
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
summary = {"phase": "measure", "region": "answer", "slots": a.slots, "wall_s": round(tm_, 1), "total_tokens": tot_tok, "aggregate_tok_s_wall": round(tot_tok / tm_, 2),
           "per_stream_tok_s": [r["decode_tok_s"] for r in res], "mean_per_stream": round(sum(r["decode_tok_s"] for r in res) / len(res), 2),
           "accept_mean": round(sum(acc) / len(acc), 3) if acc else None, "prompt_n": [r["prompt_n"] for r in res]}
print(json.dumps(summary, ensure_ascii=False), flush=True)
if a.out:
    json.dump({"args": vars(a), "warm": warm, "measure": res, "summary": summary}, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
