# 在 Orin 上运行：把重放请求并发发给本机 llama-server，逐条落盘（已完成的跳过，可断点续跑）。
# 用法: python replay_client.py <requests.jsonl> <out.jsonl> [--slot_ctx 40960] [--workers 6] [--samples 1] [--url ...]
import json, sys, time, argparse, threading, urllib.request
from concurrent.futures import ThreadPoolExecutor
ap = argparse.ArgumentParser()
ap.add_argument("req"); ap.add_argument("out")
ap.add_argument("--slot_ctx", type=int, default=40960); ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--samples", type=int, default=1); ap.add_argument("--url", default="http://192.168.55.1:8080")
ap.add_argument("--margin", type=int, default=128)
a = ap.parse_args()
reqs = [json.loads(l) for l in open(a.req, encoding="utf-8")]
done = set()
try:
    for l in open(a.out, encoding="utf-8"):
        r = json.loads(l)
        if "error" not in r: done.add((r["id"], r["sample"]))
except FileNotFoundError:
    pass
jobs = [(r, s) for s in range(a.samples) for r in reqs if (r["id"], s) not in done]
# 长提示先发，减少尾部空转
jobs.sort(key=lambda x: -(x[0].get("prompt_tokens") or 0))
lock = threading.Lock()
def run(job):
    r, s = job
    body = dict(model="qwen3.8-27b-local", messages=r["messages"], stream=False,
                max_tokens=max(256, a.slot_ctx - (r.get("prompt_tokens") or 0) - a.margin),
                chat_template_kwargs={"reasoning_effort": "compact"}, response_format={"type": "json_object"})
    t0 = time.time()
    try:
        q = urllib.request.Request(a.url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        resp = json.loads(urllib.request.urlopen(q, timeout=6 * 3600).read())
        ch = resp["choices"][0]; m = ch["message"]
        rec = dict(id=r["id"], sample=s, split=r["split"], case_id=r["case_id"], max_tokens=body["max_tokens"],
                   finish=ch.get("finish_reason"), reasoning_content=m.get("reasoning_content") or "", content=m.get("content") or "",
                   usage=resp.get("usage"), timings=resp.get("timings"), wall=round(time.time() - t0, 1))
    except Exception as e:
        rec = dict(id=r["id"], sample=s, error=repr(e), wall=round(time.time() - t0, 1))
    with lock:
        with open(a.out, "a", encoding="utf-8") as fo: fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(time.strftime("%H:%M:%S"), r["id"], s, rec.get("finish"), (rec.get("usage") or {}).get("completion_tokens"), rec.get("error", ""), flush=True)
with ThreadPoolExecutor(a.workers) as ex: list(ex.map(run, jobs))
print("ALL_DONE", flush=True)
