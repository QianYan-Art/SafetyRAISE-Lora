"""速度/接受率探针 v2:用已有 trace 的首回合提示,让 llama-server 生成 N 个 token,读 timings,并把生成文本存下来供不同构建之间做逐字比对。
与 speed_probe.py 的区别:支持 --temp(缺省=不传,与线上请求一致,走服务端默认采样)、--seed(固定随机数,同构建同数值时输出可复现)、--out(保存文本与计时)。
用法: speed_probe2.py <trace.json> [--n 600] [--rounds 3] [--temp 0] [--seed 1] [--out out.json] [--host 192.168.55.1:8080] [--trim-user-chars N]
  --trim-user-chars N:把每条 user 消息截到前 N 个字符(粗略截短,用来测不同上下文长度下的解码速度;0=不截)。
"""
import argparse, hashlib, http.client, json, os, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("trace")
ap.add_argument("--n", type=int, default=600)
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--temp", type=float, default=None)
ap.add_argument("--seed", type=int, default=None)
ap.add_argument("--out", default=None)
ap.add_argument("--host", default=os.environ.get("SR_LOCAL_HOST", "192.168.55.1:8080"))
ap.add_argument("--trim-user-chars", type=int, default=0, help="把每条 user 消息截到前 N 个字符(测短上下文;N=4000 约得 5K token 的提示)")
a = ap.parse_args()

t = json.load(open(a.trace, encoding="utf-8"))
call = next(c for c in t["calls"] if c.get("role") == "generator" and not c.get("not_sent"))
msgs = json.loads(json.dumps(call["payload"]["messages"]))
if a.trim_user_chars > 0:
    for m in msgs:
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            m["content"] = m["content"][: a.trim_user_chars]

rows = []
for r in range(a.rounds):
    body = {"model": "x", "messages": msgs, "max_tokens": a.n, "stream": False}
    if a.temp is not None:
        body["temperature"] = a.temp
    if a.seed is not None:
        body["seed"] = a.seed
    c = http.client.HTTPConnection(a.host, timeout=3600)
    t0 = time.time()
    c.request("POST", "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    d = json.loads(c.getresponse().read())
    tm = d.get("timings", {})
    msg = d["choices"][0]["message"]
    text = (msg.get("reasoning_content") or "") + "\u0000" + (msg.get("content") or "")
    row = {
        "round": r, "wall_s": round(time.time() - t0, 1), "prompt_n": tm.get("prompt_n"), "cache_n": tm.get("cache_n"),
        "prompt_per_s": round(tm.get("prompt_per_second", 0), 1), "predicted_n": tm.get("predicted_n"),
        "decode_tok_s": round(tm.get("predicted_per_second", 0), 2), "draft_n": tm.get("draft_n"),
        "draft_accepted": tm.get("draft_n_accepted"),
        "accept_rate": round(tm["draft_n_accepted"] / tm["draft_n"], 3) if tm.get("draft_n") else None,
        "finish": d["choices"][0].get("finish_reason"), "text_sha": hashlib.sha256(text.encode()).hexdigest()[:16],
    }
    rows.append((row, text))
    print(json.dumps(row, ensure_ascii=False), flush=True)

if a.out:
    json.dump({"args": vars(a), "rounds": [dict(row, text=text) for row, text in rows]}, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
