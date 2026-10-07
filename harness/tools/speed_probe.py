"""单槽位速度/接受率探针:取一条已有 trace 的首回合提示,温度 0 让服务生成 N 个 token,读 llama-server 返回的 timings(解码速度、草稿接受率)。
用法: speed_probe.py <trace.json> [max_tokens=500] [次数=2]   环境变量 SR_LOCAL_HOST(默认 192.168.55.1:8080)
"""
import http.client, json, os, sys, time
host = os.environ.get("SR_LOCAL_HOST", "192.168.55.1:8080")
t = json.load(open(sys.argv[1], encoding="utf-8"))
call = next(c for c in t["calls"] if c.get("role") == "generator" and not c.get("not_sent"))
msgs = call["payload"]["messages"]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 500
rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 2
for r in range(rounds):
    body = json.dumps({"model": "x", "messages": msgs, "max_tokens": n, "temperature": 0, "stream": False}).encode()
    c = http.client.HTTPConnection(host, timeout=1800)
    t0 = time.time()
    c.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
    d = json.loads(c.getresponse().read())
    tm = d.get("timings", {})
    print(json.dumps({"round": r, "wall_s": round(time.time() - t0, 1), "prompt_n": tm.get("prompt_n"), "prompt_per_s": round(tm.get("prompt_per_second", 0), 1),
                      "predicted_n": tm.get("predicted_n"), "decode_tok_s": round(tm.get("predicted_per_second", 0), 2),
                      "draft_n": tm.get("draft_n"), "draft_accepted": tm.get("draft_n_accepted"),
                      "accept_rate": round(tm["draft_n_accepted"] / tm["draft_n"], 3) if tm.get("draft_n") else None,
                      "finish": d["choices"][0].get("finish_reason")}, ensure_ascii=False), flush=True)
