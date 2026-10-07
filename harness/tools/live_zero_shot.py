"""零样本探针:把线上生成者第 1 回合 payload 直接发给 Orin 的 llama-server(v3f),保存原始响应(含思考)。
用法: live_zero_shot.py <payloads.json> <输出jsonl> <variant: json|plain> [max_tokens] [并发]
variant=json 带 response_format=json_object(线上同款);plain 不带。思考档位 xhigh(与评测台一致)。
"""
import concurrent.futures as cf, json, os, sys, time
import httpx

src, out, variant = sys.argv[1], sys.argv[2], sys.argv[3]
max_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 26000
workers = int(sys.argv[5]) if len(sys.argv) > 5 else 4
host = os.environ.get("SR_LOCAL_HOST", "192.168.55.1:8080")
items = json.loads(open(src, encoding="utf-8").read())


def one(it):
    body = {"model": "local", "messages": it["messages"], "max_tokens": max_tokens, "temperature": 0.6, "top_p": 0.95,
            "chat_template_kwargs": {"reasoning_effort": os.environ.get("EFFORT", "xhigh")}}
    if variant == "json":
        body["response_format"] = {"type": "json_object"}
    t0 = time.time()
    try:
        r = httpx.post(f"http://{host}/v1/chat/completions", json=body, timeout=httpx.Timeout(7200.0), trust_env=False)
        d = r.json()
        msg = d["choices"][0]["message"]
        rec = {"case_id": it["case_id"], "variant": variant, "content": msg.get("content") or "", "reasoning": msg.get("reasoning_content") or "",
               "finish": d["choices"][0].get("finish_reason"), "usage": d.get("usage"), "sec": round(time.time() - t0, 1)}
    except Exception as e:  # noqa: BLE001
        rec = {"case_id": it["case_id"], "variant": variant, "error": f"{type(e).__name__}: {e}", "sec": round(time.time() - t0, 1)}
    with open(out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


with cf.ThreadPoolExecutor(workers) as ex:
    for rec in ex.map(one, items):
        print(rec["case_id"], variant, rec.get("finish"), rec.get("usage", {}) and rec["usage"].get("completion_tokens"), rec["sec"], rec.get("error", ""), flush=True)
print("EVENT: 零样本探针完成", variant)
