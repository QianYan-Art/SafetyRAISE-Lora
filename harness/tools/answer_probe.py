"""答案段探针:把“完整提示 + 学生真实的思考文本”作为已有前缀,只测答案段(结构化 JSON,大量照抄提示里的证据原文)的解码速度与草稿接受率。
为什么需要它:常规探针只生成思考开头的几百个 token,测不到 n-gram 查表这类“抄上下文”式投机的收益,也低估 MTP 在答案段的接受率。
做法:POST /apply-template 用服务端聊天模板渲染消息(含 `<think>\\n` 生成提示)→ 拼上学生的思考文本和 `\\n</think>\\n\\n` → 用 /completion 续写 N 个 token。
用法: answer_probe.py <answer_trace.json> [--n 1500] [--rounds 2] [--temp T] [--seed S] [--host 127.0.0.1:8081] [--out out.json]
  answer_trace.json 形如 {"messages": […], "reasoning": "…", "content": "…"}(由 live 运行的首回合生成)。第 1 轮含完整提示的预填充,第 2 轮提示已缓存、只测解码。
"""
import argparse, hashlib, http.client, json, os, time

ap = argparse.ArgumentParser()
ap.add_argument("trace")
ap.add_argument("--n", type=int, default=1500)
ap.add_argument("--rounds", type=int, default=2)
ap.add_argument("--temp", type=float, default=None)
ap.add_argument("--seed", type=int, default=None)
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
rows = []
for r in range(a.rounds):
    body = {"prompt": text, "n_predict": a.n, "cache_prompt": True, "stream": False}
    if a.temp is not None:
        body["temperature"] = a.temp
    if a.seed is not None:
        body["seed"] = a.seed
    t0 = time.time()
    d = post("/completion", body)
    tm = d.get("timings", {})
    out = d.get("content", "")
    row = {"round": r, "wall_s": round(time.time() - t0, 1), "prompt_n": tm.get("prompt_n"), "cache_n": tm.get("cache_n"),
           "predicted_n": tm.get("predicted_n"), "decode_tok_s": round(tm.get("predicted_per_second", 0), 2),
           "draft_n": tm.get("draft_n"), "draft_accepted": tm.get("draft_n_accepted"),
           "accept_rate": round(tm["draft_n_accepted"] / tm["draft_n"], 3) if tm.get("draft_n") else None,
           "tok_per_draft_step": None, "text_sha": hashlib.sha256(out.encode()).hexdigest()[:16], "head": out[:60].replace("\n", " ")}
    rows.append((row, out))
    print(json.dumps(row, ensure_ascii=False), flush=True)
if a.out:
    json.dump({"args": vars(a), "rounds": [dict(row, text=o) for row, o in rows]}, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
