# 用正在运行的 v3h2 服务（同一聊天模板）把配对序列渲染并切词，写 tokens.i32 与区间 [start,end)（四、五节在序列中的位置）。
import json, sys, urllib.request, numpy as np, os
U = "http://192.168.55.1:8080"
def post(p, d): return json.loads(urllib.request.urlopen(urllib.request.Request(U + p, data=json.dumps(d).encode(), headers={"Content-Type": "application/json"})).read())
def tok(s): return post("/tokenize", {"content": s, "add_special": False})["tokens"]
esc = lambda s: json.dumps(s, ensure_ascii=False)[1:-1]
os.makedirs("cap", exist_ok=True); meta = []
for i, l in enumerate(open("pairs.jsonl", encoding="utf-8")):
    r = json.loads(l)
    p = post("/apply-template", {"messages": r["messages"], "chat_template_kwargs": {"reasoning_effort": "compact"}})["prompt"]
    head = p + "\n</think>\n\n" + '{\n  "version": 1,\n  "report_markdown": "' + esc(r["prefix"])
    full = head + esc(r["section"])
    th, tf = tok(head), tok(full)
    k = 0
    while k < len(th) and th[k] == tf[k]: k += 1          # 切词边界可能合并，取真正的公共前缀
    np.array(tf, dtype="<i4").tofile(f"cap/{i:02d}.i32")
    meta.append(dict(i=i, key=r["key"], case=r["case"], side=r["side"], n=len(tf), start=k, end=len(tf)))
    print(i, r["key"], len(tf), k, len(tf) - k, flush=True)
json.dump(meta, open("cap/meta.json", "w"), ensure_ascii=False, indent=1)
