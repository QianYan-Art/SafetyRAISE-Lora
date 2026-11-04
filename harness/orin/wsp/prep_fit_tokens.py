# 权重编辑拟合用序列：P2 无向量完整输出（提示 + 思考 + 正文），经服务 /apply-template 与 /tokenize 切词，写 fit/<case>.i32。
# 用法（交付服务在跑时）：prep_fit_tokens.py requests_p2.jsonl out_p2_5120.jsonl fit/ case1,case2,...
import json, sys, os, urllib.request, numpy as np
U = "http://192.168.55.1:8080"
def post(p, d): return json.loads(urllib.request.urlopen(urllib.request.Request(U + p, data=json.dumps(d).encode(), headers={"Content-Type": "application/json"}), timeout=600).read())
req = {json.loads(l)["case_id"]: json.loads(l) for l in open(sys.argv[1], encoding="utf-8")}
outs = [json.loads(l) for l in open(sys.argv[2], encoding="utf-8")]
od = sys.argv[3]; os.makedirs(od, exist_ok=True); meta = []
for case in sys.argv[4].split(","):
    # 取该案首个“直接写报告”的输出（content 以 { 开头且含 report_markdown）
    r = next(o for o in outs if o["case_id"] == case and "report_markdown" in (o.get("content") or ""))
    pr = post("/apply-template", {"messages": req[case]["messages"], "chat_template_kwargs": {"reasoning_effort": "compact"}})["prompt"]
    text = pr + (r.get("reasoning_content") or "") + "\n</think>\n\n" + r["content"]
    toks = post("/tokenize", {"content": text, "add_special": False})["tokens"]
    np.asarray(toks, dtype="<i4").tofile(f"{od}/{case}.i32")
    meta.append(dict(case=case, sample=r.get("sample", 0), n=len(toks), n_prompt=len(post("/tokenize", {"content": pr, "add_special": False})["tokens"])))
    print(meta[-1], flush=True)
json.dump(meta, open(f"{od}/meta.json", "w"), indent=1)
