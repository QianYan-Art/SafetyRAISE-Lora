# 续写探针客户端：/apply-template 渲染原协议提示，拼空思考与 JSON 前文，/completion 从“## 四、致因分析”续写到“## 六、”为止。
import json, sys, urllib.request, concurrent.futures as cf, time
U = "http://192.168.55.1:8080"; src, dst, tag = sys.argv[1], sys.argv[2], sys.argv[3]
def post(p, d, t=3600): return json.loads(urllib.request.urlopen(urllib.request.Request(U + p, data=json.dumps(d).encode(), headers={"Content-Type": "application/json"}), timeout=t).read())
esc = lambda s: json.dumps(s, ensure_ascii=False)[1:-1]
P = [json.loads(l) for l in open(src, encoding="utf-8")]
def run(p):
    pr = post("/apply-template", {"messages": p["messages"], "chat_template_kwargs": {"reasoning_effort": "compact"}})["prompt"]
    prompt = pr + "\n</think>\n\n" + '{\n  "version": 1,\n  "report_markdown": "' + esc(p["prefix"]) + "## 四、致因分析\n"
    t0 = time.time()
    r = post("/completion", {"prompt": prompt, "n_predict": 1100, "stop": ["## 六、", "\",\n", "\", \""], "cache_prompt": True})
    return dict(tag=tag, case=p["case"], used_for_dir=p["used_for_dir"], raw=r["content"], stop=r.get("stop_type"), n=r.get("tokens_predicted"), wall=round(time.time() - t0))
with cf.ThreadPoolExecutor(6) as ex, open(dst, "a", encoding="utf-8") as w:
    for res in ex.map(run, P):
        w.write(json.dumps(res, ensure_ascii=False) + "\n"); w.flush(); print(time.strftime("%T"), tag, res["case"], res["n"], res["stop"], flush=True)
print("PROBE_DONE", tag, flush=True)
