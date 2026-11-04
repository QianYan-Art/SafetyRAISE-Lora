# 转向筛选用的“续写探针”：原协议提示 + 空思考 + 原协议报告的一至三节作前文，让模型从“## 四、致因分析”起续写四、五节。
# 只比较不同转向配置下四五节的写法；完整协议（含思考）留给最终确认。开发 12 案中有原协议报告可作前文的都用上。
import json
Q = json.load(open("results/quality_all.json", encoding="utf-8"))
req = {json.loads(l)["case_id"]: json.loads(l) for l in open("phase0/requests.jsonl", encoding="utf-8")}
rep = {(o["case"], o["tag"]): o["rep"] for o in Q if o["kind"] == "report"}
USED = {"syn_dev_000", "syn_dev_003", "syn_dev_004", "syn_dev_006", "syn_dev_018", "syn_dev_038"}   # 取方向用过的案件
n = 0
with open("phase1/probe.jsonl", "w", encoding="utf-8") as w:
    for case in sorted(c for c in req if c.startswith("syn_dev")):
        t = next((t for t in ["Q4(5120)", "m-Q4(5120)", "Q8(5120)", "Q6(5120)", "m-Q4(无预算)", "P1(无预算)"] if (case, t) in rep and rep[(case, t)].find("## 四、") > 0), None)
        if not t: print("skip", case); continue
        r = rep[(case, t)]
        w.write(json.dumps(dict(case=case, prefix=r[:r.find("## 四、")], prefix_from=t, used_for_dir=case in USED,
                                messages=req[case]["messages"]), ensure_ascii=False) + "\n"); n += 1
print(n, "probes")
