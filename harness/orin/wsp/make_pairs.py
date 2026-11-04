# 阶段 1 配对数据：同一案件、同一提示（原协议，不含 P2 要求）、同一报告前文（一至三节），
# A = P2 下“抓核心”写法的四、五节（经人工阅读挑选），B = 原协议下 v3h2 各采样的四、五节。
import json, sys
Q = json.load(open("results/quality_all.json", encoding="utf-8"))
req = {json.loads(l)["case_id"]: json.loads(l) for l in open("phase0/requests.jsonl", encoding="utf-8")}
A_SEL = [("syn_dev_000", "P2(5120)#1"), ("syn_dev_003", "P2(5120)"), ("syn_dev_003", "P2(5120)#1"), ("syn_dev_004", "P2(5120)#1"),
         ("syn_dev_006", "P2(5120)#1"), ("syn_dev_018", "P2(5120)"), ("syn_dev_038", "P2(5120)"), ("syn_dev_038", "P2(5120)#1")]
B_TAGS = ["Q4(5120)", "m-Q4(5120)", "Q8(5120)", "Q6(5120)"]
rep = {(o["case"], o["tag"]): o["rep"] for o in Q if o["kind"] == "report"}
def split(r):
    i = r.find("## 四、"); j = r.find("## 六、")
    return (r[:i], r[i:j if j > 0 else None]) if i > 0 else (None, None)
out = []
for case in sorted({c for c, _ in A_SEL}):
    base_tag = next(t for t in B_TAGS if (case, t) in rep)      # 前文取第一份可用的原协议报告
    prefix, _ = split(rep[(case, base_tag)])
    for c, t in A_SEL:
        if c == case: out.append(dict(key=f"A|{case}|{t}", case=case, side="A", prefix=prefix, section=split(rep[(c, t)])[1]))
    for t in B_TAGS:
        if (case, t) in rep and split(rep[(case, t)])[1]:
            out.append(dict(key=f"B|{case}|{t}", case=case, side="B", prefix=prefix, section=split(rep[(case, t)])[1]))
with open("phase1/pairs.jsonl", "w", encoding="utf-8") as w:
    for o in out:
        o["messages"] = req[o["case"]]["messages"]; w.write(json.dumps(o, ensure_ascii=False) + "\n")
print(len(out), "sequences;", sum(o["side"] == "A" for o in out), "A /", sum(o["side"] == "B" for o in out), "B")
