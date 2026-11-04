# 致因+责任两节的“否定式/推断式”用词密度（粗指标，只作阅读辅助）
import json, re, sys
H = r"不能|无法|不足以|未提供|尚不能|难以"; I = r"可能|提示|首要|重点|更可能|倾向|说明|表明|若|如果"
Q = json.load(open(sys.argv[1], encoding="utf-8")); by = {}
for o in Q:
    if o["kind"] != "report": continue
    r = o["rep"]; i = r.find("## 四、"); j = r.find("## 六、"); s = r[i:j] if i >= 0 else r
    if not s: continue
    by.setdefault(o["tag"].split("#")[0], []).append((1000*len(re.findall(H, s))/len(s), 1000*len(re.findall(I, s))/len(s), len(s)))
med = lambda v: sorted(v)[len(v)//2]
for t, v in by.items():
    print(f"{t:14s} n={len(v):2d} 否定/千字 {med([x[0] for x in v]):5.1f}  推断/千字 {med([x[1] for x in v]):5.1f}  两节字数 {med([x[2] for x in v])}")
