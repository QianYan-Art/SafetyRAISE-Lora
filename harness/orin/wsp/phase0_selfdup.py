# 补充指标：排除提示里出现过的 8-gram 后，思考末尾 512 token 的“自我重复”比例；并打印高重复样例的末尾文本。
import json, glob, os, sys
from tokenizers import Tokenizer
tok = Tokenizer.from_file(sys.argv[2]); RUNS = sys.argv[1]; SHOW = sys.argv[3:] 
BM = "Thinking budget reached."
ids = lambda s: tok.encode(s, add_special_tokens=False).ids
def grams(s, n=8): return {tuple(s[i:i+n]) for i in range(len(s)-n+1)}
out = []
for f in sorted(glob.glob(os.path.join(RUNS, "live_*", "*.json"))):
    d = json.load(open(f, encoding="utf-8")); run = os.path.basename(os.path.dirname(f))
    for c in d.get("calls", []):
        if c.get("role") != "generator": continue
        msgs = c["payload"]["messages"]
        ptxt = "\n".join(m["content"] if isinstance(m.get("content"), str) else json.dumps(m.get("content"), ensure_ascii=False) for m in msgs)
        P = grams(ids(ptxt))
        t = ids((c.get("reasoning_content") or "").split(BM)[0])
        seen = set(); hit = tot = 0; st = max(0, len(t) - 512)
        for i in range(len(t) - 7):
            g = tuple(t[i:i+8])
            if i >= st and g not in P:
                tot += 1; hit += g in seen
            seen.add(g)
        out.append((run, d["case_id"], c["call_index"], len(t), round(hit / tot, 3) if tot else 0.0, tot))
import collections
by = collections.defaultdict(list)
for r in out: by[r[0]].append(r)
for k, v in by.items():
    s = sorted(x[4] for x in v)
    print(f"{k:28s} n={len(v):3d} selfdup_tail med={s[len(s)//2]:.3f} p90={s[int(.9*len(s))]:.3f} >=0.5:{sum(x>=0.5 for x in s)} >=0.3:{sum(x>=0.3 for x in s)}")
print("top final:", sorted([r for r in out if r[0]=="live_final_v3h2m_dev12"], key=lambda r:-r[4])[:8])
for spec in SHOW:
    run, case, ci = spec.split(":")
    d = json.load(open(os.path.join(RUNS, run, case + ".json"), encoding="utf-8"))
    r = d["calls"][int(ci)]["reasoning_content"].split(BM)[0]
    print("=" * 30, spec, len(r)); print(r[-1800:])
