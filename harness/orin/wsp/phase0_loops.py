# 严格循环指标：思考中存在不出现在提示里的 24-gram 出现 ≥3 次，记为“循环”；同时统计思考里起草报告所占比例。
import json, glob, os, sys, collections
from tokenizers import Tokenizer
tok = Tokenizer.from_file(sys.argv[2]); RUNS = sys.argv[1]
BM = "Thinking budget reached."; N = 24
ids = lambda s: tok.encode(s, add_special_tokens=False).ids
rows = []
for f in sorted(glob.glob(os.path.join(RUNS, "live_*", "*.json"))):
    d = json.load(open(f, encoding="utf-8")); run = os.path.basename(os.path.dirname(f))
    for c in d.get("calls", []):
        if c.get("role") != "generator": continue
        ptxt = "\n".join(m["content"] if isinstance(m.get("content"), str) else json.dumps(m.get("content"), ensure_ascii=False) for m in c["payload"]["messages"])
        p = ids(ptxt); P = {tuple(p[i:i+N]) for i in range(len(p)-N+1)}
        r = c.get("reasoning_content") or ""; think = r.split(BM)[0]; t = ids(think)
        cnt = collections.Counter(tuple(t[i:i+N]) for i in range(len(t)-N+1))
        rep = [g for g, k in cnt.items() if k >= 3 and g not in P]
        # 被循环片段覆盖的 token 比例
        cov = set()
        if rep:
            R = set(rep)
            for i in range(len(t)-N+1):
                if tuple(t[i:i+N]) in R: cov.update(range(i, i+N))
        pos = min([think.find(k) for k in ("# 交通事故分析报告", "## 一、") if k in think] or [-1])
        ch = (c.get("response") or {}).get("choices") or [{}]
        rows.append(dict(run=run, case=d["case_id"], call=c["call_index"], think_tok=len(t), hit=BM in r,
                         loop=bool(rep), loop_cov=round(len(cov)/len(t), 3) if t else 0,
                         draft=pos >= 0, draft_share=round(len(ids(think[pos:]))/len(t), 3) if pos >= 0 and t else 0.0,
                         finish=ch[0].get("finish_reason"), status=d.get("status")))
json.dump(rows, open(sys.argv[3], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
by = collections.defaultdict(list)
for r in rows: by[r["run"]].append(r)
print(f"{'run':26s}  n  hit%  loop  loop_cov>0.2  draft%  draft_share_med(drafted)  hit&draft")
for k, v in by.items():
    ds = sorted(r["draft_share"] for r in v if r["draft"])
    print(f"{k:26s} {len(v):3d} {100*sum(r['hit'] for r in v)/len(v):4.0f} {sum(r['loop'] for r in v):5d} {sum(r['loop_cov']>0.2 for r in v):8d} "
          f"{100*sum(r['draft'] for r in v)/len(v):8.0f} {ds[len(ds)//2] if ds else 0:14.3f} {sum(r['hit'] and r['draft'] for r in v):12d}")
for r in rows:
    if r["loop"]: print("LOOP", r)
