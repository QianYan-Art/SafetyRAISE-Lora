# 分析不设思考预算的重放结果：思考是否自然结束、思考长度、起草报告次数、严格循环、答案是否为合法 JSON、窗口截断。
# 用法: python phase0_analyze.py <out.jsonl> <requests.jsonl> <tokenizer.json> [ref_rows.json]
import json, sys, re, collections
from tokenizers import Tokenizer
out = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
reqs = {r["id"]: r for r in (json.loads(l) for l in open(sys.argv[2], encoding="utf-8"))}
tok = Tokenizer.from_file(sys.argv[3]); ids = lambda s: tok.encode(s, add_special_tokens=False).ids
N = 24
rows = []
for o in out:
    if "error" in o: rows.append(dict(id=o["id"], error=o["error"])); continue
    think, ans = o["reasoning_content"], o["content"]
    t = ids(think)
    ptxt = "\n".join(m["content"] if isinstance(m.get("content"), str) else json.dumps(m.get("content"), ensure_ascii=False) for m in reqs[o["id"]]["messages"])
    p = ids(ptxt); P = {tuple(p[i:i+N]) for i in range(len(p)-N+1)}
    cnt = collections.Counter(tuple(t[i:i+N]) for i in range(len(t)-N+1))
    R = {g for g, k in cnt.items() if k >= 3 and g not in P}
    cov = set()
    for i in range(len(t)-N+1):
        if tuple(t[i:i+N]) in R: cov.update(range(i, i+N))
    drafts = max(think.count("## 一、"), len(re.findall(r"一、事故(概要|概况|基本情况)", think)))
    try: json.loads(ans); ok = True
    except Exception: ok = False
    closed = bool(ans.strip())  # 有答案即思考已结束
    # 每次起草的起点（token 位置），看起草间隔
    starts = [len(ids(think[:m.start()])) for m in re.finditer(r"## 一、", think)]
    rows.append(dict(id=o["id"], split=o["split"], finish=o["finish"], think_tok=len(t), ans_tok=len(ids(ans)),
                     closed=closed, json_ok=ok, drafts=drafts, draft_starts=starts, loop_cov=round(len(cov)/len(t), 3) if t else 0,
                     max_tokens=o["max_tokens"], comp=(o.get("usage") or {}).get("completion_tokens"),
                     ref_think=reqs[o["id"]].get("ref_think"),
                     tps=round((o.get("timings") or {}).get("predicted_per_second") or 0, 2)))
json.dump(rows, open(sys.argv[1].replace(".jsonl", ".rows.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for r in sorted(rows, key=lambda r: r["id"]):
    if "error" in r: print(r); continue
    print(f"{r['id']:22s} fin={r['finish']:6s} think={r['think_tok']:6d} ans={r['ans_tok']:5d} json={int(r['json_ok'])} drafts={r['drafts']} loopcov={r['loop_cov']:.2f} ref5120={r['ref_think']} starts={r['draft_starts'][:6]}")
ok = [r for r in rows if "error" not in r]
th = sorted(r["think_tok"] for r in ok)
print(f"\nn={len(ok)} 自然结束(有答案且 finish=stop)={sum(r['closed'] and r['finish']=='stop' for r in ok)} 窗口截断(finish=length)={sum(r['finish']=='length' for r in ok)} "
      f"JSON合法={sum(r['json_ok'] for r in ok)} 思考中位={th[len(th)//2] if th else None} p90={th[int(.9*len(th))] if th else None} "
      f">5120={sum(x>5120 for x in th)} 起草>=3={sum(r['drafts']>=3 for r in ok)} loopcov>0.2={sum(r['loop_cov']>0.2 for r in ok)}")
