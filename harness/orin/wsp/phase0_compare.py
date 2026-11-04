# 阶段 0 定性对照：不设预算的首调用 vs 5120 预算版（同案首调用与最终发布版）。
# 输出每案：答案类型、报告字数、claims 数与引文校验、义务解决状态分布、起草段落长度与相互重合。
import json, sys, re, os
from tokenizers import Tokenizer
OUT, RAW, TOK = sys.argv[1], sys.argv[2], sys.argv[3]
tok = Tokenizer.from_file(TOK)
def ids(s): return tok.encode(s, add_special_tokens=False).ids
def grams(t, n=8): return {tuple(t[i:i+n]) for i in range(len(t)-n+1)}
def ov(a, b):
    A, B = grams(ids(a)), grams(ids(b)); return round(len(A & B)/len(B), 2) if B else 0
def parse(s):
    try: return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s or "", re.S)
        try: return json.loads(m.group(0)) if m else None
        except Exception: return None
def summ(j):
    if j is None: return dict(kind="bad_json")
    if j.get("tool_calls"): return dict(kind="tool", tools=[(t.get("name"), json.dumps(t.get("arguments"), ensure_ascii=False)[:60]) for t in j["tool_calls"]])
    rep = j.get("report_markdown") or ""
    cl = j.get("claims") or []
    bad = 0
    for c in cl:
        for q in (c.get("quotes") or c.get("evidence_quotes") or ([c.get("quote")] if c.get("quote") else [])):
            if isinstance(q, dict): q = q.get("text") or q.get("quote") or ""
            if q and rep.count(q) != 1: bad += 1
    res = j.get("obligation_resolutions") or []
    st = {}
    for r in res: st[r.get("status")] = st.get(r.get("status"), 0) + 1
    secs = re.findall(r"^## (.+)$", rep, re.M)
    return dict(kind="report", chars=len(rep), secs=len(secs), claims=len(cl), quote_bad=bad, res=len(res), res_st=st, rep=rep)
rows = [json.loads(l) for l in open(OUT, encoding="utf-8")]
for r in sorted(rows, key=lambda r: r["id"]):
    if not r["id"].startswith("dev:"): continue
    case = r["case_id"]
    think = r.get("reasoning_content") or ""
    a = summ(parse(r.get("content") or ""))
    d = json.load(open(os.path.join(RAW, case + ".json"), encoding="utf-8"))
    gens = [c for c in d["calls"] if c["role"] == "generator"]
    b0 = summ(parse(gens[0].get("content") or ""))
    pub = None
    for rd in reversed(d.get("rounds") or []):
        if rd.get("candidate"): pub = summ(rd["candidate"]); break
    starts = [m.start() for m in re.finditer(r"## 一、", think)]
    drafts = [think[s:e] for s, e in zip(starts, starts[1:] + [len(think)])]
    print(f"\n=== {case}  finish={r['finish']}  5120状态={d.get('status')} 生成调用数={len(gens)}")
    for nm, s in (("无预算首调用", a), ("5120首调用", b0), ("5120最终候选", pub)):
        if not s: print(f"  {nm}: 无"); continue
        if s["kind"] != "report": print(f"  {nm}: {s['kind']} {s.get('tools','')}"); continue
        print(f"  {nm}: 报告{s['chars']}字 章节{s['secs']} claims{s['claims']} 引文不唯一/缺失{s['quote_bad']} 义务{s['res']} {s['res_st']}")
    if drafts:
        print("  草稿字数:", [len(x) for x in drafts], " 相邻草稿重合:", [ov(drafts[i], drafts[i+1]) for i in range(len(drafts)-1)])
        if a.get("kind") == "report": print("  末稿→最终答案重合:", ov(drafts[-1], a["rep"]))
    if a.get("kind") == "report" and pub and pub.get("kind") == "report":
        print("  无预算首调用 vs 5120最终 报告重合:", ov(pub["rep"], a["rep"]))
