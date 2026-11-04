# 报告质量的确定性检查（不请外部评审；人工式阅读另做）。对每份首调用答案：
#   json/kind、报告字数、事实义务处置数（应 37）、claims 引文合格率、
#   事实召回：事故信息非空字段的值在正文中的字二元组覆盖≥0.6 记为已写到；
#   编造数字：正文里出现、但提示全文里没有的数字串；
#   越界责任：未加否定/条件的定责表述（全部/主要/次要/同等责任、应负、负有…责任）；
#   清单腔：正文中“字段”“记载为”“未提供”出现次数（报告应面向读者，不复述字段）。
import json, re, sys, glob, os
def bigr(s): s = re.sub(r"\s", "", s); return {s[i:i+2] for i in range(len(s)-1)}
def parse(s):
    try: return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s or "", re.S)
        try: return json.loads(m.group(0)) if m else None
        except Exception: return None
NEG = r"(不能|不宜|无法|不应|不得|不足以|尚不能|难以|不作|未|不|并非|是否|如|若|假如|如果)"
LIAB = re.compile(r"(承担|负有?|应负)(全部|主要|次要|同等)?责任|(全部|主要|次要|同等)责任")
def score(content, prompt_text, ad):
    j = parse(content)
    if j is None: return dict(kind="bad_json")
    if j.get("tool_calls"): return dict(kind="tool")
    if not j.get("report_markdown"):
        # 既不是合格工具调用也没有报告正文：自造格式（如 {"action":"retrieve"} 或裸 name/arguments）
        return dict(kind="bad_proto")
    rep = j.get("report_markdown") or ""
    cl = j.get("claims") or []
    ok = 0
    for c in cl:
        q = c.get("quote") or ""
        ok += bool(q) and rep.count(q) == 1
    vals = [v for v in ad.values() if isinstance(v, str) and len(v) >= 2]
    rb = bigr(rep)
    recall = sum(1 for v in vals if bigr(v) and len(bigr(v) & rb) / len(bigr(v)) >= 0.6) / max(1, len(vals))
    body = re.sub(r"(?m)^#+\s*\d+(?:\.\d+)*", "", rep)  # 去掉小节编号（### 4.1），免得误报为编造数字
    nums = set(re.findall(r"\d+(?:\.\d+)?", body)) - {"1","2","3","4","5","6","7"}
    fab = sorted(n for n in nums if n not in prompt_text)
    over = []
    for m in LIAB.finditer(rep):
        ctx = rep[max(0, m.start()-25):m.start()]
        sent = re.split(r"[。；\n]", ctx)[-1]
        if not re.search(NEG, sent): over.append(rep[max(0, m.start()-15):m.end()+5])
    form = sum(rep.count(k) for k in ("字段", "记载为", "未提供"))
    return dict(kind="report", chars=len(rep), obl=len(j.get("obligation_resolutions") or []),
                claims=len(cl), claim_ok=round(ok/len(cl), 2) if cl else None,
                recall=round(recall, 2), fab=fab, over=over, form=form, rep=rep)
if __name__ == "__main__":
    # 用法: quality.py <requests.jsonl> <raw_runs_dir> <out.json> [replay_out.jsonl:标签 ...]
    req = {json.loads(l)["case_id"]: json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if '"dev"' in l}
    RAW = sys.argv[2]; rows = []
    srcs = [("m-Q4(5120)", "live_final_v3h2m_dev12"), ("Q4(5120)", "live_v3h2_dev12_first"),
            ("Q8(5120)", "live_v3h2q8_dev12_first"), ("Q6(5120)", "live_v3h2q6_dev12_first")]
    for tag, run in srcs:
        for f in sorted(glob.glob(os.path.join(RAW, run, "*.json"))):
            d = json.load(open(f, encoding="utf-8"))
            g = [c for c in d["calls"] if c["role"] == "generator"][0]
            rows.append((tag, d["case_id"], g.get("content") or "", g.get("reasoning_content") or ""))
    for spec in sys.argv[4:]:
        path, tag = spec.rsplit(":", 1)
        for l in open(path, encoding="utf-8"):
            r = json.loads(l)
            if r.get("split") == "dev" and "error" not in r:
                rows.append((tag + (f"#{r['sample']}" if r.get("sample") else ""), r["case_id"], r.get("content") or "", r.get("reasoning_content") or ""))
    out = []
    for tag, case, content, think in rows:
        q = req[case]; pt = json.dumps(q["messages"], ensure_ascii=False)
        ad = json.loads(open(glob.glob(os.path.join(RAW, "live_final_v3h2m_dev12", case + ".json"))[0], encoding="utf-8").read())["case"]["accident_data"]
        s = score(content, pt, ad); s.update(tag=tag, case=case, think_chars=len(think)); out.append(s)
    json.dump(out, open(sys.argv[3], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    tags = list(dict.fromkeys(o["tag"] for o in out))
    print(f"{'来源':14s} n 报告 工具 坏JSON/自造格式 义务37 引文合格 事实召回 编造数字 越界责任 清单腔/千字")
    for t in tags:
        v = [o for o in out if o["tag"] == t]; R = [o for o in v if o["kind"] == "report"]
        med = lambda xs: sorted(xs)[len(xs)//2] if xs else None
        print(f"{t:14s} {len(v):2d} {len(R):3d} {sum(o['kind']=='tool' for o in v):3d} {sum(o['kind']=='bad_json' for o in v):4d}/{sum(o['kind']=='bad_proto' for o in v)} "
              f"{sum(o['obl']==37 for o in R):5d} {med([o['claim_ok'] for o in R if o['claim_ok'] is not None])} {med([o['recall'] for o in R])} "
              f"{sum(len(o['fab']) for o in R):6d} {sum(len(o['over']) for o in R):6d} {med([round(1000*o['form']/max(1,o['chars']),1) for o in R])}")
