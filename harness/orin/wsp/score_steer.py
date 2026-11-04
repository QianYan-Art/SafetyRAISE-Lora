# 转向探针评分：把续写的四五节（JSON 转义形态）还原，按配置汇总抓核心与安全指标；分“取方向用过的 6 案”和“其余 6 案”。
import json, re, sys
sys.path.insert(0, "scripts")
from core_focus import CORE, CAUSAL, NEG, sents, is_core
from quality import LIAB
req = {json.loads(l)["case_id"]: json.loads(l) for l in open("phase0/requests.jsonl", encoding="utf-8")}
def unesc(raw):
    s = raw.split('",')[0]
    try: return json.loads('"' + s + '"')
    except Exception: return s.replace("\n", "\n").replace('\\"', '"')
rows = []
for l in open(sys.argv[1], encoding="utf-8"):
    r = json.loads(l); txt = "## 四、致因分析\n" + unesc(r["raw"]); c = r["case"]
    ss = sents(txt); j5 = txt.find("## 五、"); cause = txt[:j5 if j5 > 0 else None]; cs = sents(cause)[1:]
    pt = json.dumps(req[c]["messages"], ensure_ascii=False)
    body = re.sub(r"(?m)^#+\s*\d+(?:\.\d+)*", "", txt)
    fab = [n for n in set(re.findall(r"\d+(?:\.\d+)?", body)) - set("1234567") if n not in pt]
    over = [m.group(0) for m in LIAB.finditer(txt) if not re.search(r"(不能|不宜|无法|不应|不得|不足以|难以|不作|未|不|并非|是否|如|若|假如|如果)", re.split(r"[。；\n]", txt[max(0, m.start()-25):m.start()])[-1])]
    rows.append(dict(tag=r["tag"], case=c, used=r["used_for_dir"], chars=len(txt), has5=j5 > 0,
        frame=bool(re.search(r"首要研判|核心机制|首要假设|最可能|主线", txt)),
        flip=sum(1 for x in ss if re.search(r"若|如果|假如", x) and re.search(r"则|更可能|上升为|转为|成为主要|不再", x)),
        layer=len(set(re.findall(r"直接原因|直接诱因|促成因素|背景条件|背景因素", txt))),
        nec=bool(re.search(r"若没有|如果没有|即使没有|即便|仍会发生|仍可能发生|能否避免|足以避免|可以避免", txt)),
        hit=any(is_core(s, c) and re.search(CAUSAL, s) for s in cs) if c in CORE else None,
        first=is_core(" ".join(cs[:2]), c) if c in CORE and cs else None,
        neg=sum(bool(re.search(NEG, s)) and not (c in CORE and is_core(s, c)) for s in ss) / max(1, len(ss)),
        fab=fab, over=over, garbage=bool(re.search(r"(.{6,})\1\1", txt)), text=txt))
json.dump(rows, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
med = lambda v: sorted(v)[len(v)//2] if v else None
print(f"{'配置':20s} 组 n 字数 有五节 框架句 翻转句 分层 必要性 核心因果句 首因素核心 否认句比 编造 越界 重复")
for t in dict.fromkeys(x["tag"] for x in rows):
    for g, sel in (("用过", True), ("未用", False)):
        v = [x for x in rows if x["tag"] == t and x["used"] == sel]
        if not v: continue
        pc = lambda k: f"{100*sum(bool(x[k]) for x in v)/len(v):3.0f}%"
        print(f"{t:20s} {g} {len(v):2d} {med([x['chars'] for x in v]):5d} {pc('has5')} {pc('frame')} {sum(x['flip'] for x in v)/len(v):5.1f} {sum(x['layer'] for x in v)/len(v):4.1f} {pc('nec')} {pc('hit')} {pc('first')} {med([x['neg'] for x in v]):5.2f} {sum(len(x['fab']) for x in v):3d} {sum(len(x['over']) for x in v):3d} {sum(x['garbage'] for x in v):3d}")
