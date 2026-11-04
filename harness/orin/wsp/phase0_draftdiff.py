# 草稿之间到底改了什么：把思考里每份草稿按"## X、"切节，与最终答案逐节比对（相同/改动字数），并统计草稿是否已是 JSON 转义形态
import json, re, sys, difflib
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
def norm(s): return s.replace("\n", "\n").replace('\\"', '"').strip()
def secs(txt):
    parts = re.split(r"(?m)^## (?=[一二三四五六七八]、)", norm(txt))
    out = {}
    for p in parts[1:]:
        k = p[:2]; body = re.split(r"\n\s*(?:Need|Now|Let's|We |But |Check|Claims|claims|\"claims\"|\",)", p)[0]
        out[k] = body.strip()
    return out
for r in sorted(rows, key=lambda r: r["id"]):
    t = r.get("reasoning_content") or ""
    try: fin = json.loads(r["content"]).get("report_markdown")
    except Exception: fin = None
    st = [m.start() for m in re.finditer(r"## 一、", t)]
    if not st: continue
    F = secs(fin) if fin else {}
    line = []
    for k, s in enumerate(st):
        e = st[k+1] if k+1 < len(st) else len(t)
        D = secs(t[s:e]); esc = "\n" in t[s:s+400]
        same = sum(1 for x in F if x in D and D[x] == F[x])
        chg = sum(int((1 - difflib.SequenceMatcher(None, D[x], F[x]).ratio()) * len(F[x])) for x in F if x in D)
        line.append(f"稿{k+1}[{len(D)}节{'/转义' if esc else ''}] 同终稿{same}/{len(F)}节 差~{chg}字")
    print(r["id"], r["finish"], " | ".join(line))
