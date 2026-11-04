# 阶段 0（离线）：对已有线上环境轨迹里每次本地生成者调用的思考段做循环/空转统计。
# 指标：思考 token 数、是否触到思考预算、思考末尾 512 token 的重复 8-gram 比例（与博文 duplicate8 同义：
# 该位置的 8-gram 在思考前文中出现过）、思考中是否起草了整份报告、答案与思考的 8-gram 重合比例。
import json, glob, os, sys, statistics as st
from tokenizers import Tokenizer

RUNS = sys.argv[1]
OUT = sys.argv[2]
tok = Tokenizer.from_file(sys.argv[3])
BUDGET_MSG = "Thinking budget reached."

def ids(s):
    return tok.encode(s, add_special_tokens=False).ids

def dup_frac(seq, start, n=8):
    # seq[start:] 中每个 n-gram 位置，是否在它之前的序列里出现过
    seen = set(); hit = tot = 0
    for i in range(len(seq) - n + 1):
        g = tuple(seq[i:i+n])
        if i >= start:
            tot += 1; hit += g in seen
        seen.add(g)
    return hit / tot if tot else 0.0

def overlap(a, b, n=8):
    A = {tuple(a[i:i+n]) for i in range(len(a)-n+1)}
    B = [tuple(b[i:i+n]) for i in range(len(b)-n+1)]
    return sum(g in A for g in B) / len(B) if B else 0.0

rows = []
for run in sorted(glob.glob(os.path.join(RUNS, "live_*"))):
    if not os.path.isdir(run): continue
    for f in sorted(glob.glob(os.path.join(run, "*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        for c in d.get("calls", []):
            if c.get("role") != "generator": continue
            ch = (c.get("response") or {}).get("choices") or [{}]
            r = c.get("reasoning_content") or ""
            content = c.get("content") or ""
            hit = BUDGET_MSG in r
            think = r.split(BUDGET_MSG)[0]
            t = ids(think); a = ids(content)
            draft = any(k in think for k in ("# 交通事故分析报告", "## 一、", "report exactly", "final report"))
            pos = min([think.find(k) for k in ("# 交通事故分析报告", "## 一、") if k in think] or [-1])
            rows.append(dict(
                run=os.path.basename(run), case=d.get("case_id"), call=c.get("call_index"),
                model=os.path.basename(str((c.get("response") or {}).get("model", ""))),
                finish=ch[0].get("finish_reason"), think_tok=len(t), ans_tok=len(a), budget_hit=hit,
                dup8_tail=round(dup_frac(t, max(0, len(t) - 512)), 4), dup8_all=round(dup_frac(t, 0), 4),
                draft_in_think=draft,
                draft_share=round((len(ids(think[pos:])) / len(t)) if pos >= 0 and t else 0.0, 3),
                ans_from_think=round(overlap(t, a), 3),
                case_status=d.get("status"),
            ))
json.dump(rows, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

def q(v, p):
    v = sorted(v); return v[min(len(v)-1, int(p*len(v)))] if v else None
by = {}
for r in rows: by.setdefault(r["run"], []).append(r)
print(f"{'run':28s} n  think_med p90  hit%  tail_dup_med p90 >=0.5  draft%  draft_share_med ans_from_think_med  finish!=stop")
for k, v in by.items():
    th = [r["think_tok"] for r in v]; du = [r["dup8_tail"] for r in v]
    print(f"{k:28s} {len(v):3d} {q(th,.5):6d} {q(th,.9):6d} {100*sum(r['budget_hit'] for r in v)/len(v):5.0f} "
          f"{q(du,.5):8.3f} {q(du,.9):6.3f} {sum(x>=0.5 for x in du):4d} {100*sum(r['draft_in_think'] for r in v)/len(v):6.0f} "
          f"{q([r['draft_share'] for r in v],.5):10.3f} {q([r['ans_from_think'] for r in v],.5):12.3f} {sum(r['finish']!='stop' for r in v):6d}")
