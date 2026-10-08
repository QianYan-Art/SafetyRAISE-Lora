"""把 region_sweep.sh 产出的 jsonl(思考区间 / 答案区间两类行)汇成 Markdown 表:每个配置的解码速度与草稿接受率,并给出相对基准配置的倍数。
用法: region_table.py <jsonl> [<jsonl> ...] [--base n5_p0]     (多个文件按顺序合并,同标签后者覆盖前者)
"""
import json, sys

args = sys.argv[1:]
base = "n5_p0"
if "--base" in args:
    i = args.index("--base"); base = args[i + 1]; args = args[:i] + args[i + 2:]
rows = {}
order = []
for path in args:
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if "error" in d:
            continue
        lab = d["label"]
        if lab not in rows:
            rows[lab] = {}; order.append(lab)
        rows[lab][d["region"]] = d["r"]


def cell(r):
    if not r:
        return "–", "–", "–"
    return f"{r['decode_tok_s']:.2f}", (f"{r['accept_rate']:.3f}" if r.get("accept_rate") is not None else "–"), str(r.get("predicted_n", "–"))


b = rows.get(base, {})
bt = (b.get("think") or {}).get("decode_tok_s")
ba = (b.get("answer") or {}).get("decode_tok_s")
print(f"基准: {base}")
print("| 配置 | 思考 tok/s | 倍数 | 接受率 | 生成 token | 答案 tok/s | 倍数 | 接受率 | 生成 token |")
print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
for lab in order:
    t = cell(rows[lab].get("think")); a = cell(rows[lab].get("answer"))
    rt = f"{float(t[0]) / bt:.2f}×" if (t[0] != "–" and bt) else "–"
    ra = f"{float(a[0]) / ba:.2f}×" if (a[0] != "–" and ba) else "–"
    print(f"| {lab} | {t[0]} | {rt} | {t[1]} | {t[2]} | {a[0]} | {ra} | {a[1]} | {a[2]} |")
