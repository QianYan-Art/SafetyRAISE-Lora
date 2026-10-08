"""把 bench_sweep.sh 产出的 jsonl 汇成 Markdown 表:每个配置两种采样(温度 0 / 服务端默认采样)的解码速度与草稿接受率,并给出相对基准配置的倍数。
用法: bench_table.py <sweep.jsonl> [基准标签=stock] [--round 1]
  --round 1 取第二轮(提示已缓存,纯解码);--round 0 含预填充后的第一轮。
"""
import json, sys, collections

path = sys.argv[1]
base = "stock"
rnd = 1
args = sys.argv[2:]
if args and not args[0].startswith("--"):
    base = args.pop(0)
if "--round" in args:
    rnd = int(args[args.index("--round") + 1])

rows = collections.OrderedDict()
errors = {}
for line in open(path, encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    d = json.loads(line)
    if "error" in d:
        errors[d["label"]] = d["error"]
        continue
    r = d["r"]
    if r.get("round") != rnd:
        continue
    key = "temp0" if "--temp 0" in d["mode"] else "sample"
    rows.setdefault(d["label"], {})[key] = r

def cell(r, ref):
    if not r:
        return "–", "–", "–"
    sp = r["decode_tok_s"]
    acc = r["accept_rate"]
    rel = f"{sp / ref:.2f}×" if ref else "–"
    return f"{sp:.2f}", f"{acc:.3f}" if acc is not None else "–", rel

b0 = (rows.get(base) or {}).get("temp0")
b1 = (rows.get(base) or {}).get("sample")
print(f"基准配置: {base}(第 {rnd} 轮)")
print("| 配置 | 温度0 tok/s | 接受率 | 倍数 | 默认采样 tok/s | 接受率 | 倍数 |")
print("| --- | --- | --- | --- | --- | --- | --- |")
for label, d in rows.items():
    c0 = cell(d.get("temp0"), b0["decode_tok_s"] if b0 else None)
    c1 = cell(d.get("sample"), b1["decode_tok_s"] if b1 else None)
    print(f"| {label} | {c0[0]} | {c0[1]} | {c0[2]} | {c1[0]} | {c1[1]} | {c1[2]} |")
for label, e in errors.items():
    print(f"| {label} | 启动失败: {e[:80]} | | | | | |")
