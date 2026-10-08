"""把 probe_multi.py 产出的 multi_<标签>.json(及同名服务日志 .log)汇成 Markdown 表:合计吞吐、每路速度、草稿接受率、各位置接受率。
用法: multi_table.py <目录(含 multi_*.json / multi_*.log)> [基准标签=stock_n5_s6]
各位置接受率取自服务日志里的 `acc per pos = (…)`(该位置被接受的验证步占比,逐位置累计),对本次测量里全部请求取平均;日志缺失则留空。
"""
import glob, json, os, re, sys

d = sys.argv[1]
base = sys.argv[2] if len(sys.argv) > 2 else "stock_n5_s6"
pos_re = re.compile(r"acc per pos = \(([^)]*)\)")
rows = []
for f in sorted(glob.glob(os.path.join(d, "multi_*.json"))):
    label = os.path.basename(f)[len("multi_"):-len(".json")]
    try:
        j = json.load(open(f, encoding="utf-8"))
    except Exception:
        continue
    s = j.get("summary", {})
    pos_avg = None
    lf = f[:-5] + ".log"
    if os.path.exists(lf):
        vecs = []
        for line in open(lf, encoding="utf-8", errors="replace"):
            m = pos_re.search(line)
            if m and m.group(1).strip():
                try:
                    vecs.append([float(x) for x in m.group(1).split(",")])
                except ValueError:
                    pass
        # 日志里同一请求只会打印一次;测量阶段是最后 N 个请求(N=槽位数)
        n = s.get("slots") or len(vecs)
        vecs = vecs[-n:] if n else vecs
        if vecs:
            L = max(len(v) for v in vecs)
            pos_avg = [sum(v[i] for v in vecs if i < len(v)) / len([v for v in vecs if i < len(v)]) for i in range(L)]
    rows.append((label, s, pos_avg))

ref = next((s.get("aggregate_tok_s_wall") for l, s, _ in rows if l == base), None)
print(f"基准配置: {base}")
print("| 配置 | 槽位 | 合计 tok/s | 倍数 | 每路平均 tok/s | 接受率 | 各位置接受率 |")
print("| --- | --- | --- | --- | --- | --- | --- |")
for label, s, pos in rows:
    agg = s.get("aggregate_tok_s_wall")
    rel = f"{agg / ref:.2f}×" if (agg and ref) else "–"
    pos_s = " / ".join(f"{x:.2f}" for x in pos) if pos else "–"
    print(f"| {label} | {s.get('slots', '–')} | {agg if agg is not None else '–'} | {rel} | {s.get('mean_per_stream', '–')} | {s.get('accept_mean', '–')} | {pos_s} |")
