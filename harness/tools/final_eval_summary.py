"""总评测的服务端侧汇总:从 llama-server 日志(评测期间新增的部分)统计调用数、生成 token、按 token 加权的单路解码速度、
合计吞吐(生成 token / 墙钟)、预填充速度、草稿接受率(含各位置)、并发槽位数分布。
用法: final_eval_summary.py <server_final_*.log>
日志时间戳形如 `129.31.469.941`(分.秒.毫秒.微秒,自服务启动起算)。
"""
import re, statistics as st, sys, collections

path = sys.argv[1]
ts = re.compile(r"^(\d+)\.(\d+)\.(\d+)\.(\d+) I ")
pe = re.compile(r"print_timing: id\s+(\d+) \| task (\d+) \| prompt eval time =\s+([\d.]+) ms /\s+(\d+) tokens")
ev = re.compile(r"print_timing: id\s+(\d+) \| task (\d+) \|\s+eval time =\s+([\d.]+) ms /\s+(\d+) tokens")
dr = re.compile(r"print_timing: id\s+(\d+) \| task (\d+) \| draft acceptance = ([\d.]+) \(\s*(\d+) accepted /\s*(\d+) generated\), mean len =\s+([\d.]+)")
pos = re.compile(r"print_timing: id\s+(\d+) \| task (\d+) \|\s+acc per pos = \(([^)]*)\)")


def stamp(line):
    m = ts.match(line)
    if not m:
        return None
    mm, ss, ms, us = map(int, m.groups())
    return mm * 60 + ss + ms / 1000 + us / 1e6


t_first = t_last = None
calls = collections.OrderedDict()
for line in open(path, encoding="utf-8", errors="replace"):
    t = stamp(line)
    if t is not None:
        t_first = t if t_first is None else t_first
        t_last = t
    m = pe.search(line)
    if m:
        c = calls.setdefault(int(m[2]), {}); c["pn"] = int(m[4]); c["pms"] = float(m[3]); continue
    m = ev.search(line)
    if m:
        c = calls.setdefault(int(m[2]), {}); c["en"] = int(m[4]); c["ems"] = float(m[3]); continue
    m = dr.search(line)
    if m:
        c = calls.setdefault(int(m[2]), {}); c["acc"] = int(m[4]); c["gen"] = int(m[5]); c["mean_len"] = float(m[6]); continue
    m = pos.search(line)
    if m and m[3].strip():
        c = calls.setdefault(int(m[2]), {}); c["pos"] = [float(x) for x in m[3].split(",")]
rows = [c for c in calls.values() if "en" in c]
wall = (t_last - t_first) if t_first is not None else 0
tot_en = sum(c["en"] for c in rows); tot_ems = sum(c["ems"] for c in rows)
tot_pn = sum(c.get("pn", 0) for c in rows); tot_pms = sum(c.get("pms", 0) for c in rows)
acc = sum(c.get("acc", 0) for c in rows); gen = sum(c.get("gen", 0) for c in rows)
long_rows = [c for c in rows if c["en"] >= 200]
print(f"日志墙钟 {wall / 60:.1f} 分钟,完成调用 {len(rows)} 次(其中生成≥200 token 的 {len(long_rows)} 次)")
print(f"生成 token {tot_en},合计吞吐(生成 token / 墙钟){tot_en / wall:.2f} token/s" if wall else "")
print(f"按 token 加权的单路解码速度 {tot_en / (tot_ems / 1000):.2f} token/s;≥200 token 的调用中位 {st.median([c['en'] / (c['ems'] / 1000) for c in long_rows]):.2f} token/s" if long_rows else "")
print(f"预填充 {tot_pn} token,按累计耗时折算 {tot_pn / (tot_pms / 1000):.1f} token/s(并发时含与他路解码/预填充的互相等待)")
print(f"草稿接受率 {acc / gen:.3f}({acc}/{gen}),平均接受长度中位 {st.median([c['mean_len'] for c in rows if 'mean_len' in c]):.2f}")
pl = [c for c in rows if c.get("pos")]
if pl:
    L = max(len(c["pos"]) for c in pl)
    w = [c.get("gen", 1) for c in pl]
    avg = [sum(c["pos"][i] * wi for c, wi in zip(pl, w) if i < len(c["pos"])) / sum(wi for c, wi in zip(pl, w) if i < len(c["pos"])) for i in range(L)]
    print("各位置接受率(按调用的草稿数加权平均):", ", ".join(f"{x:.3f}" for x in avg))
