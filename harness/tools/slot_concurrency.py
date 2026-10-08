"""从 llama-server 日志统计各槽位的状态占比:同时在生成的槽位数、同时在预填充的槽位数、有预填充时生成被拖慢的时间占比。
用来回答“交付形态(6 路并发)下大部分时间有几路在生成”,据此选草稿长度/自适应上限。
用法: slot_concurrency.py <llama-server 日志>
日志行时间戳形如 `129.31.469.941`(分.秒.毫秒.微秒,自服务启动起算);槽位生命周期取自 launch_slot_ / `prompt processing … progress = 1.00` / release 三类行。
"""
import collections, re, sys

ts = re.compile(r"^(\d+)\.(\d+)\.(\d+)\.(\d+) I ")
launch = re.compile(r"slot launch_slot_: id\s+(\d+) \| task (\d+)")
rel = re.compile(r"slot\s+release: id\s+(\d+) \| task (\d+)")
done = re.compile(r"slot print_timing: id\s+(\d+) \| task (\d+) \| prompt processing, n_tokens =\s+\d+, progress = 1\.00")


def stamp(line):
    m = ts.match(line)
    if not m:
        return None
    mm, ss, ms, us = map(int, m.groups())
    return mm * 60 + ss + ms / 1000 + us / 1e6


ev = []
for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    t = stamp(line)
    if t is None:
        continue
    for rx, kind in ((launch, "launch"), (done, "pp_done"), (rel, "rel")):
        m = rx.search(line)
        if m:
            ev.append((t, kind, int(m[1])))
            break
ev.sort()
if not ev:
    sys.exit("日志里没有槽位事件")
state, prev, dur = {}, ev[0][0], collections.Counter()
for t, kind, slot in ev:
    g = sum(1 for v in state.values() if v == "gen")
    p = sum(1 for v in state.values() if v == "pp")
    dur[(g, p)] += t - prev
    prev = t
    if kind == "launch":
        state[slot] = "pp"
    elif kind == "pp_done":
        state[slot] = "gen"
    else:
        state.pop(slot, None)
tot = sum(dur.values())
by_gen, by_pp = collections.Counter(), collections.Counter()
for (g, p), v in dur.items():
    by_gen[g] += v
    by_pp[p] += v
print(f"统计时长 {tot:.0f} 秒,槽位事件 {len(ev)} 个")
print("同时在生成的槽位数占比:", ", ".join(f"{g}路 {by_gen[g] / tot:.1%}" for g in sorted(by_gen)))
print("同时在预填充的槽位数占比:", ", ".join(f"{p}路 {by_pp[p] / tot:.1%}" for p in sorted(by_pp)))
pp_any = sum(v for (g, p), v in dur.items() if p > 0)
both = sum(v for (g, p), v in dur.items() if p > 0 and g > 0)
print(f"有预填充在进行的时间占比 {pp_any / tot:.1%};其中同时有槽位在生成(被预填充拖慢)的占比 {both / tot:.1%}")
print(f"按时间平均:在生成 {sum(g * v for (g, p), v in dur.items()) / tot:.2f} 路,在预填充 {sum(p * v for (g, p), v in dur.items()) / tot:.2f} 路")
