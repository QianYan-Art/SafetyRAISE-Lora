"""比较"完整词表草稿头"与"词表子集草稿头"两次运行的草稿候选(服务以 -lv 4 启动,日志里有 SPC_DBG 的 "draft candidate" 行)。
同一提示、温度 0、同一模型时,两次运行在草稿位置上应逐步一致,除非完整头的 top-1 不在子集里。
用法: draft_log_compare.py <完整头日志> <子集日志> <子集id文件.i32>
输出: 对齐的草稿步数、top-1 一致数、不一致位置的统计(其中"完整头 top-1 不在子集内"的占比),以及第一处不一致的上下文。
注意: 一旦某步草稿不同,后续的验证批次组成就不同,数值噪声可能让后面的序列分叉,所以只统计"第一处分叉之前"的一致性。
"""
import re, struct, sys

pat = re.compile(r"seq_id (\d+), draft candidate\s+(\d+), pos\s+(\d+): +(-?\d+) \(\s*([0-9.]+)\)")


def read(path):
    out = []  # (seq_id, pos, [(tok, p)...]) 按出现顺序
    cur = None
    for line in open(path, encoding="utf-8", errors="replace"):
        m = pat.search(line)
        if not m:
            continue
        seq, k, pos, tok, p = int(m[1]), int(m[2]), int(m[3]), int(m[4]), float(m[5])
        if k == 0:
            cur = (seq, pos, [(tok, p)])
            out.append(cur)
        elif cur is not None:
            cur[2].append((tok, p))
    return out


full = read(sys.argv[1])
sub = read(sys.argv[2])
ids = set(struct.unpack("<%di" % (len(open(sys.argv[3], "rb").read()) // 4), open(sys.argv[3], "rb").read()))
n = min(len(full), len(sub))
same = 0
first_div = None
for i in range(n):
    a, b = full[i], sub[i]
    if a[0] == b[0] and a[1] == b[1] and a[2][0][0] == b[2][0][0]:
        same += 1
    else:
        first_div = i
        break
print(f"草稿步数: 完整头 {len(full)}, 子集 {len(sub)}; 逐步对齐比较 {n} 步")
print(f"第一处分叉之前一致的步数: {same}" + (f"(第 {first_div} 步分叉)" if first_div is not None else "(未分叉)"))
if first_div is not None:
    a, b = full[first_div], sub[first_div]
    print(f"分叉处: 完整头 seq={a[0]} pos={a[1]} top-1={a[2][0]}  子集 seq={b[0]} pos={b[1]} top-1={b[2][0]}")
    print(f"完整头 top-1 {'在' if a[2][0][0] in ids else '不在'}子集内")
# 子集头的 top-1 一定在子集内, 抽查
bad = sum(1 for _, _, c in sub if c[0][0] not in ids)
print(f"子集运行中 top-1 不在子集内的步数(应为 0): {bad}")
