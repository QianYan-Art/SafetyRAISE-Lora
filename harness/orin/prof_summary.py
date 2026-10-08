"""汇总 nsys 导出的 sqlite:按内核名统计总耗时/次数/平均耗时,按类别归并,并估算 GPU 空闲(相邻内核间隙)占比。
用法: python3 prof_summary.py <report.sqlite> [top=40]
先 `nsys export --type sqlite --force-overwrite true -o report.sqlite report.nsys-rep`。
"""
import re, sqlite3, sys, collections

db = sqlite3.connect(sys.argv[1])
top = int(sys.argv[2]) if len(sys.argv) > 2 else 40
cur = db.cursor()
tables = {r[0] for r in cur.execute("select name from sqlite_master where type='table'")}
kt = "CUPTI_ACTIVITY_KIND_KERNEL"
if kt not in tables:
    sys.exit("没有内核表,导出时需要 --trace=cuda")
cols = {r[1] for r in cur.execute(f"pragma table_info({kt})")}
name_col = "demangledName" if "demangledName" in cols else "shortName"
rows = cur.execute(
    f"select s.value, k.start, k.end, k.gridX*k.gridY*k.gridZ, k.blockX*k.blockY*k.blockZ from {kt} k join StringIds s on s.id = k.{name_col} order by k.start"
).fetchall()
if not rows:
    sys.exit("空")
t0, t1 = rows[0][1], rows[-1][2]
span = (t1 - t0) / 1e6
busy = 0
prev_end = None
gaps = 0
for _, s, e, _, _ in rows:
    if prev_end is not None and s > prev_end:
        gaps += s - prev_end
    busy += e - max(s, prev_end) if (prev_end is not None and s < prev_end) else e - s
    prev_end = max(prev_end or e, e)
print(f"内核 {len(rows)} 个, 时间跨度 {span:.1f} ms, 内核累计 {busy/1e6:.1f} ms, 间隙累计 {gaps/1e6:.1f} ms ({gaps/(t1-t0):.1%})")

agg = collections.defaultdict(lambda: [0, 0.0])
for n, s, e, _, _ in rows:
    short = re.sub(r"\(.*", "", n)[:110]
    agg[short][0] += 1
    agg[short][1] += (e - s) / 1e3  # µs
cats = [
    ("mmvq 量化GEMV", r"mul_mat_vec_q"), ("mmq 量化GEMM", r"mul_mat_q"), ("mmf/mmvf 浮点矩阵乘", r"mul_mat_vec_f|mul_mat_f|gemv|gemm|cutlass|sm80|sm87"),
    ("flash-attn", r"flash_attn|fattn"), ("gated_delta_net", r"gated_delta_net"), ("ssm_conv", r"ssm_conv"),
    ("norm", r"rms_norm|l2_norm|norm_f32|group_norm"), ("q8_1 量化", r"quantize_q8_1|quantize_mmq"), ("rope", r"rope"),
    ("元素/拷贝/取行", r"k_bin_bcast|unary|silu|sigmoid|softplus|scale|cpy|concat|dup|get_rows|set_rows|fill|pad|clamp|sum_rows|mean|argsort|top_k|argmax|softmax|cumsum|tri|solve_tri|diag"),
]
cat_tot = collections.defaultdict(float)
cat_cnt = collections.defaultdict(int)
for n, (c, us) in agg.items():
    for cname, pat in cats:
        if re.search(pat, n):
            cat_tot[cname] += us
            cat_cnt[cname] += c
            break
    else:
        cat_tot["其他"] += us
        cat_cnt["其他"] += c
tot_us = sum(cat_tot.values())
print("\n按类别:")
for cname, us in sorted(cat_tot.items(), key=lambda x: -x[1]):
    print(f"  {cname:22s} {us/1e3:9.1f} ms  {us/tot_us:6.1%}  {cat_cnt[cname]:6d} 次  平均 {us/max(cat_cnt[cname],1):8.1f} µs")
print(f"\n前 {top} 个内核:")
for n, (c, us) in sorted(agg.items(), key=lambda x: -x[1][1])[:top]:
    print(f"  {us/1e3:9.1f} ms {c:6d} 次 平均 {us/c:8.1f} µs  {n}")

# 按 (内核名, 网格, 线程块) 分组: mmvq 的网格 x 就是输出行数,可据此对应到具体层(17408=FFN gate/up,5120=FFN down/ssm_out/attn_out,10240=qkv,6144=z 门,248320=输出头...)
grp = collections.defaultdict(lambda: [0, 0.0])
for n, s, e, g, b in rows:
    short = re.sub(r"\(.*", "", n)[:90]
    grp[(short, g, b)][0] += 1
    grp[(short, g, b)][1] += (e - s) / 1e3
print(f"\n按内核+网格+线程块分组的前 {top} 项:")
for (n, g, b), (c, us) in sorted(grp.items(), key=lambda x: -x[1][1])[:top]:
    print(f"  {us/1e3:9.1f} ms {c:6d} 次 平均 {us/c:8.1f} µs  grid={g} block={b}  {n}")
