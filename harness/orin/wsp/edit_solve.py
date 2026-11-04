# 权重编辑第 2 步：用候选神经元的 Gram 求每层秩 1 编辑的读出向量 v（只在候选神经元上非零）。
# 目标：对每个 token，v·a ≈ 1（a 为 down_proj 输入），这样 ΔW = c·u_l·vᵀ 在第 l 层残差上写出 ≈ c·u_l，复现控制向量。
# 岭回归 v = (G + λ·tr(G)/k·I)^{-1} s，再按拟合集均值归一使平均增益恰为 1。
# 在留出案上报告：增益均值、增益标准差/均值、汇聚位（开头 4 个位置）上的最大 |增益|。
# 用法: edit_solve.py <拟合案,..> <留出案,..> [写出 K λ]
import json, sys, numpy as np
FIT, HO = sys.argv[1].split(","), sys.argv[2].split(",")
def load(c):
    h = np.fromfile(f"fit/gram_{c}.hdr", dtype="<f8"); nl, F, K, sk = map(int, h[:4]); cnt = h[4:4 + nl]
    gs = np.fromfile(f"fit/gram_{c}.gs", dtype="<f8").reshape(nl, K)
    gg = np.fromfile(f"fit/gram_{c}.gg", dtype="<f8").reshape(nl, K, K)
    gg = np.triu(gg) + np.triu(gg, 1).transpose(0, 2, 1)
    sink = np.fromfile(f"fit/gram_{c}.gsink", dtype="<f8").reshape(nl, sk, K)
    return nl, F, K, cnt, gs, gg, sink
data = {c: load(c) for c in FIT + HO}
nl, F, K0 = data[FIT[0]][:3]
idx = np.fromfile("fit/idx.i32", dtype="<i4").reshape(nl, K0)
def agg(cs):
    N = sum(data[c][3] for c in cs); s = sum(data[c][4] for c in cs); g = sum(data[c][5] for c in cs)
    return s / N[:, None], g / N[:, None, None]
sF, gF = agg(FIT)
def solve(L, k, lam):
    G = gF[L, :k, :k]; s = sF[L, :k]
    v = np.linalg.solve(G + lam * np.trace(G) / k * np.eye(k), s)
    return v / (v @ s)               # 拟合集平均增益归一为 1
def evalv(L, k, v, cs):
    out = []
    for c in cs:
        N = data[c][3][L]; s = data[c][4][L, :k] / N; G = data[c][5][L, :k, :k] / N
        m = v @ s; sd = np.sqrt(max(v @ G @ v - m * m, 0))
        out.append((m, sd / abs(m), np.abs(data[c][6][L, :, :k] @ v).max()))
    return np.array(out)
grid = [(k, lam) for k in (64, 128, 256, 512) if k <= K0 for lam in (1e-4, 1e-3, 1e-2, 1e-1)]
print("K     λ      留出增益均值(最差)  留出相对标准差(中位)  汇聚位最大|增益|  |v|中位")
res = {}
for k, lam in grid:
    R = []; vn = []
    for L in range(nl):
        v = solve(L, k, lam); R.append(evalv(L, k, v, HO)); vn.append(np.linalg.norm(v))
    R = np.array(R)   # [nl, nHO, 3]
    worst = R[:, :, 0].flatten()[np.argmax(np.abs(R[:, :, 0].flatten() - 1))]
    res[(k, lam)] = R
    print(f"{k:4d} {lam:7.0e}  {np.median(R[:, :, 0]):6.3f} ({worst:6.3f})        {np.median(R[:, :, 1]):6.3f}            {R[:, :, 2].max():8.2f}      {np.median(vn):.4f}")
if len(sys.argv) > 4:
    k, lam = int(sys.argv[3]), float(sys.argv[4])
    V = np.zeros((nl, F), dtype=np.float32)
    for L in range(nl):
        V[L, idx[L, :k]] = solve(L, k, lam)
    np.save(f"fit/v_k{k}_l{lam:g}.npy", V)
    R = res.get((k, lam))
    print("写出", f"fit/v_k{k}_l{lam:g}.npy")
    if R is not None:
        for L in range(nl): print(f"  层 {36+L}: 留出增益 {np.round(R[L,:,0],3)}  相对标准差 {np.round(R[L,:,1],3)}  汇聚位 {np.round(R[L,:,2],2)}")
