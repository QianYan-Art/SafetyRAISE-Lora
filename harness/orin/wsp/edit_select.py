# 权重编辑第 1 步：由拟合案的逐神经元统计选候选神经元（层 36–52 各 K0 个），写 fit/idx.i32，供 dump_ffn gram 用。
# 选法：一直在激活且稳定的神经元——|均值|/标准差（信噪比）高；排除汇聚位（开头 4 个位置）或长尾上有异常大激活的神经元，
# 否则编辑会在这些位置上写出远大于 0.25·u 的量。
import json, sys, numpy as np
FIT = sys.argv[1].split(","); K0 = int(sys.argv[2]) if len(sys.argv) > 2 else 512
def load(c):
    h = np.fromfile(f"fit/stats_{c}.hdr", dtype="<f8"); nl, F = int(h[0]), int(h[1]); cnt = h[4:4 + nl]
    g = lambda e: np.fromfile(f"fit/stats_{c}.{e}", dtype="<f8").reshape(nl, F)
    return nl, F, cnt, g("sum"), g("sq"), g("mx"), g("sink")
nl, F, *_ = load(FIT[0])
S = np.zeros((nl, F)); Q = np.zeros((nl, F)); MX = np.zeros((nl, F)); SK = np.zeros((nl, F)); N = np.zeros(nl)
for c in FIT:
    _, _, cnt, s, q, mx, sk = load(c); S += s; Q += q; MX = np.maximum(MX, mx); SK = np.maximum(SK, sk); N += cnt
mean = S / N[:, None]; std = np.sqrt(np.maximum(Q / N[:, None] - mean ** 2, 1e-12)); rms = np.sqrt(Q / N[:, None])
snr = np.abs(mean) / std
# 排除：汇聚位激活超过 (|均值|+4σ) 的 3 倍；或全程最大值超过 rms 的 30 倍（长尾）
bad = (SK > 3 * (np.abs(mean) + 4 * std)) | (MX > 30 * rms)
score = np.where(bad, -1, snr)
idx = np.argsort(-score, axis=1)[:, :K0].astype("<i4")
idx.tofile("fit/idx.i32")
print("layer  n_bad  snr@1  snr@64  snr@256  snr@512  |mean|@64中位  rms全层中位")
for L in range(nl):
    o = score[L, idx[L]]
    print(f"{36+L:5d} {int(bad[L].sum()):6d} {o[0]:6.2f} {o[63]:6.2f} {o[255]:7.2f} {o[K0-1]:7.2f} {np.median(np.abs(mean[L, idx[L, :64]])):10.4f} {np.median(rms[L]):10.4f}")
json.dump(dict(fit=FIT, K0=K0, N=N.tolist()), open("fit/idx_meta.json", "w"))
np.save("fit/mean_std.npy", np.stack([mean, std]))
