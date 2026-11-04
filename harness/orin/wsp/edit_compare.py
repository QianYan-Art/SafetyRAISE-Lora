# 比较留出序列上各模型的逐层残差均值偏移：编辑 vs 控制向量（余弦、范数比），零编辑的偏移作为重量化噪声底
import numpy as np
X = {k: np.fromfile(f"fit/lout_{k}.f32", dtype="<f4").reshape(64, 5120).astype(np.float64) for k in ("orig", "cv", "wsp", "null")}
cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
u = np.load("dir_core.npy").astype(np.float64)
print("层   |Δcv|   |Δ编辑|  cos(Δ编辑,Δcv)  |Δ编辑|/|Δcv|  |Δ零编辑|  cos(Δ零编辑,Δcv)  |Δcv|/|h|")
for l in list(range(34, 64, 2)) + [63]:
    dc = X["cv"][l] - X["orig"][l]; de = X["wsp"][l] - X["orig"][l]; dn = X["null"][l] - X["orig"][l]
    print(f"{l:3d} {np.linalg.norm(dc):7.3f} {np.linalg.norm(de):7.3f}   {cos(de, dc):6.3f}        {np.linalg.norm(de)/max(np.linalg.norm(dc),1e-9):6.3f}      {np.linalg.norm(dn):7.3f}    {cos(dn, dc):6.3f}        {np.linalg.norm(dc)/np.linalg.norm(X['orig'][l]):.3f}")
