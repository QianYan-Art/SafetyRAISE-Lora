# 由逐条序列的每层均值算“抓核心 vs 原写法”方向：按案件平衡（每案 A 均值 − B 均值，再对案件取平均）。
# 输出：方向 npy、逐层统计（方向范数、相对残差范数、留一案件一致性），以及 llama.cpp 控制向量 GGUF（方向与同范数随机对照）。
import json, os, numpy as np, sys
sys.path.insert(0, os.path.expanduser("~/work/llama.cpp/gguf-py"))
import gguf
meta = json.load(open("cap/meta.json")); L, E = 64, 5120
X = {m["i"]: np.fromfile(f"cap/{m['i']:02d}.f32", dtype="<f4").reshape(L, E).astype(np.float64) for m in meta}
cases = sorted({m["case"] for m in meta}); D = {}
for c in cases:
    A = [X[m["i"]] for m in meta if m["case"] == c and m["side"] == "A"]; B = [X[m["i"]] for m in meta if m["case"] == c and m["side"] == "B"]
    D[c] = np.mean(A, 0) - np.mean(B, 0)
u = np.mean([D[c] for c in cases], 0)
resid = np.mean([X[m["i"]] for m in meta if m["side"] == "B"], 0)
cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
loo = np.array([[cos(D[c][l], np.mean([D[o][l] for o in cases if o != c], 0)) for c in cases] for l in range(L)])
print("layer |u|  |u|/|h|  LOO-cos(median over cases)")
for l in range(0, L, 4):
    print(f"{l:3d} {np.linalg.norm(u[l]):7.3f} {np.linalg.norm(u[l])/np.linalg.norm(resid[l]):6.3f} {np.median(loo[l]):6.3f}")
print("LOO 中位 0-31:", round(float(np.median(loo[1:32])), 3), " 32-63:", round(float(np.median(loo[32:63])), 3))
np.save("dir_core.npy", u.astype(np.float32))
json.dump(dict(norm=[float(np.linalg.norm(u[l])) for l in range(L)], rel=[float(np.linalg.norm(u[l])/np.linalg.norm(resid[l])) for l in range(L)],
               loo=loo.tolist(), cases=cases), open("dir_stats.json", "w"))
rng = np.random.default_rng(0)
def write(path, V):
    w = gguf.GGUFWriter(path, "controlvector")
    w.add_string("controlvector.model_hint", "qwen35"); w.add_int32("controlvector.layer_count", L - 1)
    for l in range(1, L): w.add_tensor(f"direction.{l}", V[l].astype(np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
write("cv_core.gguf", u)
R = rng.standard_normal((L, E)); R *= (np.linalg.norm(u, axis=1) / np.linalg.norm(R, axis=1))[:, None]
write("cv_rand.gguf", R)
print("wrote cv_core.gguf cv_rand.gguf")
