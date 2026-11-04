# 权重编辑第 3 步：写中间 GGUF——层 36–52 的 ffn_down 解量化后加 ΔW = c·u_l·vᵀ 并存 F16，其余张量与元数据原样复制；
# 之后用 llama-quantize（不允许重量化）按 Q4_K_M 规则只量化这 17 个 F16 张量，其余已量化张量逐字节拷贝。
# 用法: edit_build.py <src.gguf> <dst_f16.gguf> <v.npy|zero|rand> <scale> [dir.npy=dir_core.npy]
#   v=zero：零编辑（只走一遍解量化→F16→重量化，用作“重量化本身”的对照）
#   v=rand：读出向量用同一 v，但写入方向换成同范数随机方向（随机方向对照）
import os, sys, numpy as np
sys.path.insert(0, os.path.expanduser("~/work/llama.cpp/gguf-py"))
import gguf
from gguf.quants import dequantize
src, dst, vsrc, scale = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
U = np.load(sys.argv[5] if len(sys.argv) > 5 else "dir_core.npy").astype(np.float64)   # [64, n_embd]，第 l 行 = 第 l 层输出方向
LO, HI = 36, 52
if vsrc == "zero": V = None
else:
    vpath = sys.argv[6] if vsrc == "rand" else vsrc
    V = np.load(vpath).astype(np.float64)
    if vsrc == "rand":
        rng = np.random.default_rng(0); R = rng.standard_normal(U.shape)
        U = R * (np.linalg.norm(U, axis=1) / np.linalg.norm(R, axis=1))[:, None]
r = gguf.GGUFReader(src)
w = gguf.GGUFWriter(dst, r.fields["general.architecture"].contents())
for f in r.fields.values():
    if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF."): continue
    vt = f.types[0]; st = f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
    w.add_key_value(f.name, f.contents(), vt, sub_type=st)
EDIT = {f"blk.{l}.ffn_down.weight": l for l in range(LO, HI + 1)}
new = {}
for t in r.tensors:
    if t.name in EDIT:
        l = EDIT[t.name]
        W = dequantize(t.data, t.tensor_type).astype(np.float64)           # [n_embd, n_ff]
        if V is not None:
            D = scale * np.outer(U[l], V[l - LO])
            print(f"{t.name} {t.tensor_type.name} |ΔW|/|W|={np.linalg.norm(D)/np.linalg.norm(W):.2e}  max|ΔW|/max|W|={np.abs(D).max()/np.abs(W).max():.2e}", flush=True)
            W = W + D
        new[t.name] = W.astype(np.float16)
        a = new[t.name]; w.add_tensor_info(t.name, a.shape, a.dtype, a.nbytes, gguf.GGMLQuantizationType.F16)
    else:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
for t in r.tensors:
    w.write_tensor_data(new[t.name] if t.name in new else t.data, tensor_endianess=r.endianess)
w.close()
print("wrote", dst, "edited", len(new))
