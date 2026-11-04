# 换头流程自检：用同一个 MTP 头换进去后，与参照 GGUF 的 blk.64.* 逐张量比较（类型、形状；F32 归一化权重应逐元素相等，量化张量看相对误差）。
# 用法: mtp_splice_check.py <新 gguf> <参照 gguf>
import sys, os, numpy as np
sys.path.insert(0, os.path.expanduser("~/work/llama.cpp/gguf-py"))
import gguf
from gguf.quants import dequantize
a = {t.name: t for t in gguf.GGUFReader(sys.argv[1]).tensors}; b = {t.name: t for t in gguf.GGUFReader(sys.argv[2]).tensors}
assert set(a) == set(b)
for n in sorted(k for k in a if k.startswith("blk.64.")):
    x, y = a[n], b[n]
    X = dequantize(x.data, x.tensor_type).astype(np.float64); Y = dequantize(y.data, y.tensor_type).astype(np.float64)
    rel = np.linalg.norm(X - Y) / max(np.linalg.norm(Y), 1e-12)
    print(f"{n:42s} {x.tensor_type.name:5s} {y.tensor_type.name:5s} {str(X.shape==Y.shape):5s} 相对差 {rel:.2e}")
diff_other = [n for n in a if not n.startswith("blk.64.") and (a[n].tensor_type != b[n].tensor_type or a[n].data.nbytes != b[n].data.nbytes)]
print("blk.64 以外类型/大小不同的张量:", diff_other[:5], len(diff_other))
