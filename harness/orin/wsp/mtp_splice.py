# 把一个 MTP 头（train_mtp.py 导出的 mtp.safetensors，HF 命名 mtp.*）换进现有 GGUF，不重做整模型合并。
# 1) 临时 HF 目录：官方 config/分词器 + 一个 safetensors（mtp.* + embed/lm_head/norm，后三者只为让转换器通过）；
# 2) convert_hf_to_gguf.py --mtp（只导出 MTP 块，命名与 norm 变换由转换器负责）→ BF16 的 mtp GGUF；
# 3) 写中间 GGUF：src 的全部张量与元数据，blk.<n>.* 换成 mtp GGUF 里的同名张量（BF16/F32）；之后 llama-quantize Q4_K_M（不允许重量化）只量化这些张量。
# 用法: mtp_splice.py <mtp.safetensors> <src.gguf> <mid.gguf> <workdir> [outtype=q8_0]
#   缺省 q8_0：与交付流程一致（BF16→Q8_0→Q4_K_M），之后 llama-quantize 要加 --allow-requantize（同类型张量仍逐字节拷贝）
import json, os, shutil, subprocess, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/work/llama.cpp/gguf-py"))
import gguf
mtp_st, src, mid, wd = sys.argv[1:5]; OT = sys.argv[5] if len(sys.argv) > 5 else "q8_0"
B = os.path.expanduser("~/models/Qwen3.8-27B"); PY = os.path.expanduser("~/venvs/ft/bin/python")
hf = os.path.join(wd, "hf"); shutil.rmtree(hf, ignore_errors=True); os.makedirs(hf)
for f in os.listdir(B):
    if f.endswith((".json", ".txt", ".jinja", ".model")) and f != "model.safetensors.index.json":
        os.symlink(os.path.join(B, f), os.path.join(hf, f))
import torch
from safetensors import safe_open
from safetensors.torch import save_file
idx = json.load(open(f"{B}/model.safetensors.index.json"))["weight_map"]
sd = {}
with safe_open(mtp_st, "pt") as f:
    for k in f.keys(): sd[k] = f.get_tensor(k)
for k in ("model.language_model.embed_tokens.weight", "lm_head.weight", "model.language_model.norm.weight"):
    with safe_open(f"{B}/{idx[k]}", "pt") as f: sd[k] = f.get_tensor(k)
save_file(sd, os.path.join(hf, "model.safetensors"), metadata={"format": "pt"})
mg = os.path.join(wd, f"mtp-{OT}.gguf")
subprocess.run([PY, os.path.expanduser("~/work/llama.cpp/convert_hf_to_gguf.py"), hf, "--mtp", "--outtype", OT, "--outfile", mg],
               check=True, env={**os.environ, "PYTHONPATH": os.path.expanduser("~/work/llama.cpp/gguf-py")})
m = gguf.GGUFReader(mg); r = gguf.GGUFReader(src)
nb = int(r.fields[f"{r.fields['general.architecture'].contents()}.block_count"].contents()) - 1   # MTP 块下标（64）
new = {t.name: t for t in m.tensors if t.name.startswith(f"blk.{nb}.")}
old = {t.name for t in r.tensors if t.name.startswith(f"blk.{nb}.")}
assert set(new) == old, ("MTP 张量名不一致", sorted(set(new) ^ old))
w = gguf.GGUFWriter(mid, r.fields["general.architecture"].contents())
for f in r.fields.values():
    if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF."): continue
    vt = f.types[0]; st = f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
    w.add_key_value(f.name, f.contents(), vt, sub_type=st)
for t in r.tensors:
    s = new.get(t.name, t)
    w.add_tensor_info(t.name, s.data.shape, s.data.dtype, s.data.nbytes, s.tensor_type)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
for t in r.tensors:
    w.write_tensor_data(new[t.name].data if t.name in new else t.data, tensor_endianess=r.endianess)
w.close()
print("spliced", len(new), "tensors of blk.%d into" % nb, mid, {n: new[n].tensor_type.name for n in sorted(new)})
