"""把 PEFT LoRA 适配器合并进官方 BF16 检查点(CPU 流式处理,按分片读写,内存只需一个分片)。

W' = W + (alpha/r) * B @ A,在 fp32 里算后转回 bf16;未被适配的张量(视觉塔、MTP、嵌入、lm_head、归一化……)原样复制。
适配器张量名形如 base_model.model.model.layers.N.<模块>.lora_{A,B}.weight(文本模型命名),
官方检查点里对应 model.language_model.layers.N.<模块>.weight;两种前缀都兼容。
"""
import argparse, json, os, re, shutil, sys, time
import torch
from safetensors import safe_open
from safetensors.torch import save_file

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--adapter", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--mtp_overlay", default=None, help="可选:训练后的 mtp.safetensors,按键名整体替换检查点里的 mtp.* 张量")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
cfg = json.load(open(os.path.join(args.adapter, "adapter_config.json")))
scale = cfg["lora_alpha"] / cfg["r"]
ad = {}
with safe_open(os.path.join(args.adapter, "adapter_model.safetensors"), "pt") as f:
    for k in f.keys():
        m = re.match(r"base_model\.model\.model\.(layers\.\d+\..+)\.lora_([AB])\.weight$", k)
        if m:
            ad.setdefault(m.group(1), {})[m.group(2)] = f.get_tensor(k).float()
print(f"适配器模块 {len(ad)} 个,scale={scale}", flush=True)
overlay = {}
if args.mtp_overlay:
    with safe_open(args.mtp_overlay, "pt") as f:
        overlay = {k: f.get_tensor(k) for k in f.keys()}
    print(f"MTP 覆盖张量 {len(overlay)} 个", flush=True)
used_overlay = set()
used = set()
shards = sorted(f for f in os.listdir(args.base) if f.endswith(".safetensors"))
for name in os.listdir(args.base):
    if not name.endswith(".safetensors") and os.path.isfile(os.path.join(args.base, name)):
        shutil.copy2(os.path.join(args.base, name), os.path.join(args.out, name))
t0 = time.time()
for si, shard in enumerate(shards, 1):
    tensors, n_merged = {}, 0
    with safe_open(os.path.join(args.base, shard), "pt") as f:
        meta = f.metadata()
        for k in f.keys():
            t = f.get_tensor(k)
            if k in overlay:
                assert overlay[k].shape == t.shape, (k, overlay[k].shape, t.shape)
                t = overlay[k].to(t.dtype); used_overlay.add(k); n_merged += 1
                tensors[k] = t.contiguous(); continue
            m = re.match(r"model\.(?:language_model\.)?(layers\.\d+\..+)\.weight$", k)
            if m and m.group(1) in ad and set(ad[m.group(1)]) == {"A", "B"}:
                A, B = ad[m.group(1)]["A"], ad[m.group(1)]["B"]
                assert t.shape == (B.shape[0], A.shape[1]), (k, t.shape, A.shape, B.shape)
                t = (t.float() + scale * (B @ A)).to(t.dtype)
                used.add(m.group(1)); n_merged += 1
            tensors[k] = t.contiguous()
    save_file(tensors, os.path.join(args.out, shard), metadata=meta or {"format": "pt"})
    print(f"[{si}/{len(shards)}] {shard} 张量 {len(tensors)} 合并 {n_merged} 用时 {time.time()-t0:.0f}s", flush=True)
    del tensors
if overlay and set(overlay) != used_overlay:
    print("未使用的 MTP 覆盖张量:", sorted(set(overlay) - used_overlay)); sys.exit(3)
missing = set(ad) - used
if missing:
    print("未使用的适配器模块:", sorted(missing)[:5], len(missing)); sys.exit(2)
print("合并完成", len(used), "个模块 →", args.out)
