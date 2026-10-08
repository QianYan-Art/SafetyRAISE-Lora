"""诊断:MTP 头在“服务模型真实特征”上的教师强制一致率。
特征来自 dump_hidden(llama.cpp 跑合并后的 Q4_K_M GGUF 得到的末层 output_norm 之后的隐藏状态,bf16);
头用 HF 实现(与 train_mtp.py 一致):官方头与新训练的头各评一遍,指标与训练日志里的“离线一致率”同口径(第 1 步直接预测,第 2、3 步链式)。
与训练时用 NF4+LoRA 特征得到的留出集数字(官方 0.9013/0.8254/0.7656,新头 0.9071/0.8567/0.8194)对比:
  - 差不多 → 训练与服务没有明显失配,线上思考区间接受率偏低是区域难度/采样造成的;
  - 明显更低 → 失配,按服务特征重训/微调有价值。
用法: eval_head_on_served.py --samples <目录,内含 NN.i32 / NN.bf16 / NN.meta.json> [--heads official,new] [--new ~/work/mtp/out_v3h2/mtp.safetensors]
"""
import argparse, copy, glob, json, os
import numpy as np, torch, torch.nn as nn

ap = argparse.ArgumentParser()
ap.add_argument("--samples", required=True)
ap.add_argument("--bf16_dir", default=os.path.expanduser("~/models/Qwen3.8-27B"))
ap.add_argument("--new", default=os.path.expanduser("~/work/mtp/out_v3h2/mtp.safetensors"))
ap.add_argument("--steps", type=int, default=3)
ap.add_argument("--chunk", type=int, default=256)
a = ap.parse_args()
dev = "cuda"

from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5RMSNorm, Qwen3_5TextRotaryEmbedding

cfg = AutoConfig.from_pretrained(a.bf16_dir)
text_cfg = copy.deepcopy(getattr(cfg, "text_config", cfg))
text_cfg._attn_implementation = "sdpa"
H = text_cfg.hidden_size


class MTP(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(H, eps=text_cfg.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(H, eps=text_cfg.rms_norm_eps)
        self.fc = nn.Linear(2 * H, H, bias=False)
        self.layer = Qwen3_5DecoderLayer(text_cfg, 3)
        self.norm = Qwen3_5RMSNorm(H, eps=text_cfg.rms_norm_eps)

    def forward(self, h, e, pos_emb):
        x = self.fc(torch.cat([self.pre_fc_norm_embedding(e), self.pre_fc_norm_hidden(h)], dim=-1))
        x = self.layer(x, position_embeddings=pos_emb, attention_mask=None)
        return self.norm(x)


def load_official():
    idx = json.load(open(os.path.join(a.bf16_dir, "model.safetensors.index.json")))["weight_map"]
    sd = {}
    for k, shard in idx.items():
        if k.startswith("mtp."):
            with safe_open(os.path.join(a.bf16_dir, shard), "pt") as f:
                sd[k] = f.get_tensor(k)
    return sd


def to_module(sd):
    m = MTP()
    mapping = {k[len("mtp."):].replace("layers.0.", "layer."): v.float() for k, v in sd.items()}
    missing, unexpected = m.load_state_dict(mapping, strict=False)
    assert not unexpected and not missing, (missing, unexpected)
    return m.to(dev).float().eval()


def load_new():
    sd = {}
    with safe_open(a.new, "pt") as f:
        for k in f.keys():
            sd[k] = f.get_tensor(k)
    return sd


def load_shared():
    idx = json.load(open(os.path.join(a.bf16_dir, "model.safetensors.index.json")))["weight_map"]
    out = {}
    for k, shard in idx.items():
        if k.endswith("embed_tokens.weight") or k == "lm_head.weight" or k.endswith(".lm_head.weight"):
            with safe_open(os.path.join(a.bf16_dir, shard), "pt") as f:
                out[k] = f.get_tensor(k)
    emb = next(v for k, v in out.items() if "embed_tokens" in k)
    head = next((v for k, v in out.items() if "lm_head" in k), emb)
    return emb.to(dev), head.to(dev)


@torch.no_grad()
def eval_sample(mtp, rope, E, W, Hs, ids, ts):
    L = ids.shape[0]
    n = L - 2
    agree = [0] * a.steps
    cnt = [0] * a.steps
    prev = None
    for k in range(1, a.steps + 1):
        m = n - (k - 1)
        h_in = Hs[:m] if k == 1 else prev[:m]
        pos = (torch.arange(m, device=dev) + (k - 1)).view(1, 1, -1).expand(3, 1, -1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            e = E[ids[k:k + m]].unsqueeze(0)
            cos, sin = rope(h_in.unsqueeze(0), pos)
            x = mtp(h_in.unsqueeze(0), e, (cos, sin))[0]
        prev = x
        lo = max(ts - k - 1, 0)
        for s0 in range(lo, m, a.chunk):
            e0 = min(s0 + a.chunk, m)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lm = (x[s0:e0] @ W.T).float()
            lt = (Hs[s0 + k:e0 + k] @ W.T).float()
            agree[k - 1] += int((lm.argmax(-1) == lt.argmax(-1)).sum())
            cnt[k - 1] += e0 - s0
    return agree, cnt


rope = Qwen3_5TextRotaryEmbedding(config=text_cfg).to(dev)
E, W = load_shared()
heads = {"official": to_module(load_official()), "new": to_module(load_new())}
metas = sorted(glob.glob(os.path.join(a.samples, "*.meta.json")))
tot = {h: [[0] * a.steps, [0] * a.steps] for h in heads}
for mp in metas:
    base = mp[:-len(".meta.json")]
    meta = json.load(open(mp))
    ids = torch.from_numpy(np.fromfile(base + ".i32", dtype="<i4").astype(np.int64)).to(dev)
    raw = np.fromfile(base + ".bf16", dtype=np.uint16).reshape(-1, H)
    Hs = torch.from_numpy(raw.view(np.int16)).view(torch.bfloat16).to(dev)
    assert Hs.shape[0] == ids.shape[0], (Hs.shape, ids.shape)
    for name, m in heads.items():
        ag, cn = eval_sample(m, rope, E, W, Hs, ids, meta["tstart"])
        for i in range(a.steps):
            tot[name][0][i] += ag[i]
            tot[name][1][i] += cn[i]
        print(os.path.basename(base), name, [round(ag[i] / max(cn[i], 1), 4) for i in range(a.steps)], flush=True)
print("== 合计(服务模型特征,教师强制)")
for name, (ag, cn) in tot.items():
    top = [ag[i] / max(cn[i], 1) for i in range(a.steps)]
    print(name, [round(x, 4) for x in top], "mean", round(sum(top) / len(top), 4), "tokens", cn[0])
