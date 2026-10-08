"""MTP 头自蒸馏(Orin):主模型(预量化 27B + 我们的 LoRA)冻结,只训练 mtp.* 的 15 个张量。

结构(与 llama.cpp qwen35 graph_mtp / vLLM Qwen3-Next MTP 一致):
  x = fc( concat[ enorm(embed(token_{t+1})), hnorm(h_t) ] ),  h_t = 主模型最后一层 output_norm 之后的隐藏状态
  x = 全注意力解码层(q_proj 含门控、q_norm/k_norm、mrope、SiLU-MLP,带两处残差)(x)
  logits = lm_head( mtp.norm(x) )  → 预测 token_{t+2};RMSNorm 用 (1+weight) 约定
损失 = KL( 主模型在 t+1 位置的分布 || MTP 分布 ) + 0.2 * CE(真实 token_{t+2});只统计目标(助手输出)区间。
指标 top1 = MTP 的 argmax == 主模型 t+1 位置的 argmax(贪心投机解码接受率的一阶代理)。

模式:
  cache : 载入主模型+适配器,对样本做前向并缓存 h(bf16)与 ids、共享的 embed/lm_head,写到 --feat_dir
  train : 只载入 MTP + 缓存特征,训练并保存 mtp.* (safetensors, bf16)
"""
import argparse, copy, json, math, os, random, time
import torch, torch.nn as nn, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["cache", "train"], required=True)
ap.add_argument("--model", default=os.path.expanduser("~/models/Qwen3.8-27B-unsloth-bnb-4bit"))
ap.add_argument("--bf16_dir", default=os.path.expanduser("~/models/Qwen3.8-27B"), help="官方 BF16,读取初始 mtp.* 权重")
ap.add_argument("--adapter", default=os.path.expanduser("~/work/runs/v2a_r16/ckpt_final"))
ap.add_argument("--data", nargs="*", default=[], help="cache 模式:train.jsonl 文件")
ap.add_argument("--select", default="final:all,tool:0", help="如 final:all,tool:20 (kind:数量)")
ap.add_argument("--exclude_cases_from", default=None, help="排除该 train.jsonl 里出现过的 case_id(拿主模型没训练过的样本)")
ap.add_argument("--feat_dir", required=True)
ap.add_argument("--eval_feat_dir", default=None)
ap.add_argument("--out", default=os.path.expanduser("~/work/mtp/out"))
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--accum", type=int, default=4)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--ce_weight", type=float, default=0.2)
ap.add_argument("--steps", type=int, default=3)
ap.add_argument("--chunk", type=int, default=256)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
dev = "cuda"
os.makedirs(args.feat_dir, exist_ok=True)


def log(**kw):
    print(json.dumps(kw, ensure_ascii=False), flush=True)


# ---------------------------------------------------------------- cache
def select_rows(paths, spec, exclude_from):
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in open(p, encoding="utf-8")]
    if exclude_from:
        skip = {json.loads(l)["case_id"] for l in open(exclude_from, encoding="utf-8")}
        rows = [r for r in rows if r["case_id"] not in skip]
    out = []
    for item in spec.split(","):
        kind, n = item.split(":")
        sub = [r for r in rows if r["kind"] == kind]
        out += sub if n == "all" else sub[:: max(len(sub) // max(int(n), 1), 1)][: int(n)]
    return out


def do_cache():
    from transformers import AutoModelForCausalLM, AutoConfig
    from peft import PeftModel
    cfg = AutoConfig.from_pretrained(args.model)
    qc = getattr(cfg, "quantization_config", None)
    if isinstance(qc, dict): qc["bnb_4bit_compute_dtype"] = "bfloat16"
    elif qc is not None: qc.bnb_4bit_compute_dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map={"": 0}, config=cfg)
    for p in model.parameters():
        if p.dtype == torch.float16: p.data = p.data.to(torch.bfloat16)
    for b in model.buffers():
        if b.dtype == torch.float16: b.data = b.data.to(torch.bfloat16)
    pm = PeftModel.from_pretrained(model, args.adapter).eval()
    base = pm.get_base_model()
    shared = os.path.join(args.feat_dir, "shared.pt")
    if not os.path.exists(shared):
        torch.save({"embed": base.model.embed_tokens.weight.detach().cpu(), "lm_head": base.lm_head.weight.detach().cpu()}, shared)
    rows = select_rows(args.data, args.select, args.exclude_cases_from)
    log(event="cache_start", samples=len(rows), tokens=sum(len(r["input_ids"]) for r in rows))
    t0 = time.time()
    for i, r in enumerate(rows):
        path = os.path.join(args.feat_dir, f"{i:04d}.pt")
        if os.path.exists(path):
            continue
        ids = torch.tensor([r["input_ids"]], device=dev)
        with torch.no_grad():
            h = base.model(input_ids=ids, use_cache=False).last_hidden_state[0]
        tstart = next(j for j, l in enumerate(r["labels"]) if l != -100)
        torch.save({"ids": ids[0].cpu().int(), "h": h.to(torch.bfloat16).cpu(), "tstart": tstart, "sample_id": r["sample_id"], "kind": r["kind"]}, path)
        if i % 5 == 0:
            log(event="cached", i=i, of=len(rows), len=len(r["input_ids"]), min=round((time.time() - t0) / 60, 1))
    log(event="cache_done", samples=len(rows), min=round((time.time() - t0) / 60, 1))


# ---------------------------------------------------------------- MTP 模块
def build_mtp(text_cfg):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5RMSNorm
    H = text_cfg.hidden_size

    class MTP(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre_fc_norm_embedding = Qwen3_5RMSNorm(H, eps=text_cfg.rms_norm_eps)
            self.pre_fc_norm_hidden = Qwen3_5RMSNorm(H, eps=text_cfg.rms_norm_eps)
            self.fc = nn.Linear(2 * H, H, bias=False)
            self.layer = Qwen3_5DecoderLayer(text_cfg, 3)   # 层 3 是全注意力层
            self.norm = Qwen3_5RMSNorm(H, eps=text_cfg.rms_norm_eps)

        def forward(self, h, e, pos_emb):
            x = self.fc(torch.cat([self.pre_fc_norm_embedding(e), self.pre_fc_norm_hidden(h)], dim=-1))
            x = self.layer(x, position_embeddings=pos_emb, attention_mask=None)
            return self.norm(x)

    return MTP()


def load_mtp_weights(mtp, bf16_dir):
    from safetensors import safe_open
    idx = json.load(open(os.path.join(bf16_dir, "model.safetensors.index.json")))["weight_map"]
    sd = {}
    for k, shard in idx.items():
        if k.startswith("mtp."):
            with safe_open(os.path.join(bf16_dir, shard), "pt") as f:
                sd[k] = f.get_tensor(k)
    mapping = {}
    for k, v in sd.items():
        n = k[len("mtp."):]
        n = n.replace("layers.0.", "layer.")
        mapping[n] = v.float()
    missing, unexpected = mtp.load_state_dict(mapping, strict=False)
    assert not unexpected and not missing, (missing, unexpected)
    return sd.keys()


def export_names(mtp):
    out = {}
    for n, p in mtp.state_dict().items():
        k = "mtp." + n.replace("layer.", "layers.0.", 1) if n.startswith("layer.") else "mtp." + n
        out[k] = p.detach().to(torch.bfloat16).cpu().contiguous()
    return out


def do_train():
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
    cfg = AutoConfig.from_pretrained(args.bf16_dir)
    text_cfg = copy.deepcopy(getattr(cfg, "text_config", cfg))
    text_cfg._attn_implementation = "sdpa"
    mtp = build_mtp(text_cfg)
    load_mtp_weights(mtp, args.bf16_dir)
    mtp = mtp.to(dev).float()
    rope = Qwen3_5TextRotaryEmbedding(config=text_cfg).to(dev)
    shared = torch.load(os.path.join(args.feat_dir, "shared.pt"))
    E = shared["embed"].to(dev); W = shared["lm_head"].to(dev)

    def load_dir(d):
        return [os.path.join(d, f) for f in sorted(os.listdir(d)) if f[0].isdigit()]
    train_files = load_dir(args.feat_dir)
    eval_files = load_dir(args.eval_feat_dir) if args.eval_feat_dir else []
    log(event="train_start", train=len(train_files), eval=len(eval_files), params_m=round(sum(p.numel() for p in mtp.parameters()) / 1e6, 1))

    STEP_W = [1.0, 0.6, 0.4][: args.steps]

    def entry_loss(path, grad: bool):
        """K 步展开:第 k 步输入 (上一步 MTP 输出 x_{k-1}[t] (第 1 步用主模型 h_t), embed(ids[t+k])),预测 ids[t+k+1]。
        后两步对应 llama.cpp 链式草稿(用 MTP 自己的输出当 h);为省显存,链上不回传梯度。"""
        d = torch.load(path)
        ids = d["ids"].to(dev).long(); H = d["h"].to(dev); L = ids.shape[0]; ts = d["tstart"]
        n = L - 2
        total, cnt_all = 0.0, 0
        agree = [0] * args.steps; cnt = [0] * args.steps
        prev = None
        for k in range(1, args.steps + 1):
            m = n - (k - 1)
            h_in = H[:m] if k == 1 else prev[:m].detach()
            pos = (torch.arange(m, device=dev) + (k - 1)).view(1, 1, -1).expand(3, 1, -1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                e = E[ids[k:k + m]].unsqueeze(0)
                cos, sin = rope(h_in.unsqueeze(0), pos)
                x = mtp(h_in.unsqueeze(0), e, (cos, sin))[0]
            prev = x
            lo = max(ts - k - 1, 0)                  # 第一个监督条目:目标 token ids[t+k+1] 落在助手输出区间
            c_k = max(m - lo, 0)                     # 本步参与监督的条目数(逐块反传需要先知道归一化分母)
            step_loss = 0.0
            # 为省显存:逐块算损失并立即对该块的输入 xc 反传(logits 即用即弃),块梯度填入 gx,
            # 最后一次性把 gx 反传过 MTP 层(与整段损失一次反传在数学上等价)。
            gx = torch.zeros_like(x) if (grad and c_k) else None
            for s0 in range(lo, m, args.chunk):
                e0 = min(s0 + args.chunk, m)
                xc = x[s0:e0].detach().requires_grad_(True) if grad else x[s0:e0]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lm = (xc @ W.T).float()
                with torch.no_grad():
                    lt = (H[s0 + k:e0 + k] @ W.T).float()      # 主模型在位置 t+k 的分布(即对 ids[t+k+1] 的预测)
                kl = F.kl_div(F.log_softmax(lm, -1), F.log_softmax(lt, -1), log_target=True, reduction="batchmean")
                ce = F.cross_entropy(lm, ids[s0 + k + 1:e0 + k + 1])
                lc = (kl + args.ce_weight * ce) * (e0 - s0)
                if grad:
                    (STEP_W[k - 1] * lc / c_k).backward()
                    gx[s0:e0] = xc.grad
                step_loss += float(lc.detach())
                agree[k - 1] += int((lm.argmax(-1) == lt.argmax(-1)).sum()); cnt[k - 1] += e0 - s0
                del lm, lt, kl, ce, lc, xc
            if c_k:
                total += STEP_W[k - 1] * step_loss / c_k
                if grad:
                    x.backward(gx)                    # 逐步反传,释放该步激活
                    del gx
        return total, agree, cnt

    def evaluate():
        mtp.eval(); A = [0] * args.steps; C = [0] * args.steps; ls = 0.0
        with torch.no_grad():
            for f in eval_files:
                l, ag, cn = entry_loss(f, grad=False)
                ls += l
                for i in range(args.steps): A[i] += ag[i]; C[i] += cn[i]
        mtp.train()
        top = [A[i] / max(C[i], 1) for i in range(args.steps)]
        return ls / max(len(eval_files), 1), top, sum(C)

    base_loss, base_top, n_eval = evaluate() if eval_files else (0, [0] * args.steps, 0)
    base_score = sum(base_top) / len(base_top)
    log(event="eval", epoch=0, loss=round(base_loss, 4), top1_by_step=[round(x, 4) for x in base_top], eval_tokens=n_eval,
        note="原始(官方)MTP 权重在微调后主模型上的各步一致率(第 2、3 步为链式)")
    def save_best(score, ep):
        from safetensors.torch import save_file
        os.makedirs(args.out, exist_ok=True)
        tmp = os.path.join(args.out, "mtp.safetensors.tmp")
        save_file(export_names(mtp), tmp, metadata={"format": "pt", "eval_mean_top1": str(score), "epoch": str(ep)})
        os.replace(tmp, os.path.join(args.out, "mtp.safetensors"))
        log(event="checkpoint", epoch=ep, eval_mean_top1=round(score, 4))

    opt = torch.optim.AdamW(mtp.parameters(), lr=args.lr, weight_decay=0.0)
    total = args.epochs * math.ceil(len(train_files) / args.accum); step = 0
    best = (base_score, None)
    random.seed(args.seed)
    mtp.train()
    for ep in range(1, args.epochs + 1):
        order = train_files[:]; random.shuffle(order)
        t0 = time.time(); tl = 0.0; TA = [0] * args.steps; TC = [0] * args.steps
        for b in range(0, len(order), args.accum):
            for g in opt.param_groups:
                g["lr"] = args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / max(total, 1))))
            group = order[b:b + args.accum]
            for f in group:
                loss, ag, cn = entry_loss(f, grad=True)      # 梯度已在 entry_loss 内逐步累积
                tl += loss
                for i in range(args.steps): TA[i] += ag[i]; TC[i] += cn[i]
            for p_ in mtp.parameters():
                if p_.grad is not None: p_.grad /= len(group)
            torch.nn.utils.clip_grad_norm_(mtp.parameters(), 1.0)
            opt.step(); opt.zero_grad(set_to_none=True); step += 1
            if step % 5 == 0:
                log(event="step", step=step, of=total, epoch=ep, elapsed_min=round((time.time() - t0) / 60, 1),
                    top1_by_step=[round(TA[i] / max(TC[i], 1), 4) for i in range(args.steps)],
                    peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 1))
        ev = evaluate() if eval_files else (0, [0] * args.steps, 0)
        score = sum(ev[1]) / len(ev[1])
        log(event="epoch", epoch=ep, train_loss=round(tl / len(order), 4), train_top1_by_step=[round(TA[i] / max(TC[i], 1), 4) for i in range(args.steps)],
            eval_loss=round(ev[0], 4), eval_top1_by_step=[round(x, 4) for x in ev[1]], min=round((time.time() - t0) / 60, 1))
        if score >= best[0]:
            best = (score, True)
            save_best(score, ep)                     # 每次刷新最佳就立刻落盘,中途失败也不丢
    if best[1] is None:
        log(event="no_improvement", baseline_score=round(base_score, 4)); return
    log(event="saved", path=os.path.join(args.out, "mtp.safetensors"), eval_mean_top1=round(best[0], 4), baseline_mean_top1=round(base_score, 4))


if args.mode == "cache":
    do_cache()
else:
    do_train()
