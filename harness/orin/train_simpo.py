"""Orin 上的 SimPO(无参考模型)偏好优化:预量化 27B + 上半层 LoRA,一次只放一条序列。

损失(SimPO,Meng et al. 2024):  L = -log σ( β·(mean logp(y_w) - mean logp(y_l)) - γ )
  mean logp = 响应 token 的平均对数概率(长度归一化),无需参考模型;参照 EfficientThink 的"终止行为 SimPO":
  β=1.0、γ=0.2、LR 5e-7、LoRA r16 dropout 0、很少的步数(几十对到几百对、5~20 步)。
显存技巧(梯度解耦):L 只依赖标量 m = β(a_w - a_l) - γ,dL/da_w = -β·σ(-m),dL/da_l = +β·σ(-m)。
  所以先 no_grad 求出 a_w、a_l 得到系数 s=σ(-m),再分别对 y_w、y_l 各做一次带梯度的前后向(目标 = ∓β·s·a),
  峰值只有一条序列的激活(代价:每对共 4 次前向、2 次反传)。
数据 jsonl 每行:{"pair_id","prompt_ids":[...],"chosen_ids":[...],"rejected_ids":[...]},
  其中 *_ids 是响应部分的 token(含思考与 <|im_end|>),prompt_ids 是模板渲染到 "<think>\n" 为止的提示。
"""
import argparse, json, math, os, random, sys, time

os.environ.setdefault("CUDA_CACHE_MAXSIZE", "4294967296")
import torch, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.path.expanduser("~/models/Qwen3.8-27B-unsloth-bnb-4bit"))
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--init_adapter", default=None, help="可选:从已有适配器继续(如 SFT 后的适配器);不给则在基座上新建")
ap.add_argument("--beta", type=float, default=1.0)
ap.add_argument("--gamma", type=float, default=0.2)
ap.add_argument("--lr", type=float, default=5e-7)
ap.add_argument("--accum", type=int, default=4, help="每次更新的偏好对数")
ap.add_argument("--report_only", action="store_true", help="只对报告段算平均对数概率(数据行需带 c_start/r_start;两边思考相同时偏好信号只来自报告段)")
ap.add_argument("--sft_lambda", type=float, default=0.0, help="chosen 的平均 NLL 项权重(把保能力 SFT 并进同一步;只改 chosen 的梯度系数)")
ap.add_argument("--anchor_lambda", type=float, default=1.0, help="锚点行(kind 以 anchor 开头、无 rejected_ids)的权重:只对 chosen 的(最终回答段)平均 NLL 做训练")
ap.add_argument("--save_every", type=int, default=5, help="每 N 次更新存一次适配器(ckpt_u<N>)")
ap.add_argument("--epochs", type=float, default=1.0)
ap.add_argument("--longest_first", type=int, default=0, help="最长的 N 行排到最前(先暴露显存问题)")
ap.add_argument("--r", type=int, default=16)
ap.add_argument("--alpha", type=int, default=32)
ap.add_argument("--lora_from", type=int, default=32)
ap.add_argument("--max_len", type=int, default=32768)
ap.add_argument("--mem_frac", type=float, default=0.90)
ap.add_argument("--ce_chunk", type=int, default=1024)
ap.add_argument("--max_updates", type=int, default=0)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
dev = "cuda"
torch.cuda.set_per_process_memory_fraction(args.mem_frac)
random.seed(args.seed); torch.manual_seed(args.seed)


def log(**kw):
    kw["t"] = round(time.time(), 1)
    line = json.dumps(kw, ensure_ascii=False)
    print(line, flush=True)
    with open(os.path.join(args.out, "log.jsonl"), "a") as fh:
        fh.write(line + "\n")


pairs = [json.loads(l) for l in open(args.data, encoding="utf-8")]
_n0 = len(pairs)
pairs = [p for p in pairs if p["chosen_ids"] and (p["rejected_ids"] or str(p.get("kind", "")).startswith("anchor"))
         and len(p["prompt_ids"]) + max(len(p["chosen_ids"]), len(p["rejected_ids"] or [])) <= args.max_len]
if len(pairs) != _n0:
    print(f"剔除 {_n0 - len(pairs)} 对(空响应或超过 max_len={args.max_len})", flush=True)
print(f"偏好对 {len(pairs)}", flush=True)

from transformers import AutoModelForCausalLM, AutoConfig
cfg = AutoConfig.from_pretrained(args.model)
qc = getattr(cfg, "quantization_config", None)
if isinstance(qc, dict): qc["bnb_4bit_compute_dtype"] = "bfloat16"
elif qc is not None: qc.bnb_4bit_compute_dtype = torch.bfloat16
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map={"": 0}, config=cfg)
for p in model.parameters():
    if p.dtype == torch.float16: p.data = p.data.to(torch.bfloat16)
for b in model.buffers():
    if b.dtype == torch.float16: b.data = b.data.to(torch.bfloat16)
model.config.use_cache = False
n_layers = len(model.model.layers)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
from peft import LoraConfig, get_peft_model, PeftModel
if args.init_adapter:
    pm = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
else:
    layer_pat = "|".join(str(i) for i in range(args.lora_from, n_layers))
    targets = rf".*layers\.({layer_pat})\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|mlp\.(gate_proj|up_proj|down_proj))$"
    pm = get_peft_model(model, LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM", target_modules=targets))
pm.train()
_idx = sorted({int(n.split("layers.")[1].split(".")[0]) for n, p in pm.named_parameters() if p.requires_grad and "layers." in n})
if _idx and _idx[0] != args.lora_from:
    print(f"警告:可训练 LoRA 实际从第 {_idx[0]} 层开始,与 --lora_from={args.lora_from} 不同,冻结边界改用 {_idx[0]}", flush=True)
    args.lora_from = _idx[0]
base = pm.get_base_model()
if args.lora_from > 0:
    for i, layer in enumerate(base.model.layers):
        if i < args.lora_from:
            layer.gradient_checkpointing = False
            fwd = layer.forward
            def wrapped(*a, _f=fwd, **k):
                with torch.no_grad():
                    return _f(*a, **k)
            layer.forward = wrapped
trainable = [p for p in pm.parameters() if p.requires_grad]
print(f"可训练参数 {sum(p.numel() for p in trainable)/1e6:.1f}M", flush=True)
opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)


def mean_logp(prompt_ids, resp_ids, grad: bool, start: int = 0):
    ids = torch.tensor([prompt_ids + resp_ids], device=dev)
    n_p = len(prompt_ids)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        h = base.model(input_ids=ids, use_cache=False).last_hidden_state[0, n_p - 1 + start:-1]   # 预测响应(从 start 起)各 token 的位置
        y = ids[0, n_p + start:]
        w = base.lm_head.weight
        def f(hc, yc): return -F.cross_entropy((hc @ w.T).float(), yc, reduction="sum")   # = sum logp
        tot = 0.0
        for i in range(0, h.shape[0], args.ce_chunk):
            hc, yc = h[i:i + args.ce_chunk], y[i:i + args.ce_chunk]
            tot = tot + (checkpoint(f, hc, yc, use_reentrant=False) if grad else f(hc, yc))
        return tot / max(len(resp_ids) - start, 1)


steps_per_epoch = max(len(pairs) // args.accum, 1)
total_updates = int(math.ceil(steps_per_epoch * args.epochs))
if args.max_updates: total_updates = min(total_updates, args.max_updates)
log(event="start", pairs=len(pairs), accum=args.accum, sft_lambda=args.sft_lambda, total_updates=total_updates, beta=args.beta, gamma=args.gamma, lr=args.lr)
order = list(range(len(pairs))); random.shuffle(order)
if args.longest_first:  # 把最长的几行放到最前:显存/时间问题在第 1 次更新就暴露,而不是几小时后
    _len = lambda i: len(pairs[i]["prompt_ids"]) + max(len(pairs[i]["chosen_ids"]), len(pairs[i].get("rejected_ids") or []))
    _top = sorted(order, key=_len, reverse=True)[:args.longest_first]
    order = _top + [i for i in order if i not in set(_top)]
cursor = 0; done_updates = 0; attempts = 0
while done_updates < total_updates and attempts < total_updates * 2:
    attempts += 1; upd = done_updates + 1
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time(); margins, losses, accs, aws, als = [], [], [], [], []
    oom = False; anchor_n = 0
    for _ in range(args.accum):
        pr = pairs[order[cursor % len(order)]]; cursor += 1
        cs, rs = (pr["c_start"], pr.get("r_start", 0)) if args.report_only else (0, 0)
        try:
            if not pr["rejected_ids"]:           # 锚点行:一次带梯度前后向,不需要参考/拒绝样本
                lp = mean_logp(pr["prompt_ids"], pr["chosen_ids"], grad=True, start=cs)
                (-(args.anchor_lambda) * lp / args.accum).backward()
                aws.append(float(lp)); anchor_n += 1
                continue
            with torch.no_grad():
                a_w = float(mean_logp(pr["prompt_ids"], pr["chosen_ids"], grad=False, start=cs))
                a_l = float(mean_logp(pr["prompt_ids"], pr["rejected_ids"], grad=False, start=rs))
            m = args.beta * (a_w - a_l) - args.gamma
            s = 1.0 / (1.0 + math.exp(m))          # σ(-m)
            losses.append(math.log1p(math.exp(-m))); margins.append(m); accs.append(float(a_w > a_l)); aws.append(a_w); als.append(a_l)
            (-(args.beta * s + args.sft_lambda) * mean_logp(pr["prompt_ids"], pr["chosen_ids"], grad=True, start=cs) / args.accum).backward()
            ((args.beta * s) * mean_logp(pr["prompt_ids"], pr["rejected_ids"], grad=True, start=rs) / args.accum).backward()
        except torch.cuda.OutOfMemoryError:
            oom = True
            log(event="oom", update=upd, pair_id=pr["pair_id"], tokens=len(pr["prompt_ids"]) + max(len(pr["chosen_ids"]), len(pr["rejected_ids"])))
            break
    if oom:                                      # 梯度可能只累积了一半:丢弃本次更新
        opt.zero_grad(set_to_none=True); torch.cuda.empty_cache(); continue
    gn = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
    opt.step(); opt.zero_grad(set_to_none=True); done_updates += 1
    _m = lambda xs, nd=4: round(sum(xs) / len(xs), nd) if xs else None
    log(event="update", update=upd, loss=_m(losses), margin=_m(margins),
        a_w=_m(aws), a_l=_m(als), anchors=anchor_n,
        pref_acc=_m(accs, 3), grad_norm=round(gn, 3), sec=round(time.time() - t0, 1),
        peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 1))
    if args.save_every and done_updates % args.save_every == 0 and done_updates < total_updates:
        pm.save_pretrained(os.path.join(args.out, f"ckpt_u{done_updates}"))
pm.save_pretrained(os.path.join(args.out, "ckpt"))
log(event="done", updates=done_updates, attempts=attempts, planned=total_updates)
