"""Orin 上的 27B QLoRA SFT:预量化 bnb-4bit 基座 + LoRA,支持"仅上半层适配"(下半层 no_grad)、断点续训。

数据:harness `sft-build` 导出的 train.jsonl(input_ids/labels,labels 只含助手目标)。
损失:样本内对目标 token 取平均,再对一次更新里的样本取平均(工具调用短样本与长报告权重相同)。
用法见 README(harness/orin)。只做训练,不做推理评测。
"""
import argparse, json, math, os, random, re, sys, time

os.environ.setdefault("CUDA_CACHE_MAXSIZE", "4294967296")
import torch, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.path.expanduser("~/models/Qwen3.8-27B-unsloth-bnb-4bit"))
ap.add_argument("--data", required=True)
ap.add_argument("--eval_data", default=None)
ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=float, default=1.0)
ap.add_argument("--accum", type=int, default=8, help="每次更新的样本数")
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--warmup", type=float, default=0.05)
ap.add_argument("--r", type=int, default=16)
ap.add_argument("--alpha", type=int, default=32)
ap.add_argument("--dropout", type=float, default=0.05)
ap.add_argument("--lora_from", type=int, default=32, help="只在 >= 该层号的解码层加适配器;0 = 全层")
ap.add_argument("--max_len", type=int, default=24576)
ap.add_argument("--mem_frac", type=float, default=0.90)
ap.add_argument("--ce_chunk", type=int, default=1024)
ap.add_argument("--save_every", type=int, default=5, help="每 N 次更新存一次检查点")
ap.add_argument("--eval_every", type=int, default=10)
ap.add_argument("--eval_n", type=int, default=8)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max_updates", type=int, default=0, help="调试用:限制更新次数")
ap.add_argument("--report_weight", type=float, default=1.0, help="报告段(</think> 之后)token 的损失权重;行里有 report_start 才生效。思考段权重恒为 1")
ap.add_argument("--think_weight", type=float, default=1.0, help="思考段(报告开始之前的响应 token)的损失权重;学生自己生成的思考梯度多为噪声,压低它可让信号集中在报告上")
ap.add_argument("--keep_every", type=int, default=0, help="每 N 次更新另存一份 ckpt_u<N>(默认 0=只保留滚动的 ckpt)")
ap.add_argument("--init_adapter", default=None, help="从已有适配器接着训(如 v3a 的 ckpt);已有本目录检查点时以检查点续训为准")
ap.add_argument("--stop_temp", type=float, default=92.0, help="SoC 温度超过则保存并退出(°C)")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
dev = "cuda"
torch.cuda.set_per_process_memory_fraction(args.mem_frac)
random.seed(args.seed); torch.manual_seed(args.seed)

def log(**kw):
    kw["t"] = round(time.time(), 1)
    line = json.dumps(kw, ensure_ascii=False)
    print(line, flush=True)
    with open(os.path.join(args.out, "log.jsonl"), "a") as fh: fh.write(line + "\n")

def soc_temp():
    try:
        return max(int(open(f"/sys/devices/virtual/thermal/thermal_zone{i}/temp").read()) / 1000 for i in range(8))
    except Exception:
        return -1.0

def load_rows(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    return [r for r in rows if len(r["input_ids"]) <= args.max_len]

train = load_rows(args.data)
evals = load_rows(args.eval_data)[: args.eval_n] if args.eval_data else []
print(f"训练样本 {len(train)},评估样本 {len(evals)},总 token {sum(len(r['input_ids']) for r in train)}", flush=True)

# ---------- 模型 ----------
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
print(f"加载完成 {type(model).__name__} 层数 {n_layers} alloc {torch.cuda.memory_allocated()/2**30:.1f}GiB", flush=True)

from peft import LoraConfig, get_peft_model, PeftModel
layer_pat = "|".join(str(i) for i in range(args.lora_from, n_layers))
targets = rf".*layers\.({layer_pat})\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|mlp\.(gate_proj|up_proj|down_proj))$"
ckpt = os.path.join(args.out, "ckpt")
state_path = os.path.join(args.out, "state.json")
state = {"update": 0, "cursor": 0, "epoch": 0}
if args.lora_from == 0:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.enable_input_require_grads()
else:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
if os.path.exists(state_path) and os.path.exists(os.path.join(ckpt, "adapter_config.json")):
    state = json.load(open(state_path))
    pm = PeftModel.from_pretrained(model, ckpt, is_trainable=True)
    print(f"从检查点续训:update={state['update']} cursor={state['cursor']}", flush=True)
elif args.init_adapter:
    pm = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    print(f"从已有适配器接着训:{args.init_adapter}", flush=True)
else:
    pm = get_peft_model(model, LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout, bias="none",
                                          task_type="CAUSAL_LM", target_modules=targets))
pm.train()
_idx = sorted({int(n.split("layers.")[1].split(".")[0]) for n, p in pm.named_parameters() if p.requires_grad and "layers." in n})
if _idx and args.lora_from > 0 and _idx[0] != args.lora_from:   # 冻结边界按适配器实际覆盖的层定,避免 no_grad 误伤可训练层
    print(f"警告:可训练 LoRA 实际从第 {_idx[0]} 层开始,与 --lora_from={args.lora_from} 不同,冻结边界改用 {_idx[0]}", flush=True)
    args.lora_from = _idx[0]
base = pm.get_base_model()
if args.lora_from > 0:   # 下半层:不存激活、不反传
    for i, layer in enumerate(base.model.layers):
        if i < args.lora_from:
            layer.gradient_checkpointing = False
            fwd = layer.forward
            def wrapped(*a, _f=fwd, **k):
                with torch.no_grad():
                    return _f(*a, **k)
            layer.forward = wrapped
trainable = [p for p in pm.parameters() if p.requires_grad]
n_train = sum(p.numel() for p in trainable)
print(f"可训练参数 {n_train/1e6:.1f}M(lora_from={args.lora_from})", flush=True)
opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.999))
opt_path = os.path.join(args.out, "opt.pt")
if os.path.exists(opt_path) and state["update"] > 0:
    opt.load_state_dict(torch.load(opt_path, map_location="cuda"))

def sample_loss(row, grad: bool):
    ids = torch.tensor([row["input_ids"]], device=dev)
    labels = torch.tensor(row["labels"][1:], device=dev)
    h = base.model(input_ids=ids, use_cache=False).last_hidden_state[0, :-1]
    w = base.lm_head.weight
    n = (labels != -100).sum().clamp(min=1)
    wt = torch.ones(labels.shape[0], device=dev)
    rs = row.get("report_start")
    if rs is not None and (args.report_weight != 1.0 or args.think_weight != 1.0):
        wt[:] = args.think_weight                          # 标签 j 预测的是 input_ids[j+1]
        wt[max(rs - 1, 0):] = args.report_weight
    wsum = (wt * (labels != -100)).sum().clamp(min=1e-6)
    def f(hc, yc, wc): return (F.cross_entropy((hc @ w.T).float(), yc, ignore_index=-100, reduction="none") * wc).sum()
    total = 0.0
    for i in range(0, h.shape[0], args.ce_chunk):
        yc = labels[i:i + args.ce_chunk]
        if (yc != -100).any():
            hc, wc = h[i:i + args.ce_chunk], wt[i:i + args.ce_chunk]
            total = total + (checkpoint(f, hc, yc, wc, use_reentrant=False) if grad else f(hc, yc, wc))
    return total / wsum, int(n)

def run_eval():
    pm.eval(); losses = []
    with torch.no_grad():
        for r in evals:
            l, _ = sample_loss(r, grad=False); losses.append(float(l))
    pm.train()
    return sum(losses) / max(len(losses), 1)

def save(tag=""):
    pm.save_pretrained(ckpt)
    torch.save(opt.state_dict(), opt_path)
    json.dump(state, open(state_path, "w"))
    if args.keep_every and state["update"] % args.keep_every == 0 and state["update"] > 0:
        pm.save_pretrained(os.path.join(args.out, f"ckpt_u{state['update']}"))
    log(event="checkpoint", update=state["update"], cursor=state["cursor"], tag=tag)

# ---------- 训练循环 ----------
steps_per_epoch = max(len(train) // args.accum, 1)
total_updates = int(math.ceil(steps_per_epoch * args.epochs))
if args.max_updates: total_updates = min(total_updates, args.max_updates)
warm = max(int(total_updates * args.warmup), 1)
def lr_at(u):
    if u < warm: return args.lr * (u + 1) / warm
    prog = (u - warm) / max(total_updates - warm, 1)
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))
orders = {}
def order(ep):
    if ep not in orders:
        idx = list(range(len(train))); random.Random(args.seed + ep).shuffle(idx); orders[ep] = idx
    return orders[ep]

log(event="start", samples=len(train), accum=args.accum, total_updates=total_updates, lr=args.lr, r=args.r,
    lora_from=args.lora_from, max_len=args.max_len)
if evals and state["update"] == 0:
    log(event="eval", update=0, eval_loss=round(run_eval(), 4))
t_start = time.time()
while state["update"] < total_updates:
    u = state["update"]
    for g in opt.param_groups: g["lr"] = lr_at(u)
    t0 = time.time(); losses, toks = [], 0
    for _ in range(args.accum):
        ep = state["cursor"] // len(train)
        row = train[order(ep)[state["cursor"] % len(train)]]
        state["cursor"] += 1
        loss, n = sample_loss(row, grad=True)
        (loss / args.accum).backward()
        losses.append(float(loss)); toks += len(row["input_ids"])
    gn = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
    opt.step(); opt.zero_grad(set_to_none=True)
    state["update"] += 1
    dt = time.time() - t0
    log(event="update", update=state["update"], loss=round(sum(losses) / len(losses), 4), grad_norm=round(gn, 3),
        lr=round(lr_at(u), 8), tokens=toks, sec=round(dt, 1), tok_s=round(toks / dt, 1),
        peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 1), temp=soc_temp(), epoch=state["cursor"] / len(train))
    if evals and state["update"] % args.eval_every == 0:
        log(event="eval", update=state["update"], eval_loss=round(run_eval(), 4))
    if state["update"] % args.save_every == 0:
        save()
    if soc_temp() > args.stop_temp:
        save("overheat"); log(event="stop_overheat", temp=soc_temp()); sys.exit(3)
save("final")
if evals: log(event="eval", update=state["update"], eval_loss=round(run_eval(), 4))
log(event="done", updates=state["update"], hours=round((time.time() - t_start) / 3600, 2))
