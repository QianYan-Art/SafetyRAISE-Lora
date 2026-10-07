"""Orin 本地推理服务(评测用):预量化 27B + 可选 LoRA 适配器,OpenAI 兼容 /v1/chat/completions。

渲染用与训练相同的官方 Qwen3.8 模板(chat_template.jinja,tojson 同 transformers),支持 tools 与
chat_template_kwargs.reasoning_effort;把输出拆成 reasoning_content / content / tool_calls。
请求按时间窗凑批(最多 --max_batch 条,左填充一起 generate;27B bnb 解码约 2.6 token/s/步,批越大总吞吐越高);只绑定指定地址,默认仅 USB 网口地址。仅用于评测台,不是生产服务。
"""
import argparse, json, os, re, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from jinja2.sandbox import ImmutableSandboxedEnvironment
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.path.expanduser("~/models/Qwen3.8-27B-unsloth-bnb-4bit"))
ap.add_argument("--adapter", default=None)
ap.add_argument("--template", default=os.path.expanduser("~/work/data/chat_template.jinja"))
ap.add_argument("--host", default="192.168.55.1")
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--mem_frac", type=float, default=0.92)
ap.add_argument("--max_batch", type=int, default=6)
ap.add_argument("--wait", type=float, default=4.0, help="凑批等待秒数")
args = ap.parse_args()
torch.cuda.set_per_process_memory_fraction(args.mem_frac)

tok = AutoTokenizer.from_pretrained(args.model)
cfg = AutoConfig.from_pretrained(args.model)
qc = getattr(cfg, "quantization_config", None)
if isinstance(qc, dict): qc["bnb_4bit_compute_dtype"] = "bfloat16"
elif qc is not None: qc.bnb_4bit_compute_dtype = torch.bfloat16
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map={"": 0}, config=cfg)
for p in model.parameters():
    if p.dtype == torch.float16: p.data = p.data.to(torch.bfloat16)
for b in model.buffers():
    if b.dtype == torch.float16: b.data = b.data.to(torch.bfloat16)
if args.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter)
model.eval()
env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
env.filters["tojson"] = lambda v, ensure_ascii=False, indent=None, separators=None, sort_keys=False: json.dumps(v, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
def _raise(m): raise ValueError(m)
env.globals["raise_exception"] = _raise
tpl = env.from_string(open(args.template, encoding="utf-8").read())
import queue
Q = queue.Queue()
tok.padding_side = 'left'
EOS_IDS = [tok.convert_tokens_to_ids("<|im_end|>")]
print("ready", args.adapter or "base", flush=True)

TOOL_RE = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>", re.DOTALL)
PARAM_RE = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.DOTALL)

def parse_value(v: str, schema: dict):
    t = (schema or {}).get("type")
    if t == "integer":
        try: return int(v.strip())
        except ValueError: return v
    if t in ("number",):
        try: return float(v.strip())
        except ValueError: return v
    if t in ("object", "array", "boolean"):
        try: return json.loads(v)
        except ValueError: return v
    return v

def parse_output(text: str, tools):
    # 生成提示已以 <think> 加换行结尾,输出从思考正文开始;没有 </think> 说明思考被截断,不产出正文
    if "</think>" in text:
        reasoning, content = text.split("</think>", 1)
    else:
        reasoning, content = text, ""
    reasoning = reasoning.replace("<think>", "").strip()
    schemas = {t["function"]["name"]: t["function"].get("parameters", {}).get("properties", {}) for t in (tools or [])}
    calls = []
    for m in TOOL_RE.finditer(content):
        name = m.group(1).strip()
        args_ = {k.strip(): parse_value(v, schemas.get(name, {}).get(k.strip(), {})) for k, v in PARAM_RE.findall(m.group(2))}
        calls.append({"id": f"call_{len(calls)}", "type": "function", "function": {"name": name, "arguments": json.dumps(args_, ensure_ascii=False)}})
    if calls:
        content = TOOL_RE.sub("", content)
    return reasoning, content.strip(), calls

def render(body: dict) -> str:
    kwargs = body.get("chat_template_kwargs") or {}
    effort = kwargs.get("reasoning_effort") or (body.get("reasoning") or {}).get("effort") or "low"
    if effort not in ("xhigh", "medium", "low"): effort = "low"
    return tpl.render(messages=body["messages"], tools=body.get("tools"), add_generation_prompt=True,
                      enable_thinking=True, reasoning_effort=effort)

def run_batch(items):
    bodies = [it["body"] for it in items]
    prompts = [render(b) for b in bodies]
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    max_new = max(int(b.get("max_tokens") or b.get("max_completion_tokens") or 4096) for b in bodies)
    temp = bodies[0].get("temperature")
    gen = dict(max_new_tokens=max_new, eos_token_id=EOS_IDS, pad_token_id=EOS_IDS[0])
    if temp is None or temp > 0:
        gen.update(do_sample=True, temperature=temp or 0.6, top_p=0.95, top_k=20)
    else:
        gen.update(do_sample=False)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**enc, **gen)
    dt = time.time() - t0
    results = []
    for i, body in enumerate(bodies):
        new = out[i, enc.input_ids.shape[1]:]
        eos_pos = (new == EOS_IDS[0]).nonzero()
        n_new = int(eos_pos[0]) if len(eos_pos) else int(new.shape[0])
        limit = int(body.get("max_tokens") or body.get("max_completion_tokens") or 4096)
        text = tok.decode(new[:n_new], skip_special_tokens=False)
        reasoning, content, calls = parse_output(text, body.get("tools"))
        finish = "tool_calls" if calls else ("length" if (not len(eos_pos) or n_new >= limit) else "stop")
        msg = {"role": "assistant", "content": content, "reasoning_content": reasoning}
        if calls: msg["tool_calls"] = calls
        p_tok = int(enc.attention_mask[i].sum())
        results.append({"id": f"local-{int(t0)}-{i}", "object": "chat.completion", "model": "qwen3.8-27b-local",
                        "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                        "usage": {"prompt_tokens": p_tok, "completion_tokens": n_new, "cost": 0.0,
                                  "completion_tokens_details": {"reasoning_tokens": len(tok(reasoning).input_ids) if reasoning else 0}}})
    print(f"[batch] n={len(items)} padded_prompt={enc.input_ids.shape[1]} steps={out.shape[1]-enc.input_ids.shape[1]} {dt:.0f}s "
          f"new={[r['usage']['completion_tokens'] for r in results]}", flush=True)
    return results

def batcher():
    while True:
        first = Q.get()
        items = [first]; deadline = time.time() + args.wait
        while len(items) < args.max_batch and time.time() < deadline:
            try: items.append(Q.get(timeout=max(deadline - time.time(), 0.01)))
            except queue.Empty: break
        try:
            for it, res in zip(items, run_batch(items)):
                it["result"] = (200, res); it["done"].set()
        except Exception as e:  # noqa
            torch.cuda.empty_cache()
            for it in items:
                it["result"] = (500, {"error": {"message": f"{type(e).__name__}: {e}"}}); it["done"].set()

threading.Thread(target=batcher, daemon=True).start()

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); body = json.loads(self.rfile.read(n))
        item = {"body": body, "done": threading.Event(), "result": None}
        Q.put(item); item["done"].wait()
        code, resp = item["result"]
        data = json.dumps(resp, ensure_ascii=False).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(data)))
        self.end_headers(); self.wfile.write(data)

ThreadingHTTPServer((args.host, args.port), H).serve_forever()
