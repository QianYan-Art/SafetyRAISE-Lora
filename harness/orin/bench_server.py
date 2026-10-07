"""对 llama-server 做真实提示的速度基准(在笔记本上运行,走局域网明文 HTTP)。

用教师轨迹里真实的调用输入(~1 万 token 的 system+user,带/不带工具),测:
  cold  : 首次请求 = 完整预填充速度 + 首段解码
  warm  : 同一前缀再发一次(只改末尾) = 前缀/检查点复用后的预填充
  decode: 长输出解码速度(含投机解码接受率,如果服务端返回 timings)
结果只含速度与计数,不含任何正文。
"""
import argparse, glob, json, sys, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="192.168.55.1:8080")
ap.add_argument("--trace", default=None, help="教师 trace JSON;默认取 runs/*_lunadev 里 syn_dev_006")
ap.add_argument("--call", type=int, default=-1, help="trace.calls 的下标(-1 = 最终报告调用)")
ap.add_argument("--max-tokens", type=int, default=600)
ap.add_argument("--effort", default="low")
ap.add_argument("--label", default="")
args = ap.parse_args()

path = args.trace or glob.glob("<repo>/harness/runs/*_lunadev/traces/syn_dev_006__luna.json")[0]
call = json.load(open(path, encoding="utf-8"))["calls"][args.call]
msgs = call["messages"]
tools = None
if call["with_tools"]:
    sys.path.insert(0, "<repo>/harness")
    from sr_eval import prompt as P
    tools = P.retrieve_tool_schema()


OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # 局域网直连,不走环境代理


def post(messages, max_tokens, seed=1):
    body = {"model": "x", "messages": messages, "max_tokens": max_tokens, "temperature": 0.0, "seed": seed, "stream": False,
            "cache_prompt": True, "chat_template_kwargs": {"reasoning_effort": args.effort}}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(f"http://{args.host}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with OPENER.open(req, timeout=3600) as r:
        d = json.loads(r.read())
    return d, time.time() - t0


def line(tag, d, wall):
    t = d.get("timings", {}); u = d.get("usage", {})
    spec = f" draft={t.get('draft_n')} acc={t.get('draft_n_accepted')}" if t.get("draft_n") is not None else ""
    print(f"[{args.label}] {tag:7s} wall={wall:6.1f}s prompt_tok={u.get('prompt_tokens')} cached={t.get('cache_n')} "
          f"pp={t.get('prompt_per_second', 0):7.1f} t/s ({t.get('prompt_n')} tok, {t.get('prompt_ms', 0)/1000:5.1f}s) "
          f"tg={t.get('predicted_per_second', 0):5.2f} t/s ({t.get('predicted_n')} tok){spec}", flush=True)


d, w = post(msgs, args.max_tokens); line("cold", d, w)
m2 = [dict(m) for m in msgs]; m2[-1] = dict(m2[-1]); m2[-1]["content"] += "\n(复核)"
d, w = post(m2, 16); line("warm", d, w)
d, w = post(msgs, args.max_tokens); line("decode", d, w)
