"""离线估算 n-gram 查表式投机(llama.cpp 的 ngram-simple 一类)在真实输出上的潜力:
对每条学生的首回合输出(思考 + 答案),按贪心接受规则模拟——用“当前上下文(提示 + 已生成部分)的最后 n 个 token”在上下文里找最近一次出现的位置,
把它后面的 m 个 token 当草稿,与真实输出逐个比对,接受最长匹配前缀,然后前进(接受数 + 1)个 token。
输出:有草稿的步数占比、有草稿时平均接受长度、整体每步推进 token 数(没有草稿的步按 1 计),以及思考段/答案段分别的统计。
用法: ngram_potential.py <运行目录,如 harness/runs/live_v3h2_dev12_first> [--n 3] [--m 12] [--tokenizer profiles/assets/qwen3.8-official-1d4bf0f2/tokenizer.json]
注意:提示按“各条消息正文直接拼接”分词(不含聊天模板的少量特殊 token),对 n-gram 命中统计影响很小。
"""
import argparse, collections, glob, json, os, statistics as st

from tokenizers import Tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("run_dir")
ap.add_argument("--n", type=int, default=3)
ap.add_argument("--m", type=int, default=12)
ap.add_argument("--min-hits", type=int, default=1)
ap.add_argument("--quiet", action="store_true", help="只打印合计")
ap.add_argument("--tokenizer", default="profiles/assets/qwen3.8-official-1d4bf0f2/tokenizer.json")
a = ap.parse_args()
tok = Tokenizer.from_file(a.tokenizer)


def enc(s):
    return tok.encode(s, add_special_tokens=False).ids


def simulate(ctx, resp, split, n, m):
    """ctx: 提示 token;resp: 真实输出 token;split: 思考/答案分界在 resp 里的下标。返回按段统计。"""
    pos = collections.defaultdict(list)          # n-gram -> 出现位置(结尾下标+1)
    seq = list(ctx)
    for i in range(n, len(seq) + 1):
        pos[tuple(seq[i - n:i])].append(i)
    stats = {"think": [0, 0, 0, 0, 0], "answer": [0, 0, 0, 0, 0]}   # 步数, 有草稿步数, 草稿接受 token 数, 推进 token 数, 草稿 token 总数
    t = 0
    while t < len(resp):
        seg = "think" if t < split else "answer"
        s = stats[seg]
        s[0] += 1
        key = tuple(seq[-n:])
        cand = [p for p in pos.get(key, []) if p < len(seq)]    # 不含结尾本身
        acc = 0
        if cand:
            p = cand[-1]                                         # 最近一次出现
            draft = seq[p:p + m]
            # 草稿可以延伸到已生成区域之外:此处 seq 只含到当前为止的内容,所以 draft 可能不足 m 个
            for d in draft:
                if t + acc < len(resp) and resp[t + acc] == d:
                    acc += 1
                else:
                    break
            s[1] += 1
            s[2] += acc
            s[4] += len(draft)
        adv = acc + 1
        s[3] += min(adv, len(resp) - t)
        for _ in range(min(adv, len(resp) - t)):
            seq.append(resp[t])
            if len(seq) >= n:
                pos[tuple(seq[-n:])].append(len(seq))
            t += 1
    return stats


tot = {"think": [0, 0, 0, 0, 0], "answer": [0, 0, 0, 0, 0]}
for f in sorted(glob.glob(os.path.join(a.run_dir, "*.json"))):
    t = json.load(open(f, encoding="utf-8"))
    gens = [c for c in t["calls"] if c.get("role") == "generator" and not c.get("not_sent") and c.get("response")]
    if not gens:
        continue
    c = gens[0]
    ctx = []
    for msg in c["payload"]["messages"]:
        if isinstance(msg.get("content"), str):
            ctx += enc(msg["content"])
    think, ans = c.get("reasoning_content") or "", c.get("content") or ""
    rt, at = enc(think), enc(ans)
    st_ = simulate(ctx, rt + at, len(rt), a.n, a.m)
    if not a.quiet: print(os.path.basename(f), "prompt", len(ctx), "think", len(rt), "answer", len(at),
          "| 思考段: 有草稿步占 %.0f%% 接受/草稿步 %.1f" % (100 * st_["think"][1] / max(st_["think"][0], 1), st_["think"][2] / max(st_["think"][1], 1)),
          "| 答案段: 有草稿步占 %.0f%% 接受/草稿步 %.1f" % (100 * st_["answer"][1] / max(st_["answer"][0], 1), st_["answer"][2] / max(st_["answer"][1], 1)))
    for seg in tot:
        for i in range(5):
            tot[seg][i] += st_[seg][i]
print("\n合计(n=%d, m=%d):" % (a.n, a.m))
for seg, v in tot.items():
    steps, with_d, acc, adv, drafted = v
    print(f"  {seg}: 步数 {steps}, 有草稿 {with_d / max(steps, 1):.1%}, 有草稿时平均接受 {acc / max(with_d, 1):.2f} 个(草稿平均 {drafted / max(with_d, 1):.1f} 个,命中率 {acc / max(drafted, 1):.0%}), 整体每步推进 {adv / max(steps, 1):.2f} 个 token")
