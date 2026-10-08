"""为 MTP 草稿头生成"词表子集"(借鉴 ninfer 的 proposal head):按领域语料 + 通用词频先验给词表排序,取前 K 个 token id。
草稿只需要猜"下一个最可能的 token",猜不到子集外的 token 只是少一次命中,验证用的主模型头仍是完整词表,所以不影响输出,只省草稿阶段读 lm_head 的字节。

输入:
  --tokenizer  tokenizer.json(与模型同一份)
  --prior      通用词频先验(ninfer 公开的 ranking.*.counts.i64,int64 小端,第 0 行=全词表总计数);可不给
  --mtp-data   MTP 训练/留出 jsonl(取 labels != -100 的位置,即模型自己输出的 token),可多个
  --traces     线上环境运行目录(每个 json 的 calls[*].reasoning_content / content),可多个
输出:
  --out-prefix 生成 <prefix>_k<K>.i32(int32 小端 id 列表)与 <prefix>_report.json(交叉验证覆盖率)
覆盖率口径:留出案件里"模型实际输出的 token"落在子集内的比例(与 ninfer 的 argmax 覆盖口径同类;这里用的是线上协议下学生模型自己采样出的 token)。
"""
import argparse, collections, glob, json, os, random, sys

import numpy as np
from tokenizers import Tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--tokenizer", required=True)
ap.add_argument("--prior", default=None)
ap.add_argument("--mtp-data", nargs="*", default=[])
ap.add_argument("--traces", nargs="*", default=[])
ap.add_argument("--vocab", type=int, default=248320)
ap.add_argument("--ks", type=int, nargs="+", default=[16384, 24576, 32768, 49152, 65536])
ap.add_argument("--prior-weight", type=float, nargs="+", default=[0.0, 0.1, 0.3, 1.0])
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--final-weight", type=float, default=None, help="最终排序用的先验权重(缺省取交叉验证里 K=32768 覆盖率最高者)")
ap.add_argument("--out-prefix", required=True)
a = ap.parse_args()

tok = Tokenizer.from_file(a.tokenizer)
V = a.vocab

# ---- 领域语料: 按"案件"分组的 token 计数 ----
docs = {}  # group -> Counter


def add(group, ids):
    c = docs.setdefault(group, collections.Counter())
    c.update(ids)


for fn in a.mtp_data:
    for line in open(fn, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        ids = [t for t in r["labels"] if t != -100]
        add("mtp:" + str(r.get("case_id") or r.get("sample_id")), ids)

n_calls = 0
for d in a.traces:
    for fn in sorted(glob.glob(os.path.join(d, "*.json"))):
        try:
            t = json.load(open(fn, encoding="utf-8"))
        except Exception:
            continue
        for c in t.get("calls", []):
            if c.get("role") != "generator" or c.get("not_sent"):
                continue
            think, ans = c.get("reasoning_content") or "", c.get("content") or ""
            if not (think or ans):
                continue
            ids = tok.encode(think).ids + tok.encode("\n</think>\n\n").ids + tok.encode(ans).ids + tok.encode("<|im_end|>").ids
            add("trace:" + os.path.basename(fn), ids)
            n_calls += 1

total_dom = collections.Counter()
for c in docs.values():
    total_dom.update(c)
n_tok = sum(total_dom.values())
print(f"领域语料: {len(docs)} 组, {n_calls} 个 trace 调用, {n_tok} 个 token, 不同 token {len(total_dom)}")

prior = np.zeros(V, dtype=np.float64)
if a.prior:
    prior = np.fromfile(a.prior, dtype="<i8", count=V).astype(np.float64)
    print(f"先验: 总计数 {prior.sum():.3g}, 非零 {int((prior > 0).sum())}")

# 特殊 token(added tokens)一律纳入
special = sorted(i for i in (tok.get_added_tokens_decoder() or {}).keys() if 0 <= i < V)
print("特殊/新增 token 数:", len(special))


def dom_vec(counters):
    v = np.zeros(V, dtype=np.float64)
    for c in counters:
        for k, n in c.items():
            if 0 <= k < V:
                v[k] += n
    return v


def rank(dom, w):
    ds = dom / max(dom.sum(), 1.0)
    ps = prior / max(prior.sum(), 1.0)
    score = ds + w * ps
    return np.argsort(-score, kind="stable")


def pick(order, K, forced):
    forced = list(dict.fromkeys(forced))
    chosen = set(forced)
    out = list(forced)
    for i in order.tolist():
        if len(out) >= K:
            break
        if i not in chosen:
            chosen.add(i)
            out.append(i)
    return np.array(sorted(out[:K]), dtype=np.int64)


# ---- 交叉验证 ----
groups = sorted(docs)
random.Random(7).shuffle(groups)
folds = [groups[i::a.folds] for i in range(a.folds)]
cv = {}
for w in a.prior_weight:
    res = {K: [] for K in a.ks}
    for f in range(a.folds):
        held = set(folds[f])
        dom_tr = dom_vec([docs[g] for g in groups if g not in held])
        dom_te = dom_vec([docs[g] for g in held])
        order = rank(dom_tr, w)
        for K in a.ks:
            ids = pick(order, K, special)
            mask = np.zeros(V, dtype=bool)
            mask[ids] = True
            res[K].append(float(dom_te[mask].sum() / dom_te.sum()))
    cv[w] = {K: float(np.mean(v)) for K, v in res.items()}
    print(f"先验权重 {w}: " + "  ".join(f"K={K}:{cv[w][K]:.4%}" for K in a.ks))

best_w = a.final_weight
if best_w is None:
    ref = 32768 if 32768 in a.ks else a.ks[len(a.ks) // 2]
    best_w = max(cv, key=lambda w: cv[w][ref])
print("最终排序用先验权重:", best_w)

dom_all = dom_vec(docs.values())
order = rank(dom_all, best_w)
report = {"groups": len(docs), "calls": n_calls, "tokens": n_tok, "distinct_tokens": len(total_dom), "special": len(special),
          "cv_folds": a.folds, "cv_coverage": {str(w): {str(K): v for K, v in d.items()} for w, d in cv.items()},
          "final_prior_weight": best_w, "outputs": {}}
for K in a.ks:
    ids = pick(order, K, special)
    path = f"{a.out_prefix}_k{K}.i32"
    ids.astype("<i4").tofile(path)
    in_dom = int(sum(1 for i in ids.tolist() if i in total_dom))
    all_cov = float(sum(n for k, n in total_dom.items() if k in set(ids.tolist())) / n_tok)
    report["outputs"][str(K)] = {"file": path, "domain_tokens_included": in_dom, "coverage_on_all_domain_tokens": all_cov}
    print(f"写出 {path}: {len(ids)} 个 id, 含领域词 {in_dom}, 对全部领域语料覆盖 {all_cov:.4%}")
json.dump(report, open(a.out_prefix + "_report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
