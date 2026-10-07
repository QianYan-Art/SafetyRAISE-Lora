"""量化评测台检索与生产检索的差距(只在本地跑,不外发任何内容;只输出计数/比例)。

只读导入生产的 LocalJsonlRetriever(稀疏部分,含同义词/主题索引),对同一批合成案件比较:
  A 评测台自带 BM25(字二元组)——用 8 键换行查询(local_jsonl 的查询形态)
  B 生产稀疏检索——同一 8 键查询(只差检索算法/规则文件)
  C 生产稀疏检索——生产 hybrid 初始查询(HybridRetriever._build_initial_query:只认 8 个键,其中模板里只有 2 个存在)
稠密向量部分需要对查询做嵌入(外部接口),这里不做,只在结论里标明"未量化"。
用法: python harness/tools/prod_retrieval_gap.py [dev|test] [n]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "harness"))
PROD_BACKEND = Path(r"<SafetyRAISE-system-repo>\backend")
sys.path.insert(0, str(PROD_BACKEND))

from sr_eval.cli import PRODUCTION_KB_DIR, load_cases  # noqa: E402
from sr_eval.kb import load_production_kb  # noqa: E402

import types  # noqa: E402

for _m in ('numpy',):  # 评测环境没装 numpy;这里只用 HybridRetriever 的静态查询构造,稠密部分不跑
    sys.modules.setdefault(_m, types.ModuleType(_m))
from app.providers.retrieval.hybrid_retriever import HybridRetriever  # noqa: E402
from app.providers.retrieval.local_jsonl_retriever import LocalJsonlRetriever  # noqa: E402

split = sys.argv[1] if len(sys.argv) > 1 else "dev"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 50
D = Path(PRODUCTION_KB_DIR)
prod = LocalJsonlRetriever(
    manifest_path=D / "manifest.json", chunks_path=D / "kbase_chunks.jsonl", rules_path=D / "liability_rules.jsonl",
    search_index_path=D / "search_index.json", min_score=0.2, top_k_chunks=8, top_k_rules=12, max_context_chars=5200,
    prefer_enhanced_rules=True, enable_search_index=True, watch_manifest_changes=False,
)
INIT_CFG = dict(sparse_top_k_chunks=8, sparse_top_k_rules=12, dense_top_k_chunks=8, dense_top_k_rules=12, rrf_merge_top_k=24,
                rerank_top_k=6, final_context_top_k=6, max_context_chars=5200)
AGENT_CFG = dict(sparse_top_k_chunks=4, sparse_top_k_rules=6, dense_top_k_chunks=4, dense_top_k_rules=6, rrf_merge_top_k=12,
                 rerank_top_k=3, final_context_top_k=3, max_context_chars=5200)
hyb = HybridRetriever(sparse_retriever=prod, embedding_client=None, reranker_client=None, dense_index=None, dense_index_version="",
                      embedding_model="", reranker_model="", initial_config=INIT_CFG, agentic_config=AGENT_CFG)


def hybrid_sparse_half(query, cfg, balance, limit):
    """生产 hybrid 流程把稠密候选置空后的结果(用生产自己的方法:稀疏分流、RRF、类型平衡、字数预算)。"""
    sp = hyb._collect_sparse_candidates(query=query, top_k_chunks=cfg["sparse_top_k_chunks"], top_k_rules=cfg["sparse_top_k_rules"], merge_top_k=cfg["rrf_merge_top_k"])
    merged = hyb._merge_candidates(sparse_candidates=sp, dense_candidates=[], merge_top_k=cfg["rrf_merge_top_k"])
    ranked = hyb._rank_without_reranker(candidates=merged, top_n=min(cfg["rerank_top_k"], len(merged)))
    return hyb._select_final_candidates(candidates=ranked, limit=limit, max_context_chars=cfg["max_context_chars"], enforce_type_balance=balance)


print("生产稀疏检索实际规则文件:", prod.rules_path.name, "| search_index_loaded:", prod.metadata.get("search_index_loaded"))
kb_plain = load_production_kb(PRODUCTION_KB_DIR, enhanced_rules=False)
kb_enh = load_production_kb(PRODUCTION_KB_DIR, enhanced_rules=True)
ids_plain = {r.get("rule_id") for r in kb_plain.rules}
ids_enh = {r.get("rule_id") for r in kb_enh.rules}
print(f"规则条数 普通 {len(ids_plain)} / 增强 {len(ids_enh)};id 交集 {len(ids_plain & ids_enh)};仅普通 {len(ids_plain - ids_enh)};仅增强 {len(ids_enh - ids_plain)}")

K = 6


def ids(rs):
    return [str(r.get("id")) for r in rs]


def jacc(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(len(a | b), 1)


rows = []
for c in load_cases(split, n):
    ad = c["accident_data"]
    q8 = kb_enh.retrieve(ad, K)[0]
    qp = HybridRetriever._build_initial_query(ad)
    a = ids(kb_enh.search(q8, K))                   # A 评测台 BM25(增强规则文件)
    a_plain = ids(kb_plain.search(q8, K))           # A' 评测台默认(普通规则文件,与现行 run 一致)
    b = ids(prod.retrieve(ad, K))                   # B 生产 local_jsonl 路径(也是生产在稠密不可用时的降级路径),查询与评测台相同
    cc = ids(hybrid_sparse_half(qp, INIT_CFG, True, K)) if qp else []   # C 生产 hybrid 的稀疏半边:hybrid 初始查询 + 类型平衡
    rows.append(dict(case=c["case_id"], qlen8=len(q8), qlenp=len(qp), a=a, a_plain=a_plain, b=b, c=cc))


def mean(xs):
    return sum(xs) / max(len(xs), 1)


print(f"案件 {len(rows)}({split});top-{K} 片段 id 的 Jaccard 均值")
print(f"  评测台默认(普通规则)  vs 评测台(增强规则): {mean([jacc(r['a_plain'], r['a']) for r in rows]):.3f}")
print(f"  评测台 BM25 vs 生产稀疏(同一 8 键查询):   {mean([jacc(r['a'], r['b']) for r in rows]):.3f}")
print(f"  评测台 BM25(8 键) vs 生产稀疏(hybrid 查询): {mean([jacc(r['a'], r['c']) for r in rows]):.3f}")
print(f"  生产稀疏(8 键) vs 生产稀疏(hybrid 查询):    {mean([jacc(r['b'], r['c']) for r in rows]):.3f}")
print(f"  评测台默认(普通规则) vs 生产稀疏(hybrid 查询): {mean([jacc(r['a_plain'], r['c']) for r in rows]):.3f}")
print(f"  查询字数 8 键均值 {mean([r['qlen8'] for r in rows]):.0f};hybrid 初始查询均值 {mean([r['qlenp'] for r in rows]):.0f}(最短 {min(r['qlenp'] for r in rows)})")
print(f"  hybrid 初始查询为空的案件数: {sum(1 for r in rows if not r['c'])}")
types = lambda rs: sum(1 for i in rs if i.startswith("rule") or "rule" in i[:6])
# ---- 补充检索(agentic):用 run 里模型实际发出的查询,比较评测台返回与生产稀疏半边(agentic 配置:4 块+6 规则 → 取 3 条)
import glob
run_dirs = sorted(glob.glob(str(ROOT / "harness" / "runs" / "*_lunamax_dev")))
ag = []
if run_dirs:
    for f in sorted(glob.glob(run_dirs[-1] + "/traces/*.json")):
        t = json.loads(Path(f).read_text(encoding="utf-8"))
        for r in t["rounds"]:
            q = (r["query"] or "")[:120]
            h = [str(x.get("id")) for x in r["snippets"]]
            p3 = ids(hybrid_sparse_half(q, AGENT_CFG, False, 3))
            ag.append((jacc(h, p3), len(set(h) & set(p3))))
    print(f"补充检索 {len(ag)} 次(来自 {Path(run_dirs[-1]).name}):评测台返回 vs 生产稀疏半边 Jaccard 均值 {mean([a for a,_ in ag]):.3f};完全无交集占 {sum(1 for _,k in ag if k==0)/max(len(ag),1):.0%}")

out = ROOT / "harness" / "reports" / f"prod-retrieval-gap-{split}.json"
out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
print("明细(仅片段 id)已写入", out.relative_to(ROOT))
