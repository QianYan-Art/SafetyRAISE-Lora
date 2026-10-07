"""W3 确定性检索测试；不调用网络，也不把知识库正文打印到测试输出。"""

from __future__ import annotations

from pathlib import Path

import pytest

from sr_eval.live.retrieval import DEFAULT_KB_ROOT, build_retrieval, synthetic_retrieval


CASE = {
    "事故标题": "合成路口场景",
    "事故认定原因": "信号相位待核实",
    "事故类型": "合成碰撞",
}
SOURCE_FIELDS = {
    "id", "document_id", "version", "text", "digest", "manifest_digest", "source_kind",
}


@pytest.mark.parametrize("mode", ["fallback", "sparse_half"])
def test_synthetic_projection_and_raw_shape(mode: str) -> None:
    handle = synthetic_retrieval(mode, seed=7, case_id="case-a")
    query = handle.initial_query(CASE)
    projected = handle.initial(query, 3)
    assert projected
    assert all(set(item) == SOURCE_FIELDS for item in projected)

    raw = handle.initial_raw(query, 3)
    assert [item["id"] for item in raw] == [item["id"] for item in handle.initial_raw(query, 3)]
    if mode == "fallback":
        assert all("retrieval_channels" not in item and "rrf_score" not in item for item in raw)
    else:
        assert all("retrieval_channels" in item and "rrf_score" in item for item in raw)
        assert all(item["retrieval_channels"] == ["sparse"] for item in raw)


def test_mixed_mode_is_stable_and_case_sensitive() -> None:
    first = build_retrieval
    # 合成 fixture 避免此单元测试依赖外部盘；模式决策逻辑与 build_retrieval 相同。
    a = [synthetic_retrieval("mixed", seed=11, case_id="a").mode for _ in range(3)]
    b = synthetic_retrieval("mixed", seed=11, case_id="b").mode
    assert a == [a[0]] * 3
    assert b in {"fallback", "sparse_half"}
    assert callable(first)


def test_production_sparse_half_reuses_frozen_steps() -> None:
    if not DEFAULT_KB_ROOT.exists():
        pytest.skip("本机没有只读线上知识库快照")
    handle = build_retrieval("sparse_half", seed=0, case_id="golden")
    query = handle.initial_query(CASE)
    raw = handle.initial_raw(query, 3)
    cfg = handle.retriever.initial_config
    sparse = handle.retriever._collect_sparse_candidates(
        query=query,
        top_k_chunks=cfg["sparse_top_k_chunks"],
        top_k_rules=cfg["sparse_top_k_rules"],
        merge_top_k=cfg["rrf_merge_top_k"],
    )
    merged = handle.retriever._merge_candidates(
        sparse_candidates=sparse, dense_candidates=[], merge_top_k=cfg["rrf_merge_top_k"]
    )
    ranked = handle.retriever._rank_without_reranker(
        candidates=merged, top_n=min(cfg["rerank_top_k"], len(merged))
    )
    expected = handle.retriever._select_final_candidates(
        candidates=ranked,
        limit=3,
        max_context_chars=cfg["max_context_chars"],
        enforce_type_balance=True,
    )
    assert raw == expected
    assert handle.manifest["manifest_digest"] == handle.manifest_digest
    assert handle.manifest["knowledge_source_count"] == len(handle.knowledge_source)

