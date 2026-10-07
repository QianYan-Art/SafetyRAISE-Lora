"""线上报告环境的确定性检索近似。

本模块只读取冻结知识库，不复制知识库正文。``fallback`` 是线上稠密依赖不可用时
的 ``sparse_only_fallback``，``sparse_half`` 则把冻结生产 ``HybridRetriever`` 的
稠密候选明确置空，仍使用生产的稀疏分流、RRF、类型平衡和字数预算。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Iterable

from app.report_harness.contracts import canonical_digest
from app.report_harness.knowledge_assets import source_text
from app.providers.retrieval.hybrid_retriever import HybridRetriever
from app.providers.retrieval.local_jsonl_retriever import LocalJsonlRetriever


DEFAULT_KB_ROOT = Path(r"<SafetyRAISE-knowledge-repo>\kbase\data")
_MANIFEST_CACHE: dict[tuple[str, tuple[tuple[str, int, int], ...]], dict[str, Any]] = {}

_INITIAL_CONFIG = {
    "sparse_top_k_chunks": 8,
    "sparse_top_k_rules": 12,
    "dense_top_k_chunks": 8,
    "dense_top_k_rules": 12,
    "rrf_merge_top_k": 24,
    "rerank_top_k": 6,
    "final_context_top_k": 6,
    "max_context_chars": 5200,
}
_AGENTIC_CONFIG = {
    "sparse_top_k_chunks": 4,
    "sparse_top_k_rules": 6,
    "dense_top_k_chunks": 4,
    "dense_top_k_rules": 6,
    "rrf_merge_top_k": 12,
    "rerank_top_k": 3,
    "final_context_top_k": 3,
    "max_context_chars": 5200,
}


class _EmptyEmbedding:
    """只为进入冻结 HybridRetriever 的正常 RRF 分支；不产生外部调用。"""

    def embed_query(self, query: str) -> list[float]:
        if not str(query).strip():
            raise ValueError("嵌入查询不能为空")
        return [0.0]


class _EmptyDenseIndex:
    version = "sparse-half-empty-dense-v1"

    def search(self, *, query_vector: Any, top_k_chunks: int, top_k_rules: int) -> list[dict[str, Any]]:
        del query_vector, top_k_chunks, top_k_rules
        return []


def _file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl_shape(path: Path) -> tuple[int | None, list[str]]:
    count = 0
    fields: set[str] = set()
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                item = json.loads(line)
                if isinstance(item, dict):
                    count += 1
                    fields.update(str(key) for key in item)
        return count, sorted(fields)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None, []


def _manifest_for_root(root: Path) -> dict[str, Any]:
    """只登记路径、哈希和计数；不把知识库正文写进项目。"""
    paths = sorted(path for path in root.iterdir() if path.is_file())
    stat_key = tuple((path.name, path.stat().st_size, path.stat().st_mtime_ns) for path in paths)
    cache_key = (str(root), stat_key)
    cached = _MANIFEST_CACHE.get(cache_key)
    if cached is not None:
        return deepcopy(cached)

    files: list[dict[str, Any]] = []
    for path in paths:
        record_count = None
        fields: list[str] = []
        if path.suffix == ".jsonl":
            record_count, fields = _jsonl_shape(path)
        files.append({
            "name": path.name,
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": _file_digest(path),
            "record_count": record_count,
            "fields": fields,
        })

    approved_names = {
        "manifest": "manifest.json",
        "chunks": "kbase_chunks.jsonl",
        "rules": "liability_rules_enhanced.jsonl"
        if (root / "liability_rules_enhanced.jsonl").exists()
        else "liability_rules.jsonl",
        "search_index": "search_index.json",
        "dense_manifest": "dense_manifest.json",
        "dense_records": "dense_records.jsonl",
        "dense_vectors": "dense_vectors.f16.npy",
    }
    by_name = {item["name"]: item for item in files}
    # 稠密文件在本近似中不会被读取；小型测试知识库可只提供四个稀疏资产。
    required_sparse = {
        approved_names["manifest"], approved_names["chunks"],
        approved_names["rules"], approved_names["search_index"],
    }
    missing = [name for name in required_sparse if name not in by_name]
    if missing:
        raise FileNotFoundError(f"知识库缺少冻结资产: {', '.join(missing)}")
    content_digest = canonical_digest({
        key: (by_name[name]["sha256"] if name in by_name else None)
        for key, name in approved_names.items()
    })
    manifest = {
        "schema_version": 1,
        "root": str(root.resolve()),
        "files": files,
        "file_count": len(files),
        "approved_assets": approved_names,
        "content_digest": content_digest,
        "known_approximation": "未加载嵌入服务；sparse_half 的稠密候选确定性置空。",
    }
    _MANIFEST_CACHE[cache_key] = deepcopy(manifest)
    return manifest


def _knowledge_source(local: LocalJsonlRetriever, content_digest: str) -> list[dict[str, Any]]:
    """按线上 KnowledgeAssets 的字段形状生成 ControlledTools 注册表。"""
    records = [*(local._chunk_records or []), *(local._rule_records or [])]
    if not records:
        raise ValueError("知识库没有可注册的片段")

    catalog_version = str((local._manifest or {}).get("catalog_meta", {}).get("catalog_version") or content_digest)
    collection = {
        "collection_id": "safetyraise-kbase",
        "version": catalog_version,
        "content_digest": content_digest,
        "label": "原项目固定版本知识库",
    }
    manifest_digest = canonical_digest([collection])
    source_ids: dict[str, set[str]] = {}
    for record in local._chunk_records or []:
        source = record.get("source_id")
        identifier = record.get("chunk_id")
        if isinstance(source, str) and isinstance(identifier, str):
            source_ids.setdefault(source, set()).add(identifier)

    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        identifier = record.get("chunk_id") or record.get("rule_id") or record.get("source_id")
        text = record.get("content")
        if not isinstance(identifier, str) or not isinstance(text, str) or not text.strip():
            continue
        if identifier in seen:
            raise ValueError(f"知识片段 ID 重复: {identifier}")
        seen.add(identifier)
        source = str(record.get("source_id") or identifier)
        adjacent: tuple[str, ...] = ()
        prefix, separator, ordinal = identifier.rpartition("#")
        if separator and prefix == source and ordinal.isascii() and ordinal.isdigit():
            adjacent = tuple(
                f"{source}#{position:0{len(ordinal)}d}"
                for position in (int(ordinal) - 1, int(ordinal) + 1)
                if f"{source}#{position:0{len(ordinal)}d}" in source_ids.get(source, set())
            )
        source_chunks = tuple(sorted(source_ids.get(source, ())))
        original = source_text(record, identifier, source, adjacent, source_chunks)
        result.append({
            "id": identifier,
            "document_id": source,
            "version": catalog_version,
            "text": original,
            "digest": canonical_digest(original),
            "manifest_digest": manifest_digest,
            "source_kind": "rule_excerpt" if record.get("rule_id") else "source_chunk",
        })
    return result


def _source_by_id(source: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(item["id"]): item for item in source}


def _stable_mode(mode: str, seed: int, case_id: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized in {"fallback", "sparse_half"}:
        return normalized
    if normalized not in {"mixed", "auto"}:
        raise ValueError("mode 必须是 fallback、sparse_half 或 mixed")
    token = f"{int(seed)}:{case_id}".encode("utf-8")
    return "sparse_half" if sha256(token).digest()[0] % 2 else "fallback"


@dataclass
class RetrievalHandle:
    """把生产检索对象和 ControlledTools 所需的冻结来源绑定在一起。"""

    retriever: Any
    mode: str
    source_by_id: dict[str, dict[str, Any]]
    knowledge_source: list[dict[str, Any]]
    manifest: dict[str, Any]
    manifest_digest: str
    external_knowledge_source: bool = True
    compact_registry: bool = True

    def initial_query(self, accident_data: dict[str, Any]) -> str:
        if self.mode == "fallback":
            return self.retriever.sparse_retriever._build_initial_query(accident_data)
        return HybridRetriever._build_initial_query(accident_data)

    def _fallback_raw(self, query: str, top_k: int, *, initial: bool) -> list[dict[str, Any]]:
        results = self.retriever.sparse_retriever.search(query=query, top_k=top_k)
        self.retriever.metadata = self.retriever._build_metadata(
            mode="sparse_only_fallback",
            query=query,
            sparse_candidates=results,
            dense_candidates=[],
            merged_candidates=results,
            reranked_candidates=results,
            final_candidates=results,
            fallback_reason=self.retriever.fallback_reason or "稠密依赖未就绪。",
            sparse_metadata=dict(getattr(self.retriever.sparse_retriever, "metadata", {})),
            retrieval_mode="initial" if initial else "agentic",
        )
        return results

    def initial_raw(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """返回生产原始检索结果（含 sparse/rerank/RRF 字段）。"""
        query = " ".join(str(query).split()).strip()
        if not query:
            return []
        return self._fallback_raw(query, top_k, initial=True) if self.mode == "fallback" else self._sparse_half_raw(query, top_k, initial=True)

    def search_raw(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """返回生产原始追加检索结果；ControlledTools 回调用 ``search`` 投影结果。"""
        query = " ".join(str(query).split()).strip()
        if not query:
            return []
        return self._fallback_raw(query, top_k, initial=False) if self.mode == "fallback" else self._sparse_half_raw(query, top_k, initial=False)

    def _sparse_half_raw(self, query: str, top_k: int, *, initial: bool) -> list[dict[str, Any]]:
        config = self.retriever.initial_config if initial else self.retriever.agentic_config
        sparse = self.retriever._collect_sparse_candidates(
            query=query,
            top_k_chunks=config["sparse_top_k_chunks"],
            top_k_rules=config["sparse_top_k_rules"],
            merge_top_k=config["rrf_merge_top_k"],
        )
        merged = self.retriever._merge_candidates(
            sparse_candidates=sparse,
            dense_candidates=[],
            merge_top_k=config["rrf_merge_top_k"],
        )
        ranked = self.retriever._rank_without_reranker(
            candidates=merged,
            top_n=min(config["rerank_top_k"], len(merged)),
        )
        final = self.retriever._select_final_candidates(
            candidates=ranked,
            limit=min(max(int(top_k), 1), config["final_context_top_k"]),
            max_context_chars=config["max_context_chars"],
            enforce_type_balance=initial,
        )
        self.retriever.metadata = self.retriever._build_metadata(
            mode="hybrid_rrf_only",
            query=query,
            sparse_candidates=sparse,
            dense_candidates=[],
            merged_candidates=merged,
            reranked_candidates=ranked,
            final_candidates=final,
            fallback_reason="sparse_half: dense candidates intentionally empty",
            sparse_metadata=dict(getattr(self.retriever.sparse_retriever, "metadata", {})),
            retrieval_mode="initial" if initial else "agentic",
        )
        return final

    def _as_source(self, raw: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        return [deepcopy(self.source_by_id[item["id"]]) for item in raw if item.get("id") in self.source_by_id]

    def initial(self, query: str, top_k: int) -> list[dict[str, Any]]:
        query = " ".join(str(query).split()).strip()
        if not query:
            return []
        raw = self.initial_raw(query, top_k)
        return self._as_source(raw)

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        query = " ".join(str(query).split()).strip()
        if not query:
            return []
        raw = self.search_raw(query, top_k)
        return self._as_source(raw)


def _build_production_handle(mode: str, kb_root: Path, seed: int, case_id: str) -> RetrievalHandle:
    root = kb_root.resolve()
    manifest = _manifest_for_root(root)
    paths = manifest["approved_assets"]
    local = LocalJsonlRetriever(
        manifest_path=root / paths["manifest"],
        chunks_path=root / paths["chunks"],
        rules_path=root / paths["rules"],
        search_index_path=root / paths["search_index"],
        min_score=0.2,
        top_k_chunks=8,
        top_k_rules=12,
        max_context_chars=5200,
        prefer_enhanced_rules=True,
        enable_search_index=True,
        watch_manifest_changes=False,
    )
    if mode == "fallback":
        retriever = HybridRetriever(
            sparse_retriever=local,
            embedding_client=None,
            reranker_client=None,
            dense_index=None,
            dense_index_version="",
            embedding_model="",
            reranker_model="",
            initial_config=dict(_INITIAL_CONFIG),
            agentic_config=dict(_AGENTIC_CONFIG),
            fallback_reason="确定性环境未提供稠密嵌入服务。",
        )
    else:
        retriever = HybridRetriever(
            sparse_retriever=local,
            embedding_client=_EmptyEmbedding(),
            reranker_client=None,
            dense_index=_EmptyDenseIndex(),
            dense_index_version="sparse-half-empty-dense-v1",
            embedding_model="deterministic-empty",
            reranker_model="",
            initial_config=dict(_INITIAL_CONFIG),
            agentic_config=dict(_AGENTIC_CONFIG),
        )
    source = _knowledge_source(local, manifest["content_digest"])
    manifest["mode"] = mode
    manifest["requested_seed"] = int(seed)
    manifest["case_id"] = str(case_id)
    manifest["knowledge_source_count"] = len(source)
    manifest["knowledge_source_kind_counts"] = {
        kind: sum(1 for item in source if item.get("source_kind") == kind)
        for kind in ("source_chunk", "rule_excerpt")
    }
    manifest_digest = str(source[0]["manifest_digest"])
    manifest["manifest_digest"] = manifest_digest
    return RetrievalHandle(
        retriever=retriever,
        mode=mode,
        source_by_id=_source_by_id(source),
        knowledge_source=source,
        manifest=manifest,
        manifest_digest=manifest_digest,
    )


class _SyntheticSparseRetriever:
    supports_query_search = True

    def __init__(self) -> None:
        self.records = [
            {"chunk_id": "synthetic_chunk#0001", "source_id": "synthetic_source", "title": "合成路口场景", "content": "合成路口夜间通行与信号观察提示。", "category": "traffic"},
            {"chunk_id": "synthetic_chunk#0002", "source_id": "synthetic_source", "title": "合成道路场景", "content": "合成道路低能见度时应核对道路状态。", "category": "traffic"},
            {"chunk_id": "synthetic_chunk#0003", "source_id": "synthetic_source", "title": "合成证据提示", "content": "合成材料不足时保留待核实边界。", "category": "evidence"},
            {"rule_id": "synthetic_rule#0001", "source_id": "synthetic_rules", "title": "合成规则一", "content": "合成规则摘录仅用于定位，不能替代正文。", "rule_type": "synthetic"},
            {"rule_id": "synthetic_rule#0002", "source_id": "synthetic_rules", "title": "合成规则二", "content": "合成规则要求在路口核对信号与让行。", "rule_type": "synthetic"},
            {"rule_id": "synthetic_rule#0003", "source_id": "synthetic_rules", "title": "合成规则三", "content": "合成规则要求记录尚未确认的事实。", "rule_type": "synthetic"},
        ]
        self.metadata: dict[str, Any] = {"provider": "synthetic_local_jsonl"}

    def _build_initial_query(self, accident_data: dict[str, Any]) -> str:
        keys = ["事故标题", "事故类型", "事故形态", "事故认定原因", "主要违法行为", "车辆类型", "路口路段类型", "地点（包括路名，路号）"]
        values = [str(accident_data[key]).strip() for key in keys if isinstance(accident_data.get(key), str) and accident_data[key].strip()]
        return "\n".join(dict.fromkeys(values) if values else [str(v).strip() for v in accident_data.values() if isinstance(v, str) and v.strip()][:8])

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        folded = " ".join(str(query).split()).casefold()
        terms = [term for term in re.split(r"\s+", folded) if term]
        ranked: list[dict[str, Any]] = []
        for record in self.records:
            haystack = f"{record.get('title', '')} {record.get('content', '')}".casefold()
            hits = sum(1 for term in terms if term in haystack)
            score = 0.2 + 0.1 * hits if hits else 0.2
            item = {
                "id": record.get("chunk_id") or record.get("rule_id"),
                "title": record.get("title", ""),
                "content": record.get("content", ""),
                "source": record.get("source_id", ""),
                "score": round(score, 4),
                "record_type": "rule" if record.get("rule_id") else "chunk",
                "citation": record.get("chunk_id") or record.get("rule_id"),
                "url": "",
                "category": record.get("category", ""),
                "authority": "synthetic",
            }
            if hits:
                ranked.append(item)
        ranked.sort(key=lambda item: (-float(item["score"]), str(item["id"])))
        self.metadata.update({"last_query": folded, "last_results_count": min(len(ranked), int(top_k))})
        return ranked[: int(top_k)]

    def retrieve(self, accident_data: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
        return self.search(self._build_initial_query(accident_data), top_k)


def synthetic_retrieval(mode: str = "fallback", *, seed: int = 0, case_id: str = "synthetic") -> RetrievalHandle:
    """返回仅含合成片段的测试 fixture，不代表真实知识检索。"""
    resolved = _stable_mode(mode, seed, case_id)
    sparse = _SyntheticSparseRetriever()
    source: list[dict[str, Any]] = []
    for record in sparse.records:
        identifier = str(record.get("chunk_id") or record.get("rule_id"))
        text = str(record.get("content"))
        source.append({
            "id": identifier,
            "document_id": str(record.get("source_id")),
            "version": "synthetic-v1",
            "text": text,
            "digest": canonical_digest(text),
            "manifest_digest": canonical_digest({"synthetic": True, "version": "synthetic-v1"}),
            "source_kind": "rule_excerpt" if record.get("rule_id") else "source_chunk",
        })
    if resolved == "fallback":
        retriever = HybridRetriever(
            sparse_retriever=sparse, embedding_client=None, reranker_client=None, dense_index=None,
            dense_index_version="", embedding_model="", reranker_model="",
            initial_config=dict(_INITIAL_CONFIG), agentic_config=dict(_AGENTIC_CONFIG),
            fallback_reason="synthetic fixture fallback",
        )
    else:
        retriever = HybridRetriever(
            sparse_retriever=sparse, embedding_client=_EmptyEmbedding(), reranker_client=None,
            dense_index=_EmptyDenseIndex(), dense_index_version="synthetic-empty-dense-v1",
            embedding_model="synthetic-empty", reranker_model="",
            initial_config=dict(_INITIAL_CONFIG), agentic_config=dict(_AGENTIC_CONFIG),
        )
    manifest = {
        "schema_version": 1,
        "synthetic": True,
        "mode": resolved,
        "seed": int(seed),
        "case_id": str(case_id),
        "file_count": 0,
        "knowledge_source_count": len(source),
        "known_approximation": "仅供无网络单元测试，不代表真实检索。",
    }
    digest = source[0]["manifest_digest"]
    manifest["manifest_digest"] = digest
    return RetrievalHandle(
        retriever=retriever,
        mode=resolved,
        source_by_id=_source_by_id(source),
        knowledge_source=source,
        manifest=manifest,
        manifest_digest=digest,
    )


def build_retrieval(
    mode: str = "fallback",
    kb_root: str | Path | None = None,
    seed: int = 0,
    case_id: str = "",
) -> RetrievalHandle:
    """装配冻结知识库检索。

    ``mixed``/``auto`` 按 ``seed + case_id`` 的 SHA-256 首字节稳定选模式；同一输入
    重放一定得到同一模式和同一检索排序。生产知识库路径默认只读 D 盘快照。
    """
    resolved = _stable_mode(mode, seed, case_id)
    root = Path(kb_root) if kb_root is not None else DEFAULT_KB_ROOT
    return _build_production_handle(resolved, root, seed, case_id)


__all__ = ["RetrievalHandle", "build_retrieval", "synthetic_retrieval"]
