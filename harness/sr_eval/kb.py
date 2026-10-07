"""评测台用的本地知识库检索:BM25(字二元组)+ 分块/规则两路 RRF。

与生产(稀疏+稠密 RRF、同义词/主题索引)不是同一个检索器——评测台只保证
"所有被比较的模型看到同一检索器的同一结果",并保持片段字段形状与生产一致。
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

GOVERNANCE_FIELDS = (
    "effect_level",
    "usage_note",
    "jurisdiction",
    "effective_date",
    "latest_revision",
    "validity_note",
    "published_date",
    "supersedes",
)
INITIAL_QUERY_KEYS = (
    "事故标题",
    "事故类型",
    "事故形态",
    "事故认定原因",
    "主要违法行为",
    "车辆类型",
    "路口路段类型",
    "地点（包括路名，路号）",
)
MAX_CONTEXT_CHARS = 5200  # workflow.yaml: retrieval.*.max_context_chars

_CJK_RUN = re.compile(r"[一-鿿]+")
_ALNUM = re.compile(r"[a-z0-9_.:/-]{2,}")


def tokenize(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text or "").lower()
    tokens: list[str] = []
    for run in _CJK_RUN.findall(text):
        if len(run) == 1:
            continue
        tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    tokens.extend(_ALNUM.findall(text))
    return tokens


@dataclass
class _Index:
    docs: list[dict[str, Any]]
    inverted: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    lengths: list[int] = field(default_factory=list)
    avg_len: float = 0.0

    @classmethod
    def build(cls, docs: list[dict[str, Any]]) -> "_Index":
        inverted: dict[str, list[tuple[int, int]]] = defaultdict(list)
        lengths: list[int] = []
        for i, d in enumerate(docs):
            toks = tokenize(d.get("title", "")) * 2 + tokenize(d.get("content", ""))
            lengths.append(len(toks))
            for tok, tf in Counter(toks).items():
                inverted[tok].append((i, tf))
        avg = sum(lengths) / max(len(lengths), 1)
        return cls(docs=docs, inverted=dict(inverted), lengths=lengths, avg_len=avg)

    def rank(self, query_tokens: list[str], limit: int, k1: float = 1.5, b: float = 0.75) -> list[tuple[int, float]]:
        n = len(self.docs)
        scores: dict[int, float] = defaultdict(float)
        for tok in set(query_tokens):
            postings = self.inverted.get(tok)
            if not postings:
                continue
            idf = math.log(1 + (n - len(postings) + 0.5) / (len(postings) + 0.5))
            for i, tf in postings:
                norm = tf * (k1 + 1) / (tf + k1 * (1 - b + b * self.lengths[i] / max(self.avg_len, 1)))
                scores[i] += idf * norm
        return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]


def _to_snippet(record: dict[str, Any], score: float, record_type: str) -> dict[str, Any]:
    identifier = record.get("chunk_id") or record.get("rule_id") or record.get("source_id") or "unknown"
    snippet = {
        "id": identifier,
        "title": record.get("title", ""),
        "content": record.get("content", ""),
        "source": record.get("source_id", ""),
        "score": round(score, 4),
        "record_type": record_type,
        "citation": identifier,
        "url": record.get("url", ""),
        "category": record.get("category", ""),
        "authority": record.get("authority", ""),
    }
    for key in GOVERNANCE_FIELDS:
        value = record.get(key)
        if value not in (None, "", []):
            snippet[key] = value
    return snippet


def build_initial_query(accident_data: dict[str, Any]) -> str:
    """local_jsonl_retriever._build_initial_query 的复刻:优先字段去重后取前 8 个,换行连接。"""
    parts = [
        v.strip()
        for k in INITIAL_QUERY_KEYS
        if isinstance(v := accident_data.get(k), str) and v.strip()
    ]
    if not parts:
        parts = [v.strip() for v in accident_data.values() if isinstance(v, str) and v.strip()]
    return "\n".join(list(dict.fromkeys(parts))[:8])


def truncate_context(records: list[dict[str, Any]], max_chars: int = MAX_CONTEXT_CHARS) -> list[dict[str, Any]]:
    total = 0
    out: list[dict[str, Any]] = []
    for rec in records:
        content = str(rec.get("content", ""))
        remaining = max_chars - total
        if remaining <= 0:
            break
        if len(content) > remaining:
            clipped = dict(rec)
            clipped["content"] = content[: max(remaining - 1, 1)] + "…"
            clipped["content_truncated"] = True
            out.append(clipped)
            break
        out.append(rec)
        total += len(content)
    return out


class KnowledgeBase:
    """chunks(法条/文件分块)与 rules(短规则)各建一个 BM25,查询时 RRF 融合。"""

    def __init__(self, chunks: list[dict[str, Any]], rules: list[dict[str, Any]], *, name: str, outbound_class: str) -> None:
        self.name = name
        self.outbound_class = outbound_class  # "synthetic" | "public_statutes"
        self.chunks = chunks
        self.rules = rules
        self._chunk_index = _Index.build(chunks)
        self._rule_index = _Index.build(rules) if rules else None

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        q = tokenize(re.sub(r"\s+", " ", query or "").strip())
        if not q or top_k < 1:
            return []
        fused: dict[tuple[str, int], float] = defaultdict(float)
        for tag, index in (("chunk", self._chunk_index), ("rule", self._rule_index)):
            if index is None:
                continue
            for rank, (i, _score) in enumerate(index.rank(q, limit=24)):
                fused[(tag, i)] += 1.0 / (60 + rank)
        ranked = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        results = []
        for (tag, i), score in ranked:
            rec = self.chunks[i] if tag == "chunk" else self.rules[i]
            results.append(_to_snippet(rec, score * 60, tag))
        return truncate_context(results)

    def retrieve(self, accident_data: dict[str, Any], top_k: int) -> tuple[str, list[dict[str, Any]]]:
        query = build_initial_query(accident_data)
        return query, (self.search(query, top_k) if query else [])

    def get(self, snippet_id: str) -> dict[str, Any] | None:
        for rec in self.chunks:
            if rec.get("chunk_id") == snippet_id:
                return rec
        for rec in self.rules:
            if rec.get("rule_id") == snippet_id:
                return rec
        return None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_production_kb(base_dir: str | Path, *, enhanced_rules: bool = False) -> KnowledgeBase:
    """只读加载生产知识库快照(kbase_chunks.jsonl + liability_rules*.jsonl)。公开官方来源,外发需仓库所有者批准。"""
    base = Path(base_dir)
    rules_name = "liability_rules_enhanced.jsonl" if enhanced_rules else "liability_rules.jsonl"
    return KnowledgeBase(
        _load_jsonl(base / "kbase_chunks.jsonl"),
        _load_jsonl(base / rules_name),
        name=f"production:{base.name}/{rules_name}",
        outbound_class="public_statutes",
    )


# ---- 合成最小知识库:只用于联调机制,条款为虚构,不是真实法规 ----
_SYNTH = [
    ("倒车让行", "合成示例条款甲:机动车倒车时,驾驶人应当先观察车后情况,确认安全后方可倒车;倒车时应当有人指挥或使用倒车影像、警示装置。", "机动车行驶"),
    ("人行横道让行", "合成示例条款乙:机动车行经人行横道时,应当减速行驶;遇行人正在通过人行横道,应当停车让行。", "行人通行"),
    ("无信号灯路口通行", "合成示例条款丙:通过没有交通信号灯控制的路口,支路车辆让干路车辆先行;转弯的机动车让直行的车辆先行。", "路口通行"),
    ("安全车距", "合成示例条款丁:机动车在同一车道行驶时,后车应当与前车保持足以采取紧急制动措施的安全距离。", "机动车行驶"),
    ("夜间与低能见度行驶", "合成示例条款戊:夜间或能见度较低时,应当开启前照灯并降低行驶速度,不得超过道路限速。", "机动车行驶"),
    ("雨天路滑", "合成示例条款己:遇有雨天路面湿滑时,应当降低行驶速度,增大与前车间距,不得急转向、急制动。", "特殊天气"),
    ("饮酒驾驶", "合成示例条款庚:饮酒后不得驾驶机动车;驾驶人血液酒精含量达到规定数值的,应当依法处理。", "违法行为"),
    ("事故现场保护", "合成示例条款辛:发生交通事故后,当事人应当立即停车,保护现场,抢救伤者,并及时报警。", "事故处理"),
    ("证据与责任认定", "合成示例条款壬:公安机关应当根据当事人的行为对发生交通事故所起的作用以及过错的严重程度,确定当事人的责任;证据不足的,应当说明待查事项。", "责任认定"),
    ("非机动车通行", "合成示例条款癸:非机动车应当在非机动车道内行驶;在没有非机动车道的道路上,应当靠车行道的右侧行驶。", "非机动车"),
    ("安全带与头盔", "合成示例条款子:机动车驾驶人及乘坐人员应当按规定使用安全带;驾驶、乘坐摩托车应当戴安全头盔。", "安全装备"),
    ("交叉路口转弯", "合成示例条款丑:机动车在路口转弯时,应当提前开启转向灯,减速慢行,并让行人和直行车辆优先通行。", "路口通行"),
]


def synthetic_kb() -> KnowledgeBase:
    chunks = [
        {
            "chunk_id": f"synth_kb_{i:03d}",
            "source_id": "synthetic_traffic_rules",
            "title": f"【合成示例条款】{title}",
            "content": text,
            "category": cat,
            "authority": "合成(非真实法规)",
            "url": "",
            "effect_level": "合成示例,不具法律效力",
            "usage_note": "仅用于评测台联调,不得当作真实法规引用",
            "jurisdiction": "全国",
        }
        for i, (title, text, cat) in enumerate(_SYNTH, start=1)
    ]
    return KnowledgeBase(chunks, [], name="synthetic:mini12", outbound_class="synthetic")
