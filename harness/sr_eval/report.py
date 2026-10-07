"""汇总:把一次 run 目录里的 traces / gates / judgements 聚成每个模型一行的对比表。"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any


def _mean(xs: list[float]) -> float | None:
    return round(statistics.fmean(xs), 2) if xs else None


def _pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return round(xs[min(int(q * len(xs)), len(xs) - 1)], 1)


def load_run(run_dir: Path) -> dict[str, Any]:
    traces = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((run_dir / "traces").glob("*.json"))]
    gates = json.loads((run_dir / "gates.json").read_text(encoding="utf-8")) if (run_dir / "gates.json").exists() else {}
    judged = json.loads((run_dir / "judgements.json").read_text(encoding="utf-8")) if (run_dir / "judgements.json").exists() else {}
    return {"traces": traces, "gates": gates, "judgements": judged}


def summarize(run: dict[str, Any]) -> list[dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for t in run["traces"]:
        by_model.setdefault(t["model"], []).append(t)
    rows = []
    for model, traces in sorted(by_model.items()):
        keys = [f"{t['case_id']}__{model}" for t in traces]
        gates = [run["gates"].get(k) for k in keys if run["gates"].get(k)]
        hard_codes: dict[str, int] = {}
        for g in gates:
            for h in g["hard"]:
                hard_codes[h["code"]] = hard_codes.get(h["code"], 0) + 1
        judge_totals: dict[str, list[float]] = {}
        for k in keys:
            for j, res in (run["judgements"].get(k) or {}).items():
                if res.get("valid"):
                    judge_totals.setdefault(j, []).append(res["total"])
        chars = [g["metrics"]["chars"] for g in gates if g.get("metrics")]
        rows.append({
            "model": model, "n": len(traces),
            "errors": sum(t["status"] != "ok" for t in traces),
            "hard_fail_rate": round(sum(g["hard_fail"] for g in gates) / len(gates), 2) if gates else None,
            "hard_codes": hard_codes,
            "judge_mean_total": {j: _mean(v) for j, v in judge_totals.items()},
            "chars_mean": _mean(chars), "chars_p90": _pct(chars, 0.9),
            "dup_ratio_mean": _mean([g["metrics"]["dup_12gram_ratio"] for g in gates if g.get("metrics")]),
            "citations_mean": _mean([g["metrics"]["citations"] for g in gates if g.get("metrics")]),
            "rounds_mean": _mean([t["totals"]["retrieval_rounds"] for t in traces]),
            "completion_tokens_mean": _mean([t["totals"]["completion_tokens"] for t in traces]),
            "reasoning_tokens_mean": _mean([t["totals"]["reasoning_tokens"] for t in traces]),
            "prompt_tokens_mean": _mean([t["totals"]["prompt_tokens"] for t in traces]),
            "total_tokens_p90": _pct([t["totals"]["prompt_tokens"] + t["totals"]["completion_tokens"] for t in traces], 0.9),
            "latency_s_mean": _mean([t["totals"]["latency_s"] for t in traces]),
            "cost_usd_total": round(sum(t["totals"]["cost_usd"] for t in traces), 4),
        })
    return rows


def render_table(rows: list[dict[str, Any]]) -> str:
    head = "| 模型 | n | 错误 | 硬门失败率 | 评审均分(/25) | 字数均/P90 | 重复率 | 引用 | 检索轮 | 完成tok | 思考tok | 总tok P90 | 耗时s | 费用$ |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    lines = []
    for r in rows:
        judge = ";".join(f"{j}={v}" for j, v in r["judge_mean_total"].items()) or "-"
        lines.append(
            f"| {r['model']} | {r['n']} | {r['errors']} | {r['hard_fail_rate']} | {judge} | {r['chars_mean']}/{r['chars_p90']} | "
            f"{r['dup_ratio_mean']} | {r['citations_mean']} | {r['rounds_mean']} | {r['completion_tokens_mean']} | "
            f"{r['reasoning_tokens_mean']} | {r['total_tokens_p90']} | {r['latency_s_mean']} | {r['cost_usd_total']} |"
        )
    return head + "\n".join(lines) + "\n\n> 全部为自动口径(硬门 + 模型评审),**未经人工核实**;评审分数只用于相对比较。\n"
