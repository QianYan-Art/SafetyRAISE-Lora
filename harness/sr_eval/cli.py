"""命令行入口:python -m sr_eval.cli <check|gen-cases|run|gates|judge|calibrate|summary> ...

运行方式(在 <repo> 下):
  $env:PYTHONPATH="harness"; .venv\\Scripts\\python.exe -B -m sr_eval.cli check
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from . import gates as G
from . import judge as J
from . import report as R
from .cases import generate_case
from .kb import KnowledgeBase, load_production_kb, synthetic_kb
from .llm import CONFIG_DIR, HARNESS_DIR, LLMClient, LLMError, Ledger, load_models
from .policy import OutboundPolicy
from . import rationale as RA
from . import sft as SFT
from .runner import RunConfig, run_many

PRODUCTION_KB_DIR = r"<SafetyRAISE-knowledge-repo>\kbase\data"
CASES_DIR = HARNESS_DIR / "cases"
RUNS_DIR = HARNESS_DIR / "runs"


def _budget() -> dict[str, Any]:
    return json.loads((CONFIG_DIR / "budget.json").read_text(encoding="utf-8"))


def make_client() -> LLMClient:
    b = _budget()
    return LLMClient(models=load_models(), ledger=Ledger(), cap_usd=float(b["cap_usd"]), scope=b["scope"])


def preflight_openrouter(client: LLMClient, models: list[str]) -> None:
    """用到 OpenRouter 模型时先核验 key:必须存在且带 credit limit(服务商侧硬上限)。"""
    if not any(client.models[m].provider == "openrouter" for m in models):
        return
    b = _budget()
    st = client.openrouter_key_status()
    if st["limit"] is None:
        raise LLMError("key_has_no_limit", detail="该 OpenRouter key 没有 credit limit;评测台要求有可证明的费用上界(见 harness/README.md“费用与外发闸门”),请换用设置了额度上限的专用 key")
    if float(st["limit"]) > float(b["total_budget_usd"]) + 1e-9:
        raise LLMError("key_limit_above_budget", detail=f"key 额度 {st['limit']} > 总预算 {b['total_budget_usd']}")
    remaining = st["limit_remaining"]
    if remaining is not None:
        # 账本记的是该专用 key 的累计花费,所以上限要按"累计 + key 剩余额度 − 1 美元余量"算(原先直接取剩余额度,
        # 花得越多越早误拦:key 用量 24/50 时就把累计上限压到 25.85)。仍不超过预算文件里的软件上限。
        room = float(remaining)
        try:                                   # 账户总余额(含该账户下其他 key/项目的用量)可能比 key 余额更紧
            acct = client.openrouter_credits()
            if acct.get("remaining") is not None:
                room = min(room, float(acct["remaining"]))
        except Exception:
            pass
        headroom_cap = client.ledger.committed_usd(client.scope) + room - 1.0
        client.cap_usd = min(client.cap_usd, headroom_cap)
    print(f"[openrouter] key limit={st['limit']} remaining={remaining} usage={st['usage']} → 软件上限(累计口径) {client.cap_usd:.2f} USD")


def load_cases(split: str, n: int | None = None, offset: int = 0) -> list[dict[str, Any]]:
    paths = sorted((CASES_DIR / split).glob("*.json"))
    cases = [json.loads(p.read_text(encoding="utf-8")) for p in paths][offset:]
    return cases[:n] if n else cases


def pick_kb(name: str) -> KnowledgeBase:
    return synthetic_kb() if name == "synthetic" else load_production_kb(PRODUCTION_KB_DIR)


def resolve_run(name: str) -> Path:
    p = Path(name)
    if p.exists():
        return p
    exact = sorted(RUNS_DIR.glob(f"*_{name}"))   # 先按标签精确匹配(避免 lunatrain 误配 lunatrain2)
    matches = exact or sorted(RUNS_DIR.glob(f"*{name}*"))
    if not matches:
        raise SystemExit(f"找不到 run: {name}")
    return matches[-1]


def cmd_check(_a: argparse.Namespace) -> None:
    client = make_client()
    print("ledger 已用/预留(openrouter 范围):", round(client.ledger.committed_usd(client.scope), 4), "USD / 上限", client.cap_usd)
    try:
        st = client.openrouter_key_status()
        print("openrouter key:", st)
        print("openrouter 账户额度:", client.openrouter_credits())
    except LLMError as exc:
        print("openrouter key 不可用:", exc.code, exc.detail)
    print("外发闸门:", OutboundPolicy.load())


def cmd_gen_cases(a: argparse.Namespace) -> None:
    client = make_client()
    preflight_openrouter(client, [a.model])
    out = CASES_DIR / a.split
    out.mkdir(parents=True, exist_ok=True)
    seeds = [s for s in range(a.start, a.start + a.n) if not (out / f"syn_{a.split}_{s:03d}.json").exists()]

    def one(seed: int) -> str:
        for attempt in (1, 2):
            try:
                case = generate_case(client, a.model, seed, a.split, timeout=a.timeout)
            except ValueError as exc:
                msg = f"seed {seed} 第{attempt}次失败: {exc}"
                if attempt == 2:
                    return msg
                continue
            (out / f"{case['case_id']}.json").write_text(json.dumps(case, ensure_ascii=False, indent=1), encoding="utf-8")
            g = case["gen_meta"]
            return f"{case['case_id']} ok scenario={case['scenario']['type']}/{case['scenario']['richness']} 字段={g['filled_fields']} 字数={g['accident_chars']}"
        return ""

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for line in pool.map(one, seeds):
            print(line)


def cmd_run(a: argparse.Namespace) -> None:
    client = make_client()
    models = a.models.split(",")
    preflight_openrouter(client, models)
    kb = pick_kb(a.kb)
    cases = load_cases(a.cases, a.n, a.offset)
    if not cases:
        raise SystemExit(f"cases/{a.cases} 下没有案件,先运行 gen-cases")
    run_dir = RUNS_DIR / (a.resume or f"{time.strftime('%Y%m%d_%H%M%S')}_{a.tag}")
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = RunConfig(exec_variant=a.variant, max_tokens=a.max_tokens, timeout=a.timeout)
    if a.effort:
        cfg.reasoning = {"effort": a.effort}
    meta = {"kb": kb.name, "kb_class": kb.outbound_class, "models": models, "cases": [c["case_id"] for c in cases],
            "config": cfg.__dict__, "harness": "sr_eval 0.1.0 mode L"}
    (run_dir / "run.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    jobs = [(m, c) for c in cases for m in models]
    traces = run_many(client, jobs, kb, cfg, run_dir, policy=OutboundPolicy.load(), run_id=run_dir.name, workers=a.workers)
    cmd_gates(argparse.Namespace(run=str(run_dir)))
    print(f"run={run_dir.name} traces={len(traces)} 账本合计={client.ledger.committed_usd(client.scope):.4f} USD")


def cmd_gates(a: argparse.Namespace) -> None:
    run_dir = resolve_run(a.run)
    cases = {c["case_id"]: c for split in CASES_DIR.iterdir() if split.is_dir() for c in load_cases(split.name)}
    out = {}
    for p in sorted((run_dir / "traces").glob("*.json")):
        t = json.loads(p.read_text(encoding="utf-8"))
        if t.get("final_raw"):   # 早期轨迹用过简化清洗;统一按冻结的生产清洗重算
            try:
                from .runner import _resolve_final
                t["final_markdown"] = _resolve_final(t["final_raw"])
            except ValueError:
                pass
        out[f"{t['case_id']}__{t['model']}"] = G.evaluate(t, cases[t["case_id"]])
    (run_dir / "gates.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"gates: {len(out)} 份;硬门失败 {sum(v['hard_fail'] for v in out.values())} 份")


def cmd_judge(a: argparse.Namespace) -> None:
    client = make_client()
    judges = a.judges.split(",")
    preflight_openrouter(client, judges)
    run_dir = resolve_run(a.run)
    kb_class = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["kb_class"]
    OutboundPolicy.load().check(kb_class=kb_class, case_kind="synthetic")
    cases = {c["case_id"]: c for split in CASES_DIR.iterdir() if split.is_dir() for c in load_cases(split.name)}
    path = run_dir / "judgements.json"
    done: dict[str, Any] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    only = None if a.only is None else {k for k in a.only.split(",") if k}   # 只评指定条目(case_id__model),用于抽样评审
    if only is not None and not only:
        raise SystemExit("--only 给了空列表:没有要评审的条目")
    gates_all = json.loads((run_dir / "gates.json").read_text(encoding="utf-8")) if (run_dir / "gates.json").exists() else {}
    work = []
    for p in sorted((run_dir / "traces").glob("*.json")):
        t = json.loads(p.read_text(encoding="utf-8"))
        if not t.get("final_markdown"):
            continue
        key = f"{t['case_id']}__{t['model']}"
        if only is not None and key not in only:
            continue
        if a.eligible:   # 只评"合格"候选:硬门全过、无思考循环(失败样本是 rejected,不需要评审)
            g = gates_all.get(key)
            if g is None or g["hard_fail"] or "reasoning_loop" in {w["code"] for w in g.get("warn", [])}:
                continue
        for j in judges:
            if a.force or not (done.get(key, {}).get(j) or {}).get("valid"):   # 无效结果(如思考耗尽上限被截断)重跑;--force 全部重评(旧结果存到 <judge>_prev)
                work.append((key, j, t))

    def one(item):
        key, j, t = item
        return key, j, J.judge_report(client, j, cases[t["case_id"]], t["final_markdown"], t["visible_snippets"], effort=a.effort, meta={"run_id": run_dir.name, "case_id": t["case_id"]})

    # 每条结果一完成就落盘:中途被停止也不会丢掉已付费的评审
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for fut in as_completed([pool.submit(one, w) for w in work]):
            key, j, res = fut.result()
            # 多个评审进程可能同时写同一个文件(luna 与 MiniMax 并发):每次落盘前重新读取、只合并本条结果、原子替换,避免互相覆盖
            cur = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            if a.force and cur.get(key, {}).get(j) is not None and f"{j}_prev" not in cur[key]:
                cur[key][f"{j}_prev"] = cur[key][j]
            cur.setdefault(key, {})[j] = res
            tmp = path.with_suffix(f".tmp{os.getpid()}")
            tmp.write_text(json.dumps(cur, ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(tmp, path)
    print(f"judge: 新增 {len(work)} 次评审")


def cmd_calibrate(a: argparse.Namespace) -> None:
    client = make_client()
    preflight_openrouter(client, [a.judge])
    run_dir = resolve_run(a.run)
    kb_class = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["kb_class"]
    OutboundPolicy.load().check(kb_class=kb_class, case_kind="synthetic")
    cases = {c["case_id"]: c for split in CASES_DIR.iterdir() if split.is_dir() for c in load_cases(split.name)}
    traces = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((run_dir / "traces").glob("*.json"))]
    good = [t for t in traces if t.get("final_markdown")][: a.n]
    plan = J.calibration_plan([(f"{t['case_id']}__{t['model']}", t["final_markdown"]) for t in good])
    by_id = {f"{t['case_id']}__{t['model']}": t for t in good}

    def one(item):
        t = by_id[item["report_id"]]
        res = J.judge_report(client, a.judge, cases[t["case_id"]], item["text"], t["visible_snippets"],
                             effort=a.effort, meta={"run_id": run_dir.name, "case_id": t["case_id"], "calibration": item["mutation"]})
        return {**{k: item[k] for k in ("item_id", "report_id", "mutation", "target")}, **res}

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        items = list(pool.map(one, plan))
    summary = J.summarize_calibration(items)
    (run_dir / f"calibration_{a.judge}.json").write_text(json.dumps({"items": items, "summary": summary}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


def cmd_summary(a: argparse.Namespace) -> None:
    run_dir = resolve_run(a.run)
    table = R.render_table(R.summarize(R.load_run(run_dir)))
    (run_dir / "summary.md").write_text(table, encoding="utf-8")
    print(table)


def _load_cases_all() -> dict[str, dict[str, Any]]:
    return {c["case_id"]: c for split in CASES_DIR.iterdir() if split.is_dir() for c in load_cases(split.name)}


SFT_DIR = HARNESS_DIR / "sft"
EXCLUDE_WARN = {"liability_assertion_unhedged", "guidance_verbatim", "meta_language", "missing_sections", "english_leakage",
                "tool_call_multiple_tool_calls"}


def select_traces(run_dirs: list[Path], cases: dict[str, Any], min_score: int, split: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """硬门全过 + 无排除类警示 + luna 评审≥min_score(有效的 MiniMax 评审须≥min_score-3)。"""
    stats = {"traces": 0, "no_report": 0, "hard_fail": 0, "warn_excluded": 0, "no_judge": 0, "low_score": 0, "selected": 0}
    chosen = []
    for rd in run_dirs:
        gates = json.loads((rd / "gates.json").read_text(encoding="utf-8"))
        judged = json.loads((rd / "judgements.json").read_text(encoding="utf-8")) if (rd / "judgements.json").exists() else {}
        for p in sorted((rd / "traces").glob("*.json")):
            t = json.loads(p.read_text(encoding="utf-8"))
            if cases[t["case_id"]]["split"] != split:
                continue
            stats["traces"] += 1
            key = f"{t['case_id']}__{t['model']}"
            g = gates.get(key)
            if not t.get("final_markdown") or g is None:
                stats["no_report"] += 1; continue
            if g["hard_fail"]:
                stats["hard_fail"] += 1; continue
            if {w["code"] for w in g["warn"]} & EXCLUDE_WARN:
                stats["warn_excluded"] += 1; continue
            j = judged.get(key, {})
            lj, mj = j.get("luna"), j.get("minimax")
            if not lj or not lj.get("valid"):
                stats["no_judge"] += 1; continue
            if lj["total"] < min_score or (mj and mj.get("valid") and mj["total"] < min_score - 3):
                stats["low_score"] += 1; continue
            t["_judge"] = {"luna": lj["total"], "minimax": mj["total"] if mj and mj.get("valid") else None}
            chosen.append(t)
            stats["selected"] += 1
    return chosen, stats


def cmd_sft_build(a: argparse.Namespace) -> None:
    client = make_client()
    preflight_openrouter(client, [a.rationale_model])
    cases = _load_cases_all()
    run_dirs = [resolve_run(r) for r in a.runs.split(",")]
    chosen, stats = select_traces(run_dirs, cases, a.min_score, a.split)
    out = SFT_DIR / a.name
    out.mkdir(parents=True, exist_ok=True)
    tpl = SFT.QwenTemplate()
    plan = [(t, i) for t in chosen for i in SFT.case_sample_plan(t, a.keep_tool_prob)]
    cache_path = out / "thinking.json"
    cache: dict[str, Any] = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    todo = [(t, i) for t, i in plan if f"{t['case_id']}__{t['model']}__{i}" not in cache]

    def one(item):
        t, i = item
        key = f"{t['case_id']}__{t['model']}__{i}"
        return key, RA.make_thinking(client, a.rationale_model, cases[t["case_id"]], t, i, meta={"case_id": t["case_id"]})

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for key, res in pool.map(one, todo):
            cache[key] = res
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    rows, skipped = [], {"no_thinking": 0, "too_long": 0, "template": 0}
    for t, i in plan:
        key = f"{t['case_id']}__{t['model']}__{i}"
        th = cache.get(key)
        if not th or not th.get("ok"):
            skipped["no_thinking"] += 1; continue
        try:
            s = SFT.build_sample(tpl, t["calls"][i], th["text"], reasoning_effort=a.template_effort)
        except ValueError:
            skipped["template"] += 1; continue
        if len(s["input_ids"]) > a.max_len:
            skipped["too_long"] += 1; continue
        rows.append({"sample_id": key, "case_id": t["case_id"], "teacher": t["model"], "call_index": i,
                     "kind": th["kind"], "thinking_source": "luna_rationale_zh", "judge": t["_judge"],
                     "prompt_tokens": s["prompt_tokens"], "target_tokens": s["target_tokens"],
                     "input_ids": s["input_ids"], "labels": s["labels"], "target_text": s["target_text"]})
    with (out / "train.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tot = sum(len(r["input_ids"]) for r in rows)
    manifest = {"name": a.name, "split": a.split, "runs": [d.name for d in run_dirs], "selection": stats, "min_score": a.min_score,
                "reasoning_effort_in_template": a.template_effort, "keep_tool_prob": a.keep_tool_prob, "max_len": a.max_len,
                "samples": len(rows), "cases": len({r["case_id"] for r in rows}), "total_tokens": tot,
                "target_tokens": sum(r["target_tokens"] for r in rows), "skipped": skipped,
                "by_kind": {k: sum(r["kind"] == k for r in rows) for k in ("tool", "final")},
                "by_teacher": {k: sum(r["teacher"] == k for r in rows) for k in {r["teacher"] for r in rows}},
                "max_tokens": max((len(r["input_ids"]) for r in rows), default=0),
                "template_sha256": tpl.template_sha256, "tokenizer_sha256": tpl.tokenizer_sha256,
                "thinking_note": "简短中文思考由 luna 事后合理化生成(thinking_source),不是教师原始思考;全部合成案件,未经人工核实",
                "outbound_note": "仅合成案件 + 公开法规片段"}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=1))


def cmd_eval_final(a: argparse.Namespace) -> None:
    """教师强制检索 + 学生只写最终报告:沿用教师轨迹最后一次调用的输入(含已检索片段),让待测模型生成最终报告。
    用于解码很慢的本地模型(Orin HF+bnb ~2.5 token/s),只评"写报告"这一步,不评检索决策。"""
    from . import prompt as P
    client = make_client()
    preflight_openrouter(client, [a.model])
    teacher = resolve_run(a.run)
    meta = json.loads((teacher / "run.json").read_text(encoding="utf-8"))
    cases = _load_cases_all()
    run_dir = RUNS_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_{a.tag}"
    (run_dir / "traces").mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({**meta, "models": [a.model], "mode": "final_only", "teacher_run": teacher.name,
                                                  "note": "最终报告步评测:检索轨迹来自教师,非端到端"}, ensure_ascii=False, indent=1), encoding="utf-8")
    paths = sorted((teacher / "traces").glob("*.json"))
    if a.cases:
        want = set(a.cases.split(","))
        paths = [p for p in paths if p.name.split("__")[0] in want]
    for p in paths[a.offset: a.offset + a.n]:
        t = json.loads(p.read_text(encoding="utf-8"))
        if not t.get("final_markdown"):
            continue
        call = t["calls"][-1]
        tools = P.retrieve_tool_schema() if call["with_tools"] else None
        try:
            res = client.chat(a.model, call["messages"], tools=tools, max_tokens=a.max_tokens, timeout=a.timeout,
                              purpose="eval_final", meta={"case_id": t["case_id"], "run_id": run_dir.name})
            status, err = "ok", None
            md = P.sanitize_markdown_output(res.content) if res.content.strip() else ""
            if not md or res.tool_calls:
                status, err = "error", "workflow: 最终步未产出报告正文" if not md else "workflow: 最终步又调用了工具"
            rec = res.to_dict(); rec["messages"] = call["messages"]; rec["with_tools"] = call["with_tools"]
        except LLMError as exc:
            status, err, md, rec = "error", f"{exc.code}: {exc.detail[:200]}", "", None
        out = {"case_id": t["case_id"], "model": a.model, "slug": client.models[a.model].slug, "kb": t["kb"], "mode": "final_only",
               "status": status, "error": err, "calls": [rec] if rec else [], "rounds": t["rounds"], "tool_call_issues": [],
               "initial_snippets": t["initial_snippets"], "visible_snippets": t["visible_snippets"], "final_raw": rec["content"] if rec else "",
               "final_markdown": md}
        calls = out["calls"]
        out["totals"] = {"calls": len(calls), "retrieval_rounds": len(t["rounds"]), "prompt_tokens": sum(c["prompt_tokens"] for c in calls),
                         "completion_tokens": sum(c["completion_tokens"] for c in calls), "reasoning_tokens": sum(c["reasoning_tokens"] for c in calls),
                         "cost_usd": round(sum(c["cost_usd"] for c in calls), 6), "latency_s": sum(c["latency_s"] for c in calls),
                         "last_finish_reason": calls[-1]["finish_reason"] if calls else None,
                         "any_truncated": any(c["finish_reason"] == "length" for c in calls)}
        (run_dir / "traces" / f"{t['case_id']}__{a.model}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{t['case_id']} {status} tokens={out['totals']['completion_tokens']} {out['totals']['latency_s']}s", flush=True)
    cmd_gates(argparse.Namespace(run=str(run_dir)))
    print(f"run={run_dir.name}")


def cmd_sample_final(a: argparse.Namespace) -> None:
    """同一固定检索上下文(取自轨迹 run 的最后一次调用)下,让模型重复生成 K 份最终回答,供偏好对/最佳样本筛选。
    每份存为 traces/<case>__<model>.s<k>.json,trace 的 model 字段为 "<model>.s<k>",可直接走 gates / judge。"""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from . import prompt as P
    client = make_client()
    preflight_openrouter(client, [a.model])
    src = resolve_run(a.run)
    meta = json.loads((src / "run.json").read_text(encoding="utf-8"))
    OutboundPolicy.load().check(kb_class=meta["kb_class"], case_kind="synthetic")
    gates = json.loads((src / "gates.json").read_text(encoding="utf-8")) if (src / "gates.json").exists() else {}
    run_dir = RUNS_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_{a.tag}"
    (run_dir / "traces").mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({**meta, "models": [a.model], "mode": "final_k", "k": a.k, "source_run": src.name, "effort": a.effort}, ensure_ascii=False, indent=1), encoding="utf-8")
    jobs = []
    for p in sorted((src / "traces").glob("*.json")):
        t = json.loads(p.read_text(encoding="utf-8"))
        g = gates.get(f"{t['case_id']}__{t['model']}")
        if t.get("status") != "ok" or not t.get("calls") or (g is not None and g["hard_fail"]):
            continue
        jobs += [(t, k) for k in range(a.k)]
    if a.exclude_pairs:   # 排除已用于训练的案件(v3a 没见过的案件上采样,才是干净的泛化信号)
        used = {json.loads(l)["pair_id"] for l in open(a.exclude_pairs, encoding="utf-8")}
        jobs = [(t, k) for t, k in jobs if t["case_id"] not in used]
    if a.limit:
        keep = sorted({t["case_id"] for t, _ in jobs})[: a.limit]
        jobs = [(t, k) for t, k in jobs if t["case_id"] in keep]

    def one(job):
        t, k = job
        call = t["calls"][-1]
        tools = P.retrieve_tool_schema() if call["with_tools"] else None
        name = f"{a.model}.s{k}"
        try:
            res = client.chat(a.model, call["messages"], tools=tools, max_tokens=a.max_tokens, timeout=a.timeout,
                              reasoning={"effort": a.effort}, purpose="sample_final", meta={"case_id": t["case_id"], "run_id": run_dir.name})
            md = P.sanitize_markdown_output(res.content) if res.content.strip() else ""
            status, err = ("ok", None) if (md and not res.tool_calls) else ("error", "workflow: 最终步未产出报告正文" if not md else "workflow: 最终步又调用了工具")
            rec = res.to_dict(); rec["messages"] = call["messages"]; rec["with_tools"] = call["with_tools"]
        except LLMError as exc:
            status, err, md, rec = "error", f"{exc.code}: {exc.detail[:200]}", "", None
        calls = [rec] if rec else []
        out = {"case_id": t["case_id"], "model": name, "slug": client.models[a.model].slug, "kb": t["kb"], "mode": "final_k", "sample": k,
               "status": status, "error": err, "calls": calls, "rounds": t["rounds"], "tool_call_issues": [],
               "initial_snippets": t["initial_snippets"], "visible_snippets": t["visible_snippets"],
               "final_raw": rec["content"] if rec else "", "final_markdown": md,
               "totals": {"calls": len(calls), "retrieval_rounds": len(t["rounds"]), "prompt_tokens": sum(c["prompt_tokens"] for c in calls),
                          "completion_tokens": sum(c["completion_tokens"] for c in calls), "reasoning_tokens": sum(c["reasoning_tokens"] for c in calls),
                          "reasoning_chars": sum(len(c.get("reasoning") or "") for c in calls),
                          "cost_usd": round(sum(c["cost_usd"] for c in calls), 6), "latency_s": sum(c["latency_s"] for c in calls),
                          "last_finish_reason": calls[-1]["finish_reason"] if calls else None,
                          "any_truncated": any(c["finish_reason"] == "length" for c in calls)}}
        (run_dir / "traces" / f"{t['case_id']}__{name}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        return f"{t['case_id']} {name} {status} comp={out['totals']['completion_tokens']} cost={out['totals']['cost_usd']}"

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for line in pool.map(one, jobs):
            print(line, flush=True)
    cmd_gates(argparse.Namespace(run=str(run_dir)))
    print(f"run={run_dir.name}")


def cmd_revise(a: argparse.Namespace) -> None:
    """教师修订:对"硬门全过、无循环、已有有效 luna 评审且有缺陷"的候选,让 reviser 依据缺陷清单最小改动地修订报告。"""
    from . import prompt as P
    from . import revise as RV
    client = make_client()
    preflight_openrouter(client, [a.reviser])
    src_dirs = [resolve_run(r) for r in a.runs.split(",")]
    meta0 = json.loads((src_dirs[0] / "run.json").read_text(encoding="utf-8"))
    OutboundPolicy.load().check(kb_class=meta0["kb_class"], case_kind="synthetic")
    run_dir = RUNS_DIR / (a.resume or f"{time.strftime('%Y%m%d_%H%M%S')}_{a.tag}")
    (run_dir / "traces").mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps({**meta0, "models": [a.reviser], "mode": "revise", "source_runs": [d.name for d in src_dirs],
                                                  "effort": a.effort}, ensure_ascii=False, indent=1), encoding="utf-8")
    jobs = []
    for rd in src_dirs:
        gates_all = json.loads((rd / "gates.json").read_text(encoding="utf-8")) if (rd / "gates.json").exists() else {}
        judged = json.loads((rd / "judgements.json").read_text(encoding="utf-8")) if (rd / "judgements.json").exists() else {}
        for p in sorted((rd / "traces").glob("*.json")):
            t = json.loads(p.read_text(encoding="utf-8"))
            key = f"{t['case_id']}__{t['model']}"
            g, j = gates_all.get(key), (judged.get(key) or {}).get("luna")
            if a.gate_fail:   # 硬门修复模式:有报告、仅因"引用不可见/量刑措辞/数量"等可修订项失败的候选
                if g is None or not g["hard_fail"] or not t.get("calls") or not t.get("final_markdown"):
                    continue
                if not {h["code"] for h in g["hard"]} <= RV.REPAIRABLE_GATES:
                    continue
                defects = RV.gate_defects(g)
            else:
                if t.get("status") != "ok" or not t.get("final_markdown") or not t.get("calls") or g is None or g["hard_fail"]:
                    continue
                if "reasoning_loop" in {w["code"] for w in g.get("warn", [])} or not (j and j.get("valid") and j.get("defects")):
                    continue
                defects = j["defects"]
            if (run_dir / "traces" / f"{t['case_id']}__{t['model']}+rev.json").exists():
                continue
            jobs.append((rd.name, t, defects))
    if a.limit:
        jobs = jobs[: a.limit]
    print(f"待修订 {len(jobs)} 份", flush=True)

    def one(job):
        src_run, t, defects = job
        call = t["calls"][-1]
        try:
            res = client.chat(a.reviser, RV.revise_messages(call["messages"], t["final_markdown"], defects), max_tokens=a.max_tokens,
                              reasoning={"effort": a.effort}, timeout=a.timeout, purpose="revise", meta={"case_id": t["case_id"], "run_id": run_dir.name})
        except LLMError as exc:
            return f"{t['case_id']} {t['model']} error {exc.code}"
        md = P.sanitize_markdown_output(res.content) if res.content.strip() else ""
        out = RV.revised_trace(t, md, res.content, src_run=src_run, reviser=a.reviser,
                               usage={"cost_usd": res.cost_usd, "completion_tokens": res.completion_tokens, "finish_reason": res.finish_reason})
        (run_dir / "traces" / f"{t['case_id']}__{out['model']}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        return f"{t['case_id']} {out['model']} ok chars={len(md)} cost={res.cost_usd}"

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for line in pool.map(one, jobs):
            print(line, flush=True)
    cmd_gates(argparse.Namespace(run=str(run_dir)))
    print(f"run={run_dir.name}")


def cmd_revsft_build(a: argparse.Namespace) -> None:
    """把"学生原生思考 + 教师修订后的报告"构成 SFT 样本。采用条件:修订稿硬门全过且无循环、字段覆盖率不明显下降、有引用、
    各评审(至少 luna)都不比原稿差且平均提升 ≥ min-gain。每个案件按评审分取前 per-case 份;留 hold 个案件做留出集。"""
    import collections
    rev_dir = resolve_run(a.rev)
    gr_all = json.loads((rev_dir / "gates.json").read_text(encoding="utf-8"))
    jr_all = json.loads((rev_dir / "judgements.json").read_text(encoding="utf-8")) if (rev_dir / "judgements.json").exists() else {}
    tpl = SFT.QwenTemplate()
    stats, cache = collections.Counter(), {}

    def scores(j):
        return {n: r["total"] for n in ("luna", "minimax") if (r := (j or {}).get(n)) and r.get("valid")}

    cands = []
    for p in sorted((rev_dir / "traces").glob("*.json")):
        t = json.loads(p.read_text(encoding="utf-8"))
        src = t["source"]
        kr, ko = f"{t['case_id']}__{t['model']}", f"{t['case_id']}__{src['model']}"
        if src["run"] not in cache:
            sd = resolve_run(src["run"])
            cache[src["run"]] = (json.loads((sd / "gates.json").read_text(encoding="utf-8")), json.loads((sd / "judgements.json").read_text(encoding="utf-8")))
        go, jo = cache[src["run"]][0].get(ko), cache[src["run"]][1].get(ko)
        gr = gr_all.get(kr)
        if gr is None or gr["hard_fail"] or "reasoning_loop" in {w["code"] for w in gr.get("warn", [])}:
            stats["修订稿硬门/循环未过"] += 1
            continue
        cr, co = gr["metrics"].get("fact_coverage") or 0.0, (go or {}).get("metrics", {}).get("fact_coverage") or 0.0
        if cr < co - a.cov_drop:
            stats["覆盖率明显下降"] += 1
            continue
        if (gr["metrics"].get("citations") or 0) < 1:
            stats["无引用"] += 1
            continue
        sr, so = scores(jr_all.get(kr)), scores(jo)
        common = set(sr) & set(so)
        if "luna" not in common and a.abs_min and "luna" in sr and not so:
            # 原稿硬门失败、没有评审:改用绝对线(修订稿 luna 评审 ≥ abs-min)
            if sr["luna"] < a.abs_min:
                stats["修订稿评审低于绝对线"] += 1
                continue
            gains = {}
        elif "luna" not in common:
            stats["luna 评审缺失"] += 1
            continue
        else:
            gains = {n: sr[n] - so[n] for n in common}
            if min(gains.values()) < 0 or sum(gains.values()) / len(gains) < a.min_gain:
                stats["评审未提升"] += 1
                continue
        cands.append({"t": t, "key": kr, "score": sum(sr.values()) / len(sr), "judge": sr, "gains": gains, "cov": cr})
    by_case = collections.defaultdict(list)
    for c in cands:
        by_case[c["t"]["case_id"]].append(c)
    cases = sorted(by_case)
    hold = set(cases[-a.hold:]) if a.hold else set()
    train, evals = [], []
    for case_id in cases:
        picked = sorted(by_case[case_id], key=lambda c: (-c["score"], len(c["t"]["calls"][-1].get("reasoning") or "")))[: a.per_case]
        for c in picked:
            call = dict(c["t"]["calls"][-1], content=c["t"]["final_markdown"], tool_calls=[])
            thinking = (call.get("reasoning") or "").strip()
            s = SFT.build_sample(tpl, call, thinking, reasoning_effort=a.template_effort)
            if len(s["input_ids"]) > a.max_len:
                stats["超长"] += 1
                continue
            cut = s["target_text"].index("</think>") + len("</think>") + 2           # 报告正文从 </think> 后的空行之后开始
            report_start = s["prompt_tokens"] + len(tpl.encode(s["target_text"][:cut]))
            row = {"sample_id": c["key"], "case_id": case_id, "teacher": "luna_revision", "report_start": report_start, "kind": "final", "thinking_source": "student_native",
                   "judge": c["judge"], "gains": c["gains"], "coverage": c["cov"], "prompt_tokens": s["prompt_tokens"],
                   "target_tokens": s["target_tokens"], "input_ids": s["input_ids"], "labels": s["labels"], "target_text": s["target_text"]}
            (evals if case_id in hold else train).append(row)
    out = SFT_DIR / a.name
    out.mkdir(parents=True, exist_ok=True)
    for fname, rows in (("train.jsonl", train), ("eval.jsonl", evals)):
        (out / fname).write_text("".join(json.dumps(r, ensure_ascii=False) + chr(10) for r in rows), encoding="utf-8")
    man = {"name": a.name, "rev_run": rev_dir.name, "train": len(train), "eval": len(evals), "cases_train": len({r["case_id"] for r in train}),
           "tokens_train": sum(len(r["input_ids"]) for r in train), "min_gain": a.min_gain, "per_case": a.per_case, "cov_drop": a.cov_drop,
           "template_effort": a.template_effort, "template_sha256": tpl.template_sha256, "skipped": dict(stats),
           "mean_gain": round(sum(sum(r["gains"].values()) / len(r["gains"]) for r in train if r["gains"]) / max(sum(1 for r in train if r["gains"]), 1), 2),
           "abs_min_rows": sum(1 for r in train if not r["gains"]),
           "note": "合成案件;未经人工核实;思考为学生原生输出,报告为教师(luna max)依据评审缺陷的最小改动修订稿"}
    (out / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(man, ensure_ascii=False, indent=1))


def cmd_ret_contexts(a: argparse.Namespace) -> None:
    """T1 候选:最终调用还带着检索工具、却引用了"可见片段里没有"的条文,且该条文在知识库里存在。写出 contexts.json 与写手用的 writer_input.md。"""
    from . import retrieval_data as RD
    kb = pick_kb("production")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ctxs, seen = [], set()
    for r in a.runs.split(","):
        rd = resolve_run(r)
        gates_all = json.loads((rd / "gates.json").read_text(encoding="utf-8"))
        for p in sorted((rd / "traces").glob("*.json")):
            t = json.loads(p.read_text(encoding="utf-8"))
            key = f"{t['case_id']}__{t['model']}"
            g = gates_all.get(key)
            if not g or not g["hard_fail"] or not t.get("calls"):
                continue
            call = t["calls"][-1]
            if not call["with_tools"] or call["tool_calls"] or not (call.get("reasoning") or "").strip():
                continue
            picked = None
            for h in g["hard"]:
                if h["code"] != "article_not_in_visible_text":
                    continue
                for it in h.get("items", []):
                    if "条" not in it or ("《" not in it and not re.fullmatch(r"第[0-9零一二三四五六七八九十百]+条", it.strip())):
                        continue
                    tg = RD.find_targets(kb, it)
                    if tg:
                        picked = (it, tg)
                        break
                if picked:
                    break
            if not picked:
                continue
            it, tg = picked
            ident = (t["case_id"], RD.parse_item(it))
            if ident in seen:
                continue
            seen.add(ident)
            ctxs.append({"id": f"c{len(ctxs) + 1:02d}", "run": rd.name, "key": key, "case_id": t["case_id"], "item": it, "targets": tg[:3],
                         "draft": RD.cited_sentences(t.get("final_markdown") or call.get("content") or "", it),
                         "visible_titles": [s.get("title") for s in t.get("visible_snippets", [])][:9]})
    (out / "contexts.json").write_text(json.dumps(ctxs, ensure_ascii=False, indent=1), encoding="utf-8")
    view = [{"id": c["id"], "想引用的条文": c["item"], "草稿中的相关句子": c["draft"], "已可见的片段标题": c["visible_titles"],
             "知识库中该条的标题与原文开头": [{"标题": x["title"], "原文开头": x["excerpt"]} for x in c["targets"][:1]]} for c in ctxs]
    (out / "writer_input.md").write_text(RD.WRITER_INSTRUCTION + "\n## 情境\n```json\n" + json.dumps(view, ensure_ascii=False, indent=1) + "\n```\n", encoding="utf-8")
    print(f"T1 情境 {len(ctxs)} 个 → {out}(涉及案件 {len({c['case_id'] for c in ctxs})} 个)")


def cmd_ret_collect(a: argparse.Namespace) -> None:
    """收集各写手输出,用真实 kb.search 验证后写 verified.json。"""
    import collections
    from . import retrieval_data as RD
    kb = pick_kb("production")
    d = Path(a.dir)
    ctxs = {c["id"]: c for c in json.loads((d / "contexts.json").read_text(encoding="utf-8"))}
    verified, stats = {}, collections.Counter()
    for w in a.writers.split(","):
        f = d / f"out_{w}.json"
        if not f.exists():
            stats[f"{w}: 无输出文件"] += 1
            continue
        arr = RD.extract_array(f.read_text(encoding="utf-8", errors="replace"))
        if not arr:
            stats[f"{w}: 无法解析"] += 1
            continue
        for o in arr:
            c = ctxs.get(str(o.get("id")))
            if not c:
                continue
            ok, why = RD.verify_writer_item(kb, c, o)
            stats[f"{w}: {'通过' if ok else why}"] += 1
            if ok:
                verified.setdefault(c["id"], []).append({"writer": w, **{k: o[k] for k in ("closing", "query", "reason", "top_k")}})
    (d / "verified.json").write_text(json.dumps(verified, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    print(f"情境 {len(ctxs)} 个,至少一个写手验证通过的 {len(verified)} 个")


def cmd_ret_build(a: argparse.Namespace) -> None:
    """构建主动检索 SFT 数据:T1(事后重标注的"核对引用→检索")+ T2(好结果轨迹的检索调用)+ 回放(最终报告步、硬门修复)。"""
    import collections
    import random
    d = Path(a.dir)
    ctxs = {c["id"]: c for c in json.loads((d / "contexts.json").read_text(encoding="utf-8"))}
    verified = json.loads((d / "verified.json").read_text(encoding="utf-8"))
    tpl = SFT.QwenTemplate()
    stats, rows = collections.Counter(), []
    rng = random.Random(3)

    def make(call2, thinking, case_id, sample_id, kind, anchor, extra):
        s = SFT.build_sample(tpl, call2, thinking, reasoning_effort=a.template_effort)
        if len(s["input_ids"]) > a.max_len:
            stats[f"{kind}: 超长"] += 1
            return None
        idx = s["target_text"].rindex(anchor) if anchor else s["target_text"].index("</think>") + len("</think>") + 2
        rs = s["prompt_tokens"] + len(tpl.encode(s["target_text"][:idx]))
        return {"sample_id": sample_id, "case_id": case_id, "teacher": kind, "kind": kind, "thinking_source": "student_native+closing" if anchor else "student_native",
                "report_start": rs, "prompt_tokens": s["prompt_tokens"], "target_tokens": s["target_tokens"], "input_ids": s["input_ids"],
                "labels": s["labels"], "target_text": s["target_text"], **extra}

    # T1
    for cid, outs in sorted(verified.items()):
        c = ctxs[cid]
        t = json.loads((resolve_run(c["run"]) / "traces" / f"{c['key']}.json").read_text(encoding="utf-8"))
        call = t["calls"][-1]
        native = (call.get("reasoning") or "").strip()
        for o in [outs[(sum(map(ord, cid))) % len(outs)]][: a.t1_variants]:      # 按情境轮换写手,保证多模型混合
            args = {"query": o["query"][:120], "reason": o["reason"], "top_k": int(o["top_k"])}
            call2 = dict(call, content="", tool_calls=[{"name": "retrieve_knowledge", "arguments": args}])
            closing = o["closing"].strip()
            r = make(call2, native + "\n\n" + closing, c["case_id"], f"{cid}__{o['writer']}", "ret_t1", closing[:24], {"writer": o["writer"], "item": c["item"]})
            if r:
                rows.append(r)
                stats["T1 入选"] += 1
    # T2
    cands = []
    for tag in a.t2_runs.split(","):
        rd = resolve_run(tag)
        gates_all = json.loads((rd / "gates.json").read_text(encoding="utf-8"))
        judged = json.loads((rd / "judgements.json").read_text(encoding="utf-8")) if (rd / "judgements.json").exists() else {}
        for p in sorted((rd / "traces").glob("*.json")):
            t = json.loads(p.read_text(encoding="utf-8"))
            key = f"{t['case_id']}__{t['model']}"
            g, j = gates_all.get(key), (judged.get(key) or {}).get("luna")
            if t.get("status") != "ok" or not g or g["hard_fail"] or not (j and j.get("valid")) or j["total"] < a.t2_min:
                continue
            if "reasoning_loop" in {w["code"] for w in g.get("warn", [])} or not t.get("rounds"):
                continue
            retrieved = {str(s.get("id")) for r in t["rounds"] for s in r.get("snippets", []) if isinstance(s, dict)}
            if not any(i in (t.get("final_markdown") or "") for i in retrieved):   # 检索到的片段确实被最终报告引用 = 检索有用
                continue
            for ci, call in enumerate(t["calls"]):
                if call["tool_calls"] and (call.get("reasoning") or "").strip() and call["tool_calls"][0].get("arguments_valid", True):
                    cands.append((j["total"], t["case_id"], key, ci, call, tag))
    cands.sort(key=lambda x: -x[0])
    per_case = collections.Counter()
    for total, case_id, key, ci, call, tag in cands:
        if len([r for r in rows if r["kind"] == "ret_t2"]) >= a.t2_max or per_case[case_id] >= 1:
            continue
        r = make(call, (call.get("reasoning") or "").strip(), case_id, f"{key}__call{ci}", "ret_t2", None, {"judge": {"luna": total}, "src": tag})
        if r:
            rows.append(r)
            per_case[case_id] += 1
            stats["T2 入选"] += 1
    # 回放:最终报告步(v3c 数据)与硬门修复数据;重建为"原生思考 + 正例核对段落(由真实引用推出) + 修订报告",让模型见过"核对通过"的正例
    from . import retrieval_data as RD2
    for name, n, rev_tags in ((a.replay_from, a.replay_n, ("rev1", "rev2")), (a.gate_from, a.gate_n, ("rev3",))):
        if not name or n <= 0:
            continue
        pool = [json.loads(l) for l in open(SFT_DIR / name / "train.jsonl", encoding="utf-8")]
        by_case = collections.defaultdict(list)
        for r in pool:
            by_case[r["case_id"]].append(r)
        pick = [rng.choice(v) for v in by_case.values()]
        rng.shuffle(pick)
        done = 0
        for r in pick:
            if done >= n:
                break
            tr = None
            for tag in rev_tags:
                try:
                    pth = resolve_run(tag) / "traces" / f"{r['sample_id']}.json"
                except SystemExit:
                    continue
                if pth.exists():
                    tr = json.loads(pth.read_text(encoding="utf-8"))
                    break
            if tr is None:
                stats[f"回放 {name}: 找不到原轨迹"] += 1
                continue
            call = tr["calls"][-1]
            para = RD2.grounding_paragraph(tr["final_markdown"], tr.get("visible_snippets", []), int(r["case_id"][-3:]) if r["case_id"][-3:].isdigit() else 0)
            if not para:
                stats[f"回放 {name}: 推不出核对段落"] += 1
                continue
            call2 = dict(call, content=tr["final_markdown"], tool_calls=[])
            row = make(call2, (call.get("reasoning") or "").strip() + "\n\n" + para, r["case_id"], r["sample_id"], "final_grounded_replay", para[:24], {"src_replay": name})
            if row:
                rows.append(row)
                done += 1
                stats[f"回放 {name}(带核对段落)"] += 1
    rng.shuffle(rows)
    hold_cases = {r["case_id"] for r in rows if r["kind"].startswith("ret_t")}
    hold = set(sorted(hold_cases)[-a.hold:]) if a.hold else set()
    train = [r for r in rows if r["case_id"] not in hold]
    evals = [r for r in rows if r["case_id"] in hold]
    out = SFT_DIR / a.name
    out.mkdir(parents=True, exist_ok=True)
    for fname, rs in (("train.jsonl", train), ("eval.jsonl", evals)):
        (out / fname).write_text("".join(json.dumps(r, ensure_ascii=False) + chr(10) for r in rs), encoding="utf-8")
    kinds = collections.Counter(r["kind"] for r in train)
    man = {"name": a.name, "train": len(train), "eval": len(evals), "kinds": dict(kinds), "tokens_train": sum(len(r["input_ids"]) for r in train),
           "stats": dict(stats), "template_effort": a.template_effort, "note": "合成案件;未经人工核实;T1 的自检段落与检索查询由多个模型生成并经真实 kb.search 验证"}
    (out / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(man, ensure_ascii=False, indent=1))


def cmd_revpairs_build(a: argparse.Namespace) -> None:
    """报告段级偏好对:同一案件、同一提示、同一段原生思考,chosen=教师修订后通过硬门的报告,rejected=原报告。
    gate 对:原报告硬门失败(来源 --gate-revs),修订稿硬门通过且 luna ≥ abs-min;
    quality 对:原报告硬门通过但评审有缺陷(来源 --quality-revs),修订稿 luna 比原稿高 ≥ q-min、各评审不变差、覆盖率不降超过 cov-drop、有引用。
    因为两边思考完全相同,偏好信号只来自报告段(训练时只对报告段算对数概率)。"""
    import collections
    tpl = SFT.QwenTemplate()
    stats, rows = collections.Counter(), []

    def resp_and_start(call, thinking, content):
        s = SFT.build_sample(tpl, dict(call, content=content, tool_calls=[]), thinking, reasoning_effort=a.template_effort)
        resp = s["input_ids"][s["prompt_tokens"]:]
        cut = s["target_text"].index("</think>") + len("</think>") + 2
        return s["input_ids"][: s["prompt_tokens"]], resp, len(tpl.encode(s["target_text"][:cut]))

    def scores(j):
        return {n: r["total"] for n in ("luna", "minimax") if (r := (j or {}).get(n)) and r.get("valid")}

    for kind, revs in (("gate", a.gate_revs), ("quality", a.quality_revs)):
        cand = []
        for rtag in [x for x in revs.split(",") if x]:
            rd = resolve_run(rtag)
            gr_all = json.loads((rd / "gates.json").read_text(encoding="utf-8"))
            jr_all = json.loads((rd / "judgements.json").read_text(encoding="utf-8")) if (rd / "judgements.json").exists() else {}
            cache = {}
            for p in sorted((rd / "traces").glob("*.json")):
                t = json.loads(p.read_text(encoding="utf-8"))
                src = t["source"]
                kr, ko = f"{t['case_id']}__{t['model']}", f"{t['case_id']}__{src['model']}"
                if src["run"] not in cache:
                    sd = resolve_run(src["run"])
                    cache[src["run"]] = (sd, json.loads((sd / "gates.json").read_text(encoding="utf-8")), json.loads((sd / "judgements.json").read_text(encoding="utf-8")) if (sd / "judgements.json").exists() else {})
                sd, go_all, jo_all = cache[src["run"]]
                gr, go = gr_all.get(kr), go_all.get(ko)
                if gr is None or go is None or gr["hard_fail"] or "reasoning_loop" in {w["code"] for w in gr.get("warn", [])}:
                    stats[f"{kind}: 修订稿硬门/循环未过"] += 1
                    continue
                if (gr["metrics"].get("citations") or 0) < 1:
                    stats[f"{kind}: 无引用"] += 1
                    continue
                sr, so = scores(jr_all.get(kr)), scores(jo_all.get(ko))
                if "luna" not in sr:
                    stats[f"{kind}: 修订稿无评审"] += 1
                    continue
                if kind == "gate":
                    if not go["hard_fail"] or sr["luna"] < a.abs_min:
                        stats["gate: 原稿未失败或修订稿低于绝对线"] += 1
                        continue
                    gain = sr["luna"] - (so.get("luna") or 0)
                else:
                    if go["hard_fail"] or "luna" not in so:
                        stats["quality: 原稿失败或无评审"] += 1
                        continue
                    common = set(sr) & set(so)
                    gains = {n: sr[n] - so[n] for n in common}
                    if min(gains.values()) < 0 or gains["luna"] < a.q_min:
                        stats["quality: 提升不足"] += 1
                        continue
                    if (gr["metrics"].get("fact_coverage") or 0) < (go["metrics"].get("fact_coverage") or 0) - a.cov_drop:
                        stats["quality: 覆盖率下降"] += 1
                        continue
                    gain = gains["luna"]
                cand.append((gain, t, sd / "traces" / f"{ko}.json", kr))
        cand.sort(key=lambda x: -x[0])
        per_case, taken = collections.Counter(), 0
        cap = a.gate_max if kind == "gate" else a.q_max
        for gain, t, orig_path, kr in cand:
            if taken >= cap or per_case[t["case_id"]] >= 1:
                continue
            orig = json.loads(orig_path.read_text(encoding="utf-8"))
            ocall, thinking = orig["calls"][-1], (orig["calls"][-1].get("reasoning") or "").strip()
            try:
                p_c, c_ids, c_st = resp_and_start(ocall, thinking, t["final_markdown"])
                p_r, r_ids, r_st = resp_and_start(ocall, thinking, orig["final_markdown"])
            except ValueError:
                stats[f"{kind}: 模板构造失败"] += 1
                continue
            if p_c != p_r or c_st != r_st or c_ids[:c_st] != r_ids[:r_st]:
                stats[f"{kind}: 提示/思考前缀不一致"] += 1
                continue
            if len(p_c) + max(len(c_ids), len(r_ids)) > a.max_len or c_st >= len(c_ids) or r_st >= len(r_ids):
                stats[f"{kind}: 超长或空报告"] += 1
                continue
            rows.append({"pair_id": kr, "kind": kind, "case_id": t["case_id"], "gain": gain, "prompt_ids": p_c, "chosen_ids": c_ids, "rejected_ids": r_ids,
                         "c_start": c_st, "r_start": r_st})
            per_case[t["case_id"]] += 1
            taken += 1
            stats[f"{kind} 入选"] += 1
    rng = __import__("random").Random(4)
    rng.shuffle(rows)
    out = SFT_DIR / a.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "pairs.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + chr(10) for r in rows), encoding="utf-8")
    man = {"name": a.name, "pairs": len(rows), "kinds": dict(collections.Counter(r["kind"] for r in rows)), "cases": len({r["case_id"] for r in rows}),
           "mean_report_tokens_chosen": int(sum(len(r["chosen_ids"]) - r["c_start"] for r in rows) / max(len(rows), 1)),
           "mean_total_tokens": int(sum(len(r["prompt_ids"]) + len(r["chosen_ids"]) for r in rows) / max(len(rows), 1)),
           "stats": dict(stats), "note": "合成案件;未经人工核实;两边思考完全相同,仅报告段不同"}
    (out / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(man, ensure_ascii=False, indent=1))


def cmd_pairs_build(a: argparse.Namespace) -> None:
    from . import pairs as PR
    PR.MODE["judge"] = not a.no_judge
    PR.MODE["require_agree"] = a.require_agree
    run_dirs = [resolve_run(r) for r in a.runs.split(",")]
    cands = PR.collect_candidates(run_dirs)
    tpl = SFT.QwenTemplate()
    pairs, best, stats = PR.build_pair_rows(cands, tpl, min_judge=a.min_judge, margin=a.margin, effort=a.template_effort, max_len=a.max_len)
    out = SFT_DIR / a.name
    out.mkdir(parents=True, exist_ok=True)
    with (out / "pairs.jsonl").open("w", encoding="utf-8") as fh:
        for r in pairs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out / "best.jsonl").open("w", encoding="utf-8") as fh:
        for r in best:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {"name": a.name, "runs": [d.name for d in run_dirs], "stats": stats, "pairs": len(pairs), "best": len(best),
                "min_judge": a.min_judge, "margin": a.margin, "no_judge": a.no_judge, "template_effort": a.template_effort, "max_len": a.max_len,
                "pair_tokens_total": sum(len(r["prompt_ids"]) + len(r["chosen_ids"]) + len(r["rejected_ids"]) for r in pairs),
                "chosen_resp_tokens_mean": round(sum(r["chosen_resp_tokens"] for r in pairs) / max(len(pairs), 1)),
                "rejected_resp_tokens_mean": round(sum(r["rejected_resp_tokens"] for r in pairs) / max(len(pairs), 1)),
                "template_sha256": tpl.template_sha256, "note": "合成案件;未经人工核实;思考来自模型原生输出"}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=1))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="sr_eval")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check").set_defaults(fn=cmd_check)
    p = sub.add_parser("gen-cases")
    p.add_argument("--model", default="minimax"); p.add_argument("--split", default="dev"); p.add_argument("--timeout", type=float, default=900)
    p.add_argument("--start", type=int, default=0); p.add_argument("--n", type=int, default=6); p.add_argument("--workers", type=int, default=1)
    p.set_defaults(fn=cmd_gen_cases)
    p = sub.add_parser("run")
    p.add_argument("--models", required=True); p.add_argument("--cases", default="dev"); p.add_argument("--n", type=int); p.add_argument("--offset", type=int, default=0)
    p.add_argument("--kb", choices=["synthetic", "production"], default="synthetic")
    p.add_argument("--variant", choices=["production", "concise"], default="concise")
    p.add_argument("--max-tokens", type=int, default=40000); p.add_argument("--timeout", type=float, default=1800); p.add_argument("--effort"); p.add_argument("--workers", type=int, default=1)
    p.add_argument("--tag", default="run"); p.add_argument("--resume")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("gates"); p.add_argument("--run", required=True); p.set_defaults(fn=cmd_gates)
    p = sub.add_parser("judge"); p.add_argument("--run", required=True); p.add_argument("--judges", required=True); p.add_argument("--only", default=None, help="逗号分隔的 case_id__model,只评这些(给了空串会报错,避免误评全部)"); p.add_argument("--force", action="store_true", help="已有有效评审也重评(旧结果保留在 <judge>_prev)"); p.add_argument("--eligible", action="store_true", help="只评硬门全过且无思考循环的候选(需已有 gates.json)"); p.add_argument("--workers", type=int, default=1); p.add_argument("--effort", default="max")
    p.set_defaults(fn=cmd_judge)
    p = sub.add_parser("calibrate"); p.add_argument("--run", required=True); p.add_argument("--judge", required=True)
    p.add_argument("--n", type=int, default=3); p.add_argument("--workers", type=int, default=1); p.add_argument("--effort", default="max"); p.set_defaults(fn=cmd_calibrate)
    p = sub.add_parser("sft-build")
    p.add_argument("--runs", required=True); p.add_argument("--name", required=True); p.add_argument("--split", default="train")
    p.add_argument("--min-score", type=int, default=20); p.add_argument("--template-effort", default="low", help="写进 Qwen 模板的学生思考档位(xhigh/medium/low),与教师 effort 无关")
    p.add_argument("--keep-tool-prob", type=float, default=0.5); p.add_argument("--max-len", type=int, default=24000)
    p.add_argument("--rationale-model", default="luna"); p.add_argument("--workers", type=int, default=4)
    p.set_defaults(fn=cmd_sft_build)
    p = sub.add_parser("eval-final")
    p.add_argument("--run", required=True, help="教师 run(提供检索轨迹)"); p.add_argument("--model", required=True)
    p.add_argument("--cases", help="逗号分隔的 case_id"); p.add_argument("--n", type=int, default=3); p.add_argument("--offset", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=4000); p.add_argument("--timeout", type=float, default=3600); p.add_argument("--tag", default="evalfinal")
    p.set_defaults(fn=cmd_eval_final)
    p = sub.add_parser("sample-final"); p.add_argument("--exclude-pairs", default=None, help="排除该偏好对文件里已用的案件(pair_id)")
    p.add_argument("--run", required=True, help="轨迹 run(取其最后一次调用的固定上下文)"); p.add_argument("--model", default="qwen27")
    p.add_argument("--k", type=int, default=3); p.add_argument("--effort", default="xhigh"); p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=32000); p.add_argument("--timeout", type=float, default=3600)
    p.add_argument("--workers", type=int, default=8); p.add_argument("--tag", default="finalk")
    p.set_defaults(fn=cmd_sample_final)
    p = sub.add_parser("revise")
    p.add_argument("--runs", required=True, help="逗号分隔的来源 run(需已有 gates 与 luna 评审)"); p.add_argument("--reviser", default="luna")
    p.add_argument("--effort", default="max"); p.add_argument("--max-tokens", type=int, default=32000); p.add_argument("--timeout", type=float, default=1800)
    p.add_argument("--workers", type=int, default=8); p.add_argument("--limit", type=int); p.add_argument("--tag", default="rev"); p.add_argument("--resume")
    p.add_argument("--gate-fail", action="store_true", help="硬门修复模式:修订因引用不可见/量刑措辞等失败的报告(无需原稿评审)")
    p.set_defaults(fn=cmd_revise)
    p = sub.add_parser("revsft-build")
    p.add_argument("--rev", required=True); p.add_argument("--name", required=True); p.add_argument("--per-case", type=int, default=2)
    p.add_argument("--min-gain", type=float, default=1.0); p.add_argument("--cov-drop", type=float, default=0.03); p.add_argument("--hold", type=int, default=4)
    p.add_argument("--template-effort", default="xhigh"); p.add_argument("--max-len", type=int, default=32000)
    p.add_argument("--abs-min", type=float, default=0.0, help="原稿没有评审(硬门失败)时,修订稿 luna 评审的绝对下限")
    p.set_defaults(fn=cmd_revsft_build)
    p = sub.add_parser("ret-contexts")
    p.add_argument("--runs", required=True); p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_ret_contexts)
    p = sub.add_parser("ret-collect")
    p.add_argument("--dir", required=True); p.add_argument("--writers", required=True)
    p.set_defaults(fn=cmd_ret_collect)
    p = sub.add_parser("ret-build")
    p.add_argument("--dir", required=True); p.add_argument("--name", required=True)
    p.add_argument("--t1-variants", type=int, default=1); p.add_argument("--t2-runs", default="smp1,onp_v3c"); p.add_argument("--t2-min", type=float, default=18.0)
    p.add_argument("--t2-max", type=int, default=30); p.add_argument("--replay-from", default="v3c"); p.add_argument("--replay-n", type=int, default=20)
    p.add_argument("--gate-from", default="v3e_gate"); p.add_argument("--gate-n", type=int, default=10); p.add_argument("--hold", type=int, default=3)
    p.add_argument("--template-effort", default="xhigh"); p.add_argument("--max-len", type=int, default=32000)
    p.set_defaults(fn=cmd_ret_build)
    p = sub.add_parser("revpairs-build")
    p.add_argument("--name", required=True); p.add_argument("--gate-revs", default=""); p.add_argument("--quality-revs", default="")
    p.add_argument("--abs-min", type=float, default=18.0); p.add_argument("--q-min", type=float, default=3.0); p.add_argument("--cov-drop", type=float, default=0.08)
    p.add_argument("--gate-max", type=int, default=30); p.add_argument("--q-max", type=int, default=8)
    p.add_argument("--template-effort", default="xhigh"); p.add_argument("--max-len", type=int, default=28000)
    p.set_defaults(fn=cmd_revpairs_build)
    p = sub.add_parser("pairs-build")
    p.add_argument("--runs", required=True, help="逗号分隔:轨迹 run + 各 sample-final run(均需已有 gates/judgements)")
    p.add_argument("--name", required=True); p.add_argument("--min-judge", type=float, default=21.0)
    p.add_argument("--margin", type=float, default=2.0); p.add_argument("--template-effort", default="xhigh"); p.add_argument("--max-len", type=int, default=36000)
    p.add_argument("--require-agree", action="store_true", help="质量对要求 luna 与 MiniMax 都认可(chosen 比 rejected 高 ≥1 分)")
    p.add_argument("--no-judge", action="store_true", help="不依赖 LLM 评审,只用确定性指标(硬门/截断/循环/覆盖率/引用/长度)")
    p.set_defaults(fn=cmd_pairs_build)
    p = sub.add_parser("summary"); p.add_argument("--run", required=True); p.set_defaults(fn=cmd_summary)
    args = ap.parse_args(argv)
    try:
        args.fn(args)
    except LLMError as exc:
        print(f"[终止] {exc}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
