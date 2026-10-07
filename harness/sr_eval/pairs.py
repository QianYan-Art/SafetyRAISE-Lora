"""v3 路线的数据构造:同一固定检索上下文下的 K 份最终回答 → 最佳样本(阶段 A)与偏好对(阶段 B)。

候选 = 轨迹 run 里该案件的最终调用本身 + sample-final 的 K 份。全部走同一套硬门、字段覆盖率、luna(max) 评审。
chosen 必须"合格"(状态 ok、硬门全过、正常结束、无思考循环、评审有效且不低于阈值);
rejected 优先取"终止/硬门类"坏样本(截断、无答案、思考循环、硬门失败),其次取评审分明显更低者——
硬门失败者必为 rejected,不会因总分高而被选为 chosen。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .sft import EOS, QwenTemplate, assistant_message, build_sample


def _load(run_dir: Path, name: str, default: Any = None) -> Any:
    p = run_dir / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def prompt_sig(call: dict[str, Any]) -> str:
    """最终调用的完整输入(消息历史 + 是否带工具)的指纹。chosen 与 rejected 必须同一指纹,否则偏好信号不成立。"""
    return hashlib.sha256(json.dumps([call["messages"], call["with_tools"]], sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def collect_candidates(run_dirs: list[Path]) -> dict[str, list[dict[str, Any]]]:
    """case_id → 候选列表(含轨迹 run 的最终调用)。每个候选带 trace、门禁结果、评审分、覆盖率与合格性。"""
    out: dict[str, list[dict[str, Any]]] = {}
    for rd in run_dirs:
        gates = _load(rd, "gates.json", {})
        judged = _load(rd, "judgements.json", {})
        for p in sorted((rd / "traces").glob("*.json")):
            t = json.loads(p.read_text(encoding="utf-8"))
            if not t.get("calls"):
                continue
            key = f"{t['case_id']}__{t['model']}"
            g = gates.get(key)
            js = {n: r["total"] for n in ("luna", "minimax") if (r := (judged.get(key) or {}).get(n)) and r.get("valid")}   # 各评审的有效总分
            call = t["calls"][-1]
            warn = {w["code"] for w in (g or {}).get("warn", [])}
            reasons = []
            if t.get("status") != "ok" or not t.get("final_markdown"):
                reasons.append("no_report")
            if g is None or g["hard_fail"]:
                reasons.append("hard_gate")
            if call["finish_reason"] == "length":
                reasons.append("truncated")
            if "reasoning_loop" in warn:
                reasons.append("loop")
            judge_ok = "luna" in js                                    # luna 有效才算"已评审";有 MiniMax 时取两者均值
            cov = ((g or {}).get("metrics") or {}).get("fact_coverage")
            out.setdefault(t["case_id"], []).append({
                "name": f"{rd.name}:{t['model']}", "trace": t, "call": call, "reasons": reasons, "judge_ok": judge_ok, "sig": prompt_sig(call),
                "judge": (sum(js.values()) / len(js)) if judge_ok else None, "judges": js, "coverage": cov if cov is not None else 0.0,
                "reasoning_chars": len(call.get("reasoning") or ""),
                "chars": ((g or {}).get("metrics") or {}).get("chars") or 0,
                "citations": ((g or {}).get("metrics") or {}).get("citations") or 0,
            })
    return out


MODE = {"judge": True, "require_agree": False}   # False = 不依赖 LLM 评审,只用确定性指标(仓库所有者 2026-10-03:评审成本太高)


def score(c: dict[str, Any]) -> float:
    """有评审:评审分 + 3×字段覆盖率;无评审:覆盖率 − 4e-5×报告字数 − 2e-6×思考字符
    (奖励覆盖输入事实,轻罚冗长的报告与拖尾的思考;系数使两项罚分各约在 0.1~0.2,不压过覆盖率差异)。"""
    if MODE["judge"]:
        return (c["judge"] or 0.0) + 3.0 * c["coverage"]
    return c["coverage"] - 4e-5 * c["chars"] - 2e-6 * c["reasoning_chars"]


def eligible(c: dict[str, Any], min_judge: float) -> bool:
    if c["reasons"]:
        return False
    if MODE["judge"]:
        return c["judge_ok"] and c["judge"] >= min_judge
    return c["citations"] >= 1          # 无评审模式:必须带引用(无引用是已知缺陷)


def judges_agree(chosen: dict[str, Any], rejected: dict[str, Any], min_diff: float = 1.0) -> bool:
    """质量对的双评审一致性:两边都有评审的每个评审,chosen 都比 rejected 高出 min_diff;且至少有两个评审可比。"""
    common = set(chosen.get("judges", {})) & set(rejected.get("judges", {}))
    return len(common) >= 2 and all(chosen["judges"][n] - rejected["judges"][n] >= min_diff for n in common)


def pick_all(cands: list[dict[str, Any]], min_judge: float, margin: float):
    """按分数从高到低枚举 (chosen, rejected|None, kind)。rejected 只从"提示指纹与 chosen 完全相同"的候选里取;
    终止类 rejected 取思考长度与 chosen 最接近者(避免"拒绝样本更长"的长度捷径)。"""
    good = sorted((c for c in cands if eligible(c, min_judge)), key=lambda c: (score(c), -c["reasoning_chars"]), reverse=True)
    for chosen in good:
        pool = [c for c in cands if c.get("sig") == chosen.get("sig") and c is not chosen]
        bad_terminal = [c for c in pool if {"truncated", "loop", "no_report"} & set(c["reasons"])]
        bad_gate = [c for c in pool if "hard_gate" in c["reasons"] and c not in bad_terminal]
        if bad_terminal:
            yield chosen, min(bad_terminal, key=lambda c: abs(c["reasoning_chars"] - chosen["reasoning_chars"])), "terminal"
            continue
        if bad_gate:
            yield chosen, bad_gate[0], "gate"
            continue
        lower = [c for c in pool if (c["judge_ok"] or not MODE["judge"]) and score(c) <= score(chosen) - margin
                 and (not MODE["judge"] or not MODE.get("require_agree") or judges_agree(chosen, c))]
        yield (chosen, min(lower, key=score), "quality") if lower else (chosen, None, "")


def pick(cands: list[dict[str, Any]], min_judge: float, margin: float) -> tuple[dict | None, dict | None, str]:
    return next(pick_all(cands, min_judge, margin), (None, None, ""))


def response_ids(tpl: QwenTemplate, call: dict[str, Any], effort: str) -> tuple[list[int], list[int]]:
    """(prompt_ids, response_ids)。正常结束的样本用官方模板的目标文本;被截断的样本不补结束符/闭合标签。"""
    think = (call.get("reasoning") or "").strip()
    if call["finish_reason"] == "length":
        from . import prompt as P
        tools = P.retrieve_tool_schema() if call["with_tools"] else None
        prompt = tpl.render(call["messages"], tools, add_generation_prompt=True, reasoning_effort=effort)
        content = (call.get("content") or "").strip()
        resp = think + ("\n</think>\n\n" + content if content else "")
        return tpl.encode(prompt), tpl.encode(resp)
    msg_call = dict(call)
    s = build_sample(tpl, msg_call, think, reasoning_effort=effort)
    return s["input_ids"][: s["prompt_tokens"]], s["input_ids"][s["prompt_tokens"]:]


def build_pair_rows(cands_by_case: dict[str, list[dict[str, Any]]], tpl: QwenTemplate, *, min_judge: float, margin: float,
                    effort: str, max_len: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    pairs, best = [], []
    stats = {"cases": len(cands_by_case), "no_chosen": 0, "no_pair": 0, "too_long": 0, "rejected_too_long": 0, "empty": 0,
             "terminal": 0, "gate": 0, "quality": 0}
    for case_id, cands in sorted(cands_by_case.items()):
        if not any(eligible(c, min_judge) for c in cands):
            stats["no_chosen"] += 1
            continue
        picked = None
        for chosen, rejected, kind in pick_all(cands, min_judge, margin):       # 最高分超长就退到下一个合格者
            p_ids, c_ids = response_ids(tpl, chosen["call"], effort)
            if c_ids and len(p_ids) + len(c_ids) <= max_len:
                picked = (chosen, rejected, kind, p_ids, c_ids)
                break
        if picked is None:
            stats["too_long"] += 1
            continue
        chosen, rejected, kind, p_ids, c_ids = picked
        best.append({"case_id": case_id, "name": chosen["name"], "judge": chosen["judge"], "coverage": chosen["coverage"],
                     "prompt_tokens": len(p_ids), "response_tokens": len(c_ids), "input_ids": p_ids + c_ids,
                     "labels": [-100] * len(p_ids) + c_ids})
        if rejected is None:
            stats["no_pair"] += 1
            continue
        rp_ids, r_ids = response_ids(tpl, rejected["call"], effort)
        if rp_ids != p_ids:                                                      # 双方提示必须逐 token 相同
            stats["no_pair"] += 1
            continue
        if not r_ids:
            stats["empty"] += 1
            continue
        if len(p_ids) + len(r_ids) > max_len:
            # 只有"没收尾"的终止类失败,裁掉尾部才仍是同一种失败;硬门/质量类的失败证据可能在尾部,不能裁,整对丢弃
            if kind == "terminal" and (set(rejected["reasons"]) & {"truncated", "loop"} or rejected["call"]["finish_reason"] == "length"):
                r_ids = r_ids[: max_len - len(p_ids)]
            else:
                stats["rejected_too_long"] += 1
                continue
        stats[kind] += 1
        pairs.append({"pair_id": case_id, "kind": kind, "chosen": chosen["name"], "rejected": rejected["name"],
                      "chosen_score": round(score(chosen), 3), "rejected_score": None if (MODE["judge"] and not rejected["judge_ok"]) else round(score(rejected), 3),
                      "rejected_reasons": rejected["reasons"], "chosen_resp_tokens": len(c_ids), "rejected_resp_tokens": len(r_ids),
                      "prompt_ids": p_ids, "chosen_ids": c_ids, "rejected_ids": r_ids})
    return pairs, best, stats
