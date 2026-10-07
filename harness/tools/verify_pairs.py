"""偏好对数据体检(复核发现后加):chosen/rejected 的提示是否一致、有无空响应、是否被裁切、各类的长度分布。
用法: verify_pairs.py <pairs.jsonl>"""
import json, sys, statistics, collections
from pathlib import Path

RUNS = Path(__file__).resolve().parent.parent / "runs"


def trace_of(name: str, case_id: str) -> dict:
    run, model = name.split(":", 1)
    return json.loads((RUNS / run / "traces" / f"{case_id}__{model}.json").read_text(encoding="utf-8"))


def main(path: str) -> None:
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    bad_prompt, empty, cut, lens = [], [], [], collections.defaultdict(lambda: ([], []))
    for r in rows:
        tc, tr = trace_of(r["chosen"], r["pair_id"]), trace_of(r["rejected"], r["pair_id"])
        cc, cr = tc["calls"][-1], tr["calls"][-1]
        if cc["messages"] != cr["messages"] or cc["with_tools"] != cr["with_tools"]:
            bad_prompt.append(r["pair_id"])
        if not r["chosen_ids"] or not r["rejected_ids"]:
            empty.append(r["pair_id"])
        if r["rejected_resp_tokens"] != len(r["rejected_ids"]) or len(r["prompt_ids"]) + len(r["rejected_ids"]) >= 36000:
            cut.append(r["pair_id"])
        lens[r["kind"]][0].append(len(r["chosen_ids"])); lens[r["kind"]][1].append(len(r["rejected_ids"]))
    print(f"对数 {len(rows)};提示不一致 {len(bad_prompt)} {bad_prompt};空响应 {len(empty)};疑似被裁切 {len(cut)}")
    for k, (c, rj) in lens.items():
        print(f"  {k:9s} n={len(c):2d}  chosen 响应 token 中位 {int(statistics.median(c)):6d}  rejected 中位 {int(statistics.median(rj)):6d}  rejected/chosen={statistics.median(rj)/statistics.median(c):.2f}")


main(sys.argv[1])
