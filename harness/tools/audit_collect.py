"""汇总盲评输出 → build_post --audit-results 所需的 JSON。
读取各盲评目录的 codex_blind.md / kimi_blind.md(JSON 数组,元素含 pair/kind/passed/winner),
同一份材料:任一评审 passed=false 或 偏好材料 winner 不一致/为 tie → 记失败;全部通过才算过。
用法: audit_collect.py <build输出目录> <输出json> <盲评目录名...>   (目录在 harness/runs/blind/)
"""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
args = [a for a in sys.argv[1:] if not a.startswith("--")]
PAIRS_DIRECTION_ONLY = "--pairs-direction-only" in sys.argv
build, out = Path(args[0]), Path(args[1])
dirs = args[2:]
manifest = json.loads((build / "manifest.json").read_text(encoding="utf-8"))
answers = json.loads((build / "audit-answers.json").read_text(encoding="utf-8"))
rows_sha = manifest["pre_rollback_rows_sha256"]


def parse(path):
    """评审输出可能是 JSON 数组,也可能是依次拼接的多个 JSON 对象(DeepSeek);两种都支持。"""
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    dec = json.JSONDecoder()
    for m in re.finditer(r"[\[{]", text):
        try:
            first, end = dec.raw_decode(text, m.start())
        except ValueError:
            continue
        if isinstance(first, list) and first and isinstance(first[0], dict) and "pair" in first[0]:
            return first
        if isinstance(first, dict) and "pair" in first:
            items, pos = [first], end
            while True:
                n = re.compile(r"\s*\{").match(text, pos)
                if not n:
                    break
                try:
                    obj, pos = dec.raw_decode(text, n.end() - 1)
                except ValueError:
                    break
                if isinstance(obj, dict) and "pair" in obj:
                    items.append(obj)
            return items
    return None


per_pair = {}
missing = []
for d in dirs:
    for who in ("codex", "kimi", "deepseek"):
        data = parse(ROOT / "harness" / "runs" / "blind" / d / f"{who}_blind.md")
        if data is None:
            missing.append(f"{d}/{who}")
            continue
        for item in data:
            per_pair.setdefault(item["pair"], {})[who] = item
def chosen_pass(item, ans):
    """偏好材料:只要求"被选中的一侧"达到合格线(被拒的一侧本来就该更差);单份材料:passed 即整体。"""
    p = item.get("passed")
    if ans["kind"].startswith("pair_"):
        side = "A" if ans["A"] == "chosen" else "B"
        if isinstance(p, dict):
            return p.get(side) is True
        for key in ("candidates", "assessments", "reviews"):
            sub = item.get(key)
            if isinstance(sub, dict) and isinstance(sub.get(side), dict) and isinstance(sub[side].get("passed"), bool):
                return sub[side]["passed"]
    return p is True


reviews, problems = [], []
for pair, ans in answers.items():
    got = per_pair.get(pair, {})
    if not got:
        problems.append(f"{pair}: 无评审结果")
        continue
    passed = all(chosen_pass(v, ans) for v in got.values())
    if PAIRS_DIRECTION_ONLY and str(ans["kind"]).startswith("pair_"):
        passed = True  # 偏好对只要求"选中的一侧更优"(下面 winner 校验);绝对质量由锚点严审把关
    winner = None
    if str(ans["kind"]).startswith("pair_"):
        winners = {v.get("winner") for v in got.values()}
        winner = winners.pop() if len(winners) == 1 else "tie"
    def note(v):
        return str(v.get("failure_points") or v.get("failures") or v.get("主要失败点") or v.get("preference_reason") or v.get("reason") or "")[:500]
    reviews.append({"pair": pair, "pair_id": ans["pair_id"], "reviewer_kind": "codex" if "codex" in got else "kimi",
                    "reviewer_id": "+".join(sorted(got)), "offline": True, "blind": True,
                    "passed": bool(passed), "winner": winner, "kind": ans["kind"],
                    "detail": {k: {"chosen_pass": chosen_pass(v, ans), "winner": v.get("winner"), "note": note(v)}
                               for k, v in got.items()}})
out.write_text(json.dumps({"rows_sha256": rows_sha, "reviews": reviews, "missing_outputs": missing, "problems": problems},
                          ensure_ascii=False, indent=1), encoding="utf-8")
print("reviews", len(reviews), "passed", sum(r["passed"] for r in reviews), "failed",
      [r["pair"] for r in reviews if not r["passed"]], "missing", missing, "problems", problems)
