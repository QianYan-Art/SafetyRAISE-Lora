"""由 build_post 的产出目录派生"最终训练集":按排除清单剔除行(不重建、不重新分词),写出 rows/provenance/manifest。
用法: finalize_rows.py <build目录> <输出目录> <排除清单.json> <审查摘要.json>
 排除清单: {"pair_id": "原因", ...};审查摘要: 任意 JSON,原样写入 manifest.independent_review_rounds。
manifest 如实记录:源构建目录哈希、被排除的行与原因、各轮抽检结论;release_eligible 仍为 false。
"""
import hashlib, json, sys
from collections import Counter
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
exclude = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
review = json.loads(Path(sys.argv[4]).read_text(encoding="utf-8"))
if out.exists() and any(out.iterdir()):
    raise SystemExit("输出目录必须为空")
out.mkdir(parents=True, exist_ok=True)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows = [json.loads(l) for l in open(src / "rows.jsonl", encoding="utf-8")]
prov = [json.loads(l) for l in open(src / "provenance.jsonl", encoding="utf-8")]
keep = [r for r in rows if r["pair_id"] not in exclude]
keep_ids = {r["pair_id"] for r in keep}
(out / "rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep), encoding="utf-8")
(out / "provenance.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in prov if p["pair_id"] in keep_ids), encoding="utf-8")
m = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
tot = sorted(len(r["prompt_ids"]) + max(len(r["chosen_ids"]), len(r["rejected_ids"] or [])) for r in keep)
kinds = Counter(r["kind"] for r in keep)
m["derived_from"] = {"build_dir": str(src), "rows_sha256": sha(src / "rows.jsonl"), "manifest_sha256": sha(src / "manifest.json")}
m["excluded_rows"] = exclude
m["counts"] = {"selected": len(keep), "written": len(keep),
               "anchors": sum(v for k, v in kinds.items() if k.startswith("anchor_")),
               "pairs": sum(v for k, v in kinds.items() if not k.startswith("anchor_"))}
m["kinds"] = dict(kinds)
m["window_distribution"] = {"count": len(tot), "p50": tot[(len(tot) + 1) // 2 - 1], "max": tot[-1], "over_28000": sum(t > 28000 for t in tot)}
m["independent_review_rounds"] = review
m["audit_policy_note"] = ("偏好对:被选中一侧须在两位独立评审(Codex、DeepSeek-flash-high)盲评中一致胜出,绝对质量不要求满分;"
                          "锚点:须通过 Codex 全量严审且在复审中无任何评审判不合格;Kimi 因配额未参与。训练在最终抽检完成前已开始,"
                          "完成后按本清单重启。人工未核实,仅合成数据。")
m["training_forbidden"] = False
m["release_eligible"] = False
m["files"] = {"rows.jsonl": sha(out / "rows.jsonl"), "provenance.jsonl": sha(out / "provenance.jsonl")}
for k in ("audit", "pre_rollback_rows_sha256", "discarded_calls"):
    m.pop(k, None)
(out / "manifest.json").write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({"kept": len(keep), "excluded": len(rows) - len(keep), "counts": m["counts"], "window": m["window_distribution"]}, ensure_ascii=False))
