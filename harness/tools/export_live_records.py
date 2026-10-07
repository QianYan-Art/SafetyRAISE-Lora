"""把后训练最终训练集导出成"文本版"记录(与分词器无关):提示消息、学生思考、选中/被拒回答、教师溯源、评审结论。
训练直接用 rows.jsonl(token id);records.jsonl 供审阅、换分词器/模板重建、扩展数据时使用。
用法: export_live_records.py <最终集目录> <修订记录目录> <输出 records.jsonl> [评审轮次摘要.json]
"""
import json, sys
from pathlib import Path

final, revs, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
rows = [json.loads(l) for l in open(final / "rows.jsonl", encoding="utf-8")]
prov = {json.loads(l)["pair_id"]: json.loads(l) for l in open(final / "provenance.jsonl", encoding="utf-8")}
n = 0
with open(out, "w", encoding="utf-8") as fh:
    for r in rows:
        p = prov[r["pair_id"]]
        rec = json.loads((revs / (p["source_call_sha256"] + ".json")).read_text(encoding="utf-8"))
        payload = rec["payload"]
        item = {
            "pair_id": r["pair_id"], "kind": r["kind"], "case_id": p["case_id"], "derivation": p["derivation"],
            "source_call_sha256": p["source_call_sha256"], "synthetic": True, "human_verified": False,
            "request": {k: v for k, v in payload.items() if k != "messages"},
            "messages": payload["messages"],
            "student_thinking": rec["reasoning_content"],
            "chosen": rec["chosen_content"],
            "rejected": (rec.get("rejected_content") or rec.get("student_content")) if r["rejected_ids"] else None,
            "tokens": {"prompt": len(r["prompt_ids"]), "chosen": len(r["chosen_ids"]),
                       "rejected": len(r["rejected_ids"]) if r["rejected_ids"] else None, "answer_start": r.get("c_start")},
            "teacher": {"backend": rec.get("teacher_backend"),
                        "model": ((rec.get("teacher_calls") or [{}])[0].get("payload") or {}).get("model"),
                        "ledger_ids": [t.get("ledger_id") for t in rec.get("teacher_calls", [])]},
            "original_fail_signals": (rec.get("original_validation") or {}).get("fail_signals", []),
            "unknown_metrics": (rec.get("validation") or {}).get("unknown_metrics", {}),
            "diff": rec.get("diff"),
        }
        fh.write(json.dumps(item, ensure_ascii=False) + "\n")
        n += 1
print("records", n, "->", out)
