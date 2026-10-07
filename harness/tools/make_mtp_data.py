"""为 MTP 头重训构造"主模型自己的输出"序列(线上协议、compact 档位渲染):提示 + 学生思考(含预算提示语)+ 学生最终 JSON。
训练集取训练案的学生首调用输出(排除后训练 v3h 已用过的案件,避免只拟合已见数据);留出集取 dev 案的在线评测输出(终版模型自己的输出,再用 --eval-traces 指定)。
输出 train_mtp.py 可读的 jsonl:sample_id, case_id, kind=final, input_ids, labels(只覆盖回答段含思考)。
用法: make_mtp_data.py --revisions <修订记录目录> --exclude-rows <最终集 rows 的 provenance.jsonl> --n 60 --out <train.jsonl>
      make_mtp_data.py --eval-traces <live 运行目录> --n 12 --out <eval.jsonl>
运行环境 .venv-live(需要冻结的线上代码与分词器)。
"""
import argparse, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("SR_QWEN_PROFILE_DIR", str(ROOT / "profiles" / "assets" / "qwen3.8-compact-v1"))
sys.path[:0] = [str(ROOT / "harness"), str(ROOT / "harness" / "vendor" / "safetyraise_7200e30")]
sys.dont_write_bytecode = True
from sr_eval.live.samples import render_call_sample  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--revisions")
ap.add_argument("--exclude-rows")
ap.add_argument("--eval-traces")
ap.add_argument("--n", type=int, default=60)
ap.add_argument("--max-len", type=int, default=32768)
ap.add_argument("--out", required=True)
a = ap.parse_args()

items = []  # (case_id, sample_id, payload, reasoning, content)
if a.eval_traces:
    for f in sorted(Path(a.eval_traces).glob("*.json")):
        t = json.loads(f.read_text(encoding="utf-8"))
        for i, c in enumerate(t["calls"]):
            if c.get("role") == "generator" and not c.get("not_sent") and c.get("response") and c.get("content"):
                items.append((t["case_id"], f"{t['case_id']}-c{i}", c["payload"], c.get("reasoning_content", ""), c["content"]))
else:
    skip = set()
    if a.exclude_rows:
        skip = {json.loads(l)["case_id"] for l in open(a.exclude_rows, encoding="utf-8")}
    for f in sorted(Path(a.revisions).glob("*.json")):
        if f.name == "summary.json":
            continue
        r = json.loads(f.read_text(encoding="utf-8"))
        if r.get("case_id") in skip or not r.get("student_content") or r.get("call_index") not in (0, None):
            continue
        items.append((r["case_id"], r["source_call_sha256"][:12], r["payload"], r.get("reasoning_content", ""), r["student_content"]))
seen, out, toks = set(), [], 0
for case_id, sid, payload, reasoning, content in items:
    if case_id in seen:
        continue
    call = {"payload": payload, "content": content, "reasoning_content": reasoning, "reasoning": reasoning, "finish_reason": "stop", "tool_calls": []}
    s = render_call_sample({}, call, reasoning_effort="compact")
    if s.get("status") not in ("exact", "exact_bridge"):
        continue
    p, r_ = s["prompt_ids"], s["response_ids"]
    if len(p) + len(r_) > a.max_len:
        continue
    seen.add(case_id)
    out.append({"sample_id": sid, "case_id": case_id, "kind": "final", "input_ids": p + r_, "labels": [-100] * len(p) + r_})
    toks += len(p) + len(r_)
    if len(out) >= a.n:
        break
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text("".join(json.dumps(x) + "\n" for x in out), encoding="utf-8")
print(f"{len(out)} 条, {toks} token -> {a.out}")
