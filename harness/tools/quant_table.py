"""多个量化版本(同一模型)首回合评测的横评表:通过数、解读指标失败、schema 失败、答案/思考长度、体积与速度(若有 speed_probe 输出)。
用法: quant_table.py <输出前缀> <名称=运行目录[:体积GB[:速度文件]]> ...   运行环境 .venv-live
"""
import json, os, statistics as st, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("SR_QWEN_PROFILE_DIR", str(ROOT / "profiles" / "assets" / "qwen3.8-compact-v1"))
sys.path[:0] = [str(ROOT / "harness"), str(ROOT / "harness" / "vendor" / "safetyraise_7200e30")]
sys.dont_write_bytecode = True
from sr_eval.live.revise import evaluate_response, load_traces  # noqa: E402

ONLY = set(filter(None, os.environ.get("QT_ONLY", "").split(",")))  # 只统计这些案件(用于在共同案件子集上公平比较)
rows = []
for arg in sys.argv[2:]:
    name, rest = arg.split("=", 1)
    parts = rest.split(":")
    d, size, sp = parts[0], (parts[1] if len(parts) > 1 else ""), (parts[2] if len(parts) > 2 else "")
    n = ok = schema = oc = fab = attr = trunc = 0
    think, ans = [], []
    case_ids = []
    for _, t in load_traces(str(ROOT / d))[0]:
        g = [c for c in t["calls"] if c.get("role") == "generator" and not c.get("not_sent")]
        if not g or not g[0].get("response"):
            continue
        if ONLY and t["case_id"] not in ONLY:
            continue
        c = g[0]; n += 1; case_ids.append(t["case_id"])
        v = evaluate_response(t, c)
        ok += bool(v["training_eligible"])
        sig = [str(s) for s in v.get("hard_problems", []) + v.get("fail_signals", [])]
        schema += any(s.startswith("candidate_schema") for s in sig)
        ch = (v.get("metrics") or {}).get("checks", {})
        oc += ch.get("obligation_consistency", {}).get("failed", 0)
        fab += ch.get("empty_field_fabrication", {}).get("failed", 0)
        attr += ch.get("evaluation_attribution", {}).get("failed", 0)
        fin = ((c.get("response") or {}).get("choices") or [{}])[0].get("finish_reason")
        trunc += fin == "length"
        rw = c.get("response_window") or {}
        if rw.get("c_start"):
            think.append(rw["c_start"])
            if rw.get("response_tokens"):
                ans.append(rw["response_tokens"] - rw["c_start"])
    speed = ""
    if sp and Path(sp).exists():
        vals = [json.loads(l).get("decode_tok_s") for l in open(sp, encoding="utf-8") if l.startswith("{")]
        vals = [x for x in vals if x]
        speed = round(sum(vals) / len(vals), 1) if vals else ""
    rows.append({"name": name, "n": n, "eligible": ok, "schema_fail_cases": schema, "obligation_field_fails": oc, "empty_field_fab_fails": fab,
                 "eval_attr_fails": attr, "truncated": trunc, "think_med": int(st.median(think)) if think else None,
                 "answer_med": int(st.median(ans)) if ans else None, "size_gb": size, "decode_tok_s_single_slot": speed, "cases": sorted(case_ids)})
Path(sys.argv[1] + ".json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
hdr = "| 量化 | 体积(GB) | 单槽位解码(token/s) | 案件数 | 通过确定性检查 | 义务一致性字段失败 | 空字段编造字段失败 | 评价字段归因失败 | schema 失败案件 | 被截断 | 答案中位(token) |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
body = "".join(f"| {r['name']} | {r['size_gb']} | {r['decode_tok_s_single_slot']} | {r['n']} | {r['eligible']} | {r['obligation_field_fails']} | {r['empty_field_fab_fails']} | {r['eval_attr_fails']} | {r['schema_fail_cases']} | {r['truncated']} | {r['answer_med']} |\n" for r in rows)
Path(sys.argv[1] + ".md").write_text("# 量化横评(首回合,开发 12 案,同一提示/检索/档位;自动指标,未经人工核实)\n\n" + hdr + body, encoding="utf-8")
print(hdr + body)
