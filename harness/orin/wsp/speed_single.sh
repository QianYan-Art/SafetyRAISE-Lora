#!/bin/bash
# 单路速度与草稿接受率：交付版 / 编辑+旧头 / 编辑+新头，各以交付配置（1 槽位、思考预算 5120）起服务，
# 4 个开发案原协议提示，从思考开头续写 1200 token（固定种子、官方采样），记 predicted_per_second 与 draft 接受率。
W=~/work/wsp/phase1; cd $W; OUTJ=$W/speed_single.jsonl; : > $OUTJ
for spec in "delivery|$HOME/models/gguf|v3h2m-Q4_K_M" "wsp_oldhead|$HOME/work/wsp/gguf|v3h2m-wsp-Q4_K_M" "wsp_newhead|$HOME/work/wsp/gguf|v3h2m-wsp-mtp-Q4_K_M"; do
  IFS='|' read -r tag dir name <<< "$spec"
  GGUF_DIR=$dir bash ~/work/wsp/scripts/serve_delivery_exp.sh $name 1 5120 > speed_srv_$tag.log 2>&1 || { echo "SERVER_FAIL $tag"; exit 1; }
  ~/venvs/ft/bin/python - "$tag" "$OUTJ" <<'PY' || exit 1
import json, sys, urllib.request
U = "http://192.168.55.1:8080"; tag, out = sys.argv[1], sys.argv[2]
post = lambda p, d: json.loads(urllib.request.urlopen(urllib.request.Request(U + p, data=json.dumps(d).encode(), headers={"Content-Type": "application/json"}), timeout=3600).read())
rows = [json.loads(l) for l in open("requests_dev.jsonl", encoding="utf-8")]
for r in [x for x in rows if x["case_id"] in ("syn_dev_003", "syn_dev_010", "syn_dev_012", "syn_dev_027")]:
    pr = post("/apply-template", {"messages": r["messages"], "chat_template_kwargs": {"reasoning_effort": "compact"}})["prompt"]
    res = post("/completion", {"prompt": pr, "n_predict": 1200, "seed": 7, "cache_prompt": False})
    t = res["timings"]
    o = dict(tag=tag, case=r["case_id"], n=t.get("predicted_n"), tps=round(t.get("predicted_per_second", 0), 2),
             draft=t.get("draft_n", 0), acc=t.get("draft_n_accepted", 0))
    o["acc_rate"] = round(o["acc"] / max(o["draft"], 1), 3); print(o, flush=True)
    open(out, "a").write(json.dumps(o) + "\n")
PY
done
~/venvs/ft/bin/python - "$OUTJ" <<'PY'
import json, sys, collections
R = [json.loads(l) for l in open(sys.argv[1])]; g = collections.defaultdict(list)
for r in R: g[r["tag"]].append(r)
for t, v in g.items():
    print(t, "单路 token/s 均值", round(sum(x["tps"] for x in v) / len(v), 2), "草稿接受率", round(sum(x["acc"] for x in v) / max(sum(x["draft"] for x in v), 1), 3))
PY
echo SPEED_DONE
