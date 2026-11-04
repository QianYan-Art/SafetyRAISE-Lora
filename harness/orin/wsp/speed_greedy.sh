#!/bin/bash
# 旧头 / 新头的干净对比：两者主模型相同，贪心解码（temperature 0）下正文应一致，接受率差别只来自 MTP 头。
# 等 chain_final_probe.sh 的 FINAL_PROBE_DONE；6 个开发案原协议提示，从思考开头续写 1200 token；结束后恢复交付服务。
W=~/work/wsp/phase1; cd $W; OUTJ=$W/speed_greedy.jsonl; : > $OUTJ
until grep -qE "FINAL_PROBE_DONE|FAIL|NOT_OK" final_probe.log; do sleep 60; done
for spec in "wsp_oldhead|v3h2m-wsp-Q4_K_M" "wsp_newhead|v3h2m-wsp-mtp-Q4_K_M"; do
  IFS='|' read -r tag name <<< "$spec"
  GGUF_DIR=$HOME/work/wsp/gguf bash ~/work/wsp/scripts/serve_delivery_exp.sh $name 1 5120 > speedg_srv_$tag.log 2>&1 || { echo "SERVER_FAIL $tag"; exit 1; }
  ~/venvs/ft/bin/python - "$tag" "$OUTJ" <<'PY' || exit 1
import json, sys, urllib.request, hashlib
U = "http://192.168.55.1:8080"; tag, out = sys.argv[1], sys.argv[2]
post = lambda p, d: json.loads(urllib.request.urlopen(urllib.request.Request(U + p, data=json.dumps(d).encode(), headers={"Content-Type": "application/json"}), timeout=3600).read())
rows = [json.loads(l) for l in open("requests_dev.jsonl", encoding="utf-8")]
for r in [x for x in rows if x["case_id"] in ("syn_dev_000", "syn_dev_003", "syn_dev_010", "syn_dev_012", "syn_dev_018", "syn_dev_027")]:
    pr = post("/apply-template", {"messages": r["messages"], "chat_template_kwargs": {"reasoning_effort": "compact"}})["prompt"]
    res = post("/completion", {"prompt": pr, "n_predict": 1200, "temperature": 0, "cache_prompt": False})
    t = res["timings"]
    o = dict(tag=tag, case=r["case_id"], n=t.get("predicted_n"), tps=round(t.get("predicted_per_second", 0), 2),
             draft=t.get("draft_n", 0), acc=t.get("draft_n_accepted", 0), text_md5=hashlib.md5(res["content"].encode()).hexdigest()[:8])
    o["acc_rate"] = round(o["acc"] / max(o["draft"], 1), 3); print(o, flush=True)
    open(out, "a").write(json.dumps(o) + "\n")
PY
done
bash ~/work/scripts/serve_delivery.sh v3h2m-Q4_K_M > speedg_srv_restore.log 2>&1
echo SPEEDG_DONE
