#!/bin/bash
# 最终 GGUF（编辑权重 + 新 MTP 头）的行为验收：等 mtp_train_export 链 ALL_DONE 后，续写探针原提示 / P2 提示各 12 案，
# 标签 final_orig / final_p2（对照 none1、core_b36-52_s0.25 / p2_none、p2_core_b36-52_s0.25），写入 out_steer.jsonl；结束后恢复交付服务。
W=~/work/wsp/phase1; L=$W/final_probe.log; cd $W
log() { echo "$(date +%T) $*" >> $L; }
until grep -qE "ALL_DONE|FAIL|CHAIN_EXIT" ~/work/runs/wsp_mtp.status; do sleep 60; done
grep -q ALL_DONE ~/work/runs/wsp_mtp.status || { log "CHAIN_NOT_OK"; exit 1; }
log "serve final"
GGUF_DIR=~/work/wsp/gguf bash ~/work/wsp/scripts/serve_delivery_exp.sh v3h2m-wsp-mtp-Q4_K_M 6 5120 >> $L 2>&1 || { log "SERVER_FAIL"; exit 1; }
for spec in "probe.jsonl|final_orig" "probe_p2.jsonl|final_p2"; do
  IFS='|' read -r f tag <<< "$spec"
  log "probe $tag"
  ~/venvs/ft/bin/python ../scripts/probe.py $f out_steer.jsonl $tag >> $W/steer.log 2>&1 || { log "PROBE_FAIL $tag"; exit 1; }
  log "PROBE_DONE $tag"
done
bash ~/work/scripts/serve_delivery.sh v3h2m-Q4_K_M >> $L 2>&1
log FINAL_PROBE_DONE
