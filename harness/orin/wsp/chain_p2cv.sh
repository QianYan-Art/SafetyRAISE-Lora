#!/bin/bash
# 完整协议确认：交付配置 + 核心方向 0.25（层 36–52）+ P2 提示，开发 12 案 ×2 采样（与 out_p2_5120 同协议同预算，仅多控制向量）
W=~/work/wsp; L=$W/phase1/chain_p2cv.log; log() { echo "$(date +%T) $*" >> $L; }
CV=$W/phase1/cv_core.gguf
log "start v3h2 delivery + cv_core 0.25 b36-52"
EXTRA_ARGS="--control-vector-scaled $CV:0.25 --control-vector-layer-range 36 52" bash ~/work/scripts/serve_delivery.sh v3h2m-Q4_K_M 6 5120 >> $L 2>&1 || { log "server start failed"; exit 1; }
cd $W/phase1
~/venvs/ft/bin/python $W/scripts/replay_client.py requests_p2.jsonl out_p2cv_5120.jsonl --slot_ctx 40960 --workers 6 --samples 2 > client_p2cv.log 2>&1
log "p2cv client exit $?"
log P2CV_DONE
