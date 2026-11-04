#!/bin/bash
# 停底座服务 → 交付配置起 v3h2（serve_delivery.sh，预算 5120）→ P2 开发 12 案 ×2 采样
W=~/work/wsp; L=$W/phase1/chain_p2.log; log() { echo "$(date +%T) $*" >> $L; }
log "switch to v3h2 delivery (budget 5120)"
bash ~/work/scripts/serve_delivery.sh v3h2m-Q4_K_M 6 5120 >> $L 2>&1 || { log "server start failed"; exit 1; }
cd $W/phase1
~/venvs/ft/bin/python $W/scripts/replay_client.py requests_p2.jsonl out_p2_5120.jsonl --slot_ctx 40960 --workers 6 --samples 2 > client_p2.log 2>&1
log "p2 client exit $?"
