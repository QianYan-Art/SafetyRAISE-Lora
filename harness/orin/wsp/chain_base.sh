#!/bin/bash
# 串联：等 P1 重放结束、底座 GGUF 转好 → 用交付同配置（预算 5120、6×40960、KV q8_0、MTP 草稿、官方采样）起底座服务 → 跑开发 12 案 ×2 采样。
# 底座 GGUF 在 ~/work/wsp/gguf（不进交付目录）；服务脚本是交付脚本的副本，只把 GGUF 目录换成 GGUF_DIR。
W=~/work/wsp; L=$W/phase1/chain.log
log() { echo "$(date +%T) $*" >> $L; }
until grep -q ALL_DONE $W/phase1/client_p1.log 2>/dev/null; do
  pgrep -f "replay_clien[t].py requests_p1" >/dev/null || { log "P1 client gone without ALL_DONE"; exit 1; }; sleep 60; done
log "P1 done"
until grep -q GGUF_DONE $W/gguf/make_base.log 2>/dev/null; do
  pgrep -f "make_bas[e].sh" >/dev/null || { log "make_base gone without GGUF_DONE"; exit 1; }; sleep 60; done
log "GGUF done; switching server to base"
GGUF_DIR=$W/gguf bash $W/scripts/serve_delivery_exp.sh base-Q4_K_M 6 5120 >> $L 2>&1 || { log "server start failed"; exit 1; }
cd $W/phase1
~/venvs/ft/bin/python $W/scripts/replay_client.py requests_dev.jsonl out_base_5120.jsonl --slot_ctx 40960 --workers 6 --samples 2 > client_base.log 2>&1
log "base client exit $?"
