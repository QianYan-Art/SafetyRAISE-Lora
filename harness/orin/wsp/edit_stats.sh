#!/bin/bash
# 停服务 → 对 fit/ 下每条序列跑 dump_ffn stats（层 36–52），产物 fit/stats_<case>.*
W=~/work/wsp/phase1; L=$W/edit.log; cd $W
log() { echo "$(date +%T) $*" >> $L; }
P=$(pgrep -f "build/bin/llama-serve[r]"); [ -n "$P" ] && { kill $P; sleep 8; }
log "server stopped ($P)"
for f in fit/syn_dev_*.i32; do
  c=$(basename $f .i32)
  [ -f fit/stats_$c.sum ] && continue
  ~/work/wsp/scripts/dump_ffn ~/models/gguf/v3h2m-Q4_K_M.gguf $f fit/stats_$c 36 52 stats 4 512 >> $W/edit_dump.log 2>&1 || { log "STATS_FAIL $c"; exit 1; }
  log "stats $c ok"
done
log STATS_DONE
