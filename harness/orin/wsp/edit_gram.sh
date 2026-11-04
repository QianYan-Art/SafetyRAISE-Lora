#!/bin/bash
# 等 STATS_DONE → 选候选神经元（拟合 6 案，K0=512）→ 9 条序列逐条跑 dump_ffn gram
W=~/work/wsp/phase1; L=$W/edit.log; cd $W
log() { echo "$(date +%T) $*" >> $L; }
until grep -qE "STATS_DONE|STATS_FAIL" $L; do sleep 20; done
grep -q STATS_FAIL $L && exit 1
FIT=syn_dev_000,syn_dev_003,syn_dev_004,syn_dev_006,syn_dev_018,syn_dev_038
~/venvs/ft/bin/python ../scripts/edit_select.py $FIT 512 > fit/select.txt 2>&1 || { log SELECT_FAIL; exit 1; }
log "select ok"
for f in fit/syn_dev_*.i32; do
  c=$(basename $f .i32)
  [ -f fit/gram_$c.gg ] && continue
  ~/work/wsp/scripts/dump_ffn ~/models/gguf/v3h2m-Q4_K_M.gguf $f fit/gram_$c 36 52 gram fit/idx.i32 512 4 512 >> $W/edit_dump.log 2>&1 || { log "GRAM_FAIL $c"; exit 1; }
  log "gram $c ok"
done
log GRAM_DONE
