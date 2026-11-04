#!/bin/bash
# 转向筛选：逐个配置以交付配置起服务（附控制向量），跑续写探针。配置格式：标签|EXTRA_ARGS
W=~/work/wsp/phase1; L=$W/steer.log; cd $W
CV=$W/cv_core.gguf; RV=$W/cv_rand.gguf
while IFS='|' read -r tag extra; do
  [ -z "$tag" ] && continue
  grep -q "PROBE_DONE $tag$" $L 2>/dev/null && continue
  echo "$(date +%T) config $tag :: $extra" >> $L
  EXTRA_ARGS="$extra" bash ~/work/scripts/serve_delivery.sh v3h2m-Q4_K_M 6 5120 >> $L 2>&1 || { echo "SERVER_FAIL $tag" >> $L; exit 1; }
  ~/venvs/ft/bin/python $W/../scripts/probe.py probe.jsonl out_steer.jsonl "$tag" >> $L 2>&1 || { echo "PROBE_FAIL $tag" >> $L; exit 1; }
done <<CFG
none0|
none1|
core_b36-52_s0.10|--control-vector-scaled $CV:0.10 --control-vector-layer-range 36 52
core_b36-52_s0.25|--control-vector-scaled $CV:0.25 --control-vector-layer-range 36 52
core_b36-52_s0.50|--control-vector-scaled $CV:0.50 --control-vector-layer-range 36 52
rand_b36-52_s0.25|--control-vector-scaled $RV:0.25 --control-vector-layer-range 36 52
CFG
echo STEER_DONE >> $L
