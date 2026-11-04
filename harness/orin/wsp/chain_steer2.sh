#!/bin/bash
# 转向筛选第二轮：检验转向与 P2 提示能否叠加，并补 0.35 剂量。配置格式：标签|探针文件|EXTRA_ARGS
W=~/work/wsp/phase1; L=$W/steer.log; cd $W
CV=$W/cv_core.gguf
while IFS='|' read -r tag probe extra; do
  [ -z "$tag" ] && continue
  grep -q "PROBE_DONE $tag$" $L 2>/dev/null && continue
  echo "$(date +%T) config $tag :: $probe :: $extra" >> $L
  EXTRA_ARGS="$extra" bash ~/work/scripts/serve_delivery.sh v3h2m-Q4_K_M 6 5120 >> $L 2>&1 || { echo "SERVER_FAIL $tag" >> $L; exit 1; }
  ~/venvs/ft/bin/python $W/../scripts/probe.py $probe out_steer.jsonl "$tag" >> $L 2>&1 || { echo "PROBE_FAIL $tag" >> $L; exit 1; }
done <<CFG
p2_none|probe_p2.jsonl|
p2_core_b36-52_s0.25|probe_p2.jsonl|--control-vector-scaled $CV:0.25 --control-vector-layer-range 36 52
core_b36-52_s0.35|probe.jsonl|--control-vector-scaled $CV:0.35 --control-vector-layer-range 36 52
CFG
echo STEER2_DONE >> $L
