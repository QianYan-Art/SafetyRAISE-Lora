#!/bin/bash
# 编辑模型行为验证：续写探针（P2 提示）只跑编辑模型（零编辑已由数值验证覆盖）；再用编辑模型跑完整协议 P2 开发 12 案 ×1（阿颜要求缩短）。
W=~/work/wsp/phase1; L=$W/wsp.log; cd $W
log() { echo "$(date +%T) $*" >> $L; }
srv() { GGUF_DIR=~/work/wsp/gguf bash ~/work/wsp/scripts/serve_delivery_exp.sh $1 6 5120 >> $L 2>&1; }
for m in wsp; do
  tag=p2_edit_$m
  grep -q "PROBE_DONE $tag$" $W/steer.log 2>/dev/null && continue
  log "probe $tag"; srv v3h2m-$m-Q4_K_M || { log "SERVER_FAIL $m"; exit 1; }
  ~/venvs/ft/bin/python ../scripts/probe.py probe_p2.jsonl out_steer.jsonl $tag >> $W/steer.log 2>&1 || { log "PROBE_FAIL $tag"; exit 1; }
  log "PROBE_DONE $tag"
done
log "full protocol P2 on edited model"; srv v3h2m-wsp-Q4_K_M || { log "SERVER_FAIL full"; exit 1; }
~/venvs/ft/bin/python ../scripts/replay_client.py requests_p2.jsonl out_p2wsp_5120.jsonl --slot_ctx 40960 --workers 6 --samples 1 > client_p2wsp.log 2>&1
log "full client exit $?"
log WSP_DONE
