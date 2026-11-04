#!/bin/bash
# 生成编辑后的 Q4_K_M：中间 F16 GGUF（17 个 ffn_down）→ llama-quantize Q4_K_M（不允许重量化：其余张量逐字节拷贝）
# 用法: edit_make.sh <标签> <v.npy|zero> <scale>
W=~/work/wsp/phase1; G=~/work/wsp/gguf; L=$W/edit.log; cd $W
log() { echo "$(date +%T) $*" >> $L; }
TAG=$1; V=$2; S=$3
log "build $TAG ($V, scale $S)"
~/venvs/ft/bin/python ../scripts/edit_build.py ~/models/gguf/v3h2m-Q4_K_M.gguf $G/$TAG-mid.gguf $V $S dir_core.npy > $W/build_$TAG.log 2>&1 || { log "BUILD_FAIL $TAG"; exit 1; }
~/work/llama.cpp/build/bin/llama-quantize $G/$TAG-mid.gguf $G/$TAG-Q4_K_M.gguf Q4_K_M >> $W/build_$TAG.log 2>&1 || { log "QUANT_FAIL $TAG"; exit 1; }
rm -f $G/$TAG-mid.gguf
log "MADE $TAG $(stat -c %s $G/$TAG-Q4_K_M.gguf)"
