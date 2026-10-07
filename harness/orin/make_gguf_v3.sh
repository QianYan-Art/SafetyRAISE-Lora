#!/bin/bash
# 用法: make_gguf_v3.sh <adapter_dir> <tag> [mtp_overlay.safetensors]
# 合并(含可选 MTP 覆盖)→ 转 Q8_0 GGUF(磁盘峰值约 80GB,比走 BF16 的 104GB 低)→ 删合并目录 → 重量化 Q4_K_M → 删 Q8_0。
# 空间不足时才清理可重建的旧产物(v2a 的 GGUF、基座 Q8_0 对照);官方 BF16 与 bnb 基座不动。
set -e
AD=$1; TAG=$2; OV=${3:-}; PY=~/venvs/ft/bin/python; MERGED=~/models/q38-27b-$TAG-bf16; G=~/models/gguf
free_gb() { df --output=avail -BG ~ | tail -1 | tr -dc 0-9; }
if [ "$(free_gb)" -lt 95 ]; then
  echo "[0/4] 空间 $(free_gb)GB < 95GB,清理可重建的旧产物 $(date +%T)"; rm -fv $G/base-Q8_0.gguf $G/v2a-mtp-Q4_K_M.gguf $G/v2a-lora-f16.gguf
  # 合并目录 52GB + Q8_0 28GB 会同时存在,峰值需要约 80GB 空闲:仍不足时,按最旧优先删其它自定义 Q4_K_M(可由适配器/官方 BF16 重建)
  NEED=80; [ -d $MERGED ] && NEED=32   # 合并目录已存在(上次失败后重跑)时只还需放下 Q8_0
  while [ "$(free_gb)" -lt $NEED ]; do
    old=$(ls -tr $G/*-Q4_K_M.gguf 2>/dev/null | grep -v "/$TAG-Q4_K_M.gguf" | head -1)
    [ -z "$old" ] && break
    echo "[0/4] 空间仍只有 $(free_gb)GB,删除最旧的 $old"; rm -fv "$old"
  done
fi
echo "[1/4] merge $(date +%T) 空闲 $(free_gb)GB"
$PY ~/work/scripts/merge_lora.py --base ~/models/Qwen3.8-27B --adapter $AD --out $MERGED ${OV:+--mtp_overlay $OV}
echo "[2/4] convert q8_0 $(date +%T) 空闲 $(free_gb)GB"
PYTHONPATH=~/work/llama.cpp/gguf-py $PY ~/work/llama.cpp/convert_hf_to_gguf.py $MERGED --outfile $G/$TAG-Q8_0.gguf --outtype q8_0
rm -rf $MERGED
echo "[3/4] quantize Q8_0→Q4_K_M $(date +%T)"
~/work/llama.cpp/build/bin/llama-quantize --allow-requantize $G/$TAG-Q8_0.gguf $G/$TAG-Q4_K_M.gguf Q4_K_M
echo "[4/4] cleanup $(date +%T)"; [ "${KEEP_Q8:-0}" = 1 ] || rm -f $G/$TAG-Q8_0.gguf   # KEEP_Q8=1:留着 Q8_0 供量化回归(Q6_K 等)再量化用
ls -la $G; df -h ~ | tail -1; echo GGUF_DONE
