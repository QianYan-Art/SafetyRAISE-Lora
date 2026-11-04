#!/bin/bash
# 底座对照：官方 BF16 → Q8_0 GGUF → Q4_K_M（与交付同一量化流程，不合并 LoRA、不换 MTP 头）；中间 Q8_0 用完即删。
# 产物放实验目录，不放 ~/models/gguf（交付目录不动）。
set -e
PY=~/venvs/ft/bin/python; O=~/work/wsp/gguf; mkdir -p $O
echo "[1/3] convert q8_0 $(date +%T) 空闲 $(df --output=avail -BG ~ | tail -1)"
PYTHONPATH=~/work/llama.cpp/gguf-py $PY ~/work/llama.cpp/convert_hf_to_gguf.py ~/models/Qwen3.8-27B --outfile $O/base-Q8_0.gguf --outtype q8_0
echo "[2/3] quantize $(date +%T)"
~/work/llama.cpp/build/bin/llama-quantize --allow-requantize $O/base-Q8_0.gguf $O/base-Q4_K_M.gguf Q4_K_M
echo "[3/3] cleanup $(date +%T)"; rm -f $O/base-Q8_0.gguf
ls -la $O; df -h ~ | tail -1; echo GGUF_DONE
