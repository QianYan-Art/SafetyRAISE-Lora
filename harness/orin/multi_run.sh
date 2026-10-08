#!/bin/bash
# 多槽位并发吞吐的单个配置:按交付配置起服务(KV q8_0、预算 5120、6 槽位),用 probe_multi.py 同时发 N 路请求,结果写 ~/work/runs/multi_<标签>.json/.log。
# 用法: multi_run.sh <GGUF 基名> <标签> <并发路数> [环境变量 K=V ...]
#   例: multi_run.sh v3h2m-Q4_K_M sl32k_b16_s6 6 LLAMA_DIR=$HOME/work/llama-opt SR_MTP_DRAFT_IDS=$HOME/work/data/draft_ids/draft_k32768.i32 SR_SPEC_BATCH_TOKENS=16
# 与 post_chain_auto.sh 里的 multi() 同口径(同一条合成案件的提示截到约 5K token,每路 400 个 token,服务端默认采样),可在自动流程结束后补跑其它配置。
set -u
NAME=${1:?need gguf basename}; LABEL=${2:?need label}; SLOTS=${3:?need slots}; shift 3
S=$HOME/work/scripts; R=$HOME/work/runs; TR=${TR:-$HOME/work/data/probe_trace_dev000.json}
env LISTEN_HOST=127.0.0.1 LISTEN_PORT=8081 KV_TYPE=q8_0 "$@" bash "$S/serve_llama.sh" "$NAME" 6 5120 | tail -1
timeout 2400 python3 "$S/probe_multi.py" "$TR" --slots "$SLOTS" --n 400 --trim-user-chars 4000 --host 127.0.0.1:8081 --out "$R/multi_$LABEL.json" | tail -1
cp "$R/llama_${NAME}_compact.log" "$R/multi_$LABEL.log"
pkill -f "build/bin/llama-server"; sleep 5
echo "MULTI_DONE $LABEL"
