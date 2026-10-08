#!/bin/bash
# 解码速度对照实验:逐个配置起服务(交付配置:6 槽位、KV、思考预算 5120)、各测两种采样,记录解码速度与草稿接受率。
# 用法: bench_sweep.sh <GGUF 基名,如 v3h2m-Q4_K_M> <configs.txt> <trace.json> <out.jsonl>
# configs.txt 每行: 标签|环境变量(空格分隔 K=V,可空)|probe 额外参数(可空,如 "--trim-user-chars 12000")
#   常用环境变量: LLAMA_DIR=$HOME/work/llama-opt  SR_MTP_DRAFT_IDS=<id 文件>  SPEC_N_MAX=5  SPEC_P_MIN=0.5
#                 KV_TYPE=q8_0|f16  SR_MMVQ_MAX_NE11=4  GGML_CUDA_GRAPH_OPT=1  EXTRA_ARGS="--spec-draft-sampling probabilistic"
# 每个配置两种测法:温度 0(确定性,看内核速度与最好情形的接受率)和服务端默认采样(与线上请求一致:请求里没有温度)。
set -u
NAME=${1:?need gguf basename}; CFG=${2:?need configs file}; TRACE=${3:?need trace json}; OUT=${4:?need out jsonl}
PROBE="${PROBE:-$HOME/work/scripts/speed_probe2.py}"   # 多槽位并发测法: PROBE=$HOME/work/scripts/probe_multi.py MODES="--slots 6 --n 500 --trim-user-chars 4000"
MODES=${MODES:---temp 0 --n 400 --rounds 2;--seed 11 --n 700 --rounds 2}   # 用分号分隔多种测法
IFS=';' read -ra MODE_LIST <<< "$MODES"
while IFS='|' read -r LABEL ENVS PARGS; do
  case "$LABEL" in ''|\#*) continue;; esac
  echo "=== $LABEL $(date +%T)  env: $ENVS  probe: $PARGS" | tee -a "$OUT.log"
  # shellcheck disable=SC2086
  r=$(env KV_TYPE=q8_0 $ENVS LISTEN_HOST=127.0.0.1 LISTEN_PORT=8081 bash "$HOME/work/scripts/serve_llama.sh" "$NAME" 6 5120 2>&1 | tail -1)
  echo "$r" | tee -a "$OUT.log"
  case "$r" in
    SERVER_UP*) ;;
    *) echo "{\"label\":\"$LABEL\",\"error\":\"${r//\"/\'}\"}" >> "$OUT"; pkill -f "build/bin/llama-server"; sleep 3; continue;;
  esac
  for mode in "${MODE_LIST[@]}"; do
    tag=$(echo "$mode" | tr -d ' -')
    # shellcheck disable=SC2086
    timeout 1200 python3 "$PROBE" "$TRACE" $mode $PARGS --host 127.0.0.1:8081 --out "$HOME/work/runs/probe_${LABEL}_${tag}.json" \
      | sed "s|^|{\"label\":\"$LABEL\",\"mode\":\"$mode\",\"r\":|; s|\$|}|" >> "$OUT"
  done
  cp "$HOME/work/runs/llama_${NAME}_compact.log" "$HOME/work/runs/bench_${LABEL}.log"
  grep "draft acceptance" "$HOME/work/runs/llama_${NAME}_compact.log" | tail -2 | sed "s|^|[$LABEL] |" >> "$OUT.log"
  pkill -f "build/bin/llama-server"; sleep 4
done < "$CFG"
echo "SWEEP_DONE $(date +%T)" | tee -a "$OUT.log"
