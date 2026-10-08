#!/bin/bash
# 批量跑多槽位并发吞吐配置:逐行读 configs,每行 标签|并发路数|环境变量(空格分隔 K=V),按交付配置起服务(6 槽位、KV q8_0、预算 5120),
# 用 probe_multi.py 同时发 N 路请求(同一条合成案件提示截到约 5K token,每路 400 token,服务端默认采样),结果写 ~/work/runs/multi_<标签>.json/.log。
# 用法: [TRIM=4000] multi_batch.sh <GGUF 基名> <configs.txt>      TRIM=0 不截提示(完整约 15.7K token,真实上下文长度;每个配置要预填充 N 份,约多 7 分钟)
set -u
NAME=${1:?need gguf basename}; CFG=${2:?need configs}
S=$HOME/work/scripts; R=$HOME/work/runs; TR=${TR:-$HOME/work/data/probe_trace_dev000.json}
while IFS='|' read -r LABEL SLOTS ENVS; do
  case "$LABEL" in ''|\#*) continue;; esac
  echo "--- multi $LABEL slots=$SLOTS env: $ENVS $(date +%T)"
  # shellcheck disable=SC2086
  r=$(env LISTEN_HOST=127.0.0.1 LISTEN_PORT=8081 KV_TYPE=q8_0 $ENVS bash "$S/serve_llama.sh" "$NAME" 6 5120 2>&1 | tail -1)
  echo "$r"
  case "$r" in SERVER_UP*) ;; *) echo "{\"label\":\"$LABEL\",\"error\":\"fail\"}"; pkill -f "build/bin/llama-server"; sleep 3; continue;; esac
  timeout 2400 python3 "$S/probe_multi.py" "$TR" --slots "$SLOTS" --n 400 --trim-user-chars "${TRIM:-4000}" --host 127.0.0.1:8081 --out "$R/multi_$LABEL.json" | tail -1
  cp "$R/llama_${NAME}_compact.log" "$R/multi_$LABEL.log"
  pkill -f "build/bin/llama-server"; sleep 5
done < "$CFG"
echo "MULTI_BATCH_DONE $(date +%T)"
