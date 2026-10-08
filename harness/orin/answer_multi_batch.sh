#!/bin/bash
# 6 路并发的答案区间吞吐:逐个配置起服务(交付配置),answer_multi.py 同时发 N 路。每个配置要预填充 N 份 ~20K token 的提示,约 10–12 分钟。
# 用法: answer_multi_batch.sh <GGUF 基名> <configs.txt> <answer_trace.json>    configs 每行: 标签|并发路数|环境变量(空格分隔 K=V)
set -u
NAME=${1:?need gguf basename}; CFG=${2:?need configs}; ATRACE=${3:?need answer trace}
S=$HOME/work/scripts; R=$HOME/work/runs
while IFS='|' read -r LABEL SLOTS ENVS; do
  case "$LABEL" in ''|\#*) continue;; esac
  echo "--- answer_multi $LABEL slots=$SLOTS env: $ENVS $(date +%T)"
  # shellcheck disable=SC2086
  r=$(env LISTEN_HOST=127.0.0.1 LISTEN_PORT=8081 KV_TYPE=q8_0 $ENVS bash "$S/serve_llama.sh" "$NAME" 6 5120 2>&1 | tail -1)
  echo "$r"
  case "$r" in SERVER_UP*) ;; *) echo "{\"label\":\"$LABEL\",\"error\":\"fail\"}"; pkill -f "build/bin/llama-server"; sleep 3; continue;; esac
  timeout 3000 python3 "$S/answer_multi.py" "$ATRACE" --slots "$SLOTS" --n 800 --host 127.0.0.1:8081 --out "$R/amulti_$LABEL.json" | tail -1
  cp "$R/llama_${NAME}_compact.log" "$R/amulti_$LABEL.log"
  pkill -f "build/bin/llama-server"; sleep 5
done < "$CFG"
echo "ANSWER_MULTI_DONE $(date +%T)"
