#!/bin/bash
# 答案段投机对照:逐个配置起服务(交付配置:6 槽位、KV q8_0、预算 5120),用 answer_probe.py 在“完整提示 + 学生思考文本”之后续写答案段,
# 记录解码速度与草稿接受率。用于比较 MTP 单独 / n-gram 查表 + MTP 等组合(常规探针只测思考开头,测不出 n-gram 在“抄证据原文”的答案段上的收益)。
# 用法: answer_sweep.sh <GGUF 基名> <configs.txt> <answer_trace.json> <out.jsonl> [probe 额外参数,如 "--n 1500 --seed 11"]
# configs.txt 每行: 标签|环境变量(空格分隔 K=V,不能含空格的值;n-gram 参数用 SPEC_TYPES、NGRAM_N、NGRAM_M)
set -u
NAME=${1:?need gguf basename}; CFG=${2:?need configs}; TRACE=${3:?need answer trace}; OUT=${4:?need out jsonl}; PARGS=${5:---n 1500 --seed 11}
while IFS='|' read -r LABEL ENVS; do
  case "$LABEL" in ''|\#*) continue;; esac
  echo "=== $LABEL $(date +%T)  env: $ENVS" | tee -a "$OUT.log"
  # shellcheck disable=SC2086
  r=$(env KV_TYPE=q8_0 $ENVS LISTEN_HOST=127.0.0.1 LISTEN_PORT=8081 bash "$HOME/work/scripts/serve_llama.sh" "$NAME" 6 5120 2>&1 | tail -1)
  echo "$r" | tee -a "$OUT.log"
  case "$r" in
    SERVER_UP*) ;;
    *) echo "{\"label\":\"$LABEL\",\"error\":\"${r//\"/\'}\"}" >> "$OUT"; pkill -f "build/bin/llama-server"; sleep 3; continue;;
  esac
  # shellcheck disable=SC2086
  timeout 2400 python3 "$HOME/work/scripts/answer_probe.py" "$TRACE" --rounds 2 $PARGS --host 127.0.0.1:8081 --out "$HOME/work/runs/answer_${LABEL}.json" \
    | sed "s|^|{\"label\":\"$LABEL\",\"r\":|; s|\$|}|" >> "$OUT"
  cp "$HOME/work/runs/llama_${NAME}_compact.log" "$HOME/work/runs/answer_${LABEL}.log"
  grep "draft acceptance\|acc per pos" "$HOME/work/runs/llama_${NAME}_compact.log" | tail -4 | sed "s|^|[$LABEL] |" >> "$OUT.log"
  pkill -f "build/bin/llama-server"; sleep 4
done < "$CFG"
echo "ANSWER_SWEEP_DONE $(date +%T)" | tee -a "$OUT.log"
