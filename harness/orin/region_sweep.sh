#!/bin/bash
# 分区间投机对照:对每个配置起服务(交付配置:6 槽位、KV q8_0、预算 5120),先测“思考区间”再测“答案区间”(服务端默认采样、固定种子,与线上请求一致)——
#   思考区间:完整提示(约 15.6K token)从头生成 N1 个 token(speed_probe2.py);
#   答案区间:完整提示 + 学生真实的思考文本之后续写 N2 个 token(answer_probe.py)。
# 思考约占一次调用 4–5K token、答案约 4K token;两个区间的接受率差别很大(实测答案区间 MTP 接受率约 93%,思考开头约 31%),所以分开测。
# 用法: region_sweep.sh <GGUF 基名> <configs.txt> <probe_trace.json> <answer_trace.json> <out.jsonl> [N1=1500] [N2=1200]
# configs.txt 每行: 标签|环境变量(空格分隔 K=V)
set -u
NAME=${1:?need gguf basename}; CFG=${2:?need configs}; TRACE=${3:?need probe trace}; ATRACE=${4:?need answer trace}; OUT=${5:?need out jsonl}; N1=${6:-1500}; N2=${7:-1200}
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
  timeout 1500 python3 "$HOME/work/scripts/speed_probe2.py" "$TRACE" --n "$N1" --rounds 1 --seed 11 --host 127.0.0.1:8081 --out "$HOME/work/runs/think_${LABEL}.json" \
    | sed "s|^|{\"label\":\"$LABEL\",\"region\":\"think\",\"r\":|; s|\$|}|" >> "$OUT"
  timeout 1500 python3 "$HOME/work/scripts/answer_probe.py" "$ATRACE" --n "$N2" --rounds 1 --seed 11 --host 127.0.0.1:8081 --out "$HOME/work/runs/answer_${LABEL}.json" \
    | sed "s|^|{\"label\":\"$LABEL\",\"region\":\"answer\",\"r\":|; s|\$|}|" >> "$OUT"
  cp "$HOME/work/runs/llama_${NAME}_compact.log" "$HOME/work/runs/region_${LABEL}.log"
  grep "draft acceptance\|acc per pos" "$HOME/work/runs/llama_${NAME}_compact.log" | tail -4 | sed "s|^|[$LABEL] |" >> "$OUT.log"
  pkill -f "build/bin/llama-server"; sleep 4
done < "$CFG"
echo "REGION_SWEEP_DONE $(date +%T)" | tee -a "$OUT.log"
