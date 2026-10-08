#!/bin/bash
# 解码阶段性能剖析:用 nsys 抓取一段稳定解码(预填充结束后)的 CUDA 内核,导出 sqlite 并汇总(prof_summary.py)。
# 用法: prof_decode.sh <GGUF 基名> <输出前缀> <trace.json> [抓取秒数=15]
# 环境变量同 serve_llama.sh(LLAMA_DIR、KV_TYPE、SPEC_N_MAX、SR_MTP_DRAFT_IDS 等);服务监听 127.0.0.1:8081,6 槽位、预算 5120。
set -u
NAME=${1:?need gguf basename}; OUT=${2:?need output prefix}; TRACE=${3:?need trace json}; SECS=${4:-15}
export LISTEN_HOST=127.0.0.1 LISTEN_PORT=8081
export WRAP="nsys launch --session-new=srprof --trace=cuda,nvtx --cuda-graph-trace=node"
bash "$HOME/work/scripts/serve_llama.sh" "$NAME" 6 5120 | tail -1
LOG="$HOME/work/runs/llama_${NAME}_compact.log"
python3 "$HOME/work/scripts/speed_probe2.py" "$TRACE" --temp 0 --n 900 --rounds 1 --host 127.0.0.1:8081 --out "${OUT}_probe.json" > "${OUT}_probe.txt" 2>&1 &
PP=$!
for i in $(seq 1 300); do grep -q "progress = 1.00" "$LOG" && break; sleep 2; done
sleep 8
nsys start --session=srprof --sample=none --cpuctxsw=none -o "$OUT" --force-overwrite=true
sleep "$SECS"
nsys stop --session=srprof
wait $PP
pkill -f "build/bin/llama-server"; sleep 3
nsys shutdown --session=srprof > /dev/null 2>&1
nsys export --type sqlite --force-overwrite true -o "${OUT}.sqlite" "${OUT}.nsys-rep" > /dev/null 2>&1
python3 "$HOME/work/scripts/prof_summary.py" "${OUT}.sqlite" 45 > "${OUT}_summary.txt" 2>&1
echo "PROF_DONE $OUT"
