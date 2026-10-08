#!/bin/bash
# 6 槽位并发解码的 nsys 剖析:同一提示 6 路并发,先用 6 token 的短生成预热(各槽位缓存好提示),正式生成稳定后抓一段内核并汇总。
# 用法: prof_multi.sh <GGUF 基名> <输出前缀> <trace.json> [抓取秒数=20]    环境变量同 serve_llama.sh(LLAMA_DIR、SR_MTP_DRAFT_IDS、SPEC_N_MAX、SR_SPEC_BATCH_TOKENS 等)
set -u
NAME=${1:?need gguf basename}; OUT=${2:?need output prefix}; TRACE=${3:?need trace json}; SECS=${4:-20}
export LISTEN_HOST=127.0.0.1 LISTEN_PORT=8081
export WRAP="nsys launch --session-new=srprof --trace=cuda,nvtx --cuda-graph-trace=node"
bash "$HOME/work/scripts/serve_llama.sh" "$NAME" 6 5120 | tail -1
python3 "$HOME/work/scripts/probe_multi.py" "$TRACE" --slots 6 --n 700 --warm-n 6 --trim-user-chars 4000 --host 127.0.0.1:8081 --out "${OUT}_probe.json" > "${OUT}_probe.txt" 2>&1 &
PP=$!
for i in $(seq 1 900); do grep -q '"phase": "warm"' "${OUT}_probe.txt" 2>/dev/null && break; sleep 2; done
sleep 12
nsys start --session=srprof --sample=none --cpuctxsw=none -o "$OUT" --force-overwrite=true
sleep "$SECS"
nsys stop --session=srprof
wait $PP
pkill -f "build/bin/llama-server"; sleep 3
nsys shutdown --session=srprof > /dev/null 2>&1
nsys export --type sqlite --force-overwrite true -o "${OUT}.sqlite" "${OUT}.nsys-rep" > /dev/null 2>&1
python3 "$HOME/work/scripts/prof_summary.py" "${OUT}.sqlite" 45 > "${OUT}_summary.txt" 2>&1
echo "PROF_DONE $OUT"
