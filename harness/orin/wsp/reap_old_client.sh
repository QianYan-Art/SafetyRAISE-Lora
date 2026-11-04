#!/bin/bash
# 等旧客户端在服务端上的 6 个任务全部结束 → SIGCONT 让它写出结果 → 结果数达到 8 条后杀掉它（不再让它发训练案）。
OLD=7126; L=$HOME/work/runs/llama_v3h2m-Q4_K_M_compact.log; OUT=$HOME/work/wsp/phase0/out_v3h2_nobudget.jsonl
TASKS="199 2 270 0 3 4"
while true; do
  n=0; for t in $TASKS; do grep -q "task $t | stop processing" "$L" && n=$((n+1)); done
  [ $n -eq 6 ] && break
  kill -0 $OLD 2>/dev/null || { echo "$(date +%T) old client gone early"; exit 1; }
  sleep 30
done
echo "$(date +%T) all 6 old tasks released on server; resuming old client"
kill -CONT $OLD
for i in $(seq 1 120); do
  [ "$(grep -c . "$OUT")" -ge 8 ] && break
  sleep 1
done
kill $OLD; sleep 1; kill -9 $OLD 2>/dev/null
echo "$(date +%T) old client killed; lines in out=$(grep -c . "$OUT")"
tail -6 $HOME/work/wsp/phase0/client.log
