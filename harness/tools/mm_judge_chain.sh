#!/bin/bash
# MiniMax(max)评审补跑链:并发 ≤4(过载教训:16 路并发 max 档会触发 529/504),逐个 run 串行。已评的条目自动跳过。
cd <repo>; export PYTHONPATH=harness PYTHONIOENCODING=utf-8; PY=.venv/Scripts/python.exe
exec >> harness/runs/mm_judge_chain.log 2>&1
# 等已在跑的 lunamax_dev 评审结束
while powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*judge --run lunamax_dev*' } | Measure-Object | ForEach-Object Count" | grep -qv '^0'; do sleep 60; done
for r in lunamax_dev lunamax_test lunatest v3f_test50; do
  echo "=== $(date +%T) MiniMax 评审 $r"
  $PY -B -m sr_eval.cli judge --run $r --judges minimax --workers 4 --effort max
done
echo "EVENT: $(date +%T) MiniMax 补评完成(lunamax_dev/lunamax_test/lunatest/v3f_test50)"
