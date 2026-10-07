#!/bin/bash
# 用 MiniMax(max) 补全教师与 v3d/v3f 在开发集上的独立评审(luna 既是教师又是评审,可能偏向教师;MiniMax 是独立旁证)。只用 API;结果即时落盘,可重复运行只补缺。
# 日志 harness/runs/minimax_fill.log;结束输出 EVENT:。
cd <repo>; export PYTHONPATH=harness PYTHONIOENCODING=utf-8; PY=.venv/Scripts/python.exe; C="$PY -B -m sr_eval.cli"
exec >> harness/runs/minimax_fill.log 2>&1
for pass in 1 2; do
  echo "=== $(date +%T) 第 $pass 轮"
  for r in lunadev lunadev2 v3d_dev12 v3d_dev12to49 v3f_dev12 v3f_dev12to49; do
    $C judge --run $r --judges minimax --workers 10 --effort max
  done
done
echo "EVENT: MiniMax 补评完成"
