#!/bin/bash
# v3 数据放大(API 侧,不占 Orin GPU):新增训练案 → 轨迹 → 固定检索上下文重采 K 份最终回答 → 硬门 → 合格候选用 luna(max)+MiniMax(max) 评审。
# 用法: bash scale_data.sh   日志:harness/runs/scale.log。每步产出的 run 目录名固定带标签,可断点续跑(--resume)。
cd <repo>; export PYTHONPATH=harness PYTHONIOENCODING=utf-8; PY=.venv/Scripts/python.exe; C="$PY -B -m sr_eval.cli"
L=harness/runs/scale.log; exec >> $L 2>&1
echo "=== $(date +%T) 阶段1 轨迹采样"
$C run --models qwen27 --cases train --n 60 --offset 180 --kb production --effort xhigh --max-tokens 32000 --timeout 3600 --workers 12 --tag smp2a &
$C run --models qwen27 --cases train --n 40 --offset 0 --kb production --effort xhigh --max-tokens 32000 --timeout 3600 --workers 8 --tag smp2b &
wait
echo "=== $(date +%T) 阶段2 最终回答重采 K=3"
$C sample-final --run smp2a --k 3 --effort xhigh --max-tokens 32000 --timeout 3600 --workers 12 --tag finalk2a &
$C sample-final --run smp2b --k 3 --effort xhigh --max-tokens 32000 --timeout 3600 --workers 8 --tag finalk2b &
wait
echo "=== $(date +%T) 阶段3 硬门"
for r in smp2a smp2b finalk2a finalk2b finalk smp1; do $C gates --run $r; done
echo "=== $(date +%T) 阶段4 评审(合格候选;luna max + MiniMax max)"
for r in smp1 finalk smp2a smp2b finalk2a finalk2b; do
  $C judge --run $r --judges luna --eligible --workers 12 --effort max &
done
wait
for r in smp1 finalk smp2a smp2b finalk2a finalk2b; do
  $C judge --run $r --judges minimax --eligible --workers 3 --effort max &
done
wait
echo "=== $(date +%T) SCALE_DONE"
