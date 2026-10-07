#!/bin/bash
# 用 luna(max 档)按"生产真实注入"的同一条流程重新生成教师基线(开发集 50 案 + 测试集 50 案),再用 luna(max)/MiniMax(max) 评审。
# 原因:旧的教师基线(lunadev/lunadev2/lunatest,10-02~10-03)没有显式指定思考档位(run 记录 reasoning=None,用的是当时 models.json 的默认档),
# 最终报告调用的思考 token 中位仅约 1.5K,而 max 档的评审/修订调用是 16~20K,所以旧基线不是 max 档。仓库所有者要求 luna 一律用 max。
# 日志 harness/runs/teacher_max.log;结束输出 EVENT:。
cd <repo>; export PYTHONPATH=harness PYTHONIOENCODING=utf-8; PY=.venv/Scripts/python.exe; C="$PY -B -m sr_eval.cli"
exec >> harness/runs/teacher_max.log 2>&1
echo "=== $(date +%T) 生成:开发集 50 案 + 测试集 50 案(luna max,production 变体 concise,与学生评测同一条流程)"
$C run --models luna --cases dev --n 50 --kb production --variant concise --effort max --max-tokens 64000 --workers 12 --timeout 3600 --tag lunamax_dev > harness/runs/lunamax_dev.log 2>&1 &
$C run --models luna --cases test --n 50 --kb production --variant concise --effort max --max-tokens 64000 --workers 12 --timeout 3600 --tag lunamax_test > harness/runs/lunamax_test.log 2>&1 &
wait
echo "=== $(date +%T) 评审(luna max / MiniMax max)"
for r in lunamax_dev lunamax_test; do
  $C judge --run $r --judges luna --workers 24 --effort max &
  $C judge --run $r --judges minimax --workers 8 --effort max &
done
wait
for r in lunamax_dev lunamax_test; do $C judge --run $r --judges luna --workers 24 --effort max; $C summary --run $r; done
echo "EVENT: luna(max) 教师基线生成与评审完成(lunamax_dev、lunamax_test)"
