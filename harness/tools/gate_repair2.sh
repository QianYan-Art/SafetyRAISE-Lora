#!/bin/bash
# 第二份硬门修复数据:v3d 自己在新训练案件上的"可修复硬门失败"(引用不可见法条/量刑措辞…)→ 教师精准修复(rev5)→ luna(max) 评审 → revsft-build(原稿无评审,绝对线 abs-min 18)。
# 只构建 harness/sft/v3g_gate/,不上传不训练;等 v3f 评测后再决定如何用。日志 harness/runs/gate_repair2.log。
cd <repo>; export PYTHONPATH=harness PYTHONIOENCODING=utf-8; PY=.venv/Scripts/python.exe; C="$PY -B -m sr_eval.cli"
exec >> harness/runs/gate_repair2.log 2>&1
echo "=== $(date +%T) 修复 v3d 自己的硬门失败"
$C revise --runs onp_v3d_a,onp_v3d_b --gate-fail --workers 24 --tag rev5 > harness/runs/rev5.log 2>&1
grep -q "^run=" harness/runs/rev5.log || { echo "ALERT: rev5 未正常结束"; exit 1; }
echo "=== $(date +%T) 评审"
$C judge --run rev5 --judges luna --eligible --workers 24 --effort max
$C judge --run rev5 --judges luna --eligible --workers 24 --effort max
echo "=== $(date +%T) 构建 v3g_gate"
$C revsft-build --rev rev5 --name v3g_gate --per-case 1 --abs-min 18 --cov-drop 0.08 --hold 2 --max-len 32000
echo "EVENT: 第二份硬门修复数据就绪 v3g_gate"
