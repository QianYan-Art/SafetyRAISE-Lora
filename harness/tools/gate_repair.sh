#!/bin/bash
# 硬门修复数据(笔记本侧,API):等 revise --gate-fail(rev3) 结束 → luna(max) 评审修订稿 → revsft-build(原稿无评审,用绝对线 abs-min 18)。
# 只构建 harness/sft/v3e_gate/,不上传、不训练;等 v3d 评测结果出来再决定如何使用。日志 harness/runs/gate_repair.log。
cd <repo>; export PYTHONPATH=harness PYTHONIOENCODING=utf-8; PY=.venv/Scripts/python.exe; C="$PY -B -m sr_eval.cli"
exec >> harness/runs/gate_repair.log 2>&1
until grep -q "^run=" harness/runs/rev3.log 2>/dev/null; do
  powershell -NoProfile -Command "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'gate-fail' } | Measure-Object).Count" | tr -dc 0-9 | grep -q "[1-9]" || break
  sleep 30
done
grep -q "^run=" harness/runs/rev3.log || { echo "ALERT: rev3 未正常结束"; exit 1; }
echo "=== $(date +%T) 评审修订稿"
$C judge --run rev3 --judges luna --eligible --workers 24 --effort max
$C judge --run rev3 --judges luna --eligible --workers 24 --effort max
echo "=== $(date +%T) 构建 v3e_gate"
$C revsft-build --rev rev3 --name v3e_gate --per-case 1 --abs-min 18 --cov-drop 0.08 --hold 2 --max-len 32000
echo "EVENT: 硬门修复数据就绪 v3e_gate"
