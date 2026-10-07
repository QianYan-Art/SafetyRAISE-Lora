#!/bin/bash
# 教师修订后处理(笔记本侧,API 为主,不占 Orin GPU):等 revise 完成 → luna(max)+MiniMax(max) 评审修订稿(两进程并发写同一文件已修为合并写)
# → revsft-build 构建 SFT 数据 → 上传 Orin。日志 harness/runs/rev_pipeline.log。用法: bash rev_pipeline.sh [TAG=rev1] [数据名=v3b] [每案例=1]
TAG=${1:-rev1}; NAME=${2:-v3b}; PER=${3:-1}
cd <repo>; export PYTHONPATH=harness PYTHONIOENCODING=utf-8; PY=.venv/Scripts/python.exe; C="$PY -B -m sr_eval.cli"
K=<orin-ssh-key>; H=<orin-user>@192.168.55.1
exec >> harness/runs/rev_pipeline.log 2>&1
echo "=== $(date +%T) 等待 revise($TAG) 完成"
until grep -q "^run=" harness/runs/$TAG.log 2>/dev/null; do
  powershell -NoProfile -Command "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'revise --runs' } | Measure-Object).Count" | tr -dc 0-9 | grep -q "[1-9]" || break
  sleep 30
done
grep -q "^run=" harness/runs/$TAG.log || { echo "ALERT: revise 未正常结束"; exit 1; }
echo "=== $(date +%T) 评审修订稿"
$C judge --run $TAG --judges luna --eligible --workers ${LUNA_W:-12} --effort max &
$C judge --run $TAG --judges minimax --eligible --workers ${MM_W:-6} --effort max &
wait
# MiniMax 偶发无效:补一轮(只补缺)
$C judge --run $TAG --judges minimax --eligible --workers ${MM_W:-6} --effort max
echo "=== $(date +%T) 构建 SFT 数据"
$C revsft-build --rev $TAG --name $NAME --per-case $PER --min-gain 1.0 --cov-drop 0.04 --hold 4 --max-len 32000 || { echo "ALERT: revsft-build 失败"; exit 1; }
N=$(wc -l < harness/sft/$NAME/train.jsonl)
[ "$N" -ge 20 ] || { echo "ALERT: 训练样本只有 $N 条(<20),不上传,等待人工/下一步决定"; exit 1; }
echo "=== $(date +%T) 上传 Orin ($N 条)"
scp -i $K -o BatchMode=yes harness/sft/$NAME/train.jsonl $H:/home/<orin-user>/work/data/${NAME}_train.jsonl \
 && scp -i $K -o BatchMode=yes harness/sft/$NAME/eval.jsonl $H:/home/<orin-user>/work/data/${NAME}_eval.jsonl \
 && ssh -n -i $K -o BatchMode=yes $H "sha256sum ~/work/data/${NAME}_train.jsonl | cut -c1-16; wc -l ~/work/data/${NAME}_train.jsonl" \
 && sha256sum harness/sft/$NAME/train.jsonl | cut -c1-16
echo "EVENT: 修订数据就绪 $NAME train=$N"
