#!/bin/bash
# v3b 的数据(v3a 评测之后手动启动,复用 Orin 上已在运行的 v3a 服务,不花 OpenRouter 钱):
# 在 v3a 没训练过的案件上,用 smp1 里已有的检索上下文,让 v3a 自己重采最终回答 K 份 → 硬门 → 合格候选 luna(max)+MiniMax(max) 评审。
# 用法: bash onpolicy_v3a.sh [案件数=24] [K=2]    日志 harness/runs/onpolicy_v3a.log
N=${1:-24}; K=${2:-2}
cd <repo>; export PYTHONPATH=harness PYTHONIOENCODING=utf-8 SR_LOCAL_HOST=192.168.55.1:8080; PY=.venv/Scripts/python.exe; C="$PY -B -m sr_eval.cli"
exec >> harness/runs/onpolicy_v3a.log 2>&1
echo "=== $(date +%T) v3a 在未训练案件上重采 N=$N K=$K"
$C sample-final --run smp1 --model local27 --k $K --limit $N --effort xhigh --max-tokens 32000 --timeout 7200 --workers 3 --exclude-pairs harness/sft/v3nj2/pairs.filtered.jsonl --tag onp_v3a
R=onp_v3a
$C gates --run $R
$C judge --run $R --judges luna --eligible --workers 12 --effort max &
$C judge --run $R --judges minimax --eligible --workers 4 --effort max &
wait
echo "=== $(date +%T) ONPOLICY_DONE"
