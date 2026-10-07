#!/bin/bash
# 训练期间并行的评审活(API,不占 GPU;不再调用 qwen27):① 基座/teacher 的 12 案补 MiniMax(max);② 现有合格候选 luna(max)+MiniMax(max)。
cd <repo>; export PYTHONPATH=harness PYTHONIOENCODING=utf-8; PY=.venv/Scripts/python.exe; C="$PY -B -m sr_eval.cli"
exec >> harness/runs/judge_more.log 2>&1
eval "$($PY - <<'PYEOF'
import json
e=json.load(open("harness/runs/evalset12.json",encoding="utf-8"))
b=",".join(f"{c}__qwen27" for c in e["cases"])
t=",".join(f"{c}__{m}" for c,(r,m) in e["teacher"].items() if r.endswith("_lunadev"))
print(f"KB='{b}'; KT='{t}'")
PYEOF
)"
echo "=== $(date +%T) ① 基座/teacher 12 案 MiniMax(max)"
$C judge --run baseapi_dev33 --judges minimax --only "$KB" --workers 6 --effort max &
$C judge --run lunadev --judges minimax --only "$KT" --workers 6 --effort max &
wait
echo "=== $(date +%T) ② 现有合格候选 luna(max)"
for r in smp1 finalk; do $C gates --run $r; $C judge --run $r --judges luna --eligible --workers 8 --effort max & done
wait
echo "=== $(date +%T) ② 现有合格候选 MiniMax(max)"
for r in smp1 finalk; do $C judge --run $r --judges minimax --eligible --workers 4 --effort max & done
wait
echo "=== $(date +%T) JUDGE_MORE_DONE"
