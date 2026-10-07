#!/bin/bash
# 测试集(cases/test,50 案)评测:Orin 上已加载 <TAG> 的服务(4 槽位)→ 跑 test50 → luna(max)+MiniMax(max) 评审;并行把教师在测试集上的旧评审用 luna(max) 重评到同一口径。
# 用法: TAG=v3f bash eval_test.sh   日志 harness/runs/eval_test_<TAG>.log,结束或异常输出 EVENT:/ALERT:
TAG=${TAG:?}
cd <repo>; export PYTHONPATH=harness PYTHONIOENCODING=utf-8 SR_LOCAL_HOST=192.168.55.1:8080; PY=.venv/Scripts/python.exe; C="$PY -B -m sr_eval.cli"
exec >> harness/runs/eval_test_$TAG.log 2>&1
[ "$(curl -s -o /dev/null -w '%{http_code}' --noproxy '*' http://192.168.55.1:8080/health)" = "200" ] || { echo "ALERT: 服务未就绪"; exit 1; }
echo "=== $(date +%T) 教师测试集重评(luna max,旧结果保留在 luna_prev)"
$C judge --run lunatest --judges luna --force --workers 24 --effort max &
echo "=== $(date +%T) 评测 test50 ($TAG)"
$C run --models local27 --cases test --n 50 --kb production --variant concise --effort xhigh --max-tokens 32000 --timeout 7200 --workers 4 --tag ${TAG}_test50 || { echo "ALERT: 评测异常退出"; exit 1; }
wait
R=${TAG}_test50
echo "=== $(date +%T) 评审"
$C judge --run $R --judges luna --workers 24 --effort max &
$C judge --run $R --judges minimax --workers 8 --effort max &
wait
$C judge --run $R --judges luna --workers 24 --effort max
$C summary --run $R
echo "EVENT: ${TAG} 测试集评测与评审完成 run=$R"
