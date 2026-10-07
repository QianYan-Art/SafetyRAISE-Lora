#!/bin/bash
# 扩大评测:在已加载 <TAG> 的 Orin 上,把服务重启成 4 槽位,跑开发集剩余的 38 个案件(dev_012–049),然后 luna(max)+MiniMax(max) 评审并汇总。
# 用法: TAG=v3d bash eval_rest.sh      日志 harness/runs/eval_rest_<TAG>.log,结束或异常输出 EVENT:/ALERT:
TAG=${TAG:?}
cd <repo>; export PYTHONPATH=harness PYTHONIOENCODING=utf-8 SR_LOCAL_HOST=192.168.55.1:8080; PY=.venv/Scripts/python.exe; C="$PY -B -m sr_eval.cli"
exec >> harness/runs/eval_rest_$TAG.log 2>&1
echo "=== $(date +%T) 重启服务为 4 槽位 ($TAG)"
bash harness/orin/serve_llama.sh ${TAG}_np4 $TAG-Q4_K_M.gguf -c 163840 -np 4 --spec-type draft-mtp --spec-draft-n-max 5 | tail -2
code=$(curl -s -o /dev/null -w '%{http_code}' --noproxy '*' http://192.168.55.1:8080/health)
[ "$code" = "200" ] || { echo "ALERT: 服务未就绪 health=$code"; exit 1; }
echo "=== $(date +%T) 评测 dev_012–049"
$C run --models local27 --cases dev --n 38 --offset 12 --kb production --variant concise --effort xhigh --max-tokens 32000 --timeout 7200 --workers 4 --tag ${TAG}_dev12to49 || { echo "ALERT: 评测异常退出"; exit 1; }
R=${TAG}_dev12to49
echo "=== $(date +%T) 评审"
$C judge --run $R --judges luna --workers 24 --effort max &
$C judge --run $R --judges minimax --workers 8 --effort max &
wait
$C judge --run $R --judges luna --workers 24 --effort max
$C summary --run $R
echo "EVENT: ${TAG} 剩余开发案评测与评审完成 run=$R"
