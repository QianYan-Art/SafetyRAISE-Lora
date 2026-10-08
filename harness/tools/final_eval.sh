#!/bin/bash
# 终版总评测(按交付配置:思考预算 5120、KV q8_0、Q4_K_M、新 MTP 头;服务由 Orin 上 serve_delivery.sh 或等价配置提供):
# 开发 12 案全流程(生成者 + 脚本化审查者 + 最多 2 轮修订,与线上协议一致) → 与 v3f 基线、v3h2 首回合快评对比 → 取回服务日志做速度统计。
# 用法: bash harness/tools/final_eval.sh [TAG=final_v3h2m] [SERVER_LOG=llama_v3h2m-Q4_K_M_compact.log]    环境变量 WIN=窗口(每槽位上下文,缺省 40960 = 交付窗口;
#   2026-10-08 的那次总评测是在 32768 下跑的:WIN=32768 可复现那个口径;服务端必须用同样的 SLOT_CTX 起)   日志 harness/runs/live_<TAG>_dev12.log(结束写 EVENT:)
TAG=${1:-final_v3h2m}; SLOG=${2:-llama_v3h2m-Q4_K_M_compact.log}
K=<orin-ssh-key>; H=<orin-user>@192.168.55.1
cd <repo>
WIN=${WIN:-40960}
export PYTHONIOENCODING=utf-8 SR_LIVE_MAX_WINDOW=$WIN SR_LOCAL_HOST=192.168.55.1:8080 SR_LIVE_EFFORT=compact SR_LIVE_ATTEMPT_TIMEOUT=7200
export SR_QWEN_PROFILE_DIR="$PWD/profiles/assets/qwen3.8-compact-v1"
PY=.venv-live/Scripts/python.exe
ssh -i $K -o BatchMode=yes $H "curl -s -m 5 http://192.168.55.1:8080/health" | grep -q ok || { echo "ALERT: 服务不健康,先在 Orin 上启动交付配置"; exit 1; }
ssh -i $K $H "grep -m1 -E 'type_k|cache_type_k|KV buffer' ~/work/runs/$SLOG | cut -c1-200; grep -c . ~/work/runs/$SLOG" 2>/dev/null
LOGMARK=$(ssh -i $K $H "wc -l < ~/work/runs/$SLOG")   # 评测开始前日志行数,只统计新增部分
rm -rf harness/runs/live_${TAG}_dev12
$PY -B harness/tools/live_batch.py --tag ${TAG}_dev12 --cases-file harness/tools/dev12_deploy.txt --backend local --reviewer scripted --retrieval sparse_half \
  --workers 6 --stable-limit $WIN --output-reserve 1 --max-tokens 32000 > harness/runs/live_${TAG}_dev12.log 2>&1
ssh -i $K $H "tail -n +$((LOGMARK+1)) ~/work/runs/$SLOG" > harness/runs/server_${TAG}_dev12.log
python harness/tools/server_log_stats.py harness/runs/server_${TAG}_dev12.log --json harness/reports/server_stats_${TAG}.json > harness/reports/server_stats_${TAG}.txt 2>&1
ARGS=""; for kv in v3f_full=harness/runs/live_v3f_dev12_dep32k v3h2_first_f16=harness/runs/live_v3h2_dev12_first; do [ -d "${kv#*=}" ] && ARGS="$ARGS $kv"; done   # 基线运行目录是开发机上的，新环境没有就只比终版
$PY -B harness/tools/live_compare.py $ARGS final=harness/runs/live_${TAG}_dev12 > harness/reports/compare_${TAG}.json 2>harness/reports/compare_${TAG}.err
echo "EVENT: $(date +%T) 总评测完成 ${TAG}: harness/runs/live_${TAG}_dev12、harness/reports/compare_${TAG}.json、server_stats_${TAG}.txt"
