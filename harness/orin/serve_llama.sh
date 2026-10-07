#!/bin/bash
# Orin 上启动 llama-server(线上交付形态):compact 推理档位 + 思考预算兜底 + MTP 草稿。
# 用法: serve_llama.sh <GGUF 基名,如 v3f-Q4_K_M> [slots=6] [think_budget=4000]
# 说明:服务端固定 --reasoning-effort compact(线上请求里的 reasoning.effort=high 对 Qwen3.8 模板无效,由服务端档位覆盖);
#      预算耗尽时注入 ~/work/templates/budget_msg.txt 的提示语后强制收束到 </think>。
set -u
NAME=${1:?need gguf basename e.g. v3f-Q4_K_M}; NP=${2:-6}; BUDGET=${3:-4000}
cd "$HOME/work/llama.cpp" || exit 1
pkill -f "build/bin/llama-server" 2>/dev/null; sleep 3
MSG=$(cat "$HOME/work/templates/budget_msg.txt")
CTX=$((32768 * NP))
setsid nohup ./build/bin/llama-server -m "$HOME/models/gguf/$NAME.gguf" -ngl 99 -fa on -c $CTX -np $NP \
  --host 192.168.55.1 --port 8080 --jinja --reasoning-format deepseek --no-webui \
  --chat-template-file "$HOME/work/templates/compact.jinja" --reasoning-effort compact \
  --reasoning-budget "$BUDGET" --reasoning-budget-message "$MSG" \
  --spec-type draft-mtp --spec-draft-n-max 5 > "$HOME/work/runs/llama_${NAME}_compact.log" 2>&1 < /dev/null &
for i in $(seq 1 120); do
  curl -s -m 3 http://192.168.55.1:8080/health | grep -q ok && { echo "SERVER_UP $NAME np=$NP budget=$BUDGET"; exit 0; }
  pgrep -f "[b]uild/bin/llama-server" > /dev/null || { echo "FAIL: llama-server 退出,见 llama_${NAME}_compact.log"; exit 1; }
  sleep 5
done
echo "FAIL: 10 分钟内未就绪"; exit 1
