#!/bin/bash
# Orin 上启动 llama-server(线上交付形态):compact 推理档位 + 思考预算兜底 + MTP 草稿。
# 用法: [KV_TYPE=f16|q8_0] serve_llama.sh <GGUF 基名,如 v3f-Q4_K_M> [slots=6] [think_budget=4000]
# KV 缓存精度:评测/测试期间默认 f16(与历次评测同口径);**交付成品默认 q8_0**(仓库所有者 2026-10-07 的要求),用 serve_delivery.sh 启动。
# 说明:服务端固定 --reasoning-effort compact(线上请求里的 reasoning.effort=high 对 Qwen3.8 模板无效,由服务端档位覆盖);
#      预算耗尽时注入 ~/work/templates/budget_msg.txt 的提示语后强制收束到 </think>。
# 可选环境变量(缺省值=历来的行为,不设就和以前完全一样;用于解码速度对照实验):
#   LLAMA_DIR     llama.cpp 源码/构建目录(缺省 ~/work/llama.cpp;优化构建用 ~/work/llama-opt)
#   SLOT_CTX      每槽位上下文(缺省 32768,训练窗口也是这个值;总 -c = SLOT_CTX × 槽位数;放宽到更长窗口前先读 docs/deployment.md §4)
#   SPEC_N_MAX    MTP 草稿长度(缺省 5)      SPEC_P_MIN  草稿置信度下限(缺省不传,即 llama.cpp 默认 0)
#   SPEC_TYPES    投机类型,逗号分隔,**顺序即优先级**(缺省 draft-mtp;如 ngram-simple,draft-mtp 表示先查 n-gram,没命中再用 MTP 头;n-gram 参数用 EXTRA_ARGS 传 --spec-ngram-simple-size-n/-m)
#   LISTEN_HOST / LISTEN_PORT  监听地址(缺省 192.168.55.1:8080)
#   NGRAM_N / NGRAM_M  n-gram 查表的匹配长度/草稿长度(SPEC_TYPES 含 ngram-simple 时用;缺省 llama.cpp 的 12/48,我们用 3/12 一类)
#   SR_TEMP / SR_TOP_K / SR_TOP_P / SR_MIN_P  服务端默认采样(请求里没带时生效;不设=llama.cpp 内置 0.8/40/0.95/0.05;模型官方推荐 1.0/20/0.95/0,见 generation_config.json)
#   EXTRA_ARGS    追加给 llama-server 的参数    WRAP  命令前缀(如 nsys launch … 做性能剖析)    DRY_RUN=1  只打印命令不启动
set -u
NAME=${1:?need gguf basename e.g. v3f-Q4_K_M}; NP=${2:-6}; BUDGET=${3:-4000}
LLAMA_DIR=${LLAMA_DIR:-$HOME/work/llama.cpp}
LISTEN_HOST=${LISTEN_HOST:-192.168.55.1}; LISTEN_PORT=${LISTEN_PORT:-8080}
SPEC_N_MAX=${SPEC_N_MAX:-5}
SPEC_TYPES=${SPEC_TYPES:-draft-mtp}
KV_TYPE=${KV_TYPE:-f16}
MSG=$(cat "$HOME/work/templates/budget_msg.txt")
CTX=$((${SLOT_CTX:-32768} * NP))
PMIN_ARGS=()
SMP_ARGS=()
[ -n "${SR_TEMP:-}" ] && SMP_ARGS+=(--temp "$SR_TEMP")
[ -n "${SR_TOP_K:-}" ] && SMP_ARGS+=(--top-k "$SR_TOP_K")
[ -n "${SR_TOP_P:-}" ] && SMP_ARGS+=(--top-p "$SR_TOP_P")
[ -n "${SR_MIN_P:-}" ] && SMP_ARGS+=(--min-p "$SR_MIN_P")
NG_ARGS=()
[ -n "${NGRAM_N:-}" ] && NG_ARGS+=(--spec-ngram-simple-size-n "$NGRAM_N")
[ -n "${NGRAM_M:-}" ] && NG_ARGS+=(--spec-ngram-simple-size-m "$NGRAM_M")
[ -n "${SPEC_P_MIN:-}" ] && PMIN_ARGS=(--spec-draft-p-min "$SPEC_P_MIN")
CMD=("$LLAMA_DIR/build/bin/llama-server" -m "$HOME/models/gguf/$NAME.gguf" -ngl 99 -fa on -c "$CTX" -np "$NP"
  --host "$LISTEN_HOST" --port "$LISTEN_PORT" --jinja --reasoning-format deepseek --no-webui
  --chat-template-file "$HOME/work/templates/compact.jinja" --reasoning-effort compact
  --reasoning-budget "$BUDGET" --reasoning-budget-message "$MSG"
  -ctk "$KV_TYPE" -ctv "$KV_TYPE" --spec-type "$SPEC_TYPES" --spec-draft-n-max "$SPEC_N_MAX" "${PMIN_ARGS[@]}" "${NG_ARGS[@]}" "${SMP_ARGS[@]}" ${EXTRA_ARGS:-})
if [ "${DRY_RUN:-0}" = "1" ]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
cd "$LLAMA_DIR" || exit 1
pkill -f "build/bin/llama-server" 2>/dev/null; sleep 3
# shellcheck disable=SC2086
setsid nohup ${WRAP:-} "${CMD[@]}" > "$HOME/work/runs/llama_${NAME}_compact.log" 2>&1 < /dev/null &
for i in $(seq 1 120); do
  curl -s -m 3 "http://$LISTEN_HOST:$LISTEN_PORT/health" | grep -q ok && { echo "SERVER_UP $NAME np=$NP budget=$BUDGET kv=$KV_TYPE"; exit 0; }
  pgrep -f "[b]uild/bin/llama-server" > /dev/null || { echo "FAIL: llama-server 退出,见 llama_${NAME}_compact.log"; exit 1; }
  sleep 5
done
echo "FAIL: 10 分钟内未就绪"; exit 1
