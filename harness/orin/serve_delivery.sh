#!/bin/bash
# 交付成品的启动入口:compact 档位 + 思考预算 4000 + MTP 草稿 5 + 每槽位 32768 上下文 + **KV 缓存 q8_0**(仓库所有者 2026-10-07 要求的交付默认)。
# 用法: serve_delivery.sh <GGUF 基名,如 v3h2m-Q4_K_M> [slots=6] [think_budget=5120]   (思考预算 5120 为仓库所有者 2026-10-07 定的交付值)
# 需要 flash attention(serve_llama.sh 已开 -fa on);要回到 f16 KV 做对照,直接用 KV_TYPE=f16 serve_llama.sh。
KV_TYPE=q8_0 exec bash "$(dirname "$0")/serve_llama.sh" "${1:?need gguf basename}" "${2:-6}" "${3:-5120}"
