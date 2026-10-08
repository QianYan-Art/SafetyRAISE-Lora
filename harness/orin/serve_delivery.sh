#!/bin/bash
# 交付成品的启动入口(仓库所有者 2026-10-07/08 定):v3h2 合并的 Q4_K_M + 重训 MTP 头;compact 推理档位、思考预算 5120、每槽位 32768 上下文、6 槽位、KV 缓存 q8_0、MTP 草稿,
# 服务端默认采样用模型官方推荐值(temperature 1.0、top_k 20、top_p 0.95、min_p 0;见模型目录 generation_config.json;请求里带了采样参数则以请求为准)。
# 用法: serve_delivery.sh <GGUF 基名,如 v3h2m-Q4_K_M> [slots=6] [think_budget=5120]
# 解码优化(均不改变输出,GGUF 本身不变;实测见 KBase §11.44 与仓库 inference/README.md):
#   - 优化构建 ~/work/llama-opt(补丁 0001 草稿词表子集、0003 按负载自适应草稿长度,另含默认关闭的 0002/0004 开关)
#   - SR_MTP_DRAFT_IDS = 草稿词表子集 id 文件(K=32768):草稿阶段只读输出头的 32768 行,单路解码 +17%
#   - SPEC_N_MAX=5 草稿长度上限;SR_SPEC_BATCH_TOKENS=24:每次验证批的 token 预算,按在跑槽位数自动缩短草稿(6 槽位时每路 3 个,≤4 槽位时 5 个),
#     避免并发时被拒的草稿白占算力(并发是算力受限)
#   要回到库存 llama.cpp 构建、不用子集和自适应:SR_BASELINE=1 serve_delivery.sh …;各项也可用同名环境变量单独覆盖。
# 需要 flash attention(serve_llama.sh 已开 -fa on);要用 f16 KV 做对照,直接用 KV_TYPE=f16 serve_llama.sh。
HERE=$(dirname "$0")
OPT=${LLAMA_OPT_DIR:-$HOME/work/llama-opt}
IDS=${SR_MTP_DRAFT_IDS:-$HOME/work/data/draft_ids/draft_k32768.i32}
if [ "${SR_BASELINE:-0}" != 1 ] && [ -x "$OPT/build/bin/llama-server" ] && [ -f "$IDS" ]; then
  export LLAMA_DIR="${LLAMA_DIR:-$OPT}" SR_MTP_DRAFT_IDS="$IDS"
  export SPEC_N_MAX="${SPEC_N_MAX:-5}" SR_SPEC_BATCH_TOKENS="${SR_SPEC_BATCH_TOKENS:-24}"
fi
export SR_TEMP="${SR_TEMP:-1.0}" SR_TOP_K="${SR_TOP_K:-20}" SR_TOP_P="${SR_TOP_P:-0.95}" SR_MIN_P="${SR_MIN_P:-0}"
KV_TYPE=q8_0 exec bash "$HERE/serve_llama.sh" "${1:?need gguf basename}" "${2:-6}" "${3:-5120}"
