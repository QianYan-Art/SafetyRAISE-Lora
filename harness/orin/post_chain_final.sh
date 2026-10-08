#!/bin/bash
# Orin 侧最终后训练链(由 post_chain_pref.sh 改):停服务 → train_simpo.py(SimPO+锚点,--report_only)→ 合并(沿用旧 MTP 覆盖,S9 再重训)
#   → GGUF Q8_0(保留供量化回归)→ Q4_K_M → 用 serve_llama.sh(compact 档位+思考预算+MTP)起服务。
# 用法(setsid nohup 起): post_chain_final.sh <初始适配器目录> <TAG> <数据名> -- <train_simpo.py 的其余参数>
# 数据读 ~/work/data/<数据名>_pairs.jsonl;训练输出 ~/work/runs/<TAG>_sft/;状态写 ~/work/runs/<TAG>_post.status,失败写 FAIL:…。
INIT=$1; TAG=$2; NAME=$3; shift 3; [ "$1" = "--" ] && shift
OV=${MTP_OVERLAY:-$HOME/work/mtp/out_v3h2/mtp.safetensors}; ST=$HOME/work/runs/${TAG}_post.status; OUT=$HOME/work/runs/${TAG}_sft
st() { echo "$(date +%T) $1" >> $ST; }
: > $ST; st "START"
for pid in $(ps -eo pid,args | awk '/build\/bin\/llama-serve[r]/ {print $1}'); do kill $pid; done
for i in $(seq 1 30); do ps -eo args | grep -q '[b]uild/bin/llama-server' || break; sleep 2; done
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
[ -f $HOME/work/data/${NAME}_pairs.jsonl ] || { st "FAIL:没有训练数据 ${NAME}_pairs.jsonl"; exit 1; }
mkdir -p $OUT
if grep -q '"event": "done"' $OUT/log.jsonl 2>/dev/null && [ -f $OUT/ckpt/adapter_model.safetensors ]; then
  st "TRAIN_ALREADY_DONE(跳过训练)"
else
  st "TRAIN_START"
  ~/venvs/ft/bin/python $HOME/work/scripts/train_simpo.py --data $HOME/work/data/${NAME}_pairs.jsonl --out $OUT --init_adapter $INIT "$@" > $OUT/stdout.log 2>&1
  if ! grep -q '"event": "done"' $OUT/log.jsonl 2>/dev/null || [ ! -f $OUT/ckpt/adapter_model.safetensors ]; then st "FAIL:训练失败或未完成,见 ${TAG}_sft/stdout.log"; exit 1; fi
fi
st "TRAIN_DONE"
st "MERGE_CONVERT_QUANT"
if ! KEEP_Q8=1 bash $HOME/work/scripts/make_gguf_v3.sh $OUT/ckpt $TAG $([ -f "$OV" ] && echo "$OV") > $HOME/work/runs/${TAG}_gguf.log 2>&1; then st "FAIL:合并/转换/量化,见 ${TAG}_gguf.log"; exit 1; fi
[ -f $HOME/models/gguf/$TAG-Q4_K_M.gguf ] || { st "FAIL:没有产出 $TAG-Q4_K_M.gguf"; exit 1; }
st "GGUF_READY"
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
if bash $HOME/work/scripts/serve_llama.sh $TAG-Q4_K_M 6 4000 >> $ST 2>&1; then st "SERVER_UP"; else st "FAIL:服务未就绪,见 llama_${TAG}-Q4_K_M_compact.log"; exit 1; fi
