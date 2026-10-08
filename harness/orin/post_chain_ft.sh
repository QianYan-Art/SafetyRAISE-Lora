#!/bin/bash
# Orin 侧通用链(v3c 起):立即开始 → 停掉占内存的 llama-server → train_sft.py 接着已有适配器训 → 合并/量化(make_gguf_v3.sh)→ 起 llama-server → SERVER_UP。
# 用法(setsid nohup 起): post_chain_ft.sh <初始适配器目录> <TAG> <数据名> -- <train_sft.py 的其余参数>
# 数据读 ~/work/data/<数据名>_{train,eval}.jsonl;训练输出 ~/work/runs/<TAG>_sft/;状态写 ~/work/runs/<TAG>_post.status,任一步失败写 FAIL:…。
INIT=$1; TAG=$2; NAME=$3; shift 3; [ "$1" = "--" ] && shift
OV=${MTP_OVERLAY:-$HOME/work/mtp/out_v3h2/mtp.safetensors}; ST=$HOME/work/runs/${TAG}_post.status; OUT=$HOME/work/runs/${TAG}_sft
st() { echo "$(date +%T) $1" >> $ST; }
: > $ST; st "START"
# 不用 pkill -f(会匹配到自己所在的 shell):按进程列表取 pid
for pid in $(ps -eo pid,args | awk '/build\/bin\/llama-serve[r]/ {print $1}'); do kill $pid; done
for i in $(seq 1 30); do ps -eo args | grep -q '[b]uild/bin/llama-server' || break; sleep 2; done
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
[ -f $HOME/work/data/${NAME}_train.jsonl ] || { st "FAIL:没有训练数据 ${NAME}_train.jsonl"; exit 1; }
mkdir -p $OUT
if grep -q '"event": "done"' $OUT/log.jsonl 2>/dev/null && [ -f $OUT/ckpt/adapter_model.safetensors ]; then
  st "TRAIN_ALREADY_DONE(跳过训练)"
else
  st "TRAIN_START"
  ~/venvs/ft/bin/python $HOME/work/scripts/train_sft.py --data $HOME/work/data/${NAME}_train.jsonl --eval_data $HOME/work/data/${NAME}_eval.jsonl \
    --out $OUT --init_adapter $INIT "$@" > $OUT/stdout.log 2>&1
  if ! grep -q '"event": "done"' $OUT/log.jsonl 2>/dev/null || [ ! -f $OUT/ckpt/adapter_model.safetensors ]; then st "FAIL:训练失败或未完成,见 ${TAG}_sft/stdout.log"; exit 1; fi
fi
st "TRAIN_DONE"
st "MERGE_CONVERT_QUANT"
if ! bash $HOME/work/scripts/make_gguf_v3.sh $OUT/ckpt $TAG $([ -f "$OV" ] && echo "$OV") > $HOME/work/runs/${TAG}_gguf.log 2>&1; then st "FAIL:合并/转换/量化,见 ${TAG}_gguf.log"; exit 1; fi
[ -f $HOME/models/gguf/$TAG-Q4_K_M.gguf ] || { st "FAIL:没有产出 $TAG-Q4_K_M.gguf"; exit 1; }
st "GGUF_READY"
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
cd $HOME/work/llama.cpp
setsid nohup ./build/bin/llama-server -m $HOME/models/gguf/$TAG-Q4_K_M.gguf -ngl 99 -fa on -c 122880 -np 3 --host 192.168.55.1 --port 8080 --jinja --reasoning-format deepseek --no-webui --spec-type draft-mtp --spec-draft-n-max 5 > $HOME/work/runs/llama_$TAG.log 2>&1 < /dev/null &
for i in $(seq 1 120); do
  sleep 5
  code=$(curl -s -o /dev/null -w '%{http_code}' --noproxy '*' http://192.168.55.1:8080/health)
  [ "$code" = "200" ] && { st "SERVER_UP"; exit 0; }
  ps -eo args | grep -q '[b]uild/bin/llama-server' || { st "FAIL:llama-server 退出,见 llama_$TAG.log"; exit 1; }
done
st "FAIL:llama-server 10 分钟内未就绪"; exit 1
