#!/bin/bash
# Orin 侧自动链 v2(2026-10-03 午后改:同口径评审显示只修收尾的 v3a 补不上评审分差距,故 v3a 训完直接接 v3b)。
#   等 train_simpo(v3a) 结束 → 校验 → 若有 v3b 数据:train_sft.py 从 v3a 的适配器接着训(教师修订报告 + 学生原生思考)
#   → make_gguf_v3.sh(合并+MTP 覆盖→Q8_0→Q4_K_M)→ 起 llama-server(3 槽位)→ 写 SERVER_UP。
#   没有 v3b 数据时退回:直接合并并评测 v3a。状态写 ~/work/runs/<TAG>_post.status,任一步失败写 FAIL:…。
# 用法(setsid nohup 起): post_chain.sh <v3a训练目录> <最终TAG> [v3b数据名=v3b]
RUN=$1; TAG=$2; NAME=${3:-v3b}; OV=$HOME/work/mtp/out/mtp.safetensors; ST=$HOME/work/runs/${TAG}_post.status
st() { echo "$(date +%T) $1" >> $ST; }
: > $ST; st "WAIT_V3A_TRAIN"
while pgrep -f "[t]rain_simpo.py" > /dev/null; do sleep 60; done
if ! grep -q '"event": "done"' $RUN/log.jsonl || [ ! -f $RUN/ckpt/adapter_model.safetensors ]; then st "FAIL:v3a 训练未正常完成(无 done 事件或无 ckpt)"; exit 1; fi
st "V3A_DONE"
AD=$RUN/ckpt
if [ -f $HOME/work/data/${NAME}_train.jsonl ]; then
  OUT=$HOME/work/runs/${NAME}_sft; mkdir -p $OUT
  st "V3B_TRAIN_START"
  # 学习率 5e-5:接着已有适配器、样本少(几十条)、目标是改"克制/引用"行为;v2a 用 1e-4/56 次更新,EfficientThink 的保能力 SFT 用 5e-6。报告段权重 3 让教师修订的报告 token 不被长思考稀释。
  ~/venvs/ft/bin/python $HOME/work/scripts/train_sft.py --data $HOME/work/data/${NAME}_train.jsonl --eval_data $HOME/work/data/${NAME}_eval.jsonl \
    --out $OUT --init_adapter $RUN/ckpt --epochs 1 --accum 4 --lr 5e-5 --warmup 0.1 --lora_from 32 --max_len 32768 --report_weight 3.0 \
    --save_every 5 --eval_every 5 --eval_n 4 > $OUT/stdout.log 2>&1
  if ! grep -q '"event": "done"' $OUT/log.jsonl 2>/dev/null || [ ! -f $OUT/ckpt/adapter_model.safetensors ]; then st "FAIL:v3b 训练失败或未完成,见 ${NAME}_sft/stdout.log"; exit 1; fi
  st "V3B_TRAIN_DONE"; AD=$OUT/ckpt
else
  st "NO_V3B_DATA:改为直接合并评测 v3a"
fi
st "MERGE_CONVERT_QUANT"
if ! bash $HOME/work/scripts/make_gguf_v3.sh $AD $TAG $OV > $HOME/work/runs/${TAG}_gguf.log 2>&1; then st "FAIL:合并/转换/量化,见 ${TAG}_gguf.log"; exit 1; fi
[ -f $HOME/models/gguf/$TAG-Q4_K_M.gguf ] || { st "FAIL:没有产出 $TAG-Q4_K_M.gguf"; exit 1; }
st "GGUF_READY"
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
cd $HOME/work/llama.cpp
setsid nohup ./build/bin/llama-server -m $HOME/models/gguf/$TAG-Q4_K_M.gguf -ngl 99 -fa on -c 122880 -np 3 --host 192.168.55.1 --port 8080 --jinja --reasoning-format deepseek --no-webui --spec-type draft-mtp --spec-draft-n-max 5 > $HOME/work/runs/llama_$TAG.log 2>&1 < /dev/null &
for i in $(seq 1 120); do
  sleep 5
  code=$(curl -s -o /dev/null -w '%{http_code}' --noproxy '*' http://192.168.55.1:8080/health)
  [ "$code" = "200" ] && { st "SERVER_UP"; exit 0; }
  pgrep -f "[l]lama-server" > /dev/null || { st "FAIL:llama-server 退出,见 llama_$TAG.log"; exit 1; }
done
st "FAIL:llama-server 10 分钟内未就绪"; exit 1
