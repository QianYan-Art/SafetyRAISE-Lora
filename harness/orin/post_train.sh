#!/bin/bash
# Orin 侧自动链:等 train_simpo 结束 → 合并/转换/量化 → 起 llama-server(3 槽位,每槽 40960 上下文,MTP 草稿)。
# 状态写 ~/work/runs/<TAG>_post.status;任何一步失败写 FAIL:<步骤> 并退出(笔记本侧据此告警)。
# 用法(setsid nohup 起): post_train.sh <训练目录> <TAG> [mtp_overlay]
RUN=$1; TAG=$2; OV=${3:-${MTP_OVERLAY:-$HOME/work/mtp/out_v3h2/mtp.safetensors}}; ST=$HOME/work/runs/${TAG}_post.status
st() { echo "$(date +%T) $1" >> $ST; }
: > $ST; st "WAIT_TRAIN"
while pgrep -f "[t]rain_simpo.py" > /dev/null; do sleep 60; done
if ! grep -q '"event": "done"' $RUN/log.jsonl || [ ! -f $RUN/ckpt/adapter_model.safetensors ]; then st "FAIL:训练未正常完成(无 done 事件或无 ckpt)"; exit 1; fi
st "TRAIN_DONE"
st "MERGE_CONVERT_QUANT"
if ! bash $HOME/work/scripts/make_gguf_v3.sh $RUN/ckpt $TAG $OV > $HOME/work/runs/${TAG}_gguf.log 2>&1; then st "FAIL:合并/转换/量化,见 ${TAG}_gguf.log"; exit 1; fi
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
