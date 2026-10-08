#!/bin/bash
# Orin 侧 MTP 头重训链(终版适配器确定之后):停服务 → 缓存训练/留出隐藏状态 → 训练 MTP 头 → 覆盖合并 → 转 GGUF/量化 → 起服务。
# 用法(setsid nohup 起): mtp_chain.sh <适配器目录> <TAG> <训练jsonl> <留出jsonl> [默认量化=Q4_K_M]
# 量化:仓库所有者 2026-10-07 选定 Q4_K_M(只生成它;如要换,把 ~/work/runs/<TAG>_quant_choice 写成 Q6_K/Q8_0 并改脚本保留 Q8_0)。最后用交付配置起服务(serve_delivery.sh:KV q8_0、思考预算 5120、6 槽位)。
# 训练前已评测过的同 TAG GGUF(未带新 MTP 头)在合并前删除,腾出合并峰值所需的磁盘。
# 产物: ~/work/mtp/out_<TAG>/mtp.safetensors;合并后的 GGUF ~/models/gguf/<TAG>-<量化>.gguf;状态写 ~/work/runs/<TAG>_mtp.status
AD=$1; TAG=$2; TRAIN=$3; EVAL=$4; Q=${5:-Q4_K_M}
ST=$HOME/work/runs/${TAG}_mtp.status; FEAT=$HOME/work/mtp/feat_$TAG; OUT=$HOME/work/mtp/out_$TAG
PY=~/venvs/ft/bin/python; S=$HOME/work/scripts
st() { echo "$(date +%T) $1" >> $ST; }
: > $ST; st "START"
for pid in $(ps -eo pid,args | awk '/build\/bin\/llama-serve[r]/ {print $1}'); do kill $pid; done
for i in $(seq 1 30); do ps -eo args | grep -q '[b]uild/bin/llama-server' || break; sleep 2; done
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
free_gb() { df --output=avail -BG ~ | tail -1 | tr -dc 0-9; }
[ "$(free_gb)" -lt 45 ] && { st "FAIL:磁盘剩余 $(free_gb)GB < 45GB(缓存约 20GB+合并峰值),先清理"; exit 1; }
mkdir -p $FEAT $FEAT/eval $OUT
st "CACHE_TRAIN"
$PY $S/train_mtp.py --mode cache --adapter $AD --data $TRAIN --select final:all --feat_dir $FEAT > $HOME/work/runs/${TAG}_mtp_cache.log 2>&1 || { st "FAIL:训练集缓存失败,见 ${TAG}_mtp_cache.log"; exit 1; }
st "CACHE_EVAL"
$PY $S/train_mtp.py --mode cache --adapter $AD --data $EVAL --select final:all --feat_dir $FEAT/eval > $HOME/work/runs/${TAG}_mtp_cache_eval.log 2>&1 || { st "FAIL:留出集缓存失败"; exit 1; }
# 评测特征目录里的 shared.pt 与训练目录相同(同一模型的 embed/lm_head),训练脚本只读训练目录的 shared.pt
st "TRAIN_MTP"
$PY $S/train_mtp.py --mode train --feat_dir $FEAT --eval_feat_dir $FEAT/eval --out $OUT --epochs 3 --accum 4 --lr 5e-5 > $HOME/work/runs/${TAG}_mtp_train.log 2>&1 || { st "FAIL:MTP 训练失败,见 ${TAG}_mtp_train.log"; exit 1; }
[ -f $OUT/mtp.safetensors ] || { st "FAIL:没有产出 mtp.safetensors(可能无改进,见 ${TAG}_mtp_train.log)"; exit 1; }
st "MTP_DONE"
rm -rf $FEAT   # 特征缓存约 20GB,训完即删(可重建)
st "MERGE_CONVERT_QUANT"
rm -fv $HOME/models/gguf/${TAG}-*.gguf >> $ST 2>&1   # 评测用的旧构建(无新 MTP 头)
bash $S/make_gguf_v3.sh $AD ${TAG}m $OUT/mtp.safetensors > $HOME/work/runs/${TAG}m_gguf.log 2>&1 || { st "FAIL:合并/转换/量化,见 ${TAG}m_gguf.log"; exit 1; }
[ -f $HOME/models/gguf/${TAG}m-Q4_K_M.gguf ] || { st "FAIL:没有产出 ${TAG}m-Q4_K_M.gguf"; exit 1; }
st "GGUF_READY"
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
[ -s $HOME/work/runs/${TAG}_quant_choice ] && Q=$(tr -d " 
" < $HOME/work/runs/${TAG}_quant_choice)
st "QUANT=$Q"
GG=${TAG}m-$Q
if bash $S/serve_delivery.sh $GG 6 5120 >> $ST 2>&1; then st "SERVER_UP"; else st "FAIL:服务未就绪"; exit 1; fi
