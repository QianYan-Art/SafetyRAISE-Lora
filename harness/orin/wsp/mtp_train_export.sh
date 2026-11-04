#!/bin/bash
# 等特征 FEAT_DONE → 训练 MTP 头（同 v3h2 配方）→ 删特征 → 换头导出最终 Q4_K_M → 单路测速。状态 ~/work/runs/wsp_mtp.status
F=$HOME/work/mtp/feat_wspgguf; OUT=$HOME/work/mtp/out_wsp; ST=$HOME/work/runs/wsp_mtp.status; PY=~/venvs/ft/bin/python
G=$HOME/work/wsp/gguf; C=$HOME/work/wsp/mtpconv; FINAL=v3h2m-wsp-mtp-Q4_K_M
st() { echo "$(date +%T) $1" >> $ST; }
until grep -qE "FEAT_DONE|FAIL" $ST; do sleep 30; done
grep -q FAIL $ST && exit 1
st TRAIN_MTP
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY $HOME/work/scripts/train_mtp.py --mode train --feat_dir $F --eval_feat_dir $F/eval --out $OUT --epochs 3 --accum 4 --lr 5e-5 > $HOME/work/runs/wsp_mtp_train.log 2>&1 || { st "FAIL:训练"; exit 1; }
[ -f $OUT/mtp.safetensors ] || { st "FAIL:没有产出 mtp.safetensors（见 wsp_mtp_train.log）"; exit 1; }
st MTP_DONE
rm -rf $F; st "feat removed"
cd $C
$PY $HOME/work/wsp/scripts/mtp_splice.py $OUT/mtp.safetensors $G/v3h2m-wsp-Q4_K_M.gguf $C/final-mid.gguf $C q8_0 > splice_final.log 2>&1 || { st "FAIL:换头"; exit 1; }
$HOME/work/llama.cpp/build/bin/llama-quantize --allow-requantize $C/final-mid.gguf $G/$FINAL.gguf Q4_K_M >> splice_final.log 2>&1 || { st "FAIL:量化"; exit 1; }
rm -rf $C/final-mid.gguf $C/hf $C/mtp-q8_0.gguf
$PY $HOME/work/wsp/scripts/mtp_splice_check.py $G/$FINAL.gguf $G/v3h2m-wsp-Q4_K_M.gguf > splice_final_check.txt 2>&1
st "FINAL_READY $(stat -c %s $G/$FINAL.gguf) sha16=$(sha256sum $G/$FINAL.gguf | cut -c1-16)"
bash $HOME/work/wsp/scripts/speed_single.sh || { st "FAIL:测速"; exit 1; }
st ALL_DONE
