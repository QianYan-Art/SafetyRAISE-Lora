#!/bin/bash
# 把 Orin 上解码/预填充实验的结果文件取回本机(.scratch/results_decode/),便于出表和写记录。可重复运行(增量覆盖)。
# 用法: fetch_decode_results.sh [目标目录=.scratch/results_decode]
K=<orin-ssh-key>; H=<orin-user>@192.168.55.1
DST=${1:-<repo>/.scratch/results_decode}
mkdir -p "$DST"
R='~/work/runs'
FILES="bw_probe.txt prefill_study.txt kld_*.txt tbo_cublas_bf16.txt prof_*_summary.txt profm_*_summary.txt"
FILES="$FILES region_sweep_*.jsonl region_sweep_*.jsonl.log answer_sweep_1.jsonl answer_sweep_1.jsonl.log amulti_*.json amulti_*.log"
FILES="$FILES sweep_decode_1.jsonl sweep_decode_1.jsonl.log valid_*.json valid_*.log multi_*.json multi_*.log"
FILES="$FILES post_chain_auto.log v3h2_mtp.status v3h2_mtp_train.log v3h2_mtp_train_oom1.log"
for f in $FILES; do
  scp -q -i $K "$H:$R/$f" "$DST/" 2>/dev/null
done
ls -la "$DST" | tail -n +2 | awk '{print $5, $9}'
