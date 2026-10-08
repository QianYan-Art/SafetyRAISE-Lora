#!/bin/bash
# 诊断“训练特征(NF4+LoRA)与服务特征(Q4_K_M 合并 GGUF)是否失配”:取留出集前 N 条样本,用 llama.cpp 导出服务模型的末层隐藏状态,
# 再用 HF 里的官方头与新训练的头算教师强制一致率(eval_head_on_served.py),与训练日志里的离线数字对比。
# 用法: served_feature_check.sh <GGUF 基名> [样本数=3]    需要 GPU 空闲(导出时载入整个 GGUF,约 17GB + 少量 KV)
NAME=${1:?need gguf basename}; N=${2:-3}
D=$HOME/work/runs/served_feat; mkdir -p $D; rm -f $D/*
python3 - "$N" <<PY
import json, sys, numpy as np, os
n = int(sys.argv[1]); d = os.path.expanduser("~/work/runs/served_feat")
rows = [json.loads(l) for l in open(os.path.expanduser("~/work/data/eval.jsonl"), encoding="utf-8")]
for i, r in enumerate(rows[:n]):
    ids = np.array(r["input_ids"], dtype="<i4"); ids.tofile(f"{d}/{i:02d}.i32")
    ts = next(j for j, l in enumerate(r["labels"]) if l != -100)
    json.dump({"tstart": ts, "len": len(ids), "sample_id": r.get("sample_id")}, open(f"{d}/{i:02d}.meta.json", "w"))
    print(i, "len", len(ids), "tstart", ts)
PY
for f in $D/*.i32; do
  b=${f%.i32}
  echo "dump $(basename $b) $(date +%T)"
  $HOME/work/scripts/dump_hidden "$HOME/models/gguf/$NAME.gguf" "$f" "$b.bf16" 512 2>&1 | tail -2
done
~/venvs/ft/bin/python $HOME/work/scripts/eval_head_on_served.py --samples $D 2>&1 | grep -v "Warning\|warn\|CC\|hardware\|PyTorch CUDA\|_warn\|return t.to"
echo "SERVED_CHECK_DONE $(date +%T)"
