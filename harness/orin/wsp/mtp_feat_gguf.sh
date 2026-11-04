#!/bin/bash
# 编辑模型的 MTP 重训特征（真实服务环境）：用 llama.cpp 跑编辑后的 Q4_K_M（KV q8_0、FA），导出末层 output_norm 后隐藏状态，
# 转成 train_mtp.py 的缓存格式（<i>.pt：ids/h/tstart/sample_id/kind；shared.pt：embed/lm_head 取官方 BF16，LoRA 未动这两项）。
# 产物 ~/work/mtp/feat_wspgguf/{,eval/}；状态 ~/work/runs/wsp_mtp.status
GG=${GG:-$HOME/work/wsp/gguf/v3h2m-wsp-Q4_K_M.gguf}; F=$HOME/work/mtp/feat_wspgguf; ST=$HOME/work/runs/wsp_mtp.status; PY=~/venvs/ft/bin/python
st() { echo "$(date +%T) $1" >> $ST; }
mkdir -p $F/eval $F/raw; st "FEAT_START $GG"
$PY - <<'PY' || { st "FAIL:prep"; exit 1; }
import json, os, numpy as np, torch
from safetensors import safe_open
F = os.path.expanduser("~/work/mtp/feat_wspgguf"); B = os.path.expanduser("~/models/Qwen3.8-27B")
for split, src, od in (("train", "train.jsonl", F), ("eval", "eval.jsonl", F + "/eval")):
    rows = [json.loads(l) for l in open(os.path.expanduser(f"~/work/data/{src}"), encoding="utf-8") if '"final"' in l]
    rows = [r for r in rows if r["kind"] == "final"]
    for i, r in enumerate(rows):
        np.asarray(r["input_ids"], dtype="<i4").tofile(f"{F}/raw/{split}_{i:04d}.i32")
        ts = next(j for j, l in enumerate(r["labels"]) if l != -100)
        json.dump(dict(tstart=ts, sample_id=r["sample_id"], kind=r["kind"], out=f"{od}/{i:04d}.pt"), open(f"{F}/raw/{split}_{i:04d}.json", "w"))
    print(split, len(rows))
if not os.path.exists(f"{F}/shared.pt"):
    idx = json.load(open(f"{B}/model.safetensors.index.json"))["weight_map"]; sd = {}
    for k, n in (("model.language_model.embed_tokens.weight", "embed"), ("lm_head.weight", "lm_head")):
        with safe_open(f"{B}/{idx[k]}", "pt") as f: sd[n] = f.get_tensor(k)
    torch.save(sd, f"{F}/shared.pt"); torch.save(sd, f"{F}/eval/shared.pt"); print("shared", {k: tuple(v.shape) for k, v in sd.items()})
PY
st "PREP_DONE"
for t in $F/raw/*.i32; do
  b=${t%.i32}; out=$($PY -c "import json;print(json.load(open('$b.json'))['out'])")
  [ -f "$out" ] && continue
  $HOME/work/scripts/dump_hidden $GG $t $b.bf16 512 >> $HOME/work/runs/wsp_mtp_dump.log 2>&1 || { st "FAIL:dump $(basename $b)"; exit 1; }
  $PY - "$b" <<'PY' || { st "FAIL:pack"; exit 1; }
import json, sys, numpy as np, torch
b = sys.argv[1]; m = json.load(open(b + ".json"))
ids = torch.from_numpy(np.fromfile(b + ".i32", dtype="<i4").astype(np.int32))
h = torch.from_numpy(np.fromfile(b + ".bf16", dtype="<i2")).view(torch.bfloat16).view(len(ids), -1)
torch.save({"ids": ids, "h": h.clone(), "tstart": m["tstart"], "sample_id": m["sample_id"], "kind": m["kind"]}, m["out"])
PY
  rm -f $b.bf16
  st "feat $(basename $b) $(stat -c %s $t)"
done
st FEAT_DONE
