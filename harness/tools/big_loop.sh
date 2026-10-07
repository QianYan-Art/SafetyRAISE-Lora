#!/bin/bash
# 大一圈专家迭代(v3f):v3d 在 130 个从没用过的训练案件上跑完整轨迹(8 路并行)→ gates → luna(max) 评审合格轨迹 → 教师修订 → luna 评审修订稿
#   → revsft-build 新样本 → 混入旧样本回放(原样本,不加核对段落) → 上传 Orin → 接着 v3d 的适配器训 v3f → 合并 → 评测链(TAG=v3f,dev12)。
# 日志 harness/runs/big_loop.log;结束/异常输出 EVENT:/ALERT:。新样本不足 25 条时告警并停下(不训)。
cd <repo>; export PYTHONPATH=harness PYTHONIOENCODING=utf-8 SR_LOCAL_HOST=192.168.55.1:8080; PY=.venv/Scripts/python.exe; C="$PY -B -m sr_eval.cli"
K=<orin-ssh-key>; H=<orin-user>@192.168.55.1
exec >> harness/runs/big_loop.log 2>&1
alert() { echo "ALERT: $1"; exit 1; }
echo "=== $(date +%T) 把服务换成 v3d、8 个并行槽位"
bash harness/orin/serve_llama.sh v3d_np8 v3d-Q4_K_M.gguf -c 327680 -np 8 --spec-type draft-mtp --spec-draft-n-max 5 | tail -2
[ "$(curl -s -o /dev/null -w '%{http_code}' --noproxy '*' http://192.168.55.1:8080/health)" = "200" ] || alert "服务未就绪"
echo "=== $(date +%T) 采样 130 个新案件(A: syn_train_180–279;B: syn_train_310–339)"
$C run --models local27 --cases train --n 100 --offset 80 --kb production --variant concise --effort xhigh --max-tokens 32000 --timeout 7200 --workers 6 --tag onp_v3d_a > harness/runs/onp_v3d_a.log 2>&1 &
$C run --models local27 --cases train --n 30 --offset 210 --kb production --variant concise --effort xhigh --max-tokens 32000 --timeout 7200 --workers 2 --tag onp_v3d_b > harness/runs/onp_v3d_b.log 2>&1 &
wait
echo "=== $(date +%T) 采样结束:A $(ls harness/runs/*_onp_v3d_a/traces | wc -l) 条,B $(ls harness/runs/*_onp_v3d_b/traces | wc -l) 条"
for r in onp_v3d_a onp_v3d_b; do $C gates --run $r; done
echo "=== $(date +%T) luna 评审合格轨迹"
$C judge --run onp_v3d_a --judges luna --eligible --workers 24 --effort max &
$C judge --run onp_v3d_b --judges luna --eligible --workers 12 --effort max &
wait
echo "=== $(date +%T) 教师修订"
$C revise --runs onp_v3d_a,onp_v3d_b --workers 36 --tag rev4 > harness/runs/rev4.log 2>&1
grep -q "^run=" harness/runs/rev4.log || alert "修订未正常结束"
echo "=== $(date +%T) 评审修订稿"
$C judge --run rev4 --judges luna --eligible --workers 30 --effort max
$C judge --run rev4 --judges luna --eligible --workers 30 --effort max
echo "=== $(date +%T) 构建新样本"
$C revsft-build --rev rev4 --name v3f_new --per-case 1 --min-gain 1.0 --cov-drop 0.08 --hold 4 --max-len 32000 || alert "revsft-build 失败"
NEW=$(wc -l < harness/sft/v3f_new/train.jsonl)
[ "$NEW" -ge 25 ] || alert "新样本只有 $NEW 条(<25),不训练,等待决定"
echo "=== $(date +%T) 混入回放(原样本)"
$PY - <<'PYEOF'
import json, random
from pathlib import Path
base = Path("harness/sft")
rd = lambda n, f: [json.loads(l) for l in open(base / n / f, encoding="utf-8")]
new_tr, new_ev = rd("v3f_new", "train.jsonl"), rd("v3f_new", "eval.jsonl")
rng = random.Random(9)
def one_per_case(rows):
    by = {}
    for r in rows:
        by.setdefault(r["case_id"], []).append(r)
    out = [rng.choice(v) for v in by.values()]
    rng.shuffle(out)
    return out
replay = one_per_case(rd("v3c", "train.jsonl"))[:36] + one_per_case(rd("v3e_gate", "train.jsonl"))[:10]
train = new_tr + replay
rng.shuffle(train)
evals = new_ev + rd("v3c", "eval.jsonl")[:2]
out = base / "v3f"
out.mkdir(parents=True, exist_ok=True)
(out / "train.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + chr(10) for r in train), encoding="utf-8")
(out / "eval.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + chr(10) for r in evals), encoding="utf-8")
print(f"v3f: 训练 {len(train)} 条(新 {len(new_tr)} + 回放 {len(replay)}),留出 {len(evals)} 条,约 {sum(len(r['input_ids']) for r in train)} token,新样本涉及案件 {len({r['case_id'] for r in new_tr})} 个")
PYEOF
N=$(wc -l < harness/sft/v3f/train.jsonl)
echo "=== $(date +%T) 上传 Orin ($N 条)"
scp -i $K -o BatchMode=yes harness/sft/v3f/train.jsonl $H:/home/<orin-user>/work/data/v3f_train.jsonl || alert "上传失败"
scp -i $K -o BatchMode=yes harness/sft/v3f/eval.jsonl $H:/home/<orin-user>/work/data/v3f_eval.jsonl || alert "上传失败"
echo "=== $(date +%T) 启动 Orin 训练链:接着 v3d 的适配器训 v3f"
ssh -n -i $K -o BatchMode=yes -o ConnectTimeout=20 $H "setsid nohup bash /home/<orin-user>/work/scripts/post_chain_ft.sh /home/<orin-user>/work/runs/v3d_sft/ckpt v3f v3f -- --epochs 1 --accum 4 --lr 7e-5 --warmup 0.05 --lora_from 32 --max_len 32768 --think_weight 0.1 --report_weight 1.0 --save_every 6 --keep_every 6 --eval_every 6 --eval_n 4 > /home/<orin-user>/work/runs/post_chain_v3f.out 2>&1 < /dev/null & sleep 4; cat ~/work/runs/v3f_post.status"
TAG=v3f MAX_WAIT=90000 nohup bash harness/tools/auto_eval_v3.sh > harness/runs/auto_eval_v3f.log 2>&1 < /dev/null &
echo "EVENT: v3f 新样本 $NEW 条,训练链与评测链已启动(日志 harness/runs/auto_eval_v3f.log)"
