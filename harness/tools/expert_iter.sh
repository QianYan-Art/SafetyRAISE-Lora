#!/bin/bash
# 专家迭代自动链(笔记本侧;v3c 自己在新案件上跑完整轨迹之后):
#   等采样 run(onp_v3c)结束 → luna(max) 评审合格轨迹 → luna 最小改动修订 → luna 评审修订稿 → revsft-build(新样本)
#   → 混入旧样本回放 → 上传 Orin → 启动 Orin 训练链(post_chain_ft.sh,接着 v3c 的适配器)→ 启动评测链。
# 用法: bash expert_iter.sh [采样标签=onp_v3c] [新数据名=v3d] [回放条数=36]   日志 harness/runs/expert_iter.log;结束或异常输出 EVENT:/ALERT:
SRC=${1:-onp_v3c}; NAME=${2:-v3d}; REPLAY=${3:-36}
cd <repo>; export PYTHONPATH=harness PYTHONIOENCODING=utf-8; PY=.venv/Scripts/python.exe; C="$PY -B -m sr_eval.cli"
K=<orin-ssh-key>; H=<orin-user>@192.168.55.1
exec >> harness/runs/expert_iter.log 2>&1
alert() { echo "ALERT: $1"; exit 1; }
echo "=== $(date +%T) 等待采样 $SRC 结束"
until grep -q "^run=" harness/runs/$SRC.log 2>/dev/null; do
  powershell -NoProfile -Command "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'cases train --n 30' } | Measure-Object).Count" | tr -dc 0-9 | grep -q "[1-9]" || break
  sleep 60
done
grep -q "^run=" harness/runs/$SRC.log || alert "采样未正常结束"
$C gates --run $SRC
echo "=== $(date +%T) luna 评审合格轨迹"
$C judge --run $SRC --judges luna --eligible --workers 24 --effort max || alert "评审失败"
echo "=== $(date +%T) 教师修订"
$C revise --runs $SRC --workers 36 --tag rev2 > harness/runs/rev2.log 2>&1 || alert "修订失败"
grep -q "^run=" harness/runs/rev2.log || alert "修订未正常结束"
echo "=== $(date +%T) 评审修订稿(luna)"
$C judge --run rev2 --judges luna --eligible --workers 24 --effort max
$C judge --run rev2 --judges luna --eligible --workers 24 --effort max   # 补缺一轮
echo "=== $(date +%T) 构建新样本"
$C revsft-build --rev rev2 --name ${NAME}_new --per-case 1 --min-gain 1.0 --cov-drop 0.08 --hold 3 --max-len 32000 || alert "revsft-build 失败"
NEW=$(wc -l < harness/sft/${NAME}_new/train.jsonl)
[ "$NEW" -ge 8 ] || alert "新样本只有 $NEW 条(<8),不训练,等待人工决定"
echo "=== $(date +%T) 混入旧样本回放 $REPLAY 条"
$PY - "$NAME" "$REPLAY" <<'PYEOF'
import json, random, sys
from pathlib import Path
name, replay = sys.argv[1], int(sys.argv[2])
base = Path("harness/sft")
new_tr = [json.loads(l) for l in open(base / f"{name}_new" / "train.jsonl", encoding="utf-8")]
new_ev = [json.loads(l) for l in open(base / f"{name}_new" / "eval.jsonl", encoding="utf-8")]
old_tr = [json.loads(l) for l in open(base / "v3c" / "train.jsonl", encoding="utf-8")]
old_ev = [json.loads(l) for l in open(base / "v3c" / "eval.jsonl", encoding="utf-8")]
rng = random.Random(5)
by_case = {}
for r in old_tr:
    by_case.setdefault(r["case_id"], []).append(r)
old_pick = [rng.choice(v) for v in by_case.values()]            # 每个旧案件取一条,再抽 replay 条
rng.shuffle(old_pick)
train = new_tr + old_pick[:replay]
rng.shuffle(train)
evals = new_ev + old_ev[:3]
out = base / name
out.mkdir(parents=True, exist_ok=True)
(out / "train.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + chr(10) for r in train), encoding="utf-8")
(out / "eval.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + chr(10) for r in evals), encoding="utf-8")
print(f"{name}: 训练 {len(train)} 条(新 {len(new_tr)} + 回放 {min(replay, len(old_pick))}),留出 {len(evals)} 条,约 {sum(len(r['input_ids']) for r in train)} token")
PYEOF
N=$(wc -l < harness/sft/$NAME/train.jsonl)
echo "=== $(date +%T) 上传 Orin ($N 条)"
scp -i $K -o BatchMode=yes harness/sft/$NAME/train.jsonl $H:/home/<orin-user>/work/data/${NAME}_train.jsonl || alert "上传失败"
scp -i $K -o BatchMode=yes harness/sft/$NAME/eval.jsonl $H:/home/<orin-user>/work/data/${NAME}_eval.jsonl || alert "上传失败"
echo "=== $(date +%T) 启动 Orin 训练链(接着 v3c 的适配器)"
ssh -n -i $K -o BatchMode=yes -o ConnectTimeout=20 $H "setsid nohup bash /home/<orin-user>/work/scripts/post_chain_ft.sh /home/<orin-user>/work/runs/v3c_sft/ckpt $NAME $NAME -- --epochs 1 --accum 4 --lr 5e-5 --warmup 0.05 --lora_from 32 --max_len 32768 --think_weight 0.1 --report_weight 1.0 --save_every 6 --keep_every 6 --eval_every 6 --eval_n 4 > /home/<orin-user>/work/runs/post_chain_$NAME.out 2>&1 < /dev/null & sleep 3; cat ~/work/runs/${NAME}_post.status"
TAG=$NAME MAX_WAIT=70000 nohup bash harness/tools/auto_eval_v3.sh > harness/runs/auto_eval_$NAME.log 2>&1 < /dev/null &
echo "EVENT: 专家迭代数据就绪并已启动训练链 $NAME(训练 $N 条);评测链日志 harness/runs/auto_eval_$NAME.log"
