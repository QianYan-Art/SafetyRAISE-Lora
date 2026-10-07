#!/bin/bash
# 仓库所有者不在场期间的自动推进(第二轮训练 v3h2 训完后):① 首回合快评 ② 量化回归(Q8_0 对 Q4_K_M)③ MTP 头重训并合并起服务 ④ 终版速度/接受率探针与首回合复评。
# 每个里程碑写一行 "EVENT:",失败写 "ALERT:" 并退出(唤醒会话)。用法: TAG=v3h2 bash harness/tools/auto_final.sh  日志 harness/runs/auto_final_<TAG>.log
TAG=${TAG:-v3h2}; K=<orin-ssh-key>; H=<orin-user>@192.168.55.1
cd <repo>
export PYTHONIOENCODING=utf-8 SR_LOCAL_HOST=192.168.55.1:8080 SR_LIVE_EFFORT=compact SR_LIVE_ATTEMPT_TIMEOUT=7200
export SR_QWEN_PROFILE_DIR="$PWD/profiles/assets/qwen3.8-compact-v1"
PY=.venv-live/Scripts/python.exe
alert() { echo "ALERT: $1"; exit 0; }
ev() { echo "EVENT: $(date +%T) $1"; }
ssh_() { ssh -i $K -o BatchMode=yes -o ConnectTimeout=20 $H "$@"; }
wait_status() {  # $1=~/work/runs 下的状态文件名 $2=成功标记 $3=最长等待秒
  local end=$((SECONDS+$3)) last="" line
  while [ $SECONDS -lt $end ]; do
    line=$(ssh_ "tail -1 ~/work/runs/$1" 2>/dev/null) || line="(ssh 暂不可用)"
    [ "$line" != "$last" ] && { echo "[$(date +%T)] $1: $line"; last="$line"; }
    case "$line" in *FAIL*) return 2;; *$2*) return 0;; esac
    sleep 90
  done; return 3
}
run_first() {  # $1=tag
  rm -rf harness/runs/live_$1
  $PY -B harness/tools/live_batch.py --tag $1 --cases-file .scratch/dev12_deploy.txt --backend local --reviewer scripted --retrieval sparse_half \
    --workers 6 --stable-limit 32768 --output-reserve 1 --max-tokens 32000 --first-call-only > harness/runs/live_$1.log 2>&1
}
elig() { $PY -c "import json,sys;d=json.load(open(sys.argv[1],encoding='utf-8'));print(d['first_call_deterministic_eligible'])" "$1" 2>/dev/null || echo 0; }

# ---- ① 等训练链就绪 → 首回合快评
wait_status ${TAG}_post.status SERVER_UP 90000; r=$?
[ $r -ne 0 ] && alert "训练链未就绪($r):$(ssh_ "tail -2 ~/work/runs/${TAG}_post.status")"
run_first ${TAG}_dev12_first
$PY -B harness/tools/live_compare.py v3f=harness/runs/live_v3f_dev12_dep32k ${TAG}=harness/runs/live_${TAG}_dev12_first > harness/reports/compare_${TAG}_vs_v3f_first.json 2>&1
ev "①首回合快评完成:harness/reports/compare_${TAG}_vs_v3f_first.json"

# ---- ② 量化回归:Q8_0 同口径首回合,规则=Q8_0 合格数比 Q4_K_M 多 ≥3 才改选 Q8_0
out=$(ssh_ "bash ~/work/scripts/serve_llama.sh ${TAG}-Q8_0 6 4000" 2>&1 | tail -1); echo "$out"
echo "$out" | grep -q SERVER_UP || alert "Q8_0 服务没起来:$out"
run_first ${TAG}q8_dev12_first
$PY -B harness/tools/live_compare.py q4=harness/runs/live_${TAG}_dev12_first q8=harness/runs/live_${TAG}q8_dev12_first > harness/reports/compare_${TAG}_q8_vs_q4.json 2>&1
$PY - <<PYEOF
import json,re
t=open("harness/reports/compare_${TAG}_q8_vs_q4.json",encoding="utf-8").read()
dec=json.JSONDecoder(); i=0; res=[]
while i<len(t):
    while i<len(t) and t[i]!="{": i+=1
    if i>=len(t): break
    o,j=dec.raw_decode(t,i); res.append(o); i=j
q4=next(o for o in res if o["name"]=="q4"); q8=next(o for o in res if o["name"]=="q8")
pick="Q8_0" if q8["first_call_deterministic_eligible"]>=q4["first_call_deterministic_eligible"]+3 else "Q4_K_M"
json.dump({"q4_eligible":q4["first_call_deterministic_eligible"],"q8_eligible":q8["first_call_deterministic_eligible"],"rule":"Q8_0 合格数比 Q4_K_M 多 ≥3 才改选 Q8_0","pick":pick},open("harness/reports/quant_decision_${TAG}.json","w",encoding="utf-8"),ensure_ascii=False)
print(pick)
PYEOF
PICK=$($PY -c "import json;print(json.load(open('harness/reports/quant_decision_${TAG}.json',encoding='utf-8'))['pick'])")
ev "②量化回归完成,选定 $PICK(见 harness/reports/quant_decision_${TAG}.json)"

# ---- ③ MTP 头重训:留出集用终版首回合输出,训练集用已备好的 60 条学生线上协议输出
$PY -B harness/tools/make_mtp_data.py --eval-traces harness/runs/live_${TAG}_dev12_first --n 10 --out harness/reports/mtp_data/eval.jsonl || alert "留出集构造失败"
scp -i $K harness/reports/mtp_data/train.jsonl harness/reports/mtp_data/eval.jsonl $H:~/work/data/ || alert "上传 MTP 数据失败"
scp -i $K harness/orin/mtp_chain.sh harness/orin/train_mtp.py harness/orin/make_gguf_v3.sh harness/orin/serve_llama.sh $H:~/work/scripts/ || alert "上传脚本失败"
ssh_ "chmod +x ~/work/scripts/*.sh; cd ~/work && (setsid nohup bash ~/work/scripts/mtp_chain.sh ~/work/runs/${TAG}_sft/ckpt ${TAG} ~/work/data/train.jsonl ~/work/data/eval.jsonl $PICK > ~/work/runs/${TAG}_mtp_chain.out 2>&1 < /dev/null &)"
wait_status ${TAG}_mtp.status SERVER_UP 40000; r=$?
[ $r -ne 0 ] && alert "MTP 链未就绪($r):$(ssh_ "tail -2 ~/work/runs/${TAG}_mtp.status")"
ev "③MTP 重训并合并完成,服务已起(${TAG}m-$PICK)"

# ---- ④ 终版:速度/接受率探针 + 首回合复评
TR=$(ls harness/runs/live_${TAG}_dev12_first/*.json | head -1)
$PY -B harness/tools/speed_probe.py "$TR" 500 2 > harness/reports/speed_probe_${TAG}m.txt 2>&1
run_first ${TAG}m_dev12_first
$PY -B harness/tools/live_compare.py v3f=harness/runs/live_v3f_dev12_dep32k ${TAG}=harness/runs/live_${TAG}_dev12_first ${TAG}m=harness/runs/live_${TAG}m_dev12_first > harness/reports/compare_${TAG}m_final.json 2>&1
ev "④终版探针与首回合复评完成:harness/reports/speed_probe_${TAG}m.txt、compare_${TAG}m_final.json"
