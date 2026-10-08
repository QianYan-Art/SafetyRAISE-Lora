#!/bin/bash
# 仓库所有者 2026-10-07:按 v3h2 交付;量化只做 Q4_K_M / Q6_K / Q8_0 三者的首回合短测横评,横评出来后由仓库所有者选;MTP 重训与量化无关(用 BF16+LoRA 的隐藏状态),
# 横评一做完就接着训 MTP(不等仓库所有者的选择,避免机器空闲);训完在链路最后一步读 ~/work/runs/v3h2_quant_choice 决定合并后服务哪个量化。
# 用法: TAG=v3h2 bash harness/tools/auto_quant_mtp.sh   日志 harness/runs/auto_quant_mtp_<TAG>.log(EVENT:/ALERT: 行)
TAG=${TAG:-v3h2}; K=<orin-ssh-key>; H=<orin-user>@192.168.55.1
cd <repo>
export PYTHONIOENCODING=utf-8 SR_LOCAL_HOST=192.168.55.1:8080 SR_LIVE_EFFORT=compact SR_LIVE_ATTEMPT_TIMEOUT=7200
export SR_QWEN_PROFILE_DIR="$PWD/profiles/assets/qwen3.8-compact-v1"
PY=.venv-live/Scripts/python.exe
alert() { echo "ALERT: $1"; exit 0; }
ev() { echo "EVENT: $(date +%T) $1"; }
ssh_() { ssh -i $K -o BatchMode=yes -o ConnectTimeout=20 $H "$@"; }
wait_status() { local end=$((SECONDS+$3)) last="" line
  while [ $SECONDS -lt $end ]; do
    line=$(ssh_ "tail -1 ~/work/runs/$1" 2>/dev/null) || line="(ssh 暂不可用)"
    [ "$line" != "$last" ] && { echo "[$(date +%T)] $1: $line"; last="$line"; }
    case "$line" in *FAIL*) return 2;; *$2*) return 0;; esac; sleep 90
  done; return 3; }
run_first() { rm -rf harness/runs/live_$1
  $PY -B harness/tools/live_batch.py --tag $1 --cases-file harness/tools/dev12_deploy.txt --backend local --reviewer scripted --retrieval sparse_half \
    --workers 6 --stable-limit 32768 --output-reserve 1 --max-tokens 32000 --first-call-only > harness/runs/live_$1.log 2>&1; }
serve() { local out; out=$(ssh_ "bash ~/work/scripts/serve_llama.sh $1 6 4000" 2>&1 | tail -1); echo "$out"; echo "$out" | grep -q SERVER_UP; }
TR=$(ls harness/runs/live_${TAG}_dev12_first/*.json | head -1)

# ① 等已在跑的 Q8_0 首回合评测结束,再测 Q8_0 速度
end=$((SECONDS+9000))
until grep -q "批量采样完成 ${TAG}q8_dev12_first" harness/runs/live_${TAG}q8_dev12_first.log 2>/dev/null; do [ $SECONDS -ge $end ] && alert "Q8_0 评测超时"; sleep 60; done
$PY -B harness/tools/speed_probe.py "$TR" 400 2 > harness/reports/speed_${TAG}_Q8_0.txt 2>&1
ev "Q8_0 首回合评测与速度探针完成"
# ② 等 Q6_K 量化完成,服务 Q6_K 评测 + 速度
until [ "$(ssh_ 'ls ~/models/gguf/'${TAG}'-Q6_K.gguf >/dev/null 2>&1 && ! ps -eo args | grep -q "[l]lama-quantize" && echo ok')" = "ok" ]; do sleep 30; done
serve ${TAG}-Q6_K || alert "Q6_K 服务没起来"
$PY -B harness/tools/speed_probe.py "$TR" 400 2 > harness/reports/speed_${TAG}_Q6_K.txt 2>&1
run_first ${TAG}q6_dev12_first
ev "Q6_K 首回合评测完成"
# ③ Q4_K_M 只补速度探针(首回合评测已有)
serve ${TAG}-Q4_K_M || alert "Q4_K_M 服务没起来"
$PY -B harness/tools/speed_probe.py "$TR" 400 2 > harness/reports/speed_${TAG}_Q4_K_M.txt 2>&1
# ④ 三者横评表
SZ() { ssh_ "ls -l ~/models/gguf/${TAG}-$1.gguf | awk '{printf \"%.1f\", \$5/1073741824}'"; }
$PY -B harness/tools/quant_table.py harness/reports/quant_compare_${TAG} \
  "Q4_K_M=harness/runs/live_${TAG}_dev12_first:$(SZ Q4_K_M):harness/reports/speed_${TAG}_Q4_K_M.txt" \
  "Q6_K=harness/runs/live_${TAG}q6_dev12_first:$(SZ Q6_K):harness/reports/speed_${TAG}_Q6_K.txt" \
  "Q8_0=harness/runs/live_${TAG}q8_dev12_first:$(SZ Q8_0):harness/reports/speed_${TAG}_Q8_0.txt" > harness/reports/quant_compare_${TAG}.stdout 2>&1
ev "量化横评完成:harness/reports/quant_compare_${TAG}.md —— 等仓库所有者选择(选定后写入 ~/work/runs/${TAG}_quant_choice);MTP 重训已同步开始"
# ⑤ 不等选择,直接开始 MTP 重训(与量化无关)
$PY -B harness/tools/make_mtp_data.py --eval-traces harness/runs/live_${TAG}_dev12_first --n 10 --out harness/reports/mtp_data/eval.jsonl || alert "留出集构造失败"
[ -s harness/reports/mtp_data/eval.jsonl ] || alert "留出集为空"
scp -i $K harness/reports/mtp_data/train.jsonl harness/reports/mtp_data/eval.jsonl $H:~/work/data/ || alert "上传 MTP 数据失败"
scp -i $K harness/orin/mtp_chain.sh harness/orin/train_mtp.py harness/orin/make_gguf_v3.sh harness/orin/serve_llama.sh $H:~/work/scripts/ || alert "上传脚本失败"
ssh_ "chmod +x ~/work/scripts/*.sh; cd ~/work && (setsid nohup bash ~/work/scripts/mtp_chain.sh ~/work/runs/${TAG}_sft/ckpt ${TAG} ~/work/data/train.jsonl ~/work/data/eval.jsonl Q4_K_M > ~/work/runs/${TAG}_mtp_chain.out 2>&1 < /dev/null &)"
wait_status ${TAG}_mtp.status SERVER_UP 40000; r=$?
[ $r -ne 0 ] && alert "MTP 链未就绪($r):$(ssh_ "tail -2 ~/work/runs/${TAG}_mtp.status")"
ev "MTP 重训并合并完成,服务已起($(ssh_ "grep QUANT ~/work/runs/${TAG}_mtp.status | tail -1"))"
# ⑥ 终版:速度/接受率探针 + 首回合复评
$PY -B harness/tools/speed_probe.py "$TR" 500 2 > harness/reports/speed_probe_${TAG}m.txt 2>&1
run_first ${TAG}m_dev12_first
$PY -B harness/tools/quant_table.py harness/reports/final_compare_${TAG}m "v3h2(MTP前)=harness/runs/live_${TAG}_dev12_first" "v3h2m(终版)=harness/runs/live_${TAG}m_dev12_first:::harness/reports/speed_probe_${TAG}m.txt" > harness/reports/final_compare_${TAG}m.stdout 2>&1
ev "终版探针与首回合复评完成:harness/reports/speed_probe_${TAG}m.txt、final_compare_${TAG}m.md"
