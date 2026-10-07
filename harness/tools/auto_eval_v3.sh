#!/bin/bash
# 笔记本侧自动链:等 Orin 的 post_train 链就绪(SERVER_UP)→ 评测 12 个开发案 → 硬门 → luna(max)+MiniMax(max) 评审 → 输出汇总。
# 失败/超时输出 ALERT 后退出(唤醒会话)。用法: TAG=v3a bash auto_eval_v3.sh
TAG=${TAG:-v3a}; K=<orin-ssh-key>; H=<orin-user>@192.168.55.1
cd <repo>; export PYTHONPATH=harness PYTHONIOENCODING=utf-8 SR_LOCAL_HOST=192.168.55.1:8080; PY=.venv/Scripts/python.exe; C="$PY -B -m sr_eval.cli"
alert() { echo "ALERT: $1"; exit 0; }
end=$((SECONDS+${MAX_WAIT:-36000})); last=""
while [ $SECONDS -lt $end ]; do
  s=$(ssh -i $K -o BatchMode=yes -o ConnectTimeout=20 $H "tail -1 ~/work/runs/${TAG}_post.status; echo FREE=\$(df --output=avail -BG ~ | tail -1 | tr -dc 0-9)" 2>/dev/null) || alert "SSH 连不上 Orin"
  line=$(echo "$s" | head -1); free=$(echo "$s" | sed -n 's/^FREE=//p')
  [ "$line" != "$last" ] && { echo "[$(date +%T)] Orin: $line (剩余 ${free}GB)"; last="$line"; }
  case "$line" in *FAIL*) alert "Orin 链失败:$line";; *SERVER_UP*) break;; esac
  # 合并/转换阶段磁盘会临时降到个位数 GB(合并目录 52GB + Q8_0 28GB 同时存在),该阶段不按磁盘低告警
  case "$line" in *MERGE_CONVERT_QUANT*) ;; *) [ "${free:-99}" -lt 12 ] && alert "Orin 剩余空间只有 ${free}GB";; esac
  sleep 120
done
echo "$line" | grep -q SERVER_UP || alert "等待 Orin 就绪超时(最后状态:$line)"
echo "=== $(date +%T) 评测开始"
$C run --models local27 --cases dev --n 12 --kb production --variant concise --effort xhigh --max-tokens 32000 --timeout 7200 --workers 3 --tag ${TAG}_dev12 || alert "评测命令异常退出"
R=${TAG}_dev12
echo "=== $(date +%T) 评审"
$C judge --run $R --judges luna --workers 12 --effort max &
$C judge --run $R --judges minimax --workers 4 --effort max &
wait
$C summary --run $R
echo "EVENT: ${TAG} 评测与评审完成 run=$R"
