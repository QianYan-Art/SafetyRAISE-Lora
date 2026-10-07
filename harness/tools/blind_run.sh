#!/bin/bash
# 对一批盲评目录依次/并行运行 Codex(gpt-6.1-sol xhigh)与 Kimi(k3-256k)盲评,输出到各目录的 codex_blind.md / kimi_blind.md。
# 用法: blind_run.sh <codex|kimi|deepseek|both|all> <盲评目录名> [<盲评目录名> ...]   (目录在 harness/runs/blind/ 下)
#   both=codex+kimi;all=codex+kimi+deepseek;deepseek=DeepSeek Harness 无头模式(只读沙箱,需先在其 Models 页存好 DEEPSEEK_API_KEY),输出 deepseek_blind.md
# Codex 各目录并行;Kimi 各目录串行(配额更紧)。结束时在 harness/runs/blind/blind_run.log 追加 EVENT: 行。
WHO=$1; shift
cd <repo>/harness/runs/blind || exit 1
LOG=blind_run.log
PROMPT_C="这是一次独立的只读盲评任务,请忽略所在目录之外的项目规则文件。请先阅读本目录的 TASK.md 和 RUBRIC.md,然后按要求完成盲评,最后输出 JSON 数组。不要修改任何文件。"
PROMPT_K="这是一次独立的只读盲评任务,请忽略所在目录之外的项目规则文件。先阅读本目录的 TASK.md 和 RUBRIC.md,然后自己逐个阅读 pair_*.md 并按要求完成盲评,最后输出 JSON 数组。不要使用子代理(sub-agent),不要修改任何文件,不要联网。"
run_codex() {
  ( cd "$1" && codex exec -m gpt-6.1-sol -c model_reasoning_effort="xhigh" -s read-only --skip-git-repo-check --ephemeral -o codex_blind.md "$PROMPT_C" > codex_blind.log 2>&1 )
  echo "$(date +%T) codex 完成 $1 -> $(wc -c < $1/codex_blind.md 2>/dev/null) 字节" >> $LOG
}
run_deepseek() {
  # DeepSeek Harness 无头模式:deepseek-flash + 推理强度 high(补丁层固定,不换模型);只读沙箱;任务走标准输入(命令行参数含空格会被 .cmd 弄坏)
  ( cd "$1" && printf '%s' "$PROMPT_K" | DSH_PERMISSION_MODE=read-only "<home>/AppData/Local/Programs/DeepSeek Harness/resources/runtime/cli/bin/dsh.cmd" --profile headless --patch '<repo>/harness/tools/dsh_review.patch.yml' > deepseek_blind.md 2> deepseek_blind.log )
  echo "$(date +%T) deepseek 完成 $1 -> $(wc -c < $1/deepseek_blind.md 2>/dev/null) 字节" >> $LOG
}
run_kimi() {
  ( cd "$1" && <home>/.kimi-code/bin/kimi.exe -m kimi-code/k3-256k -p "$PROMPT_K" > kimi_blind.md 2> kimi_blind.log )
  echo "$(date +%T) kimi 完成 $1 -> $(wc -c < $1/kimi_blind.md 2>/dev/null) 字节" >> $LOG
}
if [ "$WHO" = codex ] || [ "$WHO" = both ] || [ "$WHO" = all ]; then
  for d in "$@"; do run_codex "$d" & done
fi
if [ "$WHO" = deepseek ] || [ "$WHO" = all ]; then
  for d in "$@"; do run_deepseek "$d" & done
fi
if [ "$WHO" = kimi ] || [ "$WHO" = both ] || [ "$WHO" = all ]; then
  ( for d in "$@"; do run_kimi "$d"; done ) &
fi
wait
echo "EVENT: $(date +%T) 盲评($WHO)完成: $*" >> $LOG
