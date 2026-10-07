#!/bin/bash
# 阻塞直到日志里的 EVENT:/ALERT: 行数超过已见数量,然后打印新增行(用于 run_in_background 的可唤醒等待)。用法: wait_event.sh <日志> <已见行数>
LOG=$1; SEEN=${2:-0}
while [ "$(grep -cE '^(EVENT|ALERT):' "$LOG" 2>/dev/null)" -le "$SEEN" ]; do sleep 60; done
grep -E '^(EVENT|ALERT):' "$LOG" | tail -n +$((SEEN+1)) | cut -c1-400
