#!/bin/bash
# 阶段 1 取激活：切词（需服务在线）→ 停服务 → 逐条 dump_layers（交付 GGUF）→ 写 DONE
W=~/work/wsp/phase1; L=$W/capture.log; cd $W
~/venvs/ft/bin/python $W/../scripts/prep_tokens.py >> $L 2>&1 || { echo PREP_FAIL >> $L; exit 1; }
pkill -f "build/bin/llama-server"; sleep 5
M=~/models/gguf/v3h2m-Q4_K_M.gguf
~/venvs/ft/bin/python - <<'PY' > cap/jobs.txt
import json
for m in json.load(open("cap/meta.json")): print(m["i"], m["start"], m["end"])
PY
while read i s e; do
  f=$(printf "%02d" $i)
  [ -f cap/$f.f32 ] && continue
  echo "$(date +%T) seq $f [$s,$e)" >> $L
  ~/work/wsp/scripts/dump_layers $M cap/$f.i32 cap/$f.f32 $s $e 512 >> $L 2>&1 || { echo "DUMP_FAIL $f" >> $L; exit 1; }
done < cap/jobs.txt
echo CAPTURE_DONE >> $L
