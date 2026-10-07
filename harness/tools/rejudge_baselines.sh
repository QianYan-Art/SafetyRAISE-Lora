#!/bin/bash
# 把教师/基座/v2a 在开发集 dev_012–049 上的轨迹用 luna(max) 重评,与新模型同口径(旧结果保留在 luna_prev)。API 为主,不占 GPU。
# 教师轨迹在 lunadev(20 案)+lunadev2(30 案);基座(API)与 v2a 只覆盖 dev_000–032。日志 harness/runs/rejudge_baselines.log。
cd <repo>; export PYTHONPATH=harness PYTHONIOENCODING=utf-8; PY=.venv/Scripts/python.exe; C="$PY -B -m sr_eval.cli"
exec >> harness/runs/rejudge_baselines.log 2>&1
keys() { $PY - "$@" <<'PYEOF'
import glob, json, sys
tag, model = sys.argv[1], sys.argv[2]
d = glob.glob(f"harness/runs/*_{tag}")[0]
ks = []
for p in sorted(glob.glob(d + "/traces/*.json")):
    t = json.load(open(p, encoding="utf-8"))
    n = int(t["case_id"].split("_")[-1])
    if t["case_id"].startswith("syn_dev_") and n >= 12 and t.get("final_markdown"):
        ks.append(f"{t['case_id']}__{t['model']}")
print(",".join(ks))
PYEOF
}
for spec in "lunadev:luna" "lunadev2:luna" "baseapi_dev33:qwen27" "sft_dev50:local27"; do
  tag=${spec%%:*}; model=${spec##*:}; K=$(keys $tag $model)
  [ -n "$K" ] || { echo "$tag 没有 dev_012+ 的轨迹,跳过"; continue; }
  echo "=== $(date +%T) $tag ($(echo $K | tr ',' '\n' | wc -l) 份)"
  $C judge --run $tag --judges luna --only "$K" --force --workers 24 --effort max
done
echo "EVENT: 基线重评完成"
