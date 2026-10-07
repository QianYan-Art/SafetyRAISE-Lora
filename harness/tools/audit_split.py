"""把 build_post 产出的 audit/ 抽检包拆成 N 份盲评目录(放到 harness/runs/blind/<前缀>_a/b/c),供 blind_run.sh 并行评审。
保持 pair_XX 文件名不变(答案映射 audit-answers.json 仍有效);每份重写 TASK.md 里的材料清单。
用法: audit_split.py <build输出目录> <前缀> [份数=3]
"""
import re, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
pack = Path(sys.argv[1]) / "audit"
prefix = sys.argv[2]
parts = int(sys.argv[3]) if len(sys.argv) > 3 else 3
files = sorted(pack.glob("pair_*.md"))
task = (pack / "TASK.md").read_text(encoding="utf-8")
out_dirs = []
for i in range(parts):
    d = ROOT / "harness" / "runs" / "blind" / f"{prefix}_{chr(97 + i)}"
    if d.exists():
        raise SystemExit(f"目录已存在,拒绝覆盖: {d}")
    d.mkdir(parents=True)
    mine = files[i::parts]
    for f in mine:
        shutil.copy2(f, d / f.name)
    shutil.copy2(pack / "RUBRIC.md", d / "RUBRIC.md")
    names = "、".join(f"`{f.name}`" for f in mine)
    t = re.sub(r"本目录有 \d+ 份抽检材料[^。]*。", f"本目录有 {len(mine)} 份抽检材料：{names}。", task, count=1)
    (d / "TASK.md").write_text(t, encoding="utf-8")
    out_dirs.append(d.name)
print(" ".join(out_dirs))
