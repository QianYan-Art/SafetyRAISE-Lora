"""测试临时产物留在工作区，不改变项目外权限。"""

pytest_plugins = ("sr_eval.live.pytest_compat",)

# 2026-10-06:后训练行用 compact 档位模板(官方模板的超集,xhigh/medium/low 渲染不变)
import os as _os
from pathlib import Path as _Path
_os.environ.setdefault("SR_QWEN_PROFILE_DIR", str(_Path(__file__).resolve().parents[2] / "profiles" / "assets" / "qwen3.8-compact-v1"))
