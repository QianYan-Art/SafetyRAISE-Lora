"""仅测试进程内修正 Windows 受管沙箱的临时目录 ACL 继承。"""

from __future__ import annotations

import os
from pathlib import Path

_ORIGINAL_MKDIR = os.mkdir
_ROOT = Path(__file__).resolve().parents[3]


def _mkdir(path, mode=0o777, *, dir_fd=None):
    # CPython 的 0700 会撤掉沙箱附加身份的继承权限；只处理本项目的测试临时目录。
    absolute = Path(path).resolve()
    if (os.name == "nt" and mode == 0o700 and absolute.is_relative_to(_ROOT)
            and any("pytest" in part or part.startswith("live-test-tmp") for part in absolute.parts)):
        mode = 0o777
    if dir_fd is None:
        return _ORIGINAL_MKDIR(path, mode)
    return _ORIGINAL_MKDIR(path, mode, dir_fd=dir_fd)


def pytest_configure(config):
    if os.name == "nt":
        os.mkdir = _mkdir


def pytest_unconfigure(config):
    if os.mkdir is _mkdir:
        os.mkdir = _ORIGINAL_MKDIR
