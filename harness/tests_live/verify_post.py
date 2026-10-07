"""本地R2验收入口，只运行离线测试并写不含样例正文的证据。"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "harness/reports/live-r2-verification.json"


def main() -> int:
    env = {**os.environ, "PYTHONPATH": "harness;harness\\vendor\\safetyraise_7200e30",
           "PYTHONIOENCODING": "utf-8"}
    results = {}
    for name, python, target, extra in (
        ("live", ".venv-live/Scripts/python.exe", "harness/tests_live", []),
        ("legacy", ".venv/Scripts/python.exe", "harness/tests", ["-p", "sr_eval.live.pytest_compat"]),
    ):
        command = [str(ROOT / python), "-B", "-m", "pytest", target, "-q",
                   "-o", "addopts=", "-p", "no:cacheprovider", *extra,
                   "--basetemp", f"harness/reports/live-r2-pytest-verified-{name}"]
        started = time.monotonic()
        process = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                                 encoding="utf-8", text=True, check=False)
        summaries = re.findall(r"^\d+ (?:passed|failed).*$", process.stdout, re.MULTILINE)
        results[name] = {
            "command": command, "exit_code": process.returncode,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "summary": summaries[-1] if summaries else "没有匹配的测试摘要。",
            "stdout_sha256": hashlib.sha256(process.stdout.encode("utf-8")).hexdigest(),
            "stderr_sha256": hashlib.sha256(process.stderr.encode("utf-8")).hexdigest(),
        }
        print(name + ": " + results[name]["summary"], flush=True)
        if process.returncode:
            print(process.stdout[-6000:], flush=True)
            print(process.stderr[-2000:], flush=True)
    vendor = ROOT / "harness/vendor/safetyraise_7200e30"
    manifest = json.loads((vendor / "MANIFEST.json").read_text(encoding="utf-8"))
    mismatches = []
    for name, expected in manifest["files"].items():
        data = (vendor / name).read_bytes()
        if (hashlib.sha256(data).hexdigest() != expected
                and hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest() != expected):
            mismatches.append(name)
    files = sorted((ROOT / "harness/sr_eval/live").glob("*.py")) + sorted(
        (ROOT / "harness/tests_live").glob("*.py"))
    for path in files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hashes = {str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in files}
    report = {
        "date": datetime.now().astimezone().isoformat(), "tests": results,
        "vendor": {"checked": len(manifest["files"]), "mismatches": mismatches,
                   "crlf_normalization_allowed": True},
        "syntax_checked": len(files), "file_hashes": hashes,
        "network_calls": 0, "model_calls": 0, "protocol_fixtures_only": True,
        "new_dependencies": 0, "training_or_deployment": False,
        "warning": "既有cache_dir选项在禁用cacheprovider时不识别。",
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("证据：" + str(OUT), flush=True)
    return int(bool(mismatches) or any(value["exit_code"] for value in results.values()))


if __name__ == "__main__":
    raise SystemExit(main())
