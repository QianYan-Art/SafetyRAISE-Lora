"""本轮限定写集的验收证据；不提交、不联网、不改冻结来源。"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
from collections import Counter
from pathlib import Path

from .env import ROOT, VENDOR


def main():
    declared = json.loads((VENDOR / "MANIFEST.json").read_text(encoding="utf-8"))["files"]
    vendor_problems = []
    for relative, expected in declared.items():
        path = (VENDOR / relative).resolve()
        if not path.is_relative_to(VENDOR.resolve()):
            raise ValueError("冻结清单越界。")
        raw = path.read_bytes()
        digests = {
            hashlib.sha256(raw).hexdigest(),
            hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest(),
        }
        if expected not in digests:
            vendor_problems.append(relative)
    code_paths = sorted((ROOT / "harness/sr_eval/live").glob("*.py"))
    test_paths = sorted((ROOT / "harness/tests_live").glob("*.py"))
    for path in [*code_paths, *test_paths]:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    results = []
    for name, python, target, plugin in [
        ("live", ROOT / ".venv-live/Scripts/python.exe", "harness/tests_live", []),
        ("legacy", ROOT / ".venv/Scripts/python.exe", "harness/tests",
         ["-p", "sr_eval.live.pytest_compat"]),
    ]:
        temp = (ROOT / f"harness/reports/live-pytest-verify-{name}").resolve()
        if not temp.is_relative_to((ROOT / "harness/reports").resolve()):
            raise ValueError("测试临时目录越界。")
        args = [str(python), "-B", "-m", "pytest", target, "-q", "-o", "addopts=",
                "-p", "no:cacheprovider", *plugin, "--basetemp", str(temp)]
        env = {**os.environ, "PYTHONPATH": "harness;harness/vendor/safetyraise_7200e30",
               "PYTHONIOENCODING": "utf-8"}
        completed = subprocess.run(args, cwd=ROOT, env=env, text=True, encoding="utf-8",
                                   errors="replace", capture_output=True, check=False)
        output = completed.stdout + completed.stderr
        match = re.search(r"(\d+) passed", output)
        result = {"suite": name, "command": args, "returncode": completed.returncode,
                  "passed": int(match.group(1)) if match else None,
                  "output_excerpt": "\n".join(output.strip().splitlines()[-10:])}
        results.append(result)
        print(json.dumps({key: result[key] for key in ("suite", "returncode", "passed")},
                         ensure_ascii=False))
    case_counts = {}
    split_hashes = {}
    for split in ("train", "dev", "test"):
        cases = [json.loads(p.read_text(encoding="utf-8"))
                 for p in (ROOT / f"harness/cases/{split}").glob("*.json")]
        case_counts[split] = {
            "count": len(cases), "synthetic": sum(c.get("kind") == "synthetic" for c in cases),
            "field_counts": sorted({len(c["accident_data"]) for c in cases}),
        }
        nonempty = sorted(sum(bool(str(v).strip()) for v in c["accident_data"].values()) for c in cases)
        value_chars = sorted(sum(len(str(v).strip()) for v in c["accident_data"].values()) for c in cases)
        for name, values in (("nonempty_fields", nonempty), ("value_chars", value_chars)):
            case_counts[split][name] = {
                "min": min(values), "p50": values[len(values) // 2],
                "p95": values[min(len(values) - 1, int(len(values) * .95))], "max": max(values),
            }
        case_counts[split]["field_presence"] = {
            field: sum(bool(str(c["accident_data"].get(field, "")).strip()) for c in cases)
            for field in cases[0]["accident_data"]
        }
        case_counts[split]["strata"] = {
            field: dict(Counter(c["accident_data"].get(field, "") for c in cases))
            for field in ("事故类型", "道路类型", "天气", "照明条件", "车辆类型")
        }
        split_hashes[split] = {
            hashlib.sha256(json.dumps(c["accident_data"], ensure_ascii=False, sort_keys=True,
                                      separators=(",", ":")).encode("utf-8")).hexdigest()
            for c in cases
        }
    split_overlap = {
        f"{left}:{right}": len(split_hashes[left] & split_hashes[right])
        for left, right in (("train", "dev"), ("train", "test"), ("dev", "test"))
    }
    result = {
        "date": "2026-10-06", "network_model_calls": 0,
        "vendor": {"declared_files": len(declared), "mismatches": vendor_problems,
                   "comparison": "原始字节或仅CRLF归一"},
        "syntax_checked_files": len(code_paths) + len(test_paths),
        "implementation_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [*code_paths, *test_paths]
        },
        "tests": results, "cases": case_counts,
        "case_exact_hash_overlap": split_overlap,
        "external_validation": "MiniMax/学生服务/SSH/训练均未执行",
    }
    destination = ROOT / "harness/reports/live-r1-verification.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"artifact": str(destination), "vendor_mismatches": len(vendor_problems),
                      "syntax_checked": result["syntax_checked_files"]}, ensure_ascii=False))
    return 0 if not vendor_problems and all(item["returncode"] == 0 for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
