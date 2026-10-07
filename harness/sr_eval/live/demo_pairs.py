"""ScriptedBackend 的单类错误偏好对冒烟；只产生隔离夹具。"""

from __future__ import annotations

import asyncio
import json

from .backends import ScriptedBackend, openai_response
from .dataset import build_dataset
from .env import LiveEnvironment, ROOT, VENDOR, production_budget
from .scripted import SyntheticRetrieval, passing_review, wire_candidate


def main():
    accident = {key: "" for key in json.loads(
        (VENDOR / "config/input_accident_template.json").read_text(encoding="utf-8"),
    )}
    accident.update({"事故标题": "虚构偏好导出测试事故", "天气": "晴"})
    case = {"case_id": "fixture-env-pair-weather", "kind": "synthetic", "split": "train",
            "accident_data": accident, "guidance": {"建议": "只处理输入合成字段，不编造。"}}
    destination = ROOT / "harness/reports/live-dataset/live-pair-demo/traces"
    policy = production_budget().model_copy(update={"max_revision_rounds": 0})
    for rejected in (False, True):
        def script(role, payload, index):
            ctx = json.loads(payload["messages"][-1]["content"])
            if role == "generator":
                value = wire_candidate(ctx)
                if rejected:
                    original = "天气：晴。"
                    replacement = "天气：雨。"
                    value["report_markdown"] = value["report_markdown"].replace(original, replacement)
                    for claim in value["claims"]:
                        if claim["quote"] == original:
                            claim["quote"] = replacement
                return openai_response(value, reasoning="合成学生前缀；只验证最终 JSON 的偏好导出。")
            value = passing_review(ctx)
            if rejected:
                value["completed_checks"][0]["passed"] = False
                value["issues"] = [{
                    "issue_id": "fixture-weather-error", "category": "facts", "severity": "major",
                    "target": "天气断言", "explanation": "输入晴，候选雨，合成单类错误。",
                    "source_refs": ["accident:/天气"], "closure_condition": "天气与合成输入一致。",
                    "status": "open",
                }]
            return value

        trace = asyncio.run(LiveEnvironment(
            ScriptedBackend(script), SyntheticRetrieval(), budget=policy,
        ).run(case))
        trace["pair_kind"] = "seed_error"
        trace["sample_origin"] = "程序单类错误植入；合成协议夹具，未经人工核实"
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / ("rejected.json" if rejected else "chosen.json")
        path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = build_dataset(destination, ROOT / "harness/reports/live-dataset/live-pair-demo/dry-run",
                             dry_run=True, reasoning_effort="xhigh")
    print(json.dumps({"source": "隔离夹具，不是正式训练数据",
                      "artifact": "harness/reports/live-dataset/live-pair-demo/dry-run/manifest.json",
                      "counts": manifest.get("counts", manifest.get("stats"))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
