"""把教师轨迹(runner 产出的 trace)导出成 Qwen3.8 官方模板下的 SFT 样本。

形态沿用生产旧路径:每次模型调用 = 一个独立样本(system=渲染后的报告提示,user=执行提示,
带不带工具与生产一致),监督目标 = 该次调用的助手输出(简短思考 + 工具调用或最终报告)。
思考文本来源单独记录(thinking_source),不冒充教师原始思考。
"""

from __future__ import annotations

import hashlib
import os
import json
from pathlib import Path
from typing import Any

from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

from . import prompt as P

DATASET_DIR = Path(__file__).resolve().parent.parent.parent
PROFILE_DIR = Path(os.environ["SR_QWEN_PROFILE_DIR"]) if os.environ.get("SR_QWEN_PROFILE_DIR") else DATASET_DIR / "profiles" / "assets" / "qwen3.8-official-1d4bf0f2"   # 可用环境变量切到 compact 档位模板(profiles/assets/qwen3.8-compact-v1)
EOS = "<|im_end|>"


def _tojson(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


class QwenTemplate:
    """官方 chat_template.jinja;tojson 与 transformers 一致(ensure_ascii=False)。"""

    def __init__(self, profile_dir: Path = PROFILE_DIR) -> None:
        self.template_text = (profile_dir / "chat_template.jinja").read_text(encoding="utf-8")
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        env.filters["tojson"] = _tojson

        def raise_exception(message: str) -> None:
            raise ValueError(message)

        env.globals["raise_exception"] = raise_exception
        self._tpl = env.from_string(self.template_text)
        self.tokenizer = Tokenizer.from_file(str(profile_dir / "tokenizer.json"))
        self.template_sha256 = hashlib.sha256(self.template_text.encode("utf-8")).hexdigest()
        self.tokenizer_sha256 = hashlib.sha256((profile_dir / "tokenizer.json").read_bytes()).hexdigest()

    def render(self, messages, tools, *, add_generation_prompt: bool, reasoning_effort: str) -> str:
        return self._tpl.render(messages=messages, tools=tools, add_generation_prompt=add_generation_prompt,
                                enable_thinking=True, reasoning_effort=reasoning_effort)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids


def assistant_message(call: dict[str, Any], thinking: str) -> dict[str, Any]:
    """trace 里的一次调用 → 模板可渲染的 assistant 消息。"""
    if call["tool_calls"]:
        tc = call["tool_calls"][0]  # 生产只处理第一个调用,训练目标也只保留第一个
        args = {k: tc["arguments"][k] for k in ("query", "reason", "top_k") if k in tc["arguments"]}
        return {"role": "assistant", "content": (call.get("content") or "").strip(), "reasoning_content": thinking,
                "tool_calls": [{"type": "function", "function": {"name": tc["name"], "arguments": args}}]}
    return {"role": "assistant", "content": call["content"].strip(), "reasoning_content": thinking}


def build_sample(tpl: QwenTemplate, call: dict[str, Any], thinking: str, *, reasoning_effort: str = "low") -> dict[str, Any]:
    msgs = call["messages"]
    tools = P.retrieve_tool_schema() if call["with_tools"] else None
    prompt = tpl.render(msgs, tools, add_generation_prompt=True, reasoning_effort=reasoning_effort)
    full = tpl.render(msgs + [assistant_message(call, thinking)], tools, add_generation_prompt=False,
                      reasoning_effort=reasoning_effort)
    if not full.startswith(prompt):
        raise ValueError("模板渲染的提示不是完整对话的前缀")
    target = full[len(prompt):]
    if not target.endswith(EOS + "\n"):
        raise ValueError("目标未以 <|im_end|> 结尾")
    target = target[:-1]  # 去掉模板在 <|im_end|> 后追加的换行,监督到 <|im_end|> 为止
    p_ids, t_ids = tpl.encode(prompt), tpl.encode(target)
    return {"prompt_tokens": len(p_ids), "target_tokens": len(t_ids), "input_ids": p_ids + t_ids,
            "labels": [-100] * len(p_ids) + t_ids, "target_text": target}


def case_sample_plan(trace: dict[str, Any], keep_tool_prob: float) -> list[int]:
    """每个案件:最终报告调用必选;工具调用按案件+序号的哈希确定性抽样(便于控制总 token)。"""
    keep = []
    for i, call in enumerate(trace["calls"]):
        if not call["tool_calls"]:
            keep.append(i)
            continue
        h = int(hashlib.sha256(f"{trace['case_id']}:{trace['model']}:{i}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        if i == 0 or h < keep_tool_prob:  # 首轮检索决定总保留(学习"何时/怎么检索")
            keep.append(i)
    return keep
