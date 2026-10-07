"""外发闸门:调用外部模型前,按知识库类别与案件类别核对 config/outbound.json。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .llm import CONFIG_DIR, LLMError


class OutboundDenied(LLMError):
    def __init__(self, detail: str) -> None:
        super().__init__("outbound_denied", billing_state="not_sent", detail=detail)


@dataclass(frozen=True)
class OutboundPolicy:
    synthetic_cases: bool
    public_statutes: bool
    real_cases: bool

    @classmethod
    def load(cls, path: Path | None = None) -> "OutboundPolicy":
        d: dict[str, Any] = json.loads((path or CONFIG_DIR / "outbound.json").read_text(encoding="utf-8"))
        return cls(bool(d.get("synthetic_cases_external_send")), bool(d.get("public_statutes_external_send")),
                   bool(d.get("real_case_external_send")))

    def check(self, *, kb_class: str, case_kind: str) -> None:
        """kb_class: synthetic | public_statutes;case_kind: synthetic | real。外部发送前调用。"""
        if case_kind != "synthetic" and not self.real_cases:
            raise OutboundDenied("真实案件不得外发")
        if case_kind == "synthetic" and not self.synthetic_cases:
            raise OutboundDenied("合成案件外发未获授权")
        if kb_class == "public_statutes" and not self.public_statutes:
            raise OutboundDenied("公开法规/知识库片段外发未获仓库所有者批准(config/outbound.json)")
        if kb_class not in {"synthetic", "public_statutes"}:
            raise OutboundDenied(f"未知知识库类别: {kb_class}")
