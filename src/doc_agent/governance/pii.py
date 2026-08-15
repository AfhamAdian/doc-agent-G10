"""Governance — PII detection + redaction (mandatory)"""

from __future__ import annotations

import re
from typing import Any

from ..contracts import Answer, Chunk

_PATTERNS = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "phone": re.compile(r"(?<!\d)(?:\+?88)?01[3-9]\d{8}(?!\d)"),
}


def detect(text: str) -> list[tuple[int, int, str]]:
    """Return non-overlapping email and Bangladesh phone-number spans."""
    spans = [
        (match.start(), match.end(), kind)
        for kind, pattern in _PATTERNS.items()
        for match in pattern.finditer(text)
    ]
    return sorted(spans, key=lambda span: (span[0], span[1]))


def redact(text: str) -> str:
    redacted = text
    for start, end, kind in reversed(detect(text)):
        redacted = f"{redacted[:start]}[REDACTED_{kind.upper()}]{redacted[end:]}"
    return redacted


def register(hooks: Any) -> None:
    """Wire the same deterministic redaction into OCR, answers, and logs."""

    def _scrub(ctx: dict) -> dict:
        if "chunks" in ctx:
            ctx["chunks"] = [
                (
                    chunk.model_copy(update={"text": redact(chunk.text)})
                    if isinstance(chunk, Chunk)
                    else chunk
                )
                for chunk in ctx["chunks"]
            ]
        answer = ctx.get("answer")
        if isinstance(answer, Answer):
            ctx["answer"] = answer.model_copy(update={"text": redact(answer.text)})
        if isinstance(ctx.get("message"), str):
            ctx["message"] = redact(ctx["message"])
        return ctx

    hooks.register(hooks.AFTER_OCR, _scrub)  # scrub extracted text before indexing
    hooks.register(hooks.BEFORE_ANSWER, _scrub)  # scrub the outgoing answer
    hooks.register(hooks.ON_LOG, _scrub)  # scrub logs
