"""Helpers for structured tool responses."""

from __future__ import annotations

import json
from typing import Any


def tool_ok(tool: str, summary: str, **data: Any) -> str:
    return json.dumps(
        {
            "ok": True,
            "tool": tool,
            "summary": summary,
            "data": data,
        },
        ensure_ascii=False,
    )


def tool_error(tool: str, summary: str, **data: Any) -> str:
    return json.dumps(
        {
            "ok": False,
            "tool": tool,
            "summary": summary,
            "data": data,
        },
        ensure_ascii=False,
    )
