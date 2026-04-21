"""Approval policies for side-effecting tools."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from .tool_policy import evaluate_tool_calls
from tools.metadata import ToolMetadata


SIDE_EFFECT_TOOLS: set[str] = {
    "create_calendar_event",
    "update_calendar_event",
    "delete_calendar_event",
    "set_reminder",
    "delete_reminder",
    "send_email",
    "create_task_list",
    "create_task",
    "complete_task",
    "delete_task",
}

_AFFIRMATIVE_PHRASES = {
    "yes",
    "y",
    "yeah",
    "yep",
    "sure",
    "ok",
    "okay",
    "please do",
    "go ahead",
    "confirm",
    "confirmed",
    "批准",
    "确认",
    "可以",
    "好",
    "好的",
    "行",
    "继续",
    "执行",
    "发送",
    "创建",
}

_NEGATIVE_PHRASES = {
    "no",
    "n",
    "nope",
    "cancel",
    "stop",
    "don't",
    "do not",
    "not now",
    "拒绝",
    "取消",
    "不用",
    "不要",
    "算了",
    "先别",
    "不需要",
}


def classify_confirmation_response(text: str) -> Optional[str]:
    """Return approve/reject when a user reply clearly resolves a pending action."""
    normalized = " ".join(text.strip().lower().split())
    if not normalized:
        return None
    if normalized in _AFFIRMATIVE_PHRASES or any(
        phrase in normalized for phrase in ("go ahead", "please do", "可以执行", "确认执行")
    ):
        return "approve"
    if normalized in _NEGATIVE_PHRASES or any(
        phrase in normalized for phrase in ("don't", "do not", "先别", "不需要", "先不要")
    ):
        return "reject"
    return None


def requires_confirmation(tool_calls: Iterable[Dict[str, Any]]) -> bool:
    """Return True when any requested tool mutates external state."""
    decision = evaluate_tool_calls(tool_calls)
    return decision.requires_confirmation


_SENSITIVE_ARG_KEYS = {
    "body",
    "content",
    "message",
    "html",
    "token",
    "access_token",
    "refresh_token",
    "password",
}


def _redact_args(args: dict[str, Any]) -> dict[str, Any]:
    redacted: dict[str, Any] = {}
    for key, value in args.items():
        if key.lower() in _SENSITIVE_ARG_KEYS:
            text = str(value)
            redacted[key] = f"[redacted:{len(text)} chars]"
        else:
            redacted[key] = value
    return redacted


def build_tool_preview(call: Dict[str, Any], metadata: ToolMetadata | None = None) -> Dict[str, Any]:
    """Build a safe, structured preview for a pending side effect."""
    args = dict(call.get("args", {}))
    safe_args = _redact_args(args)
    name = str(call.get("name") or "unknown")
    preview_parts = [f"{key}={value}" for key, value in list(safe_args.items())[:4]]
    return {
        "tool": name,
        "summary": f"{name}({', '.join(preview_parts)})",
        "args": safe_args,
        "side_effect": metadata.side_effect if metadata else "write",
        "supports_dry_run": bool(metadata.supports_dry_run) if metadata else False,
        "dry_run_handler": metadata.dry_run_handler if metadata else None,
    }


def build_confirmation_request(tool_calls: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a serializable confirmation payload for pending side effects."""
    serialized_calls = list(tool_calls)
    decision = evaluate_tool_calls(serialized_calls)
    copied_calls: List[Dict[str, Any]] = []
    previews: List[Dict[str, Any]] = []
    for call in serialized_calls:
        metadata = decision.metadata.get(call.get("name", ""))
        preview = build_tool_preview(call, metadata)
        copied = {
            "id": call.get("id"),
            "name": call.get("name"),
            "args": preview["args"],
            "type": call.get("type"),
            "side_effect": metadata.side_effect if metadata else "read",
            "supports_dry_run": bool(metadata.supports_dry_run) if metadata else False,
            "dry_run_handler": metadata.dry_run_handler if metadata else None,
        }
        copied_calls.append(copied)
        previews.append(preview)

    return {
        "tool_calls": copied_calls,
        "preview": "; ".join(preview["summary"] for preview in previews),
        "preview_payloads": previews,
        "requested_tools": [call.get("name") for call in copied_calls],
        "highest_side_effect": decision.highest_side_effect,
        "dry_run_candidates": decision.dry_run_candidates,
        "requires_explicit_confirmation": decision.requires_confirmation,
    }
