"""Execution inspection helpers for the runtime layer."""

from __future__ import annotations

import json
from typing import Any, Iterable, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from .types import PendingConfirmation, ToolArtifact, ToolCallSpec
from ..policy.approval import build_confirmation_request, requires_confirmation
from ..utils import get_message_text


def latest_user_text(messages: Iterable[BaseMessage]) -> str:
    """Return the newest real user message, ignoring generated summaries."""
    for message in reversed(list(messages)):
        if not isinstance(message, HumanMessage):
            continue
        if getattr(message, "additional_kwargs", {}).get("summary_generated"):
            continue
        return get_message_text(message)
    return ""


def parse_tool_payload(content: Any) -> Optional[dict[str, Any]]:
    """Parse a JSON tool payload when available."""
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        return None
    text = content.strip()
    if not text.startswith("{"):
        return None
    try:
        parsed = json.loads(text)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def collect_tool_artifacts(messages: Iterable[BaseMessage]) -> list[dict[str, Any]]:
    """Convert ToolMessages into structured artifacts for later reasoning."""
    artifacts: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        payload = parse_tool_payload(message.content)
        if payload is not None:
            payload.setdefault("tool_call_id", message.tool_call_id)
            artifacts.append(ToolArtifact(
                tool=payload.get("tool"),
                tool_call_id=payload.get("tool_call_id"),
                ok=bool(payload.get("ok", True)),
                summary=str(payload.get("summary", "")),
                data=dict(payload.get("data", {})),
            ).to_payload())
            continue
        artifacts.append(
            ToolArtifact(
                tool=getattr(message, "name", None),
                tool_call_id=message.tool_call_id,
                ok=True,
                summary=str(message.content),
                data={"raw": str(message.content)},
            ).to_payload()
        )
    return artifacts


def extract_ai_tool_calls(message: AIMessage) -> list[ToolCallSpec]:
    """Normalize an AIMessage's tool calls into serializable tool specs."""
    return [
        ToolCallSpec(
            id=tool_call.get("id"),
            name=tool_call.get("name", ""),
            args=dict(tool_call.get("args", {})),
            type=tool_call.get("type"),
        )
        for tool_call in message.tool_calls or []
    ]


def build_pending_confirmation(message: AIMessage) -> PendingConfirmation | None:
    """Create a pending confirmation payload when the model requested side effects."""
    tool_calls = extract_ai_tool_calls(message)
    if not tool_calls:
        return None
    serialized = [tool_call.to_payload() for tool_call in tool_calls]
    if not requires_confirmation(serialized):
        return None
    payload = build_confirmation_request(serialized)
    return PendingConfirmation(tool_calls=tool_calls, preview=payload["preview"])
