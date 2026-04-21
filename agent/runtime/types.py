"""Typed runtime data structures shared across the agent control plane."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

if False:  # pragma: no cover
    from .router import ModelRouteDecision
    from .workspace import RuntimeWorkspace


@dataclass(frozen=True)
class ToolCallSpec:
    """Serializable representation of a model-requested tool call."""

    id: str | None
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    type: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "args": dict(self.args),
            "type": self.type,
        }


@dataclass(frozen=True)
class PendingConfirmation:
    """Pending side effect awaiting explicit user approval."""

    tool_calls: list[ToolCallSpec] = field(default_factory=list)
    preview: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {
            "tool_calls": [tool_call.to_payload() for tool_call in self.tool_calls],
            "preview": self.preview,
            "requested_tools": [tool_call.name for tool_call in self.tool_calls],
        }


@dataclass(frozen=True)
class ToolArtifact:
    """Structured result returned by a tool execution."""

    tool: str | None
    tool_call_id: str | None
    ok: bool
    summary: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "tool_call_id": self.tool_call_id,
            "ok": self.ok,
            "summary": self.summary,
            "data": dict(self.data),
        }


@dataclass(frozen=True)
class RuntimeInspection:
    """High-level per-turn runtime metadata."""

    latest_user_text: str = ""
    task_type: str = "chat"
    selected_model: str = ""
    pending_confirmation: PendingConfirmation | None = None
    confirmation_resolution: str = ""
    tool_artifacts: list[dict[str, Any]] = field(default_factory=list)
    workspace: "RuntimeWorkspace | None" = None
    route_decision: "ModelRouteDecision | None" = None


@dataclass(frozen=True)
class TurnResult:
    """Unified result for one completed runtime turn."""

    response_messages: list[dict[str, Any]] = field(default_factory=list)
    final_response: str = ""
    chunk_count: int = 0
