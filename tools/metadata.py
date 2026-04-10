"""Static metadata for tools used by the runtime and policy layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

SideEffectLevel = Literal["read", "write", "external_send", "destructive"]
RetryPolicy = Literal["none", "safe_read_retry", "idempotent_write_retry"]


@dataclass(frozen=True)
class ToolMetadata:
    """Execution policy metadata for one tool."""

    name: str
    capability: str = "general"
    side_effect: SideEffectLevel = "read"
    requires_confirmation: bool = False
    supports_dry_run: bool = False
    timeout_seconds: int = 30
    retry_policy: RetryPolicy = "none"
    output_schema: dict[str, Any] = field(default_factory=dict)
    dry_run_handler: str | None = None

    def to_payload(self) -> dict[str, Any]:
        """Return a serializable metadata payload."""
        return {
            "name": self.name,
            "capability": self.capability,
            "side_effect": self.side_effect,
            "requires_confirmation": self.requires_confirmation,
            "supports_dry_run": self.supports_dry_run,
            "timeout_seconds": self.timeout_seconds,
            "retry_policy": self.retry_policy,
            "output_schema": dict(self.output_schema),
            "dry_run_handler": self.dry_run_handler,
        }


DEFAULT_TOOL_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["ok", "tool", "summary"],
    "properties": {
        "ok": {"type": "boolean"},
        "tool": {"type": "string"},
        "summary": {"type": "string"},
        "data": {"type": "object"},
    },
}


def _meta(
    name: str,
    *,
    capability: str = "general",
    side_effect: SideEffectLevel = "read",
    requires_confirmation: bool = False,
    supports_dry_run: bool = False,
    timeout_seconds: int = 30,
    retry_policy: RetryPolicy | None = None,
    dry_run_handler: str | None = None,
) -> ToolMetadata:
    if retry_policy is None:
        retry_policy = "safe_read_retry" if side_effect == "read" else "none"
    return ToolMetadata(
        name=name,
        capability=capability,
        side_effect=side_effect,
        requires_confirmation=requires_confirmation,
        supports_dry_run=supports_dry_run,
        timeout_seconds=timeout_seconds,
        retry_policy=retry_policy,
        output_schema=DEFAULT_TOOL_OUTPUT_SCHEMA,
        dry_run_handler=dry_run_handler,
    )


_TOOL_METADATA: dict[str, ToolMetadata] = {
    "calculator": _meta("calculator", capability="utility"),
    "get_weather": _meta("get_weather", capability="weather"),
    "translator": _meta("translator", capability="translation"),
    "web_reader": _meta("web_reader", capability="research", timeout_seconds=45),
    "web_searcher": _meta("web_searcher", capability="research", timeout_seconds=45),
    "file_system_search": _meta("file_system_search", capability="local_search"),
    "list_calendar_events": _meta("list_calendar_events", capability="calendar"),
    "find_free_slots": _meta("find_free_slots", capability="calendar"),
    "create_calendar_event": _meta(
        "create_calendar_event",
        capability="calendar",
        side_effect="write",
        requires_confirmation=True,
        supports_dry_run=True,
        dry_run_handler="preview_calendar_event",
    ),
    "update_calendar_event": _meta(
        "update_calendar_event",
        capability="calendar",
        side_effect="write",
        requires_confirmation=True,
        supports_dry_run=True,
        dry_run_handler="preview_calendar_update",
    ),
    "delete_calendar_event": _meta(
        "delete_calendar_event",
        capability="calendar",
        side_effect="destructive",
        requires_confirmation=True,
    ),
    "list_reminders": _meta("list_reminders", capability="reminder"),
    "set_reminder": _meta(
        "set_reminder",
        capability="reminder",
        side_effect="write",
        requires_confirmation=True,
        supports_dry_run=True,
        dry_run_handler="preview_reminder",
    ),
    "delete_reminder": _meta(
        "delete_reminder",
        capability="reminder",
        side_effect="destructive",
        requires_confirmation=True,
    ),
    "list_emails": _meta("list_emails", capability="mail"),
    "search_emails": _meta("search_emails", capability="mail"),
    "read_email": _meta("read_email", capability="mail"),
    "send_email": _meta(
        "send_email",
        capability="mail",
        side_effect="external_send",
        requires_confirmation=True,
        supports_dry_run=True,
        dry_run_handler="preview_email",
    ),
    "list_task_lists": _meta("list_task_lists", capability="tasks"),
    "list_tasks": _meta("list_tasks", capability="tasks"),
    "create_task_list": _meta(
        "create_task_list",
        capability="tasks",
        side_effect="write",
        requires_confirmation=True,
        supports_dry_run=True,
        dry_run_handler="preview_task_list",
    ),
    "create_task": _meta(
        "create_task",
        capability="tasks",
        side_effect="write",
        requires_confirmation=True,
        supports_dry_run=True,
        dry_run_handler="preview_task",
    ),
    "complete_task": _meta(
        "complete_task",
        capability="tasks",
        side_effect="write",
        requires_confirmation=True,
    ),
    "delete_task": _meta(
        "delete_task",
        capability="tasks",
        side_effect="destructive",
        requires_confirmation=True,
    ),
    "save_preference": _meta("save_preference", capability="memory", side_effect="write"),
}


def get_tool_metadata(name: str) -> ToolMetadata:
    """Return metadata for one tool, defaulting to a safe read-only policy."""
    return _TOOL_METADATA.get(name, _meta(name))


def list_tool_metadata() -> dict[str, ToolMetadata]:
    """Return a copy of the tool metadata registry."""
    return dict(_TOOL_METADATA)
