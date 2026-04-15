"""Ephemeral task scratchpad helpers for the current turn."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TaskScratchpad:
    """Structured per-turn scratchpad that should not persist across sessions."""

    user_query: str
    session_id: str
    task_type: str
    tool_artifacts: list[dict[str, Any]] = field(default_factory=list)

    def render(self) -> str:
        """Render the scratchpad into a compact prompt section."""
        lines = [
            "## Current task scratchpad",
            f"- active_session: {self.session_id or 'unknown'}",
            f"- task_type: {self.task_type or 'chat'}",
        ]
        if self.user_query:
            lines.append(f"- user_goal: {self.user_query}")
        if self.tool_artifacts:
            lines.append(f"- known_artifacts: {len(self.tool_artifacts)}")
        return "\n".join(lines)


def build_task_scratchpad(
    *,
    user_query: str,
    session_id: str,
    task_type: str,
    tool_artifacts: list[dict[str, Any]] | None = None,
) -> TaskScratchpad:
    """Create the per-turn scratchpad object."""
    return TaskScratchpad(
        user_query=user_query.strip(),
        session_id=session_id,
        task_type=task_type or "chat",
        tool_artifacts=list(tool_artifacts or []),
    )
