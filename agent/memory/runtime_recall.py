"""Compose layered memory context for runtime prompts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .profile_store import ProfileMemoryRecord, build_profile_memory_block, search_profile_memories
from .task_scratchpad import TaskScratchpad, build_task_scratchpad


@dataclass(frozen=True)
class RuntimeMemoryContext:
    """Layered memory context assembled for one runtime turn."""

    profile_records: list[ProfileMemoryRecord] = field(default_factory=list)
    session_recall_block: str = ""
    task_scratchpad: TaskScratchpad | None = None

    def render(self) -> str:
        """Render all active memory layers into one prompt block."""
        sections: list[str] = []
        profile_block = build_profile_memory_block(self.profile_records)
        if profile_block:
            sections.append(profile_block)
        if self.session_recall_block.strip():
            sections.append(self.session_recall_block.strip())
        if self.task_scratchpad is not None:
            sections.append(self.task_scratchpad.render())
        return "\n\n".join(section for section in sections if section.strip())


async def build_runtime_memory_context(
    *,
    user_id: str,
    query: str,
    exclude_session_id: str,
    include_preferences: bool,
    task_type: str,
    logger: Any,
    is_memory_enabled_fn: Callable[[], bool],
    get_memory_manager_fn: Callable[[], Any],
    build_session_recall_block_fn: Callable[..., str],
) -> RuntimeMemoryContext:
    """Assemble durable profile memory, session recall, and a task scratchpad."""
    profile_records: list[ProfileMemoryRecord] = []
    session_recall_block = ""

    if include_preferences and is_memory_enabled_fn() and user_id:
        try:
            profile_records = await search_profile_memories(
                user_id=user_id,
                query="user preferences",
                limit=10,
                manager_factory=get_memory_manager_fn,
            )
        except Exception as exc:
            logger.warning(f"Failed to inject preferences: {exc}")

    if query:
        try:
            session_recall_block = build_session_recall_block_fn(
                user_id=user_id,
                query=query,
                limit=3,
                exclude_session_id=exclude_session_id,
            )
        except Exception as exc:
            logger.warning(f"Failed to inject session recall: {exc}")

    return RuntimeMemoryContext(
        profile_records=profile_records,
        session_recall_block=session_recall_block,
        task_scratchpad=build_task_scratchpad(
            user_query=query,
            session_id=exclude_session_id,
            task_type=task_type,
        ),
    )


async def build_runtime_recall_context(
    *,
    user_id: str,
    query: str,
    exclude_session_id: str,
    include_preferences: bool,
    task_type: str,
    logger: Any,
    is_memory_enabled_fn: Callable[[], bool],
    get_memory_manager_fn: Callable[[], Any],
    build_session_recall_block_fn: Callable[..., str],
) -> str:
    """Backward-compatible string renderer for layered runtime memory context."""
    context = await build_runtime_memory_context(
        user_id=user_id,
        query=query,
        exclude_session_id=exclude_session_id,
        include_preferences=include_preferences,
        task_type=task_type,
        logger=logger,
        is_memory_enabled_fn=is_memory_enabled_fn,
        get_memory_manager_fn=get_memory_manager_fn,
        build_session_recall_block_fn=build_session_recall_block_fn,
    )
    return context.render()
