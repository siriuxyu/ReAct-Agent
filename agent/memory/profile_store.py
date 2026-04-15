"""Helpers for durable profile memory retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from agent.interfaces import StorageType


@dataclass(frozen=True)
class ProfileMemoryRecord:
    """One durable user profile memory surfaced to the runtime."""

    content: str
    created_at: str = ""
    memory_type: str = ""
    source: str = "profile"

    @property
    def line(self) -> str:
        """Render one prompt-friendly bullet line."""
        date_str = self.created_at[:10] if self.created_at else ""
        date_prefix = f"[{date_str}] " if date_str else ""
        type_prefix = f"[{self.memory_type}] " if self.memory_type else ""
        return f"- {date_prefix}{type_prefix}{self.content}"


async def search_profile_memories(
    *,
    user_id: str,
    query: str,
    limit: int,
    manager_factory: Callable[[], Any],
) -> list[ProfileMemoryRecord]:
    """Load durable user preference/profile memories from long-term storage."""
    if not user_id:
        return []

    manager = manager_factory()
    items = await manager.search_user_memories(
        user_id=user_id,
        query=query,
        limit=limit,
    )

    results: list[ProfileMemoryRecord] = []
    for item in items:
        metadata = item.get("metadata", {})
        if (
            metadata.get("document_type") != StorageType.USER_PREFERENCE.value
            and not metadata.get("preference_type")
        ):
            continue
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        results.append(
            ProfileMemoryRecord(
                content=content,
                created_at=str(item.get("created_at", "")),
                memory_type=str(metadata.get("preference_type", "")),
            )
        )
    return results


def build_profile_memory_block(records: list[ProfileMemoryRecord]) -> str:
    """Render durable profile memory records as one prompt section."""
    if not records:
        return ""
    return "## Known user preferences\n" + "\n".join(record.line for record in records)
