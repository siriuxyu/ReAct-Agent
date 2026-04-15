"""Helpers for session transcript recall blocks."""

from __future__ import annotations

from .session_store import SessionStore, get_session_store


def build_session_recall_block(
    *,
    user_id: str,
    query: str,
    limit: int = 3,
    exclude_session_id: str | None = None,
    store: SessionStore | None = None,
) -> str:
    """Build a prompt block containing recalled cross-session snippets."""
    active_store = store or get_session_store()
    results = active_store.search_messages(
        user_id=user_id,
        query=query,
        limit=limit,
        exclude_session_id=exclude_session_id,
    )
    if not results:
        return ""

    lines = []
    for result in results:
        created = result.get("created_at", "")
        date_prefix = f"[{created[:10]}] " if created else ""
        session_prefix = f"[{result['session_id']}] "
        role_prefix = f"[{result['role']}] "
        lines.append(f"- {date_prefix}{session_prefix}{role_prefix}{result['content'][:220]}")
    return "## Recalled session context\n" + "\n".join(lines)
