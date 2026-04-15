"""SQLite-backed transcript store for cross-session recall."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any, Optional

from agent.utils import get_logger

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]{2,}")


class SessionStore:
    """Persist assistant transcripts and support simple lexical recall."""

    def __init__(self, db_path: str | Path | None = None):
        raw_path = db_path or os.environ.get("SESSION_STORE_PATH", "./session_recall.db")
        self.db_path = Path(raw_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS transcript_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_transcript_user_created
                    ON transcript_messages(user_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_transcript_session_created
                    ON transcript_messages(session_id, created_at DESC);
                """
            )
            self._conn.commit()

    def add_message(
        self,
        *,
        user_id: str,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Persist one transcript message."""
        text = (content or "").strip()
        if not text:
            return
        payload = json.dumps(metadata or {}, ensure_ascii=False)
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO transcript_messages (user_id, session_id, role, content, metadata)
                VALUES (?, ?, ?, ?, ?)
                """,
                (user_id, session_id, role, text, payload),
            )
            self._conn.commit()

    def search_messages(
        self,
        *,
        user_id: str,
        query: str,
        limit: int = 5,
        exclude_session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return the top lexical transcript matches for a user."""
        tokens = [token.lower() for token in _TOKEN_RE.findall(query or "") if len(token) >= 2]
        if not tokens and not (query or "").strip():
            return []

        sql = """
            SELECT session_id, role, content, metadata, created_at
            FROM transcript_messages
            WHERE user_id = ?
        """
        params: list[Any] = [user_id]
        if exclude_session_id:
            sql += " AND session_id != ?"
            params.append(exclude_session_id)
        sql += " ORDER BY created_at DESC LIMIT 300"

        with self._lock:
            rows = list(self._conn.execute(sql, params))

        query_text = (query or "").strip().lower()
        scored: list[dict[str, Any]] = []
        for row in rows:
            content = str(row["content"])
            lowered = content.lower()
            score = sum(lowered.count(token) for token in tokens)
            if query_text and query_text in lowered:
                score += max(2, len(tokens) or 1)
            if score <= 0 and tokens:
                continue
            metadata = row["metadata"]
            try:
                parsed_metadata = json.loads(metadata) if metadata else {}
            except Exception:
                parsed_metadata = {}
            scored.append(
                {
                    "session_id": row["session_id"],
                    "role": row["role"],
                    "content": content,
                    "metadata": parsed_metadata,
                    "created_at": row["created_at"],
                    "score": float(score),
                }
            )

        scored.sort(key=lambda item: (item["score"], item["created_at"]), reverse=True)
        return scored[:limit]


_session_store: SessionStore | None = None


def get_session_store() -> SessionStore:
    """Return a process-wide session store singleton."""
    global _session_store
    if _session_store is None:
        _session_store = SessionStore()
    return _session_store
