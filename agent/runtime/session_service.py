"""Session lifecycle and checkpoint-state management for the runtime layer."""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Awaitable, Callable


class SessionService:
    """Manage session ids, checkpoint state, expiry, and reset flows."""

    def __init__(
        self,
        *,
        app: Any,
        logger: Any,
        session_timeout_seconds: int,
        session_sweep_interval: int,
        is_memory_enabled_fn: Callable[[], bool],
        get_memory_manager_fn: Callable[[], Any],
        force_extract_preferences_fn: Callable[..., Awaitable[int]],
    ) -> None:
        self.app = app
        self.logger = logger
        self.session_timeout_seconds = session_timeout_seconds
        self.session_sweep_interval = session_sweep_interval
        self.is_memory_enabled_fn = is_memory_enabled_fn
        self.get_memory_manager_fn = get_memory_manager_fn
        self.force_extract_preferences_fn = force_extract_preferences_fn

        self._active_sessions: dict[str, dict[str, Any]] = {}
        self._session_aliases: dict[str, str] = {}
        self._finalized_sessions: set[str] = set()

    def build_session_id(self, userid: str) -> str:
        return f"{userid}_{uuid.uuid4().hex[:12]}"

    def is_session_owned_by_user(self, userid: str, session_id: str) -> bool:
        return session_id == userid or session_id.startswith(f"{userid}_")

    def resolve_session_id(self, session_id: str) -> str:
        current = session_id
        seen = set()
        while current in self._session_aliases and current not in seen:
            seen.add(current)
            current = self._session_aliases[current]
        return current

    def session_config(self, session_id: str) -> dict[str, dict[str, str]]:
        return {"configurable": {"thread_id": self.resolve_session_id(session_id)}}

    def get_state(self, session_id: str) -> Any:
        return self.app.get_state(self.session_config(session_id))

    @staticmethod
    def has_session_messages(state: Any) -> bool:
        return bool(state and state.values and state.values.get("messages"))

    def session_exists(self, session_id: str) -> bool:
        return self.has_session_messages(self.get_state(session_id))

    def create_or_resume_session(self, userid: str, session_id: str | None = None) -> tuple[str, bool]:
        target = session_id or self.build_session_id(userid)
        if not self.is_session_owned_by_user(userid, target):
            raise ValueError("session_id must match userid namespace")
        effective_session_id = self.resolve_session_id(target)
        is_new = not self.session_exists(effective_session_id)
        return effective_session_id, is_new

    def touch_session(self, session_id: str, userid: str) -> None:
        effective_session_id = self.resolve_session_id(session_id)
        self._active_sessions[effective_session_id] = {
            "userid": userid,
            "last_activity": time.time(),
        }

    def roll_session_forward(self, session_id: str, userid: str) -> str:
        effective_session_id = self.resolve_session_id(session_id)
        replacement_session_id = self.build_session_id(userid)

        alias_sources = [
            alias
            for alias in list(self._session_aliases.keys())
            if self.resolve_session_id(alias) == effective_session_id
        ]
        alias_sources.extend([session_id, effective_session_id])
        for alias in set(alias_sources):
            if alias != replacement_session_id:
                self._session_aliases[alias] = replacement_session_id

        self._active_sessions.pop(session_id, None)
        self._active_sessions.pop(effective_session_id, None)
        self._finalized_sessions.discard(replacement_session_id)
        return replacement_session_id

    async def expire_session(self, session_id: str, userid: str) -> None:
        """Force-extract preferences and rotate an idle session forward."""
        effective_session_id = self.resolve_session_id(session_id)
        self.logger.info(
            f"Session timeout: extracting preferences for {effective_session_id}",
            extra={"session_id": effective_session_id, "userid": userid},
        )
        try:
            state = self.get_state(effective_session_id)
            messages = state.values.get("messages", []) if state and state.values else []
            if (
                messages
                and self.is_memory_enabled_fn()
                and effective_session_id not in self._finalized_sessions
            ):
                count = await self.force_extract_preferences_fn(
                    messages=messages,
                    user_id=userid,
                    model="anthropic/claude-haiku-4-5-20251001",
                )
                self._finalized_sessions.add(effective_session_id)
                self.logger.info(
                    f"Timeout extraction: {count} preferences saved for {effective_session_id}"
                )
            replacement_session_id = self.roll_session_forward(effective_session_id, userid)
            self.logger.info(
                "Session expired and rolled forward",
                extra={
                    "session_id": effective_session_id,
                    "replacement_session_id": replacement_session_id,
                    "userid": userid,
                },
            )
        except Exception as exc:
            self.logger.warning(f"Timeout extraction failed for {effective_session_id}: {exc}")
        finally:
            self._active_sessions.pop(session_id, None)
            self._active_sessions.pop(effective_session_id, None)

    async def session_sweep_loop(self) -> None:
        """Background loop that expires idle sessions."""
        self.logger.info(
            f"Session sweep started (timeout={self.session_timeout_seconds}s, "
            f"interval={self.session_sweep_interval}s)"
        )
        while True:
            await asyncio.sleep(self.session_sweep_interval)
            now = time.time()
            expired = [
                (sid, info["userid"])
                for sid, info in list(self._active_sessions.items())
                if now - info["last_activity"] > self.session_timeout_seconds
            ]
            if expired:
                self.logger.info(f"Sweeping {len(expired)} expired sessions")
                await asyncio.gather(*[self.expire_session(sid, uid) for sid, uid in expired])

    async def reset_session(
        self,
        *,
        session_id: str,
        userid: str | None = None,
        preserve_memory: bool = True,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Reset short-term state for a session and optionally clear long-term memory."""
        request_id = uuid.uuid4().hex
        memory_userid = userid or session_id
        effective_session_id = self.resolve_session_id(session_id)
        self.logger.info(
            f"Resetting session {effective_session_id}",
            extra={
                "request_id": request_id,
                "user_id": memory_userid,
                "session_id": effective_session_id,
                "preserve_memory": preserve_memory,
            },
        )

        current_state = self.get_state(effective_session_id)
        current_messages = []
        if current_state and current_state.values:
            current_messages = current_state.values.get("messages", [])
        message_count = len(current_messages)

        prefs_extracted = 0
        if (
            preserve_memory
            and userid
            and self.is_memory_enabled_fn()
            and current_messages
            and effective_session_id not in self._finalized_sessions
        ):
            pref_model = model or "anthropic/claude-haiku-4-5-20251001"
            try:
                prefs_extracted = await self.force_extract_preferences_fn(
                    messages=current_messages,
                    user_id=userid,
                    model=pref_model,
                )
            except Exception as exc:
                self.logger.warning(f"force_extract_and_persist failed: {exc}")
            self._finalized_sessions.add(effective_session_id)

        memory_cleared = False
        if not preserve_memory and self.is_memory_enabled_fn() and userid:
            manager = self.get_memory_manager_fn()
            memory_cleared = await manager.clear_user_memories(userid)

        replacement_session_id = self.roll_session_forward(
            effective_session_id,
            userid or memory_userid,
        )

        self.logger.info(
            f"Session reset for {effective_session_id}",
            extra={
                "request_id": request_id,
                "user_id": memory_userid,
                "session_id": effective_session_id,
                "replacement_session_id": replacement_session_id,
                "messages_cleared": message_count,
                "memory_cleared": memory_cleared,
            },
        )
        return {
            "status": "success",
            "userid": userid,
            "session_id": session_id,
            "effective_session_id": effective_session_id,
            "replacement_session_id": replacement_session_id,
            "messages_cleared": message_count,
            "preferences_extracted": prefs_extracted,
            "memory_cleared": memory_cleared,
        }
