"""Runtime service that coordinates request preparation and turn execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from agent.adapters.api_adapter import PreparedAgentRun
from .types import TurnResult


@dataclass(frozen=True)
class RuntimeServiceResult:
    """Prepared request plus the executed turn result."""

    prepared: PreparedAgentRun
    turn: TurnResult


class AgentRuntimeService:
    """High-level runtime orchestrator for API-facing request flows."""

    def __init__(
        self,
        *,
        agent_runtime: Any,
        app: Any,
        context_cls: type,
        logger: Any,
        prepare_agent_run_fn: Callable[..., Awaitable[PreparedAgentRun]],
        invoke_graph_fn: Callable[..., Awaitable[tuple[list[dict[str, Any]], str, int]]],
        extract_text_fn: Callable[[dict[str, Any]], str | None],
        persist_transcript_fn: Callable[..., None],
        transcript_store_factory: Callable[[], Any],
        resolve_session_id: Callable[[str], str],
        is_session_owned_by_user: Callable[[str, str], bool],
        session_config: Callable[[str], dict[str, Any]],
        get_session_state: Callable[[str], Any],
        has_session_messages: Callable[[Any], bool],
        is_memory_enabled_fn: Callable[[], bool],
        get_memory_manager_fn: Callable[[], Any],
        build_session_recall_block_fn: Callable[..., str],
    ) -> None:
        self.agent_runtime = agent_runtime
        self.app = app
        self.context_cls = context_cls
        self.logger = logger
        self.prepare_agent_run_fn = prepare_agent_run_fn
        self.invoke_graph_fn = invoke_graph_fn
        self.extract_text_fn = extract_text_fn
        self.persist_transcript_fn = persist_transcript_fn
        self.transcript_store_factory = transcript_store_factory
        self.resolve_session_id = resolve_session_id
        self.is_session_owned_by_user = is_session_owned_by_user
        self.session_config = session_config
        self.get_session_state = get_session_state
        self.has_session_messages = has_session_messages
        self.is_memory_enabled_fn = is_memory_enabled_fn
        self.get_memory_manager_fn = get_memory_manager_fn
        self.build_session_recall_block_fn = build_session_recall_block_fn

    async def prepare_request(self, req: Any) -> PreparedAgentRun:
        """Prepare one API request for runtime execution."""
        return await self.prepare_agent_run_fn(
            req,
            context_cls=self.context_cls,
            logger=self.logger,
            resolve_session_id=self.resolve_session_id,
            is_session_owned_by_user=self.is_session_owned_by_user,
            session_config=self.session_config,
            get_session_state=self.get_session_state,
            has_session_messages=self.has_session_messages,
            is_memory_enabled_fn=self.is_memory_enabled_fn,
            get_memory_manager_fn=self.get_memory_manager_fn,
            build_session_recall_block_fn=self.build_session_recall_block_fn,
        )

    async def invoke_chat_request(
        self,
        *,
        message: str,
        user_id: str,
        system_prompt: str,
        model: str,
    ) -> TurnResult:
        """Execute a single-turn ad hoc chat request via the shared runtime."""
        context = self.context_cls(
            system_prompt=system_prompt,
            model=model,
            user_id=user_id,
        )
        turn = await self.agent_runtime.run_turn(
            app=self.app,
            messages=[{"role": "user", "content": message}],
            config={"configurable": {"thread_id": user_id}},
            context=context,
            request_messages=[type("ChatMessage", (), {"role": "user", "content": message})()],
            user_id=user_id,
            session_id=user_id,
            invoke_graph_fn=self.invoke_graph_fn,
            transcript_store=self.transcript_store_factory(),
            persist_transcript_fn=self.persist_transcript_fn,
        )
        return turn

    async def invoke_request(self, req: Any) -> RuntimeServiceResult:
        """Prepare and execute one non-streaming request."""
        prepared = await self.prepare_request(req)
        return await self.invoke_prepared_request(req, prepared=prepared)

    async def invoke_prepared_request(
        self,
        req: Any,
        *,
        prepared: PreparedAgentRun,
    ) -> RuntimeServiceResult:
        """Execute one non-streaming request after preparation is already complete."""
        turn = await self.agent_runtime.run_turn(
            app=self.app,
            messages=prepared.messages,
            config=prepared.config,
            context=prepared.context,
            request_messages=req.messages,
            user_id=req.userid,
            session_id=prepared.effective_session_id,
            invoke_graph_fn=self.invoke_graph_fn,
            transcript_store=self.transcript_store_factory(),
            persist_transcript_fn=self.persist_transcript_fn,
        )
        return RuntimeServiceResult(prepared=prepared, turn=turn)

    async def stream_request(
        self,
        req: Any,
        *,
        on_text_fn: Callable[[str], Awaitable[None]],
    ) -> RuntimeServiceResult:
        """Prepare and execute one streaming request."""
        prepared = await self.prepare_request(req)
        return await self.stream_prepared_request(req, prepared=prepared, on_text_fn=on_text_fn)

    async def stream_prepared_request(
        self,
        req: Any,
        *,
        prepared: PreparedAgentRun,
        on_text_fn: Callable[[str], Awaitable[None]],
    ) -> RuntimeServiceResult:
        """Execute one streaming request after preparation is already complete."""
        turn = await self.agent_runtime.run_stream_turn(
            app=self.app,
            messages=prepared.messages,
            config=prepared.config,
            context=prepared.context,
            request_messages=req.messages,
            user_id=req.userid,
            session_id=prepared.effective_session_id,
            extract_text_fn=self.extract_text_fn,
            transcript_store=self.transcript_store_factory(),
            persist_transcript_fn=self.persist_transcript_fn,
            on_text_fn=on_text_fn,
        )
        return RuntimeServiceResult(prepared=prepared, turn=turn)
