"""Framework-agnostic runtime facade for per-turn decisions and execution."""

from __future__ import annotations

from typing import Any, AsyncIterator, Awaitable, Callable, Iterable, Sequence

from langchain_core.messages import AIMessage, BaseMessage

from .executor import build_pending_confirmation, collect_tool_artifacts, latest_user_text
from .router import ModelRouteDecision, classify_task_type, explain_model_route
from .types import RuntimeInspection, TurnResult
from .workspace import RuntimeDecisionTrace, build_runtime_workspace
from ..policy.approval import classify_confirmation_response


class AgentRuntime:
    """Inspect messages and produce routing/control metadata for one turn."""

    def inspect_messages(
        self,
        messages: Iterable[BaseMessage],
        *,
        default_model: str,
        selected_model: str = "",
        pending_confirmation: dict | None = None,
    ) -> RuntimeInspection:
        history = list(messages)
        user_text = latest_user_text(history)
        artifacts = collect_tool_artifacts(history)
        task_type = classify_task_type(user_text)
        route_decision = explain_model_route(
            default_model,
            task_type=task_type,
            step_name="assistant",
            latest_user_text=user_text,
            has_tool_results=bool(artifacts),
        )
        if selected_model:
            route_decision = ModelRouteDecision(
                selected_model=selected_model,
                default_model=default_model,
                step_name="assistant",
                task_type=task_type,
                reason="request_override",
                signals={**route_decision.signals, "router_suggestion": route_decision.selected_model},
            )
        resolution = ""
        if pending_confirmation:
            resolution = classify_confirmation_response(user_text) or ""
        workspace = build_runtime_workspace(
            goal=user_text,
            task_type=task_type,
            artifacts=artifacts,
            pending_confirmation=pending_confirmation,
            confirmation_resolution=resolution,
            decision_trace=[
                RuntimeDecisionTrace(
                    kind="model_route",
                    decision=route_decision.selected_model,
                    reason=route_decision.reason,
                    signals=route_decision.to_payload(),
                )
            ],
        )
        return RuntimeInspection(
            latest_user_text=user_text,
            task_type=task_type,
            selected_model=route_decision.selected_model,
            pending_confirmation=None,
            confirmation_resolution=resolution,
            tool_artifacts=artifacts,
            workspace=workspace,
            route_decision=route_decision,
        )

    def inspect_model_response(self, response: AIMessage) -> RuntimeInspection:
        """Inspect a model response for pending confirmations."""
        pending_confirmation = build_pending_confirmation(response)
        workspace = build_runtime_workspace(
            goal="",
            task_type="chat",
            pending_confirmation=pending_confirmation,
        )
        return RuntimeInspection(
            pending_confirmation=pending_confirmation,
            workspace=workspace,
        )

    async def run_turn(
        self,
        *,
        app: Any,
        messages: list[dict[str, Any]],
        config: dict[str, Any],
        context: Any,
        request_messages: Sequence[Any],
        user_id: str,
        session_id: str,
        invoke_graph_fn: Callable[..., Awaitable[tuple[list[dict[str, Any]], str, int]]],
        transcript_store: Any,
        persist_transcript_fn: Callable[..., None],
    ) -> TurnResult:
        """Execute one non-streaming turn and persist transcript side effects."""
        response_messages, final_response, chunk_count = await invoke_graph_fn(
            app,
            messages=messages,
            config=config,
            context=context,
        )
        persist_transcript_fn(
            store=transcript_store,
            user_id=user_id,
            session_id=session_id,
            request_messages=request_messages,
            final_response=final_response,
        )
        return TurnResult(
            response_messages=response_messages,
            final_response=final_response,
            chunk_count=chunk_count,
        )

    async def run_stream_turn(
        self,
        *,
        app: Any,
        messages: list[dict[str, Any]],
        config: dict[str, Any],
        context: Any,
        request_messages: Sequence[Any],
        user_id: str,
        session_id: str,
        extract_text_fn: Callable[[dict[str, Any]], str | None],
        transcript_store: Any,
        persist_transcript_fn: Callable[..., None],
        on_text_fn: Callable[[str], Awaitable[None]],
    ) -> TurnResult:
        """Execute one streaming turn, forwarding text chunks and persisting the result."""
        final_response = ""
        chunk_count = 0

        async for chunk in self._astream_app(app, messages=messages, config=config, context=context):
            chunk_count += 1
            text = extract_text_fn(chunk)
            if not text:
                continue
            final_response = text
            await on_text_fn(text)

        persist_transcript_fn(
            store=transcript_store,
            user_id=user_id,
            session_id=session_id,
            request_messages=request_messages,
            final_response=final_response,
        )
        return TurnResult(
            response_messages=[],
            final_response=final_response,
            chunk_count=chunk_count,
        )

    async def _astream_app(
        self,
        app: Any,
        *,
        messages: list[dict[str, Any]],
        config: dict[str, Any],
        context: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        """Small seam around framework-specific streaming for easier testing."""
        async for chunk in app.astream({"messages": messages}, config=config, context=context):
            yield chunk
