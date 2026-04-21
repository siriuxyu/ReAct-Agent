"""LangGraph node implementations backed by the shared runtime layer."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Literal, Optional, cast

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.runtime import Runtime

from agent.summarizer import compress_messages, needs_compression
from tools.save_preference import make_save_preference

from ..context import Context
from ..state import State
from ..utils import load_chat_model
from .agent_runtime import AgentRuntime
from .executor import collect_tool_artifacts, latest_user_text
from .router import select_model_for_step
from .tool_execution import execute_tool_phase, resolve_runtime_tools

if hasattr(datetime, "UTC"):
    UTC = datetime.UTC
else:
    UTC = timezone.utc


def build_tools_with_memory(
    *,
    user_id: Optional[str],
    tool_registry: Any,
    logger: Any,
    is_memory_enabled_fn: Callable[[], bool],
    get_memory_manager_fn: Callable[[], Any],
) -> List[Any]:
    """Build the runtime tool list, including per-user memory tools."""

    def _dynamic_tools(target_user_id: Optional[str]) -> List[Any]:
        dynamic_tools: List[Any] = []
        if is_memory_enabled_fn() and target_user_id:
            manager = get_memory_manager_fn()
            memory_tools = manager.get_tools_for_user(target_user_id)
            dynamic_tools.extend(memory_tools)
            dynamic_tools.append(make_save_preference(target_user_id))
            logger.debug(
                "Added dynamic memory tools",
                extra={
                    "function": "build_tools_with_memory",
                    "user_id": target_user_id,
                    "details": {"memory_tool_count": len(memory_tools)},
                },
            )
        return dynamic_tools

    return tool_registry.build_runtime_tools(
        user_id=user_id,
        dynamic_tool_builder=_dynamic_tools,
    )


def create_call_model_node(
    *,
    runtime_facade: AgentRuntime,
    tool_registry: Any,
    logger: Any,
    is_memory_enabled_fn: Callable[[], bool],
    is_storage_available_fn: Callable[[], bool],
    get_memory_manager_fn: Callable[[], Any],
):
    """Create the LangGraph model node while keeping graph.py thin."""

    async def call_model(
        state: State,
        runtime: Runtime[Context],
    ) -> Dict[str, List[AIMessage]]:
        start_time = time.time()
        context = runtime.context
        user_id = context.user_id if context and context.user_id else None

        latest_user = latest_user_text(state.messages)
        selected_model = state.selected_model or select_model_for_step(
            context.model,
            task_type=state.task_type,
            step_name="assistant",
            latest_user_text=latest_user,
            has_tool_results=bool(state.tool_artifacts),
        )

        logger.info(
            "Calling model",
            extra={
                "function": "call_model",
                "details": {
                    "model": selected_model,
                    "message_count": len(state.messages),
                    "is_last_step": state.is_last_step,
                    "user_id": user_id,
                    "memory_enabled": is_memory_enabled_fn(),
                    "storage_available": is_storage_available_fn(),
                    "task_type": state.task_type,
                },
            },
        )

        tools = build_tools_with_memory(
            user_id=user_id,
            tool_registry=tool_registry,
            logger=logger,
            is_memory_enabled_fn=is_memory_enabled_fn,
            get_memory_manager_fn=get_memory_manager_fn,
        )
        base_model = load_chat_model(selected_model)

        if context.enable_web_search and selected_model.startswith("anthropic/"):
            try:
                from langchain_anthropic import ChatAnthropic

                if isinstance(base_model, ChatAnthropic):
                    web_search_tool_def = {
                        "type": "web_search_20250305",
                        "name": "web_search",
                        "max_uses": context.web_search_max_uses,
                    }
                    model = base_model.bind_tools(tools)
                    model._anthropic_web_search = web_search_tool_def
                    if hasattr(model, "kwargs") and isinstance(model.kwargs, dict):
                        existing_tools = model.kwargs.get("tools", [])
                        if not isinstance(existing_tools, list):
                            existing_tools = []
                        existing_tools.append(web_search_tool_def)
                        model.kwargs["tools"] = existing_tools
                    logger.info(
                        "Web search enabled for Claude model",
                        extra={
                            "function": "call_model",
                            "details": {
                                "web_search_max_uses": context.web_search_max_uses,
                                "model": selected_model,
                            },
                        },
                    )
                else:
                    model = base_model.bind_tools(tools)
            except Exception as exc:
                logger.warning(
                    f"Failed to enable web search: {exc}, falling back to regular tools",
                    extra={"function": "call_model", "error": str(exc)},
                )
                model = base_model.bind_tools(tools)
        else:
            model = base_model.bind_tools(tools)

        logger.debug(
            "Model loaded and tools bound",
            extra={
                "function": "call_model",
                "details": {
                    "model": selected_model,
                    "available_tools": [getattr(tool, "name", str(tool)) for tool in tools],
                    "total_tools": len(tools),
                },
            },
        )

        system_message = context.system_prompt.format(
            system_time=datetime.now(tz=UTC).isoformat()
        )
        if is_memory_enabled_fn() and user_id:
            system_message += (
                "\n\nYou have access to long-term memory. "
                "Use 'search_memory' to recall relevant information from past conversations."
            )

        logger.debug(
            "Prepared system prompt",
            extra={
                "function": "call_model",
                "details": {
                    "system_prompt_length": len(system_message),
                    "system_time": datetime.now(tz=UTC).isoformat(),
                },
            },
        )

        messages_to_use = state.messages
        if needs_compression(state.messages):
            logger.info(
                "Compressing long context",
                extra={"function": "call_model", "details": {"message_count": len(state.messages)}},
            )
            messages_to_use = await compress_messages(state.messages, model=selected_model)

        invoke_messages = [{"role": "system", "content": system_message}, *messages_to_use]

        from agent.model_router import invoke_with_fallback

        response = cast(
            AIMessage,
            await invoke_with_fallback(
                model,
                invoke_messages,
                tools,
                primary_spec=selected_model,
            ),
        )

        model_duration_ms = (time.time() - start_time) * 1000
        logger.info(
            "Model response received",
            extra={
                "function": "call_model",
                "duration_ms": round(model_duration_ms, 2),
                "details": {
                    "has_tool_calls": bool(response.tool_calls),
                    "tool_calls_count": len(response.tool_calls) if response.tool_calls else 0,
                    "response_length": len(response.content) if response.content else 0,
                    "response_preview": response.content[:200] if response.content else None,
                },
            },
        )

        if response.tool_calls:
            logger.debug(
                "Model requested tool calls",
                extra={
                    "function": "call_model",
                    "details": {
                        "tool_calls": [
                            {
                                "name": tc.get("name", "unknown"),
                                "args_preview": {
                                    key: str(value)[:50]
                                    for key, value in tc.get("args", {}).items()
                                },
                            }
                            for tc in response.tool_calls
                        ]
                    },
                },
            )

        if state.is_last_step and response.tool_calls:
            logger.warning(
                "Model requested tools on last step, returning fallback message",
                extra={
                    "function": "call_model",
                    "details": {"tool_calls_count": len(response.tool_calls)},
                },
            )
            return {
                "messages": [
                    AIMessage(
                        id=response.id,
                        content=(
                            "Sorry, I could not find an answer to your question in the "
                            "specified number of steps."
                        ),
                    )
                ]
            }

        runtime_facade.inspect_model_response(response)
        return {
            "messages": [response],
            "enable_preference_extraction": context.enable_preference_extraction,
            "selected_model": selected_model,
        }

    return call_model


def create_planner_node(
    *,
    runtime_facade: AgentRuntime,
    logger: Any,
):
    """Create the planning node used ahead of model execution."""

    async def plan_next_step(
        state: State,
        runtime: Runtime[Context],
    ) -> Dict[str, Any]:
        inspection = runtime_facade.inspect_messages(
            state.messages,
            default_model=runtime.context.model,
            selected_model=state.selected_model,
            pending_confirmation=state.pending_confirmation,
        )

        updates: Dict[str, Any] = {
            "task_type": inspection.task_type,
            "selected_model": inspection.selected_model,
            "confirmation_resolution": "",
            "confirmation_response_text": "",
            "confirmed_tool_calls": [],
            "workspace": inspection.workspace.to_payload() if inspection.workspace else {},
        }

        if state.pending_confirmation and inspection.latest_user_text:
            if inspection.confirmation_resolution == "approve":
                updates["confirmed_tool_calls"] = state.pending_confirmation.get("tool_calls", [])
                updates["pending_confirmation"] = None
                updates["confirmation_resolution"] = "approved"
                updates["confirmation_response_text"] = state.pending_confirmation.get(
                    "preview", ""
                )
            elif inspection.confirmation_resolution == "reject":
                updates["pending_confirmation"] = None
                updates["confirmation_resolution"] = "cancelled"
                updates["confirmation_response_text"] = state.pending_confirmation.get(
                    "preview", ""
                )

        logger.info(
            "Planned next step",
            extra={
                "function": "plan_next_step",
                "details": {
                    "task_type": inspection.task_type,
                    "selected_model": inspection.selected_model,
                    "has_pending_confirmation": bool(state.pending_confirmation),
                    "confirmation_resolution": updates["confirmation_resolution"],
                },
            },
        )
        return updates

    return plan_next_step


def create_tools_node(
    *,
    tool_registry: Any,
    logger: Any,
    is_memory_enabled_fn: Callable[[], bool],
    get_memory_manager_fn: Callable[[], Any],
):
    """Create the tool-execution node for the LangGraph backend."""

    async def execute_tools(
        state: State,
        runtime: Runtime[Context],
    ) -> Dict[str, Any]:
        context = runtime.context
        user_id = context.user_id if context and context.user_id else None
        all_tools = resolve_runtime_tools(
            user_id,
            lambda target_user_id: build_tools_with_memory(
                user_id=target_user_id,
                tool_registry=tool_registry,
                logger=logger,
                is_memory_enabled_fn=is_memory_enabled_fn,
                get_memory_manager_fn=get_memory_manager_fn,
            ),
        )

        tool_names = [getattr(tool, "name", str(tool)) for tool in all_tools]
        logger.info(
            f"Executing tools for user {user_id}",
            extra={
                "function": "execute_tools",
                "details": {
                    "user_id": user_id,
                    "available_tools": tool_names,
                    "total_tools": len(all_tools),
                },
            },
        )

        result = await execute_tool_phase(
            messages=list(state.messages),
            all_tools=all_tools,
            confirmed_tool_calls=list(state.confirmed_tool_calls),
        )

        if result.get("pending_confirmation"):
            pending = result["pending_confirmation"]
            logger.info(
                "Intercepted side-effecting tool call pending user confirmation",
                extra={
                    "function": "execute_tools",
                    "details": {
                        "requested_tools": pending["requested_tools"],
                        "preview": pending["preview"],
                    },
                },
            )
        return result

    return execute_tools


async def collect_tool_outputs_node(
    state: State,
    runtime: Runtime[Context],
) -> Dict[str, Any]:
    """Parse recent tool messages into structured artifacts for later reasoning."""
    recent_tool_messages: List[ToolMessage] = []
    for message in reversed(state.messages):
        if isinstance(message, ToolMessage):
            recent_tool_messages.append(message)
            continue
        break
    recent_tool_messages.reverse()
    artifacts = collect_tool_artifacts(recent_tool_messages)
    return {"tool_artifacts": artifacts}


async def confirmation_cancelled_node(
    state: State,
    runtime: Runtime[Context],
) -> Dict[str, List[AIMessage]]:
    """Return the assistant message after the user rejects a pending action."""
    preview = state.confirmation_response_text or "the requested action"
    return {
        "messages": [AIMessage(content=f"Okay, I will not execute {preview}.")],
        "confirmation_resolution": "",
        "confirmation_response_text": "",
    }


def route_after_planner(state: State) -> Literal["call_model", "tools", "confirmation_cancelled"]:
    """Route planner output toward model execution, tools, or cancellation."""
    if state.confirmed_tool_calls:
        return "tools"
    if state.confirmation_resolution == "cancelled":
        return "confirmation_cancelled"
    return "call_model"


def create_model_output_route(*, logger: Any):
    """Create the LangGraph route function for model output."""

    def route_model_output(state: State) -> Literal["__end__", "tools", "extract_preferences"]:
        last_message = state.messages[-1]
        if not isinstance(last_message, AIMessage):
            raise ValueError(
                f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
            )

        has_tool_calls = bool(last_message.tool_calls)
        logger.debug(
            "Routing model output",
            extra={
                "function": "route_model_output",
                "details": {
                    "has_tool_calls": has_tool_calls,
                    "tool_calls_count": len(last_message.tool_calls)
                    if last_message.tool_calls
                    else 0,
                },
            },
        )

        if not has_tool_calls:
            if not state.enable_preference_extraction:
                return "__end__"
            logger.info(
                "No tool calls detected, routing to preference extraction",
                extra={
                    "function": "route_model_output",
                    "details": {"next_node": "extract_preferences"},
                },
            )
            return "extract_preferences"

        logger.info(
            "Tool calls detected, routing to tools",
            extra={
                "function": "route_model_output",
                "details": {
                    "next_node": "tools",
                    "tool_calls": [
                        {"name": tc.get("name", "unknown")}
                        for tc in last_message.tool_calls[:3]
                    ],
                },
            },
        )
        return "tools"

    return route_model_output
