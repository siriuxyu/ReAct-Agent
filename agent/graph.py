"""LangGraph backend wiring for the shared agent runtime."""

from __future__ import annotations

import os
import sys
from datetime import timezone

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph

from .context import Context
from .preference import extract_preferences
from .runtime import (
    AgentRuntime,
    collect_tool_outputs_node,
    confirmation_cancelled_node,
    create_call_model_node,
    create_model_output_route,
    create_planner_node,
    create_tools_node,
    route_after_planner,
)
from .state import InputState, State
from .utils import get_logger
from tools import get_tool_registry

load_dotenv()

_tracing_enabled = os.environ.get("LANGCHAIN_TRACING_V2", "").lower() in ("true", "1")

if sys.version_info >= (3, 11):
    from datetime import UTC
else:
    UTC = timezone.utc


def is_memory_enabled():
    try:
        from .memory.memory_manager import is_memory_enabled as _is_memory_enabled

        return _is_memory_enabled()
    except Exception:
        return False


def is_storage_available():
    try:
        from .memory.memory_manager import is_storage_available as _is_storage_available

        return _is_storage_available()
    except Exception:
        return False


def get_memory_manager():
    try:
        from .memory.memory_manager import get_memory_manager as _get_memory_manager

        return _get_memory_manager()
    except Exception:
        return None


RedisSaver = None
redis = None

logger = get_logger(__name__)
runtime_facade = AgentRuntime()
tool_registry = get_tool_registry()
logger.info(
    "LangSmith tracing %s",
    f"enabled (project={os.environ.get('LANGCHAIN_PROJECT', 'default')})"
    if _tracing_enabled
    else "disabled (set LANGCHAIN_TRACING_V2=true to enable)",
)

call_model = create_call_model_node(
    runtime_facade=runtime_facade,
    tool_registry=tool_registry,
    logger=logger,
    is_memory_enabled_fn=is_memory_enabled,
    is_storage_available_fn=is_storage_available,
    get_memory_manager_fn=get_memory_manager,
)
plan_next_step = create_planner_node(runtime_facade=runtime_facade, logger=logger)
execute_tools = create_tools_node(
    tool_registry=tool_registry,
    logger=logger,
    is_memory_enabled_fn=is_memory_enabled,
    get_memory_manager_fn=get_memory_manager,
)
collect_tool_outputs = collect_tool_outputs_node
respond_to_cancelled_confirmation = confirmation_cancelled_node
route_model_output = create_model_output_route(logger=logger)

builder = StateGraph(State, input_schema=InputState, context_schema=Context)
builder.add_node("planner", plan_next_step)
builder.add_node(call_model)
builder.add_node("tools", execute_tools)
builder.add_node("collect_tool_outputs", collect_tool_outputs)
builder.add_node("confirmation_cancelled", respond_to_cancelled_confirmation)
builder.add_node("extract_preferences", extract_preferences)

builder.add_edge("__start__", "planner")
builder.add_conditional_edges(
    "planner",
    route_after_planner,
    {
        "call_model": "call_model",
        "tools": "tools",
        "confirmation_cancelled": "confirmation_cancelled",
    },
)
builder.add_conditional_edges(
    "call_model",
    route_model_output,
    {"tools": "tools", "extract_preferences": "extract_preferences", "__end__": "__end__"},
)
builder.add_edge("tools", "collect_tool_outputs")
builder.add_edge("collect_tool_outputs", "call_model")
builder.add_edge("confirmation_cancelled", "__end__")
builder.add_edge("extract_preferences", "__end__")

_redis_url = os.environ.get("REDIS_URL")
if _redis_url:
    try:
        from langgraph.checkpoint.redis import RedisSaver

        checkpointer = RedisSaver.from_conn_string(_redis_url)
        checkpointer.setup()
        logger.info("Using Redis checkpointer", extra={"redis_url": _redis_url})
    except Exception as exc:
        logger.warning(f"Redis checkpointer unavailable ({exc}), falling back to MemorySaver")
        checkpointer = MemorySaver()
else:
    logger.info("Using in-memory checkpointer (set REDIS_URL to persist sessions)")
    checkpointer = MemorySaver()

try:
    memory_manager = get_memory_manager()
    logger.info(
        "Memory configuration",
        extra={
            "memory_enabled": is_memory_enabled(),
            "storage_available": is_storage_available(),
            "has_persistent_storage": (
                memory_manager.has_persistent_storage if memory_manager else False
            ),
        },
    )
except Exception as exc:
    logger.warning(f"Could not initialize memory manager: {exc}")

logger.info("Compiling graph with checkpointer")
graph = builder.compile(
    name="Cliriux Agent",
    checkpointer=checkpointer,
)
