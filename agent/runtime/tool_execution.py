"""Tool execution helpers used by the runtime and LangGraph adapter."""

from __future__ import annotations

import json
from typing import Any, Callable

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode

from ..policy import build_confirmation_request, requires_confirmation


def resolve_runtime_tools(user_id: str | None, get_tools_fn: Callable[[str | None], list]) -> list:
    """Resolve the full tool list for a user-scoped runtime."""
    return list(get_tools_fn(user_id))


async def execute_tool_phase(
    *,
    messages: list[Any],
    all_tools: list,
    confirmed_tool_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    """Execute tools or return a pending-confirmation intercept payload."""
    if confirmed_tool_calls:
        synthetic_ai = AIMessage(content="", tool_calls=confirmed_tool_calls)
        tool_node = ToolNode(all_tools)
        state_dict = {"messages": [*messages, synthetic_ai]}
        result = await tool_node.ainvoke(state_dict)
        return {
            "messages": [synthetic_ai, *result.get("messages", [])],
            "confirmed_tool_calls": [],
            "pending_confirmation": None,
            "confirmation_resolution": "",
            "confirmation_response_text": "",
        }

    last_message = messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError("Tool execution expected the latest message to be an AIMessage.")

    if requires_confirmation(last_message.tool_calls):
        pending = build_confirmation_request(last_message.tool_calls)
        tool_messages = [
            ToolMessage(
                content=json.dumps(
                    {
                        "ok": False,
                        "tool": call.get("name", "unknown"),
                        "summary": "Confirmation required before executing this action.",
                        "data": {
                            "requires_confirmation": True,
                            "preview": pending["preview"],
                            "preview_payloads": pending.get("preview_payloads", []),
                            "highest_side_effect": pending.get("highest_side_effect"),
                        },
                    },
                    ensure_ascii=False,
                ),
                tool_call_id=call.get("id", ""),
                name=call.get("name"),
            )
            for call in last_message.tool_calls
        ]
        return {
            "messages": tool_messages,
            "pending_confirmation": pending,
            "confirmed_tool_calls": [],
        }

    tool_node = ToolNode(all_tools)
    return await tool_node.ainvoke({"messages": messages})
