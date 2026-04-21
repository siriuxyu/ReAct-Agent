"""Thin adapter over LangGraph streaming output."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from langchain_core.messages import AIMessage, ToolMessage


def _collect_chunk_messages(chunk: Dict[str, Any]) -> List[Any]:
    """Normalize different LangGraph chunk shapes into one message list."""
    if "messages" in chunk:
        return list(chunk["messages"])
    for key in ("call_model", "tools", "collect_tool_outputs", "confirmation_cancelled"):
        node_output = chunk.get(key)
        if isinstance(node_output, dict) and "messages" in node_output:
            return list(node_output["messages"])
    return []


def _stringify_message_content(content: Any) -> str:
    """Convert model content blocks into plain text for API responses."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    content_parts: List[str] = []
    for item in content:
        if isinstance(item, str):
            content_parts.append(item)
            continue
        if not isinstance(item, dict):
            content_parts.append(str(item))
            continue
        if item.get("type") in {"web_search_tool_result", "server_tool_use"}:
            continue
        if "encrypted_content" in item:
            continue
        if item.get("type") == "text" and "text" in item:
            content_parts.append(item["text"])
            continue
        nested_content = item.get("content")
        if isinstance(nested_content, list):
            for nested_item in nested_content:
                if isinstance(nested_item, dict) and "encrypted_content" in nested_item:
                    continue
                if isinstance(nested_item, dict) and nested_item.get("type") == "text" and "text" in nested_item:
                    content_parts.append(nested_item["text"])
                elif isinstance(nested_item, str):
                    content_parts.append(nested_item)
            continue
        if nested_content is not None:
            content_parts.append(str(nested_content))
    return "".join(content_parts)


def collect_response_messages(chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert one LangGraph chunk into API-facing assistant messages."""
    response_messages: List[Dict[str, Any]] = []
    for msg in _collect_chunk_messages(chunk):
        if isinstance(msg, ToolMessage):
            continue
        msg_role = "assistant"
        if hasattr(msg, "type"):
            msg_role = msg.type if msg.type in {"user", "assistant", "system"} else "assistant"
        elif hasattr(msg, "role"):
            msg_role = msg.role
        response_messages.append(
            {
                "role": msg_role,
                "content": _stringify_message_content(getattr(msg, "content", str(msg))),
                "tool_calls": getattr(msg, "tool_calls", None),
                "_message_obj": msg,
            }
        )
    return response_messages


def extract_text_from_chunk(chunk: Dict[str, Any]) -> str | None:
    """Pull the final assistant text out of a LangGraph chunk, if any."""
    for item in collect_response_messages(chunk):
        msg = item["_message_obj"]
        if isinstance(msg, AIMessage) and item["content"] and not getattr(msg, "tool_calls", None):
            return item["content"]
    return None


async def invoke_graph(
    app: Any,
    *,
    messages: List[Dict[str, Any]],
    config: Dict[str, Any],
    context: Any,
) -> Tuple[List[Dict[str, Any]], str, int]:
    """Run a LangGraph app and collect response messages plus the final answer."""
    response_messages: List[Dict[str, Any]] = []
    final_response = ""
    chunk_count = 0

    async for chunk in app.astream({"messages": messages}, config=config, context=context):
        chunk_count += 1
        for item in collect_response_messages(chunk):
            msg = item.pop("_message_obj", None)
            response_messages.append(item)
            if isinstance(msg, AIMessage) and item["content"] and not getattr(msg, "tool_calls", None):
                final_response = item["content"]
    return response_messages, final_response, chunk_count
