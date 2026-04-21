"""Helpers for API-layer request preparation and transcript persistence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Sequence, Union

from fastapi import HTTPException

from agent.memory import build_runtime_recall_context


@dataclass(frozen=True)
class PreparedAgentRun:
    """Prepared runtime inputs for one API request."""

    config: dict[str, Any]
    context: Any
    messages: list[dict[str, Any]]
    effective_session_id: str


def content_preview(content: Union[str, List[Any]], max_chars: int = 100) -> str:
    """Return a short text preview of a message content payload."""
    if isinstance(content, str):
        return content[:max_chars]
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block[:max_chars])
        elif isinstance(block, dict):
            btype = block.get("type", "")
            if btype == "text":
                parts.append(block.get("text", "")[:max_chars])
            elif btype == "image":
                src = block.get("source", {})
                parts.append(f"[{src.get('media_type', 'image')}]")
            elif btype == "document":
                src = block.get("source", {})
                parts.append(f"[{src.get('media_type', 'application/pdf')}]")
            else:
                parts.append(f"[{btype}]")
    return " ".join(parts)[:max_chars]


def latest_request_text(messages: Sequence[Any]) -> str:
    """Extract the latest user-facing text from request messages."""
    for message in reversed(messages):
        preview = content_preview(getattr(message, "content", ""), max_chars=400).strip()
        if preview:
            return preview
    return ""


def persist_transcript(
    *,
    store: Any,
    user_id: str,
    session_id: str,
    request_messages: Sequence[Any],
    final_response: str,
) -> None:
    """Persist request/response text to the session transcript store."""
    for message in request_messages:
        preview = content_preview(getattr(message, "content", ""), max_chars=2000).strip()
        if not preview:
            continue
        store.add_message(
            user_id=user_id,
            session_id=session_id,
            role=getattr(message, "role", "user"),
            content=preview,
            metadata={"source": "api_request"},
        )
    if final_response.strip():
        store.add_message(
            user_id=user_id,
            session_id=session_id,
            role="assistant",
            content=final_response.strip(),
            metadata={"source": "agent_response"},
        )


async def prepare_agent_run(
    req: Any,
    *,
    context_cls: type,
    logger: Any,
    resolve_session_id: Callable[[str], str],
    is_session_owned_by_user: Callable[[str, str], bool],
    session_config: Callable[[str], dict[str, Any]],
    get_session_state: Callable[[str], Any],
    has_session_messages: Callable[[Any], bool],
    is_memory_enabled_fn: Callable[[], bool],
    get_memory_manager_fn: Callable[[], Any],
    build_session_recall_block_fn: Callable[..., str],
) -> PreparedAgentRun:
    """Resolve session, inject memory/recall context, and build Context."""
    session_id = req.session_id or req.userid
    if not is_session_owned_by_user(req.userid, session_id):
        raise HTTPException(status_code=400, detail="session_id must match userid namespace")

    effective_session_id = resolve_session_id(session_id)
    config = session_config(effective_session_id)
    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    existing_state = get_session_state(effective_session_id)
    is_new_session = not has_session_messages(existing_state)

    base_prompt = req.system_prompt or "You are a helpful AI assistant."
    recall_query = latest_request_text(req.messages)
    from agent.runtime.router import classify_task_type

    recall_context = await build_runtime_recall_context(
        user_id=req.userid,
        query=recall_query,
        exclude_session_id=effective_session_id,
        include_preferences=is_new_session,
        task_type=classify_task_type(recall_query),
        logger=logger,
        is_memory_enabled_fn=is_memory_enabled_fn,
        get_memory_manager_fn=get_memory_manager_fn,
        build_session_recall_block_fn=build_session_recall_block_fn,
    )
    if recall_context:
        base_prompt += "\n\n" + recall_context

    if req.enable_preference_extraction is not None:
        run_extraction = req.enable_preference_extraction
    else:
        existing_msgs = (
            existing_state.values.get("messages", [])
            if existing_state and getattr(existing_state, "values", None) else []
        )
        from langchain_core.messages import HumanMessage

        human_turns = sum(1 for message in existing_msgs if isinstance(message, HumanMessage))
        run_extraction = ((human_turns + 1) % 10 == 0)

    context_kwargs = dict(
        system_prompt=base_prompt,
        model=req.model or "anthropic/claude-sonnet-4-5-20250929",
        max_search_results=req.max_search_results or 10,
        user_id=req.userid,
        enable_preference_extraction=run_extraction,
    )
    if req.enable_web_search is not None:
        context_kwargs["enable_web_search"] = req.enable_web_search
    context = context_cls(**context_kwargs)

    return PreparedAgentRun(
        config=config,
        context=context,
        messages=messages,
        effective_session_id=effective_session_id,
    )
