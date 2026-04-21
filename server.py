from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import AsyncIterator, List, Dict, Any, Optional, Union
import asyncio
import json
import os
import uuid
import time
from agent.adapters import (
    RECURSION_ERROR_MESSAGE,
    extract_text_from_chunk,
    is_graph_recursion_error,
    invoke_graph,
    persist_transcript,
    prepare_agent_run,
)
from agent.graph import graph
from agent.context import Context
from agent.runtime import AgentRuntime, AgentRuntimeService, SessionService
from agent.utils import setup_logging, get_logger
from agent.memory.memory_manager import (
    get_memory_manager,
    is_memory_enabled,
    is_storage_available,
)
from agent.memory import build_session_recall_block, get_session_store
import auth.google_oauth as _google_oauth
import services.scheduler as _scheduler_svc

# Import LangGraph errors for proper handling
try:
    from langgraph.errors import GraphRecursionError
except ImportError:
    GraphRecursionError = None  # type: ignore

# Initialize logging
setup_logging()
logger = get_logger(__name__)
agent_runtime = AgentRuntime()
from agent.preference import force_extract_and_persist
from dotenv import load_dotenv

load_dotenv()

SESSION_TIMEOUT_SECONDS = int(os.environ.get("SESSION_TIMEOUT_SECONDS", 1800))  # 30 min
SESSION_SWEEP_INTERVAL  = int(os.environ.get("SESSION_SWEEP_INTERVAL",  300))   # 5 min

session_service = SessionService(
    app=graph,
    logger=logger,
    session_timeout_seconds=SESSION_TIMEOUT_SECONDS,
    session_sweep_interval=SESSION_SWEEP_INTERVAL,
    is_memory_enabled_fn=is_memory_enabled,
    get_memory_manager_fn=get_memory_manager,
    force_extract_preferences_fn=force_extract_and_persist,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(session_service.session_sweep_loop())
    await _scheduler_svc.start()
    try:
        yield
    finally:
        await _scheduler_svc.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


api = FastAPI(
    title="Cliriux Agent API",
    description="Cliriux personal assistant with long-term memory support",
    version="2.0.0",
    lifespan=lifespan,
)


########################################################
# Define the request body format
########################################################
# ContentBlock mirrors the Anthropic multimodal content-block schema.
# For images:   {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "<b64>"}}
# For PDFs:     {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "<b64>"}}
# For text:     {"type": "text", "text": "..."}
ContentBlock = Dict[str, Any]

class Message(BaseModel):
    role: str
    # content can be plain text OR a multimodal list of content blocks
    content: Union[str, List[ContentBlock]]

class CliriuxRequest(BaseModel):
    messages: List[Message]
    userid: str        # Stable user identity — scopes long-term memory (ChromaDB)
    session_id: Optional[str] = None  # Optional: explicit short-term session; falls back to userid
    system_prompt: Optional[str] = None
    model: Optional[str] = None
    max_search_results: Optional[int] = None
    enable_web_search: Optional[bool] = None
    enable_preference_extraction: Optional[bool] = None


class SessionCreateRequest(BaseModel):
    userid: str
    session_id: Optional[str] = None  # Optional: reuse/verify an existing session id

class SessionResponse(BaseModel):
    session_id: str
    userid: str
    is_new: bool  # False if session_id already existed (resume)

class CliriuxResponse(BaseModel):
    messages: List[Dict[str, Any]]
    final_response: str


class ChatRequest(BaseModel):
    message: str


class MemorySearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 10


class MemoryStoreRequest(BaseModel):
    key: str
    content: str
    document_type: Optional[str] = None  # e.g. "extracted_fact", "user_preference"


class InjectRequest(BaseModel):
    content: str                          # Raw text to inject (conversation, notes, docs, etc.)
    source: Optional[str] = None          # Free-form label, e.g. "chat_history", "notes"
    session_date: Optional[str] = None    # ISO date string, annotated in chunk headers
    chunk_size: int = 1500                # Max chars per chunk
    extract_facts: bool = True            # Extract EXTRACTED_FACT via LLM
    extract_preferences: bool = False     # Extract USER_PREFERENCE via LLM (off by default for raw docs)
    model: str = "anthropic/claude-haiku-4-5-20251001"


class InjectResponse(BaseModel):
    chunks_stored: int
    facts_extracted: int
    preferences_extracted: int
    total_chars: int


def _build_session_id(userid: str) -> str:
    return session_service.build_session_id(userid)


def _is_session_owned_by_user(userid: str, session_id: str) -> bool:
    return session_service.is_session_owned_by_user(userid, session_id)


def _session_config(session_id: str) -> Dict[str, Dict[str, str]]:
    return session_service.session_config(session_id)


def _get_session_state(session_id: str) -> Any:
    return session_service.get_state(session_id)


def _session_exists(session_id: str) -> bool:
    return session_service.session_exists(session_id)


runtime_service = AgentRuntimeService(
    agent_runtime=agent_runtime,
    app=graph,
    context_cls=Context,
    logger=logger,
    prepare_agent_run_fn=prepare_agent_run,
    invoke_graph_fn=invoke_graph,
    extract_text_fn=extract_text_from_chunk,
    persist_transcript_fn=persist_transcript,
    transcript_store_factory=get_session_store,
    resolve_session_id=session_service.resolve_session_id,
    is_session_owned_by_user=session_service.is_session_owned_by_user,
    session_config=session_service.session_config,
    get_session_state=session_service.get_state,
    has_session_messages=session_service.has_session_messages,
    is_memory_enabled_fn=is_memory_enabled,
    get_memory_manager_fn=get_memory_manager,
    build_session_recall_block_fn=build_session_recall_block,
)


def _log_invoke_request(request_id: str, req: CliriuxRequest) -> None:
    logger.info(
        "Received invoke request",
        extra={
            "request_id": request_id,
            "user_id": req.userid,
            "details": {
                "message_count": len(req.messages),
                "model": req.model or "default",
                "memory_enabled": is_memory_enabled(),
            },
        },
    )


def _log_turn_success(
    *,
    event_name: str,
    request_id: str,
    duration_ms: float,
    turn,
    user_id: str | None = None,
) -> None:
    extra: dict[str, Any] = {
        "request_id": request_id,
        "duration_ms": round(duration_ms, 2),
        "details": {
            "chunks_processed": turn.chunk_count,
            "response_length": len(turn.final_response),
            "response_preview": turn.final_response[:200] if turn.final_response else None,
        },
    }
    if user_id:
        extra["user_id"] = user_id
        extra["details"]["response_messages_count"] = len(turn.response_messages)
    logger.info(event_name, extra=extra)


def _log_runtime_failure(
    *,
    event_name: str,
    request_id: str,
    duration_ms: float,
    exc: Exception,
    user_id: str | None = None,
) -> None:
    extra: dict[str, Any] = {
        "request_id": request_id,
        "duration_ms": round(duration_ms, 2),
    }
    if user_id:
        extra["user_id"] = user_id
    logger.error(event_name, extra=extra, exc_info=True)


def _invoke_error_response(
    *,
    request_id: str,
    user_id: str,
    duration_ms: float,
    exc: Exception,
) -> CliriuxResponse:
    if is_graph_recursion_error(exc, GraphRecursionError):
        logger.warning(
            "Agent hit recursion limit",
            extra={
                "request_id": request_id,
                "user_id": user_id,
                "duration_ms": round(duration_ms, 2),
                "error_type": "GraphRecursionError",
            },
        )
        return CliriuxResponse(messages=[], final_response=RECURSION_ERROR_MESSAGE)

    _log_runtime_failure(
        event_name="Request failed",
        request_id=request_id,
        user_id=user_id,
        duration_ms=duration_ms,
        exc=exc,
    )
    return CliriuxResponse(messages=[], final_response=f"Error: {str(exc)}")


def _touch_session(session_id: str, userid: str) -> None:
    session_service.touch_session(session_id, userid)


########################################################
# Google OAuth endpoints
########################################################
@api.get("/auth/google")
async def auth_google_start():
    """引导用户授权 Google Calendar。"""
    url, _state = _google_oauth.get_auth_url()
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url=url, status_code=307)


@api.get("/auth/google/callback")
async def auth_google_callback(code: str, state: str):
    """OAuth 回调，交换授权码为 token。"""
    try:
        _google_oauth.exchange_code(code, state)
        return {"message": "Authorization successful. You can close this tab."}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


########################################################
# Session management
########################################################
@api.post("/sessions", response_model=SessionResponse, tags=["Agent"])
async def create_session(req: SessionCreateRequest):
    """Create a new session or verify an existing one.

    Call this before /invoke to obtain a session_id.
    Pass the returned session_id to all subsequent /invoke calls to maintain
    conversation history. Pass an existing session_id to resume that session.
    Omit session_id to always get a fresh session.
    """
    try:
        effective_session_id, is_new = session_service.create_or_resume_session(
            req.userid,
            req.session_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    logger.info(
        "Session created",
        extra={
            "userid": req.userid,
            "session_id": effective_session_id,
            "is_new": is_new,
        },
    )
    return SessionResponse(
        session_id=effective_session_id,
        userid=req.userid,
        is_new=is_new,
    )


########################################################
# Shared helpers for invoke & stream
########################################################
async def _sse_generator(
    req: CliriuxRequest, request_id: str
) -> AsyncIterator[str]:
    """
    Yield SSE-formatted events:
      data: {"type": "session",  "session_id": "..."}
      data: {"type": "chunk",    "content": "..."}
      data: {"type": "done",     "final_response": "..."}
      data: {"type": "error",    "message": "..."}
    """
    def event(payload: Dict) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    try:
        prepared = await runtime_service.prepare_request(req)
        yield event({"type": "session", "session_id": prepared.effective_session_id})

        queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()

        async def on_text(text: str) -> None:
            await queue.put(("chunk", text))

        async def run_stream_task() -> None:
            try:
                result = await runtime_service.stream_prepared_request(
                    req,
                    prepared=prepared,
                    on_text_fn=on_text,
                )
                await queue.put(("done", result.turn.final_response))
            except Exception as exc:
                await queue.put(("error", str(exc)))

        task = asyncio.create_task(run_stream_task())
        try:
            while True:
                kind, payload = await queue.get()
                if kind == "chunk":
                    yield event({"type": "chunk", "content": payload})
                    continue
                if kind == "done":
                    _touch_session(prepared.effective_session_id, req.userid)
                    yield event({"type": "done", "final_response": payload})
                    break
                yield event({"type": "error", "message": payload})
                break
        finally:
            await task

    except Exception as e:
        logger.error(f"SSE error [{request_id}]: {e}", exc_info=True)
        yield event({"type": "error", "message": str(e)})


########################################################
# Stream endpoint (SSE)
########################################################
@api.post("/stream")
async def stream(req: CliriuxRequest):
    """
    Stream agent responses as Server-Sent Events.

    Event types:
      session  — first event, carries session_id (save this for resume)
      chunk    — incremental assistant text
      done     — final complete response
      error    — something went wrong

    Example (curl):
      curl -X POST http://localhost:8000/stream \\
           -H 'Content-Type: application/json' \\
           -d '{"userid":"alice","messages":[{"role":"user","content":"hi"}]}'
    """
    request_id = uuid.uuid4().hex
    return StreamingResponse(
        _sse_generator(req, request_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering
        },
    )


########################################################
# Invoke Cliriux with user context
########################################################
@api.post("/invoke", response_model=CliriuxResponse, tags=["Agent"])
async def invoke(req: CliriuxRequest):
    """Invoke Cliriux with the provided messages and context.

    userid scopes long-term memory.
    session_id scopes short-term conversation state; if omitted, userid is used
    as a backwards-compatible single-session fallback.
    """
    request_id = uuid.uuid4().hex
    start_time = time.time()
    
    try:
        _log_invoke_request(request_id, req)

        logger.info("Starting agent execution", extra={
            'request_id': request_id,
            'user_id': req.userid,
            'function': 'AgentRuntimeService.invoke_request'
        })
        result = await runtime_service.invoke_request(req)
        
        duration_ms = (time.time() - start_time) * 1000
        _log_turn_success(
            event_name="Request completed successfully",
            request_id=request_id,
            user_id=req.userid,
            duration_ms=duration_ms,
            turn=result.turn,
        )
        
        _touch_session(result.prepared.effective_session_id, req.userid)
        return CliriuxResponse(
            messages=result.turn.response_messages,
            final_response=result.turn.final_response
        )
        
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        return _invoke_error_response(
            request_id=request_id,
            user_id=req.userid,
            duration_ms=duration_ms,
            exc=e,
        )


########################################################
# Simple chat endpoint (no context history)
########################################################
@api.post("/chat", tags=["Agent"])
async def chat(req: ChatRequest):
    """Simple chat endpoint for single message interactions without context history."""
    request_id = uuid.uuid4().hex
    start_time = time.time()
    
    try:
        logger.info(
            "Received chat request",
            extra={
                "request_id": request_id,
                "details": {
                    "message_length": len(req.message),
                    "message_preview": req.message[:100],
                },
            },
        )
        
        temp_thread_id = f"temp_{uuid.uuid4().hex}"

        logger.debug("Starting chat execution", extra={
            'request_id': request_id,
            'details': {'temp_thread_id': temp_thread_id}
        })

        turn = await runtime_service.invoke_chat_request(
            message=req.message,
            user_id=temp_thread_id,
            system_prompt="You are a helpful AI assistant.",
            model="anthropic/claude-sonnet-4-5-20250929",
        )
        
        duration_ms = (time.time() - start_time) * 1000
        _log_turn_success(
            event_name="Chat request completed",
            request_id=request_id,
            duration_ms=duration_ms,
            turn=turn,
        )
        
        return {"response": turn.final_response}
        
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        _log_runtime_failure(
            event_name="Chat request failed",
            request_id=request_id,
            duration_ms=duration_ms,
            exc=e,
        )
        return {"response": f"Error: {str(e)}"}


########################################################
# Check state endpoint
########################################################
@api.get("/state/session/{session_id}", tags=["Session"])
async def check_session_state(session_id: str):
    """Check the conversation state for a specific session."""
    try:
        effective_session_id = session_service.resolve_session_id(session_id)
        state = session_service.get_state(effective_session_id)
        
        if state:
            return {
                "session_id": session_id,
                "effective_session_id": effective_session_id,
                "state": state.values,
                "next_node": state.next,
                "config": state.config
            }
        else:
            return {"error": f"Session {effective_session_id} not found"}
    except Exception as e:
        return {"error": str(e)}


@api.get("/state/{userid}", tags=["Session"], deprecated=True)
async def check_state(userid: str):
    """Legacy alias for session state, using userid as the session id."""
    return await check_session_state(userid)


########################################################
# Reset short-term session (clear conversation history)
########################################################
@api.post("/reset/session/{session_id}", tags=["Session"])
async def reset_short_term_session(
    session_id: str,
    userid: Optional[str] = None,
    preserve_memory: bool = True,
    model: Optional[str] = None,
):
    """
    Reset the short-term conversation history for a session.
    
    Args:
        session_id: The session identifier
        userid: Optional user identifier for clearing long-term memory
        preserve_memory: If True (default), keep long-term memories intact.
                        If False, also clear long-term memories.
    
    This clears the checkpoint state but optionally preserves long-term memories.
    """
    try:
        return await session_service.reset_session(
            session_id=session_id,
            userid=userid,
            preserve_memory=preserve_memory,
            model=model,
        )
    except Exception as e:
        logger.error(f"Failed to reset session for {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/reset/{userid}", tags=["Session"], deprecated=True)
async def reset_user_session(userid: str, preserve_memory: bool = True, model: Optional[str] = None):
    """Legacy alias for resetting the userid-scoped default session."""
    return await reset_short_term_session(
        session_id=userid,
        userid=userid,
        preserve_memory=preserve_memory,
        model=model,
    )


########################################################
# Memory injection endpoint
########################################################
@api.post("/memory/{userid}/inject", response_model=InjectResponse, tags=["Memory"])
async def inject_memory(userid: str, req: InjectRequest):
    """
    Inject raw text into a user's long-term memory.

    Chunks the content, stores each chunk as LONG_TERM_CONTEXT, then optionally
    runs LLM-based fact and/or preference extraction on the full text.

    Useful for bulk-loading conversation history, documents, or notes without
    going through the agent conversation flow.
    """
    if not is_memory_enabled():
        raise HTTPException(status_code=503, detail="Memory is not enabled")

    from agent.interfaces import StorageType
    from agent.memory.memory_manager import get_memory_manager

    manager = get_memory_manager()
    content = req.content.strip()
    if not content:
        raise HTTPException(status_code=400, detail="content must not be empty")

    # ── 1. Chunk and store as LONG_TERM_CONTEXT ──────────────────────────────
    header = ""
    if req.session_date:
        header = f"[{req.session_date}]"
    if req.source:
        header = f"{header} [{req.source}]".strip() if header else f"[{req.source}]"

    chunks = _split_into_chunks(content, req.chunk_size)
    chunks_stored = 0
    for idx, chunk in enumerate(chunks):
        chunk_content = f"{header}\n{chunk}".strip() if header else chunk
        key = f"inject_{userid}_{uuid.uuid4().hex[:12]}_c{idx}"
        ok = await manager.store_user_memory(
            user_id=userid,
            key=key,
            content=chunk_content,
            metadata={"source": req.source or "inject", "chunk_index": idx},
            document_type=StorageType.LONG_TERM_CONTEXT,
        )
        if ok:
            chunks_stored += 1

    # ── 2. Extract facts from the full content ────────────────────────────────
    facts_extracted = 0
    if req.extract_facts and chunks_stored > 0:
        try:
            from agent.extraction.fact_extraction import extract_session_observations
            facts_text = await extract_session_observations(content, req.model)
            if facts_text:
                facts_content = f"{header}\n[Facts]\n{facts_text}".strip() if header else f"[Facts]\n{facts_text}"
                key = f"inject_{userid}_{uuid.uuid4().hex[:12]}_facts"
                ok = await manager.store_user_memory(
                    user_id=userid,
                    key=key,
                    content=facts_content,
                    metadata={"source": req.source or "inject", "type": "extracted_fact"},
                    document_type=StorageType.EXTRACTED_FACT,
                )
                if ok:
                    facts_extracted = len(facts_text.splitlines())
        except Exception as e:
            logger.warning(f"Fact extraction failed during inject for {userid}: {e}")

    # ── 3. Extract preferences (optional, off by default) ────────────────────
    preferences_extracted = 0
    if req.extract_preferences and chunks_stored > 0:
        try:
            from langchain_core.messages import HumanMessage as _HM
            from agent.preference import force_extract_and_persist
            preferences_extracted = await force_extract_and_persist(
                messages=[_HM(content=content)],
                user_id=userid,
                model=req.model,
            )
        except Exception as e:
            logger.warning(f"Preference extraction failed during inject for {userid}: {e}")

    logger.info(
        f"Inject complete for {userid}",
        extra={
            "userid": userid,
            "chunks_stored": chunks_stored,
            "facts_extracted": facts_extracted,
            "preferences_extracted": preferences_extracted,
        },
    )
    return InjectResponse(
        chunks_stored=chunks_stored,
        facts_extracted=facts_extracted,
        preferences_extracted=preferences_extracted,
        total_chars=len(content),
    )


def _split_into_chunks(text: str, chunk_size: int) -> List[str]:
    """Split text into chunks of at most chunk_size chars, preferring newlines."""
    chunks, current, current_len = [], [], 0
    for line in text.splitlines(keepends=True):
        remaining = line
        while remaining:
            space_left = chunk_size - current_len
            if space_left <= 0 and current:
                chunks.append("".join(current))
                current, current_len = [], 0
                space_left = chunk_size

            if len(remaining) <= space_left:
                current.append(remaining)
                current_len += len(remaining)
                remaining = ""
            else:
                if current:
                    chunks.append("".join(current))
                    current, current_len = [], 0
                    space_left = chunk_size
                current.append(remaining[:space_left])
                chunks.append("".join(current))
                current, current_len = [], 0
                remaining = remaining[space_left:]
    if current:
        chunks.append("".join(current))
    return chunks


########################################################
# Memory Management Endpoints
########################################################
@api.get("/memory/{userid}", tags=["Memory"])
async def list_memories(userid: str, limit: int = 100):
    """
    List all memories for a specific user.
    
    Args:
        userid: The user identifier
        limit: Maximum number of memories to return (default: 100)
    """
    if not is_memory_enabled():
        raise HTTPException(status_code=503, detail="Memory is not enabled")
    
    manager = get_memory_manager()
    memories = await manager.list_user_memories(userid, limit=limit)
    
    return {
        "userid": userid,
        "count": len(memories),
        "memories": memories,
    }


@api.post("/memory/{userid}/search", tags=["Memory"])
async def search_memories(userid: str, req: MemorySearchRequest):
    """
    Search memories for a specific user.
    
    Args:
        userid: The user identifier
        req: Search request with query and limit
    """
    if not is_memory_enabled():
        raise HTTPException(status_code=503, detail="Memory is not enabled")
    
    manager = get_memory_manager()
    results = await manager.search_user_memories(
        userid,
        query=req.query,
        limit=req.limit or 10,
    )
    
    return {
        "userid": userid,
        "query": req.query,
        "count": len(results),
        "results": results,
    }


@api.post("/memory/{userid}/store", tags=["Memory"])
async def store_memory(userid: str, req: MemoryStoreRequest):
    """
    Store a memory for a specific user.
    
    Args:
        userid: The user identifier
        req: Store request with key and content
    """
    if not is_memory_enabled():
        raise HTTPException(status_code=503, detail="Memory is not enabled")
    
    from agent.interfaces import StorageType
    manager = get_memory_manager()
    doc_type = StorageType.LONG_TERM_CONTEXT
    if req.document_type:
        try:
            doc_type = StorageType(req.document_type)
        except ValueError:
            pass
    success = await manager.store_user_memory(
        userid,
        key=req.key,
        content=req.content,
        document_type=doc_type,
    )
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to store memory")
    
    return {
        "status": "success",
        "userid": userid,
        "key": req.key,
    }


@api.delete("/memory/{userid}/{key}", tags=["Memory"])
async def delete_memory(userid: str, key: str):
    """
    Delete a specific memory for a user.
    
    Args:
        userid: The user identifier
        key: Memory key to delete
    """
    if not is_memory_enabled():
        raise HTTPException(status_code=503, detail="Memory is not enabled")
    
    manager = get_memory_manager()
    success = await manager.delete_user_memory(userid, key)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete memory")
    
    return {
        "status": "success",
        "userid": userid,
        "key": key,
    }


@api.delete("/memory/{userid}", tags=["Memory"])
async def clear_memories(userid: str):
    """
    Clear all memories for a specific user.
    
    Args:
        userid: The user identifier
    """
    if not is_memory_enabled():
        raise HTTPException(status_code=503, detail="Memory is not enabled")
    
    manager = get_memory_manager()
    success = await manager.clear_user_memories(userid)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to clear memories")
    
    return {
        "status": "success",
        "userid": userid,
        "message": "All memories cleared",
    }


########################################################
# Health check and status endpoints
########################################################
@api.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "memory_enabled": is_memory_enabled(),
    }


@api.get("/status", tags=["System"])
async def status():
    """Get system status including memory and storage configuration."""
    manager = get_memory_manager()
    
    return {
        "status": "running",
        "memory": {
            "enabled": is_memory_enabled(),
            "available": manager.is_available,
        },
        "storage": {
            "chromadb_available": is_storage_available(),
            "persistent": manager.has_persistent_storage,
        },
        "endpoints": {
            "sessions": "POST /sessions - Create a new session or verify an existing one",
            "invoke": "POST /invoke - Main agent endpoint",
            "chat": "POST /chat - Simple chat without history",
            "state_session": "GET /state/session/{session_id} - Check session state",
            "state_legacy": "GET /state/{userid} - Check legacy userid-scoped state",
            "reset_session": "POST /reset/session/{session_id} - Reset session state",
            "reset_legacy": "POST /reset/{userid} - Reset legacy userid-scoped session",
            "memory_list": "GET /memory/{userid} - List user memories",
            "memory_search": "POST /memory/{userid}/search - Search memories",
            "memory_store": "POST /memory/{userid}/store - Store memory",
            "memory_delete": "DELETE /memory/{userid}/{key} - Delete memory",
            "memory_clear": "DELETE /memory/{userid} - Clear all memories",
            "storage_stats": "GET /storage/stats - Get storage statistics",
        }
    }


@api.get("/storage/stats", tags=["System"])
async def storage_stats(userid: Optional[str] = None):
    """
    Get storage statistics.
    
    Args:
        userid: Optional user ID to filter stats
    """
    if not is_storage_available():
        raise HTTPException(status_code=503, detail="Storage backend not available")
    
    manager = get_memory_manager()
    stats = await manager.get_storage_stats(userid)
    
    return stats


########################################################
# Main entry point
########################################################
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Cliriux server on http://0.0.0.0:8000")
    logger.info(f"Memory enabled: {is_memory_enabled()}")
    uvicorn.run(api, host="0.0.0.0", port=8000)
