from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
import os
import uuid
import time
from agent.graph import graph
from agent.context import Context
from agent.utils import setup_logging, get_logger
from agent.memory.langmem_adapter import (
    get_langmem_manager,
    is_langmem_enabled,
    is_storage_available,
)
from langchain_core.messages import ToolMessage, AIMessage

# Import LangGraph errors for proper handling
try:
    from langgraph.errors import GraphRecursionError
except ImportError:
    GraphRecursionError = None  # type: ignore

# Initialize logging
setup_logging()
logger = get_logger(__name__)

from dotenv import load_dotenv
load_dotenv()


# ---------------------------------------------------------------------------
# Session activity tracker
# Maps session_id → {"userid": str, "last_activity": float (epoch seconds)}
# ---------------------------------------------------------------------------
_active_sessions: Dict[str, Dict[str, Any]] = {}
_session_aliases: Dict[str, str] = {}
_finalized_sessions: set[str] = set()

SESSION_TIMEOUT_SECONDS = int(os.environ.get("SESSION_TIMEOUT_SECONDS", 1800))  # 30 min
SESSION_SWEEP_INTERVAL  = int(os.environ.get("SESSION_SWEEP_INTERVAL",  300))   # 5 min


def _touch_session(session_id: str, userid: str) -> None:
    """Record or refresh the last-activity timestamp for a session."""
    effective_session_id = _resolve_session_id(session_id)
    _active_sessions[effective_session_id] = {
        "userid": userid,
        "last_activity": time.time(),
    }


def _resolve_session_id(session_id: str) -> str:
    """Follow any reset/timeout aliases to the current effective session id."""
    current = session_id
    seen = set()
    while current in _session_aliases and current not in seen:
        seen.add(current)
        current = _session_aliases[current]
    return current


def _has_session_messages(state: Any) -> bool:
    return bool(state and state.values and state.values.get("messages"))


def _roll_session_forward(session_id: str, userid: str) -> str:
    """
    Rotate a session to a fresh underlying thread id.

    The original session_id remains usable at the API layer and is aliased to the
    new thread, so clients effectively see a cleared session.
    """
    effective_session_id = _resolve_session_id(session_id)
    replacement_session_id = _build_session_id(userid)

    alias_sources = [
        alias
        for alias in list(_session_aliases.keys())
        if _resolve_session_id(alias) == effective_session_id
    ]
    alias_sources.append(session_id)
    alias_sources.append(effective_session_id)
    for alias in set(alias_sources):
        if alias != replacement_session_id:
            _session_aliases[alias] = replacement_session_id

    _active_sessions.pop(session_id, None)
    _active_sessions.pop(effective_session_id, None)
    _finalized_sessions.discard(replacement_session_id)
    return replacement_session_id


async def _expire_session(session_id: str, userid: str) -> None:
    """Force-extract preferences then remove session from the activity tracker."""
    effective_session_id = _resolve_session_id(session_id)
    logger.info(
        f"Session timeout: extracting preferences for {effective_session_id}",
        extra={"session_id": effective_session_id, "userid": userid},
    )
    try:
        state = _get_session_state(effective_session_id)
        messages = state.values.get("messages", []) if state and state.values else []
        if (
            messages
            and is_langmem_enabled()
            and effective_session_id not in _finalized_sessions
        ):
            from agent.preference import force_extract_and_persist
            n = await force_extract_and_persist(
                messages=messages,
                user_id=userid,
                model="anthropic/claude-haiku-4-5-20251001",
            )
            _finalized_sessions.add(effective_session_id)
            logger.info(
                f"Timeout extraction: {n} preferences saved for {effective_session_id}"
            )
        replacement_session_id = _roll_session_forward(effective_session_id, userid)
        logger.info(
            "Session expired and rolled forward",
            extra={
                "session_id": effective_session_id,
                "replacement_session_id": replacement_session_id,
                "userid": userid,
            },
        )
    except Exception as e:
        logger.warning(f"Timeout extraction failed for {effective_session_id}: {e}")
    finally:
        _active_sessions.pop(session_id, None)
        _active_sessions.pop(effective_session_id, None)


async def _session_sweep_loop() -> None:
    """Background task: periodically expire idle sessions."""
    logger.info(
        f"Session sweep started (timeout={SESSION_TIMEOUT_SECONDS}s, "
        f"interval={SESSION_SWEEP_INTERVAL}s)"
    )
    while True:
        await asyncio.sleep(SESSION_SWEEP_INTERVAL)
        now = time.time()
        expired = [
            (sid, info["userid"])
            for sid, info in list(_active_sessions.items())
            if now - info["last_activity"] > SESSION_TIMEOUT_SECONDS
        ]
        if expired:
            logger.info(f"Sweeping {len(expired)} expired sessions")
            await asyncio.gather(*[_expire_session(sid, uid) for sid, uid in expired])


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_session_sweep_loop())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


api = FastAPI(
    title="CSE291-A Agent API",
    description="React Agent with LangMem long-term memory support",
    version="2.0.0",
    lifespan=lifespan,
)


########################################################
# Define the request body format
########################################################
class Message(BaseModel):
    role: str
    content: str

class ReactRequest(BaseModel):
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

class ReactResponse(BaseModel):
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
    return f"{userid}_{uuid.uuid4().hex[:12]}"


def _is_session_owned_by_user(userid: str, session_id: str) -> bool:
    return session_id == userid or session_id.startswith(f"{userid}_")


def _session_config(session_id: str) -> Dict[str, Dict[str, str]]:
    return {"configurable": {"thread_id": _resolve_session_id(session_id)}}


def _get_session_state(session_id: str) -> Any:
    return graph.get_state(_session_config(session_id))


def _session_exists(session_id: str) -> bool:
    return _has_session_messages(_get_session_state(session_id))


########################################################
# Session management
########################################################
@api.post("/sessions", response_model=SessionResponse)
async def create_session(req: SessionCreateRequest):
    """Create a new session or verify an existing one.

    Call this before /invoke to obtain a session_id.
    Pass the returned session_id to all subsequent /invoke calls to maintain
    conversation history. Pass an existing session_id to resume that session.
    Omit session_id to always get a fresh session.
    """
    session_id = req.session_id or _build_session_id(req.userid)
    if not _is_session_owned_by_user(req.userid, session_id):
        raise HTTPException(
            status_code=400,
            detail="session_id must match userid namespace",
        )

    effective_session_id = _resolve_session_id(session_id)
    is_new = not _session_exists(effective_session_id)
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
# Invoke the React Agent with user context
########################################################
@api.post("/invoke", response_model=ReactResponse)
async def invoke(req: ReactRequest):
    """Invoke the React agent with the provided messages and context.

    userid scopes long-term memory.
    session_id scopes short-term conversation state; if omitted, userid is used
    as a backwards-compatible single-session fallback.
    """
    request_id = uuid.uuid4().hex
    start_time = time.time()
    
    try:
        logger.info("Received invoke request", extra={
            'request_id': request_id,
            'user_id': req.userid,
            'details': {
                'message_count': len(req.messages),
                'model': req.model or "default",
                'has_system_prompt': req.system_prompt is not None,
                'langmem_enabled': is_langmem_enabled(),
                'first_message_preview': req.messages[0].content[:100] if req.messages else None
            }
        })
        
        # Convert messages to the format expected by the graph
        messages = [{"role": msg.role, "content": msg.content} for msg in req.messages]
        
        logger.debug("Processing messages", extra={
            'request_id': request_id,
            'user_id': req.userid,
            'details': {'messages': [{'role': m['role'], 'content': m['content'][:100]} for m in messages]}
        })
        
        # session_id scopes the short-term message history (LangGraph checkpoint).
        # userid scopes long-term memory (ChromaDB) and never changes.
        # Older clients may omit session_id; in that case, reuse userid as the
        # single session key to preserve the legacy behavior.
        session_id = req.session_id or req.userid
        if not _is_session_owned_by_user(req.userid, session_id):
            raise HTTPException(
                status_code=400,
                detail="session_id must match userid namespace",
            )
        effective_session_id = _resolve_session_id(session_id)
        config = _session_config(effective_session_id)

        # Check if this is a new session (no prior messages for this session_id)
        existing_state = _get_session_state(effective_session_id)
        is_new_session = not _has_session_messages(existing_state)

        # Build system prompt, injecting known preferences for new sessions
        base_prompt = req.system_prompt or "You are a helpful AI assistant."
        if is_new_session and is_langmem_enabled() and req.userid:
            try:
                from agent.interfaces import StorageType
                manager = get_langmem_manager()
                prefs = await manager.search_user_memory(
                    user_id=req.userid,
                    query="user preferences",
                    limit=10,
                    document_type=StorageType.USER_PREFERENCE,
                )
                if prefs:
                    pref_text = "\n".join(f"- {p['content']}" for p in prefs)
                    base_prompt += f"\n\n## Known user preferences\n{pref_text}"
                    logger.info(
                        f"Injected {len(prefs)} preferences into system prompt for new session",
                        extra={"request_id": request_id, "user_id": req.userid},
                    )
            except Exception as _e:
                logger.warning(f"Failed to inject preferences: {_e}")

        # Decide whether to run preference extraction this turn.
        # Client override takes precedence; otherwise extract every 10 human turns.
        if req.enable_preference_extraction is not None:
            run_extraction = req.enable_preference_extraction
        else:
            existing_msgs = (
                existing_state.values.get("messages", [])
                if existing_state and existing_state.values else []
            )
            from langchain_core.messages import HumanMessage as _HumanMessage
            human_turns = sum(1 for m in existing_msgs if isinstance(m, _HumanMessage))
            # This request adds turn N+1; extract when (N+1) is a multiple of 10
            run_extraction = ((human_turns + 1) % 10 == 0)

        # Create context with optional parameters
        context_kwargs = dict(
            system_prompt=base_prompt,
            model=req.model or "anthropic/claude-sonnet-4-5-20250929",
            max_search_results=req.max_search_results or 10,
            user_id=req.userid,
            enable_preference_extraction=run_extraction,
        )
        if req.enable_web_search is not None:
            context_kwargs["enable_web_search"] = req.enable_web_search
        context = Context(**context_kwargs)

        logger.debug("Context created", extra={
            'request_id': request_id,
            'user_id': req.userid,
            'session_id': session_id,
            'effective_session_id': effective_session_id,
            'is_new_session': is_new_session,
            'details': {
                'system_prompt': context.system_prompt[:200],
                'model': context.model,
            }
        })
        
        # Stream the response from the graph
        response_messages = []
        final_response = ""
        chunk_count = 0
        
        logger.info("Starting agent execution", extra={
            'request_id': request_id,
            'user_id': req.userid,
            'function': 'graph.astream'
        })
        
        async for chunk in graph.astream(
            {"messages": messages},
            config=config,
            context=context
        ):
            chunk_count += 1
            
            # Debug: log chunk structure
            logger.debug(f"Chunk received (count: {chunk_count})", extra={
                'request_id': request_id,
                'user_id': req.userid,
                'details': {'chunk': str(chunk)[:200]}  # Truncate for readability
            })
            
            # Handle different chunk structures
            msgs_to_process = []
            
            if "messages" in chunk:
                msgs_to_process = chunk["messages"]
            elif "call_model" in chunk and "messages" in chunk["call_model"]:
                msgs_to_process = chunk["call_model"]["messages"]
                logger.debug("Processing model call chunk", extra={
                    'request_id': request_id,
                    'user_id': req.userid,
                    'details': {'chunk_type': 'call_model'}
                })
            elif "tools" in chunk and "messages" in chunk["tools"]:
                msgs_to_process = chunk["tools"]["messages"]
                logger.debug("Processing tool execution chunk", extra={
                    'request_id': request_id,
                    'user_id': req.userid,
                    'details': {'chunk_type': 'tools'}
                })
            
            for msg in msgs_to_process:
                # Skip ToolMessage - these contain encrypted_content from web_search
                # and should not be returned to the user
                if isinstance(msg, ToolMessage):
                    logger.debug("Skipping ToolMessage", extra={
                        'request_id': request_id,
                        'user_id': req.userid,
                        'details': {
                            'message_type': 'ToolMessage',
                            'tool_call_id': getattr(msg, 'tool_call_id', None)
                        }
                    })
                    continue
                
                # Get message content based on message type
                msg_content_raw = ""
                if hasattr(msg, 'content'):
                    msg_content_raw = msg.content
                elif hasattr(msg, 'data') and 'content' in msg.data:
                    msg_content_raw = msg.data['content']
                else:
                    msg_content_raw = str(msg)
                
                # Convert content to string if it's a list or other non-string type
                msg_content = ""
                if isinstance(msg_content_raw, str):
                    msg_content = msg_content_raw
                elif isinstance(msg_content_raw, list):
                    # Handle list of content blocks (e.g., from Claude API)
                    content_parts = []
                    for item in msg_content_raw:
                        if isinstance(item, str):
                            content_parts.append(item)
                        elif isinstance(item, dict):
                            # Skip web_search_tool_result blocks that contain encrypted_content
                            if item.get('type') == 'web_search_tool_result':
                                logger.debug("Skipping web_search_tool_result block with encrypted_content", extra={
                                    'request_id': request_id,
                                    'user_id': req.userid
                                })
                                continue
                            # Skip blocks that contain encrypted_content
                            if 'encrypted_content' in item:
                                logger.debug("Skipping content block with encrypted_content", extra={
                                    'request_id': request_id,
                                    'user_id': req.userid,
                                    'block_type': item.get('type', 'unknown')
                                })
                                continue
                            # Skip server_tool_use blocks (tool call information)
                            if item.get('type') == 'server_tool_use':
                                logger.debug("Skipping server_tool_use block", extra={
                                    'request_id': request_id,
                                    'user_id': req.userid,
                                    'tool_name': item.get('name', 'unknown')
                                })
                                continue
                            # Handle structured content blocks
                            if item.get('type') == 'text' and 'text' in item:
                                content_parts.append(item['text'])
                            elif 'content' in item:
                                # Recursively check nested content for encrypted_content
                                nested_content = item['content']
                                if isinstance(nested_content, list):
                                    for nested_item in nested_content:
                                        if isinstance(nested_item, dict) and 'encrypted_content' in nested_item:
                                            continue  # Skip nested encrypted content
                                        elif isinstance(nested_item, dict) and nested_item.get('type') == 'text' and 'text' in nested_item:
                                            content_parts.append(nested_item['text'])
                                        elif isinstance(nested_item, str):
                                            content_parts.append(nested_item)
                                else:
                                    content_parts.append(str(nested_content))
                            else:
                                content_parts.append(str(item))
                        else:
                            content_parts.append(str(item))
                    msg_content = "".join(content_parts)
                else:
                    msg_content = str(msg_content_raw)
                
                # Debug: log message content
                logger.debug("Processing message", extra={
                    'request_id': request_id,
                    'user_id': req.userid,
                    'details': {
                        'message_type': type(msg).__name__,
                        'has_content': hasattr(msg, 'content'),
                        'has_tool_calls': hasattr(msg, 'tool_calls'),
                        'content_type': type(msg_content_raw).__name__,
                        'content_preview': str(msg_content)[:200] if msg_content else None,
                        'tool_calls': [{'name': tc.get('name', 'unknown'), 'args': tc.get('args', {})} for tc in getattr(msg, 'tool_calls', [])[:3]]
                    }
                })
                
                # Get role
                msg_role = "assistant"
                if hasattr(msg, 'type'):
                    msg_role = msg.type if msg.type in ['user', 'assistant', 'system'] else "assistant"
                elif hasattr(msg, 'role'):
                    msg_role = msg.role
                
                response_messages.append({
                    "role": msg_role,
                    "content": msg_content,
                    "tool_calls": getattr(msg, 'tool_calls', None)
                })
                
                # Get the final response (last non-tool message)
                # Only update final_response for AIMessage without tool_calls
                if isinstance(msg, AIMessage) and msg_content and not getattr(msg, 'tool_calls', None):
                    final_response = msg_content
        
        duration_ms = (time.time() - start_time) * 1000
        
        logger.info("Request completed successfully", extra={
            'request_id': request_id,
            'user_id': req.userid,
            'duration_ms': round(duration_ms, 2),
            'details': {
                'chunks_processed': chunk_count,
                'response_messages_count': len(response_messages),
                'final_response_length': len(final_response),
                'final_response_preview': final_response[:200] if final_response else None
            }
        })
        
        _touch_session(effective_session_id, req.userid)
        return ReactResponse(
            messages=response_messages,
            final_response=final_response
        )
        
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        
        # Check if it's a GraphRecursionError (agent called tools too many times)
        is_recursion_error = (
            GraphRecursionError is not None and isinstance(e, GraphRecursionError)
        ) or "GraphRecursionError" in type(e).__name__ or "recursion limit" in str(e).lower()
        
        if is_recursion_error:
            logger.warning("Agent hit recursion limit", extra={
                'request_id': request_id,
                'user_id': req.userid,
                'duration_ms': round(duration_ms, 2),
                'error_type': 'GraphRecursionError'
            })
            return ReactResponse(
                messages=[],
                final_response="Error: Agent reached maximum tool call limit (25 iterations). The question may be too complex or require information not available in memory."
            )
        
        logger.error("Request failed", extra={
            'request_id': request_id,
            'user_id': req.userid,
            'duration_ms': round(duration_ms, 2)
        }, exc_info=True)
        return ReactResponse(
            messages=[],
            final_response=f"Error: {str(e)}"
        )


########################################################
# Simple chat endpoint (no context history)
########################################################
@api.post("/chat")
async def chat(req: ChatRequest):
    """Simple chat endpoint for single message interactions without context history."""
    request_id = uuid.uuid4().hex
    start_time = time.time()
    
    try:
        logger.info("Received chat request", extra={
            'request_id': request_id,
            'details': {
                'message_length': len(req.message),
                'message_preview': req.message[:100]
            }
        })
        
        messages = [{"role": "user", "content": req.message}]
        
        # Generate a unique temporary thread_id for this single-use chat
        temp_thread_id = f"temp_{uuid.uuid4().hex}"
        
        context = Context(
            system_prompt="You are a helpful AI assistant.",
            model="anthropic/claude-sonnet-4-5-20250929",
            user_id=temp_thread_id,  # Use temp_thread_id as user_id for memory
        )
        
        config = {"configurable": {"thread_id": temp_thread_id}}
        
        logger.debug("Starting chat execution", extra={
            'request_id': request_id,
            'details': {'temp_thread_id': temp_thread_id}
        })
        
        final_response = ""
        chunk_count = 0
        
        async for chunk in graph.astream(
            {"messages": messages},
            config=config,
            context=context
        ):
            chunk_count += 1
            
            # Debug: log chunk structure
            logger.debug(f"Chat chunk received (count: {chunk_count})", extra={
                'request_id': request_id,
                'details': {'chunk': str(chunk)[:200]}
            })
            
            # Handle different chunk structures
            msgs_to_process = []
            
            if "messages" in chunk:
                msgs_to_process = chunk["messages"]
            elif "call_model" in chunk and "messages" in chunk["call_model"]:
                msgs_to_process = chunk["call_model"]["messages"]
            elif "tools" in chunk and "messages" in chunk["tools"]:
                msgs_to_process = chunk["tools"]["messages"]
            
            for msg in msgs_to_process:
                # Skip ToolMessage - these contain encrypted_content from web_search
                # and should not be returned to the user
                if isinstance(msg, ToolMessage):
                    logger.debug("Skipping ToolMessage in chat", extra={
                        'request_id': request_id,
                        'details': {
                            'message_type': 'ToolMessage',
                            'tool_call_id': getattr(msg, 'tool_call_id', None)
                        }
                    })
                    continue
                
                # Debug: log message structure
                logger.debug("Processing chat message", extra={
                    'request_id': request_id,
                    'details': {
                        'message_type': type(msg).__name__,
                        'has_content': hasattr(msg, 'content'),
                        'has_tool_calls': hasattr(msg, 'tool_calls')
                    }
                })
                
                # Get message content based on message type
                msg_content_raw = ""
                if hasattr(msg, 'content'):
                    msg_content_raw = msg.content
                elif hasattr(msg, 'data') and 'content' in msg.data:
                    msg_content_raw = msg.data['content']
                else:
                    msg_content_raw = str(msg)
                
                # Convert content to string if it's a list or other non-string type
                msg_content = ""
                if isinstance(msg_content_raw, str):
                    msg_content = msg_content_raw
                elif isinstance(msg_content_raw, list):
                    # Handle list of content blocks (e.g., from Claude API)
                    content_parts = []
                    for item in msg_content_raw:
                        if isinstance(item, str):
                            content_parts.append(item)
                        elif isinstance(item, dict):
                            # Skip web_search_tool_result blocks that contain encrypted_content
                            if item.get('type') == 'web_search_tool_result':
                                logger.debug("Skipping web_search_tool_result block with encrypted_content in chat", extra={
                                    'request_id': request_id
                                })
                                continue
                            # Skip blocks that contain encrypted_content
                            if 'encrypted_content' in item:
                                logger.debug("Skipping content block with encrypted_content in chat", extra={
                                    'request_id': request_id,
                                    'block_type': item.get('type', 'unknown')
                                })
                                continue
                            # Skip server_tool_use blocks (tool call information)
                            if item.get('type') == 'server_tool_use':
                                logger.debug("Skipping server_tool_use block in chat", extra={
                                    'request_id': request_id,
                                    'tool_name': item.get('name', 'unknown')
                                })
                                continue
                            # Handle structured content blocks
                            if item.get('type') == 'text' and 'text' in item:
                                content_parts.append(item['text'])
                            elif 'content' in item:
                                # Recursively check nested content for encrypted_content
                                nested_content = item['content']
                                if isinstance(nested_content, list):
                                    for nested_item in nested_content:
                                        if isinstance(nested_item, dict) and 'encrypted_content' in nested_item:
                                            continue  # Skip nested encrypted content
                                        elif isinstance(nested_item, dict) and nested_item.get('type') == 'text' and 'text' in nested_item:
                                            content_parts.append(nested_item['text'])
                                        elif isinstance(nested_item, str):
                                            content_parts.append(nested_item)
                                else:
                                    content_parts.append(str(nested_content))
                            else:
                                content_parts.append(str(item))
                        else:
                            content_parts.append(str(item))
                    msg_content = "".join(content_parts)
                else:
                    msg_content = str(msg_content_raw)
                
                # Debug: log message content
                logger.debug("Chat message content", extra={
                    'request_id': request_id,
                    'details': {
                        'content_preview': str(msg_content)[:200] if msg_content else None,
                        'tool_calls': [{'name': tc.get('name', 'unknown'), 'args': tc.get('args', {})} for tc in getattr(msg, 'tool_calls', [])[:3]]
                    }
                })
                
                # Get final response (last non-tool message)
                # Only update final_response for AIMessage without tool_calls
                if isinstance(msg, AIMessage) and msg_content and not getattr(msg, 'tool_calls', None):
                    final_response = msg_content
        
        duration_ms = (time.time() - start_time) * 1000
        
        logger.info("Chat request completed", extra={
            'request_id': request_id,
            'duration_ms': round(duration_ms, 2),
            'details': {
                'chunks_processed': chunk_count,
                'response_length': len(final_response),
                'response_preview': final_response[:200] if final_response else None
            }
        })
        
        return {"response": final_response}
        
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.error("Chat request failed", extra={
            'request_id': request_id,
            'duration_ms': round(duration_ms, 2)
        }, exc_info=True)
        return {"response": f"Error: {str(e)}"}


########################################################
# Check state endpoint
########################################################
@api.get("/state/session/{session_id}")
async def check_session_state(session_id: str):
    """Check the conversation state for a specific session."""
    try:
        effective_session_id = _resolve_session_id(session_id)
        state = _get_session_state(effective_session_id)
        
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


@api.get("/state/{userid}")
async def check_state(userid: str):
    """Legacy alias for session state, using userid as the session id."""
    return await check_session_state(userid)


########################################################
# Reset short-term session (clear conversation history)
########################################################
@api.post("/reset/session/{session_id}")
async def reset_short_term_session(
    session_id: str,
    userid: Optional[str] = None,
    preserve_memory: bool = True,
):
    """
    Reset the short-term conversation history for a session.
    
    Args:
        session_id: The session identifier
        userid: Optional user identifier for clearing long-term memory
        preserve_memory: If True (default), keep long-term memories intact.
                        If False, also clear LangMem memories.
    
    This clears the checkpoint state but optionally preserves long-term memories.
    """
    request_id = uuid.uuid4().hex
    memory_userid = userid or session_id
    
    try:
        effective_session_id = _resolve_session_id(session_id)
        logger.info(f"Resetting session {effective_session_id}", extra={
            'request_id': request_id,
            'user_id': memory_userid,
            'session_id': effective_session_id,
            'preserve_memory': preserve_memory,
        })
        
        current_state = _get_session_state(effective_session_id)
        current_messages = []
        if current_state and current_state.values:
            current_messages = current_state.values.get("messages", [])
        message_count = len(current_messages)

        # Force preference extraction before wiping the session
        prefs_extracted = 0
        if (
            preserve_memory
            and userid
            and is_langmem_enabled()
            and current_messages
            and effective_session_id not in _finalized_sessions
        ):
            from agent.preference import force_extract_and_persist
            prefs_extracted = await force_extract_and_persist(
                messages=current_messages,
                user_id=userid,
                model="anthropic/claude-haiku-4-5-20251001",
            )
            _finalized_sessions.add(effective_session_id)

        # If not preserving memory, clear LangMem instead
        memory_cleared = False
        if not preserve_memory and is_langmem_enabled() and userid:
            manager = get_langmem_manager()
            memory_cleared = await manager.clear_user_memories(userid)

        replacement_session_id = _roll_session_forward(
            effective_session_id,
            userid or memory_userid,
        )
        
        logger.info(f"Session reset for {effective_session_id}", extra={
            'request_id': request_id,
            'user_id': memory_userid,
            'session_id': effective_session_id,
            'replacement_session_id': replacement_session_id,
            'messages_cleared': message_count,
            'memory_cleared': memory_cleared,
        })
        
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
        
    except Exception as e:
        logger.error(f"Failed to reset session for {session_id}: {e}", extra={
            'request_id': request_id,
            'user_id': memory_userid,
            'session_id': session_id,
        }, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/reset/{userid}")
async def reset_user_session(userid: str, preserve_memory: bool = True):
    """Legacy alias for resetting the userid-scoped default session."""
    return await reset_short_term_session(
        session_id=userid,
        userid=userid,
        preserve_memory=preserve_memory,
    )


########################################################
# Memory injection endpoint
########################################################
@api.post("/memory/{userid}/inject", response_model=InjectResponse)
async def inject_memory(userid: str, req: InjectRequest):
    """
    Inject raw text into a user's long-term memory.

    Chunks the content, stores each chunk as LONG_TERM_CONTEXT, then optionally
    runs LLM-based fact and/or preference extraction on the full text.

    Useful for bulk-loading conversation history, documents, or notes without
    going through the agent conversation flow.
    """
    if not is_langmem_enabled():
        raise HTTPException(status_code=503, detail="LangMem is not enabled")

    from agent.interfaces import StorageType
    from agent.memory.langmem_adapter import get_langmem_manager

    manager = get_langmem_manager()
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
            from locomo_memory_runner import extract_session_observations
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
# LangMem Memory Management Endpoints
########################################################
@api.get("/memory/{userid}")
async def list_memories(userid: str, limit: int = 100):
    """
    List all memories for a specific user.
    
    Args:
        userid: The user identifier
        limit: Maximum number of memories to return (default: 100)
    """
    if not is_langmem_enabled():
        raise HTTPException(status_code=503, detail="LangMem is not enabled")
    
    manager = get_langmem_manager()
    memories = await manager.list_user_memories(userid, limit=limit)
    
    return {
        "userid": userid,
        "count": len(memories),
        "memories": memories,
    }


@api.post("/memory/{userid}/search")
async def search_memories(userid: str, req: MemorySearchRequest):
    """
    Search memories for a specific user.
    
    Args:
        userid: The user identifier
        req: Search request with query and limit
    """
    if not is_langmem_enabled():
        raise HTTPException(status_code=503, detail="LangMem is not enabled")
    
    manager = get_langmem_manager()
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


@api.post("/memory/{userid}/store")
async def store_memory(userid: str, req: MemoryStoreRequest):
    """
    Store a memory for a specific user.
    
    Args:
        userid: The user identifier
        req: Store request with key and content
    """
    if not is_langmem_enabled():
        raise HTTPException(status_code=503, detail="LangMem is not enabled")
    
    from agent.interfaces import StorageType
    manager = get_langmem_manager()
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


@api.delete("/memory/{userid}/{key}")
async def delete_memory(userid: str, key: str):
    """
    Delete a specific memory for a user.
    
    Args:
        userid: The user identifier
        key: Memory key to delete
    """
    if not is_langmem_enabled():
        raise HTTPException(status_code=503, detail="LangMem is not enabled")
    
    manager = get_langmem_manager()
    success = await manager.delete_user_memory(userid, key)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete memory")
    
    return {
        "status": "success",
        "userid": userid,
        "key": key,
    }


@api.delete("/memory/{userid}")
async def clear_memories(userid: str):
    """
    Clear all memories for a specific user.
    
    Args:
        userid: The user identifier
    """
    if not is_langmem_enabled():
        raise HTTPException(status_code=503, detail="LangMem is not enabled")
    
    manager = get_langmem_manager()
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
@api.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "langmem_enabled": is_langmem_enabled(),
    }


@api.get("/status")
async def status():
    """Get system status including LangMem and storage configuration."""
    manager = get_langmem_manager()
    
    return {
        "status": "running",
        "langmem": {
            "enabled": is_langmem_enabled(),
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


@api.get("/storage/stats")
async def storage_stats(userid: Optional[str] = None):
    """
    Get storage statistics.
    
    Args:
        userid: Optional user ID to filter stats
    """
    if not is_storage_available():
        raise HTTPException(status_code=503, detail="Storage backend not available")
    
    manager = get_langmem_manager()
    stats = await manager.get_storage_stats(userid)
    
    return stats


########################################################
# Main entry point
########################################################
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting React Agent server on http://0.0.0.0:8000")
    logger.info(f"LangMem enabled: {is_langmem_enabled()}")
    uvicorn.run(api, host="0.0.0.0", port=8000)
