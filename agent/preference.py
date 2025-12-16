import time
import uuid
from typing import Dict, Any, List, cast

from langchain_core.messages import SystemMessage, AIMessage, ToolMessage, BaseMessage
from langgraph.runtime import Runtime

from agent.context import Context
from agent.state import State
from agent.utils import load_chat_model, get_logger
from agent.schemas import UserProfileUpdate
from agent.prompts import PREFERENCE_EXTRACTION_SYSTEM_PROMPT

logger = get_logger(__name__)


def filter_messages_for_extraction(messages: List[BaseMessage]) -> List[BaseMessage]:
    """
    Filter messages to only include Human and AI messages (without tool calls).
    This avoids the 'unexpected tool_use_id' error from incomplete tool sequences.
    """
    filtered = []
    for msg in messages:
        # Skip ToolMessages entirely
        if isinstance(msg, ToolMessage):
            continue
        # For AIMessages, only include if they don't have tool_calls
        if isinstance(msg, AIMessage):
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                # Skip AI messages that contain tool calls
                continue
            # Include AI message content only if it has actual content
            if msg.content:
                filtered.append(msg)
        else:
            # Include HumanMessage and others
            filtered.append(msg)
    return filtered


async def extract_preferences(
        state: State, runtime: Runtime[Context]
) -> Dict[str, Any]:
    """
    LangGraph Node: Analyzes the conversation to extract user preferences.
    """
    start_time = time.time()

    # prepare model
    model = load_chat_model(runtime.context.model)
    structured_llm = model.with_structured_output(UserProfileUpdate)

    # 2. prepare recent Messages - filter out tool-related messages
    recent_messages = state.messages[-10:]
    filtered_messages = filter_messages_for_extraction(recent_messages)

    messages_for_extraction = [
        SystemMessage(content=PREFERENCE_EXTRACTION_SYSTEM_PROMPT),
        *filtered_messages
    ]

    logger.info("Starting preference extraction", extra={
        'function': 'extract_preferences',
        'details': {
            'raw_message_count': len(recent_messages),
            'filtered_message_count': len(filtered_messages),
        }
    })

    try:
        result = await structured_llm.ainvoke(messages_for_extraction)
        result = cast(UserProfileUpdate, result)

        duration = (time.time() - start_time) * 1000

        if result.extracted_preferences:
            logger.info("Preferences extracted", extra={
                'function': 'extract_preferences',
                'duration_ms': round(duration, 2),
                'details': {
                    'count': len(result.extracted_preferences),
                    'types': [p.preference_type for p in result.extracted_preferences],
                    'reasoning': result.reasoning
                }
            })
            # Persist to ChromaDB automatically (passive storage)
            await _persist_preferences(
                result.extracted_preferences,
                user_id=runtime.context.user_id,
            )
            return {"extracted_preferences": result.extracted_preferences}
        else:
            logger.debug("No preferences found", extra={'function': 'extract_preferences'})
            return {"extracted_preferences": []}

    except Exception as e:
        logger.error("Error during preference extraction", exc_info=True)
        return {"extracted_preferences": []}


async def force_extract_and_persist(
    messages: List[BaseMessage],
    user_id: str,
    model: str = "anthropic/claude-haiku-4-5-20251001",
) -> int:
    """
    Extract preferences from a message list and persist to ChromaDB.
    Called directly (e.g. on session reset) without going through the LangGraph node.
    Returns the number of preferences persisted.
    """
    filtered = filter_messages_for_extraction(messages[-20:])
    if not filtered:
        return 0

    llm = load_chat_model(model)
    structured_llm = llm.with_structured_output(UserProfileUpdate)
    msgs = [SystemMessage(content=PREFERENCE_EXTRACTION_SYSTEM_PROMPT), *filtered]

    try:
        result = cast(UserProfileUpdate, await structured_llm.ainvoke(msgs))
        if result.extracted_preferences:
            await _persist_preferences(result.extracted_preferences, user_id)
            logger.info(
                f"Force-extracted {len(result.extracted_preferences)} preferences on session end",
                extra={"user_id": user_id},
            )
            return len(result.extracted_preferences)
    except Exception as e:
        logger.error(f"force_extract_and_persist failed: {e}")
    return 0


async def _persist_preferences(preferences, user_id: str) -> None:
    """Persist extracted preferences to ChromaDB as USER_PREFERENCE documents."""
    try:
        from agent.memory.langmem_adapter import get_langmem_manager
        from agent.interfaces import StorageType
        manager = get_langmem_manager()
        for pref in preferences:
            content = f"[{pref.preference_type.value}] {pref.content}"
            key = f"pref_{uuid.uuid4().hex[:12]}"
            await manager.store_user_memory(
                user_id=user_id,
                key=key,
                content=content,
                metadata={"confidence": pref.confidence, "source": "auto_extraction"},
                document_type=StorageType.USER_PREFERENCE,
            )
        logger.info(f"Persisted {len(preferences)} preferences for user {user_id}")
    except Exception as e:
        logger.error(f"Failed to persist preferences: {e}")