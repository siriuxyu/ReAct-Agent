import re
import time
import uuid
from typing import Dict, Any, List, cast

from langchain_core.messages import SystemMessage, AIMessage, ToolMessage, BaseMessage
from langgraph.runtime import Runtime

from agent.context import Context
from agent.model_router import select_model_for_step
from agent.state import State
from agent.utils import load_chat_model, get_logger
from agent.schemas import UserProfileUpdate
from agent.prompts import PREFERENCE_EXTRACTION_SYSTEM_PROMPT

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Layer 1: Regex-based fast extraction (zero LLM cost)
# ---------------------------------------------------------------------------

_RULE_PATTERNS: List[tuple] = [
    # Personal / Identity
    (r'(?:我叫|我的名字是|my\s+name\s+is|i\s*am\s+\S+\s*[,，])\s*([一-龥]{2,4}|\w{2,20})', 'personal', 'name'),
    (r'(?:我|I[\'\']?m)\s*(\d{1,3})\s*(?:岁|years?\s*old)', 'personal', 'age'),
    (r'(?:我住在|我来自|I\s+(?:live\s+in|am\s+from))\s*([一-龥]+|[\w\s]+)', 'personal', 'location'),
    (r'(?:我是|I\s+am\s+a[n]?\s+)([一-龥]+|[\w\s]+)', 'personal', 'occupation'),
    (r'(?:我会说|我说|I\s+speak)\s*([一-龥]+|[\w\s]+)', 'personal', 'language'),
    # Likes / Preferences
    (r'(?:我喜欢|我爱|I\s+(?:like|love|enjoy|prefer))\s*(.+)', 'personal', 'likes'),
    (r'(?:我不喜欢|我讨厌|I\s+(?:dislike|hate|don\'t\s+like))\s*(.+)', 'personal', 'dislikes'),
    (r'(?:我的最爱|my\s+favorite)\s*(?:是|is)?\s*(.+)', 'personal', 'favorite'),
    # Constraints
    (r'(?:不要|别|never|don\'t|do\s+not)\s*(.+)', 'constraint', 'avoid'),
    (r'(?:总是|always|请|please)\s*(.+)', 'constraint', 'rule'),
]


def _try_regex_extract(text: str) -> List[Dict[str, Any]]:
    """Fast rule-based preference extraction. Returns list of raw preference dicts."""
    results = []
    for pattern, ptype, subtype in _RULE_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            raw = match.group(1).strip() if match.lastindex else match.group(0).strip()
            if len(raw) < 2 or len(raw) > 200:
                continue
            # Clean up trailing punctuation
            raw = raw.rstrip('。，！？.!?;；')
            if not raw:
                continue
            results.append({
                "preference_type": ptype,
                "subtype": subtype,
                "content": raw,
                "confidence": 0.85,
            })
    return results


async def _persist_raw_preference(
    pref: Dict[str, Any],
    user_id: str,
    source: str = "regex"
) -> bool:
    """Persist a raw preference dict (from regex or tool call) to ChromaDB."""
    try:
        from agent.memory.memory_manager import get_memory_manager
        from agent.interfaces import StorageType
        manager = get_memory_manager()
        key = f"pref_{source}_{uuid.uuid4().hex[:8]}"
        content = pref["content"]
        await manager.store_user_memory(
            user_id=user_id,
            key=key,
            content=content,
            metadata={
                "preference_type": pref.get("preference_type", "personal"),
                "confidence": pref.get("confidence", 0.8),
                "source": source,
            },
            document_type=StorageType.USER_PREFERENCE,
        )
        return True
    except Exception as e:
        logger.error(f"Failed to persist raw preference: {e}")
        return False


# ---------------------------------------------------------------------------
# Message filtering
# ---------------------------------------------------------------------------

def filter_messages_for_extraction(messages: List[BaseMessage]) -> List[BaseMessage]:
    """
    Filter messages to only include Human and AI messages (without tool calls).
    This avoids the 'unexpected tool_use_id' error from incomplete tool sequences.
    """
    filtered = []
    for msg in messages:
        if getattr(msg, "additional_kwargs", {}).get("summary_generated"):
            continue
        if isinstance(msg, ToolMessage):
            continue
        if isinstance(msg, AIMessage):
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                continue
            if msg.content:
                filtered.append(msg)
        else:
            filtered.append(msg)
    return filtered


# ---------------------------------------------------------------------------
# Layer 3: LLM-based extraction (fallback)
# ---------------------------------------------------------------------------

async def extract_preferences(
        state: State, runtime: Runtime[Context]
) -> Dict[str, Any]:
    """
    LangGraph Node: Analyzes the conversation to extract user preferences.
    Runs Layer 1 (regex) first, then Layer 3 (LLM) if needed.
    """
    start_time = time.time()
    user_id = runtime.context.user_id

    # --- Layer 1: Regex extraction on the newest human message ---
    regex_extracted = 0
    recent_messages = state.messages[-3:]  # Only check last 3 for speed
    for msg in recent_messages:
        if isinstance(msg, BaseMessage) and getattr(msg, 'type', None) == 'human':
            if getattr(msg, "additional_kwargs", {}).get("summary_generated"):
                continue
            text = str(msg.content)
            hits = _try_regex_extract(text)
            for hit in hits:
                if await _persist_raw_preference(hit, user_id, source="regex"):
                    regex_extracted += 1

    if regex_extracted:
        logger.info(f"Layer 1 regex extracted {regex_extracted} preferences", extra={
            'function': 'extract_preferences', 'layer': 1
        })

    # --- Layer 3: LLM fallback for complex / implicit preferences ---
    # Skip if regex already found something and we want to save tokens
    # But always run if enable_preference_extraction is True (for thoroughness)
    model_spec = select_model_for_step(
        runtime.context.model,
        task_type=state.task_type,
        step_name="preference_extraction",
        latest_user_text="",
        has_tool_results=bool(state.tool_artifacts),
    )
    model = load_chat_model(model_spec)
    structured_llm = model.with_structured_output(UserProfileUpdate)

    recent_messages = state.messages[-10:]
    filtered_messages = filter_messages_for_extraction(recent_messages)

    messages_for_extraction = [
        SystemMessage(content=PREFERENCE_EXTRACTION_SYSTEM_PROMPT),
        *filtered_messages
    ]

    logger.info("Starting Layer 3 LLM preference extraction", extra={
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
            logger.info("Layer 3 LLM extracted preferences", extra={
                'function': 'extract_preferences',
                'duration_ms': round(duration, 2),
                'details': {
                    'count': len(result.extracted_preferences),
                    'types': [p.preference_type for p in result.extracted_preferences],
                    'reasoning': result.reasoning
                }
            })
            await _persist_llm_preferences(
                result.extracted_preferences,
                user_id=user_id,
            )
            return {"extracted_preferences": result.extracted_preferences}
        else:
            logger.debug("No LLM preferences found", extra={'function': 'extract_preferences'})
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
    # Layer 1: Regex on all messages
    regex_count = 0
    for msg in messages:
        if isinstance(msg, BaseMessage) and getattr(msg, 'type', None) == 'human':
            if getattr(msg, "additional_kwargs", {}).get("summary_generated"):
                continue
            text = str(msg.content)
            for hit in _try_regex_extract(text):
                if await _persist_raw_preference(hit, user_id, source="regex"):
                    regex_count += 1

    # Layer 3: LLM fallback
    filtered = filter_messages_for_extraction(messages[-20:])
    if not filtered:
        return regex_count

    llm = load_chat_model(model)
    structured_llm = llm.with_structured_output(UserProfileUpdate)
    msgs = [SystemMessage(content=PREFERENCE_EXTRACTION_SYSTEM_PROMPT), *filtered]

    try:
        result = cast(UserProfileUpdate, await structured_llm.ainvoke(msgs))
        if result.extracted_preferences:
            await _persist_llm_preferences(result.extracted_preferences, user_id)
            logger.info(
                f"Force-extracted {len(result.extracted_preferences)} preferences on session end",
                extra={"user_id": user_id},
            )
            return regex_count + len(result.extracted_preferences)
    except Exception as e:
        logger.error(f"force_extract_and_persist failed: {e}")
    return regex_count


async def _persist_llm_preferences(preferences, user_id: str) -> None:
    """Persist LLM-extracted preferences to ChromaDB as USER_PREFERENCE documents."""
    try:
        from agent.memory.memory_manager import get_memory_manager
        from agent.interfaces import StorageType
        manager = get_memory_manager()
        for pref in preferences:
            content = pref.content
            key = f"pref_{uuid.uuid4().hex[:12]}"
            await manager.store_user_memory(
                user_id=user_id,
                key=key,
                content=content,
                metadata={
                    "preference_type": pref.preference_type.value,
                    "confidence": pref.confidence,
                    "source": "llm_extraction",
                },
                document_type=StorageType.USER_PREFERENCE,
            )
        logger.info(f"Persisted {len(preferences)} LLM preferences for user {user_id}")
    except Exception as e:
        logger.error(f"Failed to persist LLM preferences: {e}")
