"""
Context compression for long conversations.

When a conversation exceeds COMPRESSION_THRESHOLD messages, older messages are
summarized into a single SystemMessage to keep the context window manageable
while preserving key user facts across extended interactions.
"""

from typing import List, Tuple

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from agent.utils import get_logger, load_chat_model

logger = get_logger(__name__)

COMPRESSION_THRESHOLD = 14  # compress when len(messages) > this
KEEP_RECENT = 6             # always keep this many recent messages verbatim

_SUMMARIZE_PROMPT = """Summarize the conversation below into a compact context note.
Preserve all factual information about the user (name, location, occupation, preferences, constraints, topics discussed).
Output only the summary text — no preamble or explanation.

Conversation:
{conversation}
"""


def needs_compression(messages: List[BaseMessage]) -> bool:
    """Return True when the message list is long enough to warrant compression."""
    return len(messages) > COMPRESSION_THRESHOLD


def split_messages(
    messages: List[BaseMessage], keep_recent: int = KEEP_RECENT
) -> Tuple[List[BaseMessage], List[BaseMessage]]:
    """Split messages into (older_to_summarize, recent_to_keep)."""
    if len(messages) <= keep_recent:
        return [], list(messages)
    return list(messages[:-keep_recent]), list(messages[-keep_recent:])


async def compress_messages(
    messages: List[BaseMessage],
    model: str,
) -> List[BaseMessage]:
    """
    Summarize older messages into a single SystemMessage prefix.

    Returns [SystemMessage(summary)] + messages[-KEEP_RECENT:].
    Falls back to returning the original messages if the LLM call fails.
    """
    old_msgs, recent = split_messages(messages)

    if not old_msgs:
        return list(messages)

    # Build plain-text transcript of the older messages
    lines = []
    for m in old_msgs:
        if isinstance(m, HumanMessage):
            lines.append(f"User: {m.content}")
        elif isinstance(m, AIMessage) and m.content:
            lines.append(f"Assistant: {m.content}")

    conversation_text = "\n".join(lines)
    prompt = _SUMMARIZE_PROMPT.format(conversation=conversation_text)

    llm = load_chat_model(model)
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        summary_text = response.content.strip()
    except Exception as e:
        logger.warning(f"Context compression failed, keeping full history: {e}")
        return list(messages)

    summary_msg = SystemMessage(
        content=f"[Conversation Summary]\n{summary_text}"
    )
    compressed = [summary_msg] + recent

    logger.info(
        "Context compressed",
        extra={
            "function": "compress_messages",
            "details": {
                "original_count": len(messages),
                "compressed_count": len(compressed),
                "summarized_turns": len(old_msgs),
            },
        },
    )
    return compressed
