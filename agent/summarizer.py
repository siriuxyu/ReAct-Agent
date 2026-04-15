"""
Context compression for long conversations.

Inspired by Hermes trajectory_compressor.py:
- Trigger by token count (not message count)
- Protect head turns (system + first human + first AI + first tool)
- Protect tail turns (last N messages)
- Compress only the middle region, as much as needed
- Insert summary as a HumanMessage (not SystemMessage)
- Use a cheap model for summarization
"""

from typing import Any, List, Tuple, Union

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage

from agent.utils import get_logger, load_chat_model

logger = get_logger(__name__)

# Token budget (cl100k_base approximate)
TARGET_MAX_TOKENS = 8000      # compress when estimated tokens exceed this
SUMMARY_TARGET_TOKENS = 600   # budget for the summary text itself
SUMMARY_MODEL = "anthropic/claude-haiku-4-5-20251001"  # cheap model for summarization

# Protected turns
PROTECT_HEAD = 4   # keep first 4 turns (system, human, AI, tool)
PROTECT_TAIL = 6   # keep last 6 turns


def _estimate_tokens(text: str) -> int:
    """Fast token estimation without loading tiktoken. ~4 chars per token."""
    return len(text) // 4


def _extract_text(content: Union[str, List[Any]]) -> str:
    """Extract plain text from a message content that may be str or a multimodal list."""
    if isinstance(content, str):
        return content
    parts: List[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            btype = block.get("type", "")
            if btype == "text":
                parts.append(block.get("text", ""))
            elif btype == "image":
                parts.append("[image]")
            elif btype == "document":
                parts.append("[document]")
    return " ".join(parts)


def _message_to_line(m: BaseMessage) -> str:
    """Convert a message to a log line for summarization prompt."""
    if isinstance(m, SystemMessage):
        return f"[System] {_extract_text(m.content)}"
    if isinstance(m, HumanMessage):
        return f"[User] {_extract_text(m.content)}"
    if isinstance(m, AIMessage):
        if getattr(m, "tool_calls", None):
            calls = ", ".join(tc.get("name", "?") for tc in m.tool_calls)
            text = _extract_text(m.content) if m.content else ""
            return f"[Assistant] tool_calls=[{calls}] {text}".strip()
        return f"[Assistant] {_extract_text(m.content)}"
    if isinstance(m, ToolMessage):
        return f"[Tool result] {_extract_text(m.content)[:500]}"
    return f"[Message] {_extract_text(m.content)}"


def count_message_tokens(m: BaseMessage) -> int:
    """Estimate tokens for a single message."""
    return _estimate_tokens(_message_to_line(m))


def count_messages_tokens(messages: List[BaseMessage]) -> int:
    """Estimate total tokens for a list of messages."""
    return sum(count_message_tokens(m) for m in messages)


def needs_compression(messages: List[BaseMessage]) -> bool:
    """Return True when estimated tokens exceed the target budget."""
    return count_messages_tokens(messages) > TARGET_MAX_TOKENS


def _find_compressible_region(
    messages: List[BaseMessage],
) -> Tuple[int, int]:
    """
    Find the middle region that can be compressed.

    Returns (compress_start, compress_end) as indices into messages.
    Head [0:PROTECT_HEAD] and tail [-PROTECT_TAIL:] are protected.
    """
    n = len(messages)
    protected_head = set(range(min(PROTECT_HEAD, n)))
    protected_tail = set(range(max(0, n - PROTECT_TAIL), n))
    protected = protected_head | protected_tail

    # Find first compressible index after head
    compress_start = PROTECT_HEAD
    while compress_start in protected and compress_start < n:
        compress_start += 1

    # Find last compressible index before tail
    compress_end = n - PROTECT_TAIL
    while compress_end - 1 in protected and compress_end > compress_start:
        compress_end -= 1

    return compress_start, compress_end


_SUMMARIZE_PROMPT = """Summarize the following conversation segment concisely.
This summary will replace these turns in the conversation history so the assistant
can continue working with the right context.

Include:
1. Actions the assistant took (tool calls, searches, calculations, etc.)
2. Key information or results obtained
3. Important decisions or findings
4. Relevant data, file names, values, or outputs
5. User facts worth remembering (name, preferences, constraints)

Keep it factual and informative. Target ~{target} tokens.

---
CONVERSATION SEGMENT:
{conversation}
---

Write only the summary, starting with "[CONTEXT SUMMARY]:" prefix."""


async def compress_messages(
    messages: List[BaseMessage],
    model: str,
) -> List[BaseMessage]:
    """
    Compress messages by summarizing the middle region.

    Strategy:
    1. If total tokens <= TARGET_MAX_TOKENS, skip compression
    2. Protect head and tail turns
    3. Calculate how many tokens need to be saved
    4. Accumulate middle turns from start until savings are sufficient
    5. Replace compressed region with a single HumanMessage summary

    Falls back to returning original messages if summarization fails.
    """
    total_tokens = count_messages_tokens(messages)

    if total_tokens <= TARGET_MAX_TOKENS:
        return list(messages)

    compress_start, compress_end = _find_compressible_region(messages)

    if compress_start >= compress_end:
        # Nothing to compress (too short after protection)
        logger.warning(
            "Messages too short to compress after protecting head/tail",
            extra={"function": "compress_messages", "message_count": len(messages)},
        )
        return list(messages)

    # Calculate how many tokens we need to save
    tokens_to_save = total_tokens - TARGET_MAX_TOKENS
    target_compress_tokens = tokens_to_save + SUMMARY_TARGET_TOKENS

    # Accumulate turns from compress_start until we have enough savings
    accumulated_tokens = 0
    actual_compress_end = compress_start

    turn_tokens = [count_message_tokens(m) for m in messages]

    for i in range(compress_start, compress_end):
        accumulated_tokens += turn_tokens[i]
        actual_compress_end = i + 1
        if accumulated_tokens >= target_compress_tokens:
            break

    # If still not enough, compress the entire compressible region
    if accumulated_tokens < target_compress_tokens and actual_compress_end < compress_end:
        actual_compress_end = compress_end
        accumulated_tokens = sum(turn_tokens[compress_start:actual_compress_end])

    # Build transcript of the region to summarize
    lines = [_message_to_line(messages[i]) for i in range(compress_start, actual_compress_end)]
    conversation_text = "\n".join(lines)

    # Truncate very long text for the summary prompt
    if len(conversation_text) > 8000:
        conversation_text = conversation_text[:4000] + "\n...[truncated]...\n" + conversation_text[-2000:]

    prompt = _SUMMARIZE_PROMPT.format(
        target=SUMMARY_TARGET_TOKENS,
        conversation=conversation_text,
    )

    try:
        from agent.model_router import select_model_for_step
        summary_model = select_model_for_step(
            SUMMARY_MODEL,
            task_type="chat",
            step_name="summarizer",
        )
        llm = load_chat_model(summary_model)
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        summary_text = (response.content or "").strip()
        if not summary_text:
            raise ValueError("Empty summary")
        if not summary_text.startswith("[CONTEXT SUMMARY]"):
            summary_text = f"[CONTEXT SUMMARY]: {summary_text}"
    except Exception as e:
        logger.warning(f"Context compression failed, keeping full history: {e}")
        return list(messages)

    # Build compressed message list:
    # [head] + [summary as HumanMessage] + [tail]
    compressed: List[BaseMessage] = []

    # Head: turns before compression region
    for i in range(compress_start):
        compressed.append(messages[i])

    # Summary inserted as a generated SystemMessage so downstream extractors can ignore it.
    compressed.append(
        SystemMessage(
            content=summary_text,
            additional_kwargs={"summary_generated": True},
        )
    )

    # Tail: turns after compression region
    for i in range(actual_compress_end, len(messages)):
        compressed.append(messages[i])

    compressed_tokens = count_messages_tokens(compressed)
    tokens_saved = total_tokens - compressed_tokens

    logger.info(
        "Context compressed",
        extra={
            "function": "compress_messages",
            "details": {
                "original_count": len(messages),
                "compressed_count": len(compressed),
                "summarized_turns": actual_compress_end - compress_start,
                "original_tokens": total_tokens,
                "compressed_tokens": compressed_tokens,
                "tokens_saved": tokens_saved,
                "summary_model": summary_model,
            },
        },
    )

    return compressed
