import asyncio
import pytest
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


def _make_long_msgs(count: int, chars_per_msg: int = 40) -> list:
    """Create messages long enough to accumulate tokens quickly."""
    return [HumanMessage(content="x" * chars_per_msg) for _ in range(count)]


def test_needs_compression_true():
    from agent.summarizer import needs_compression, TARGET_MAX_TOKENS
    # Each 40-char msg is ~10 tokens. Need > TARGET_MAX_TOKENS / 10 messages.
    needed = TARGET_MAX_TOKENS // 10 + 2
    msgs = _make_long_msgs(needed, chars_per_msg=40)
    assert needs_compression(msgs) is True


def test_needs_compression_false():
    from agent.summarizer import needs_compression
    msgs = [HumanMessage(content="short") for _ in range(10)]
    assert needs_compression(msgs) is False


def test_compress_messages_produces_summary():
    """compress_messages must insert a SystemMessage summary in the middle."""
    from agent.summarizer import compress_messages, TARGET_MAX_TOKENS
    from unittest.mock import AsyncMock, patch

    # Create enough long messages to trigger compression
    needed = TARGET_MAX_TOKENS // 10 + 5
    msgs = (
        [HumanMessage(content="I am Alice, a nurse from Boston."),
         AIMessage(content="Nice to meet you Alice!")]
        + _make_long_msgs(needed, chars_per_msg=40)
    )

    fake_summary = "Alice is a nurse from Boston."
    mock_response = AsyncMock()
    mock_response.content = fake_summary
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    with patch("agent.summarizer.load_chat_model", return_value=mock_llm):
        result = asyncio.run(compress_messages(msgs, model="anthropic/claude-haiku-4-5-20251001"))

    # Find the summary SystemMessage in the result
    summary_msgs = [m for m in result if isinstance(m, SystemMessage)]
    assert len(summary_msgs) == 1
    assert fake_summary in summary_msgs[0].content


def test_compress_messages_keeps_tail():
    """Tail messages must be preserved verbatim after compression."""
    from agent.summarizer import compress_messages, TARGET_MAX_TOKENS, PROTECT_TAIL
    from unittest.mock import AsyncMock, patch

    needed = TARGET_MAX_TOKENS // 10 + 5
    tail_content = "tail-msg"
    msgs = _make_long_msgs(needed, chars_per_msg=40)
    msgs[-1] = HumanMessage(content=tail_content)

    mock_response = AsyncMock()
    mock_response.content = "summary"
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    with patch("agent.summarizer.load_chat_model", return_value=mock_llm):
        result = asyncio.run(compress_messages(msgs, model="anthropic/claude-haiku-4-5-20251001"))

    # The last PROTECT_TAIL messages should include our tail_content
    tail_texts = [m.content for m in result[-PROTECT_TAIL:] if hasattr(m, "content")]
    assert tail_content in tail_texts


def test_compress_messages_fallback_on_error():
    """If LLM call fails, return the original messages unchanged."""
    from agent.summarizer import compress_messages, TARGET_MAX_TOKENS
    from unittest.mock import AsyncMock, patch

    needed = TARGET_MAX_TOKENS // 10 + 5
    msgs = _make_long_msgs(needed, chars_per_msg=40)

    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM error"))

    with patch("agent.summarizer.load_chat_model", return_value=mock_llm):
        result = asyncio.run(compress_messages(msgs, model="anthropic/claude-haiku-4-5-20251001"))

    assert result == msgs
