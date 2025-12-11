import asyncio
import pytest
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


def test_needs_compression_true():
    from agent.summarizer import needs_compression
    msgs = [HumanMessage(content=f"msg {i}") for i in range(15)]
    assert needs_compression(msgs) is True


def test_needs_compression_false():
    from agent.summarizer import needs_compression
    msgs = [HumanMessage(content=f"msg {i}") for i in range(10)]
    assert needs_compression(msgs) is False


def test_needs_compression_boundary():
    from agent.summarizer import needs_compression, COMPRESSION_THRESHOLD
    msgs = [HumanMessage(content=f"msg {i}") for i in range(COMPRESSION_THRESHOLD)]
    assert needs_compression(msgs) is False  # exactly at threshold — no compression
    msgs.append(HumanMessage(content="one more"))
    assert needs_compression(msgs) is True


def test_split_messages_returns_correct_lengths():
    from agent.summarizer import split_messages
    msgs = [HumanMessage(content=f"msg {i}") for i in range(20)]
    old, recent = split_messages(msgs, keep_recent=6)
    assert len(recent) == 6
    assert len(old) == 14
    assert recent == msgs[-6:]


def test_split_messages_smaller_than_keep():
    from agent.summarizer import split_messages
    msgs = [HumanMessage(content="only")]
    old, recent = split_messages(msgs, keep_recent=6)
    assert old == []
    assert recent == msgs


def test_compress_messages_produces_summary_prefix():
    """compress_messages must return a list starting with a SystemMessage summary."""
    from agent.summarizer import compress_messages, KEEP_RECENT
    from unittest.mock import AsyncMock, patch

    # Use more messages than KEEP_RECENT so old messages exist to summarize
    msgs = (
        [HumanMessage(content="I am Alice, a nurse from Boston."),
         AIMessage(content="Nice to meet you Alice!")]
        + [HumanMessage(content=f"filler {i}") for i in range(KEEP_RECENT)]
    )

    fake_summary = "Alice is a nurse from Boston."
    mock_response = AsyncMock()
    mock_response.content = fake_summary
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    with patch("agent.summarizer.load_chat_model", return_value=mock_llm):
        result = asyncio.run(
            compress_messages(msgs, model="anthropic/claude-haiku-4-5-20251001")
        )

    assert isinstance(result[0], SystemMessage)
    assert fake_summary in result[0].content


def test_compress_messages_keeps_recent():
    """The last KEEP_RECENT messages must be preserved verbatim."""
    from agent.summarizer import compress_messages, KEEP_RECENT
    from unittest.mock import AsyncMock, patch

    msgs = [HumanMessage(content=f"msg {i}") for i in range(KEEP_RECENT + 4)]

    mock_response = AsyncMock()
    mock_response.content = "summary"
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    with patch("agent.summarizer.load_chat_model", return_value=mock_llm):
        result = asyncio.run(
            compress_messages(msgs, model="anthropic/claude-haiku-4-5-20251001")
        )

    # Result: [SystemMessage] + last KEEP_RECENT messages
    assert len(result) == KEEP_RECENT + 1
    assert result[1:] == msgs[-KEEP_RECENT:]


def test_compress_messages_fallback_on_error():
    """If LLM call fails, return the original messages unchanged."""
    from agent.summarizer import compress_messages
    from unittest.mock import AsyncMock, patch

    msgs = [HumanMessage(content=f"msg {i}") for i in range(10)]

    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM error"))

    with patch("agent.summarizer.load_chat_model", return_value=mock_llm):
        result = asyncio.run(
            compress_messages(msgs, model="anthropic/claude-haiku-4-5-20251001")
        )

    assert result == msgs
