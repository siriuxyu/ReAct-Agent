import asyncio
import pytest
from langchain_core.messages import HumanMessage, AIMessage

from agent.extraction.extractor import ContextExtractor
from agent.interfaces import PreferenceType


def run(coro):
    return asyncio.run(coro)


def test_extract_preferences_finds_prefer_keyword():
    extractor = ContextExtractor()
    msgs = [HumanMessage(content="I prefer concise answers.")]
    prefs = run(extractor.extract_preferences(msgs, "u1"))
    assert len(prefs) == 1
    assert prefs[0].confidence_score >= 0.9


def test_extract_preferences_ignores_ai_messages():
    extractor = ContextExtractor()
    msgs = [AIMessage(content="I prefer this response format.")]
    prefs = run(extractor.extract_preferences(msgs, "u1"))
    assert prefs == []


def test_extract_preferences_empty_messages():
    extractor = ContextExtractor()
    prefs = run(extractor.extract_preferences([], "u1"))
    assert prefs == []


def test_extract_preferences_no_keywords():
    extractor = ContextExtractor()
    msgs = [HumanMessage(content="What is the capital of France?")]
    prefs = run(extractor.extract_preferences(msgs, "u1"))
    assert prefs == []


def test_classify_communication_style():
    extractor = ContextExtractor()
    result = extractor._classify_preference_type("I like concise answers")
    assert result == PreferenceType.COMMUNICATION_STYLE


def test_classify_domain_interest():
    extractor = ContextExtractor()
    result = extractor._classify_preference_type("I prefer python code examples")
    assert result == PreferenceType.DOMAIN_INTEREST


def test_classify_language():
    extractor = ContextExtractor()
    result = extractor._classify_preference_type("I prefer french responses")
    assert result == PreferenceType.LANGUAGE_PREFERENCE


def test_classify_tool_preference():
    extractor = ContextExtractor()
    result = extractor._classify_preference_type("I want to use the calculator tool")
    assert result == PreferenceType.TOOL_PREFERENCE


def test_classify_response_format():
    extractor = ContextExtractor()
    result = extractor._classify_preference_type("I like bullet lists")
    assert result == PreferenceType.RESPONSE_FORMAT


def test_merge_preferences_increases_frequency():
    extractor = ContextExtractor()
    msgs = [HumanMessage(content="I prefer bullet points.")]
    old = run(extractor.extract_preferences(msgs, "u1"))
    new = run(extractor.extract_preferences(msgs, "u1"))
    merged = run(extractor.merge_preferences(old, new))
    assert merged[0].frequency == 2


def test_merge_preferences_distinct_kept_separate():
    extractor = ContextExtractor()
    old = run(extractor.extract_preferences(
        [HumanMessage(content="I prefer concise answers.")], "u1"
    ))
    new = run(extractor.extract_preferences(
        [HumanMessage(content="I like python examples.")], "u1"
    ))
    merged = run(extractor.merge_preferences(old, new))
    assert len(merged) == 2


def test_generate_summary_nonempty():
    extractor = ContextExtractor()
    msgs = [
        HumanMessage(content="Hello"),
        HumanMessage(content="Goodbye"),
    ]
    summary = run(extractor.generate_summary(msgs))
    assert "Hello" in summary
    assert len(summary) <= 403  # max_length=400 + "..."


def test_generate_summary_empty():
    extractor = ContextExtractor()
    summary = run(extractor.generate_summary([]))
    assert summary == "No messages to summarize."
