import os
import sys
import asyncio
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_content_preview_handles_multimodal_blocks():
    from agent.adapters.api_adapter import content_preview

    preview = content_preview(
        [
            {"type": "image", "source": {"media_type": "image/png"}},
            {"type": "text", "text": "这是正文"},
        ]
    )

    assert "[image/png]" in preview
    assert "这是正文" in preview


def test_persist_transcript_writes_request_and_response():
    from agent.adapters.api_adapter import persist_transcript

    calls = []

    class FakeStore:
        def add_message(self, **kwargs):
            calls.append(kwargs)

    request_messages = [SimpleNamespace(role="user", content="hello world")]
    persist_transcript(
        store=FakeStore(),
        user_id="alice",
        session_id="alice_1",
        request_messages=request_messages,
        final_response="hi there",
    )

    assert calls[0]["role"] == "user"
    assert calls[1]["role"] == "assistant"
    assert calls[1]["content"] == "hi there"


def test_prepare_agent_run_injects_recall_and_preferences():
    from agent.adapters.api_adapter import prepare_agent_run
    from langchain_core.messages import HumanMessage

    class DummyContext:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class DummyManager:
        async def search_user_memories(self, **kwargs):
            return [
                {
                    "content": "喜欢简洁总结",
                    "created_at": "2026-04-01T00:00:00Z",
                    "metadata": {"preference_type": "style"},
                }
            ]

    req = SimpleNamespace(
        userid="alice",
        session_id="alice_1",
        messages=[SimpleNamespace(role="user", content="总结一下上次讨论")],
        system_prompt=None,
        model=None,
        max_search_results=None,
        enable_web_search=None,
        enable_preference_extraction=None,
    )
    existing_state = SimpleNamespace(values={"messages": [HumanMessage(content="old")]})
    prepared = asyncio.run(
        prepare_agent_run(
            req,
            context_cls=DummyContext,
            logger=SimpleNamespace(warning=lambda *_args, **_kwargs: None),
            resolve_session_id=lambda session_id: session_id,
            is_session_owned_by_user=lambda userid, session_id: True,
            session_config=lambda session_id: {"configurable": {"thread_id": session_id}},
            get_session_state=lambda _session_id: existing_state,
            has_session_messages=lambda state: bool(state.values["messages"]),
            is_memory_enabled_fn=lambda: True,
            get_memory_manager_fn=lambda: DummyManager(),
            build_session_recall_block_fn=lambda **_kwargs: "## Recalled session context\n- earlier summary",
        )
    )

    assert prepared.effective_session_id == "alice_1"
    assert "Recalled session context" in prepared.context.system_prompt
    assert prepared.context.enable_preference_extraction is False
