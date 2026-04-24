import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_search_profile_memories_filters_preference_records():
    from agent.memory.profile_store import search_profile_memories

    class DummyManager:
        async def search_user_memories(self, **kwargs):
            assert kwargs["user_id"] == "alice"
            return [
                {
                    "content": "喜欢简洁回答",
                    "created_at": "2026-04-21T08:30:00Z",
                    "metadata": {"preference_type": "style"},
                },
                {
                    "content": "上周讨论了 benchmark",
                    "created_at": "2026-04-21T08:31:00Z",
                    "metadata": {"document_type": "long_term_context"},
                },
            ]

    records = asyncio.run(
        search_profile_memories(
            user_id="alice",
            query="preferences",
            limit=5,
            manager_factory=lambda: DummyManager(),
        )
    )

    assert len(records) == 1
    assert records[0].content == "喜欢简洁回答"
    assert records[0].memory_type == "style"


def test_task_scratchpad_renders_current_turn_context():
    from agent.memory.task_scratchpad import build_task_scratchpad

    scratchpad = build_task_scratchpad(
        user_query="帮我回顾上次的计划",
        session_id="alice_session",
        task_type="memory",
        tool_artifacts=[{"tool": "search_memory"}],
    )

    rendered = scratchpad.render()
    assert "Current task scratchpad" in rendered
    assert "active_session: alice_session" in rendered
    assert "task_type: memory" in rendered
    assert "known_artifacts: 1" in rendered
