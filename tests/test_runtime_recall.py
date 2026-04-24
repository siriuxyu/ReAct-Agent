import asyncio
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_build_runtime_recall_context_combines_preferences_and_session_recall():
    from agent.memory.runtime_recall import build_runtime_memory_context, build_runtime_recall_context

    class DummyManager:
        async def search_user_memories(self, **kwargs):
            return [
                {
                    "content": "偏好简洁回复",
                    "created_at": "2026-04-20T12:00:00Z",
                    "metadata": {"preference_type": "style"},
                }
            ]

    context = asyncio.run(
        build_runtime_memory_context(
            user_id="alice",
            query="上次的总结",
            exclude_session_id="alice_1",
            include_preferences=True,
            task_type="memory",
            logger=SimpleNamespace(warning=lambda *_args, **_kwargs: None),
            is_memory_enabled_fn=lambda: True,
            get_memory_manager_fn=lambda: DummyManager(),
            build_session_recall_block_fn=lambda **_kwargs: "## Recalled session context\n- earlier summary",
        )
    )

    result = context.render()
    assert "Known user preferences" in result
    assert "偏好简洁回复" in result
    assert "Recalled session context" in result
    assert "Current task scratchpad" in result


def test_build_runtime_recall_context_skips_preferences_when_disabled():
    from agent.memory.runtime_recall import build_runtime_recall_context

    result = asyncio.run(
        build_runtime_recall_context(
            user_id="alice",
            query="",
            exclude_session_id="alice_1",
            include_preferences=False,
            task_type="chat",
            logger=SimpleNamespace(warning=lambda *_args, **_kwargs: None),
            is_memory_enabled_fn=lambda: True,
            get_memory_manager_fn=lambda: None,
            build_session_recall_block_fn=lambda **_kwargs: "",
        )
    )

    assert "Current task scratchpad" in result
