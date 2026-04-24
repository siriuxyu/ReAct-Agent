import asyncio
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.runtime.session_service import SessionService


def _make_service():
    async def fake_extract(**kwargs):
        return 2

    class FakeApp:
        def __init__(self):
            self.state = SimpleNamespace(values={"messages": ["m1", "m2"]})

        def get_state(self, config):
            return self.state

    return SessionService(
        app=FakeApp(),
        logger=SimpleNamespace(info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None),
        session_timeout_seconds=10,
        session_sweep_interval=10,
        is_memory_enabled_fn=lambda: True,
        get_memory_manager_fn=lambda: SimpleNamespace(clear_user_memories=lambda userid: asyncio.sleep(0, result=True)),
        force_extract_preferences_fn=fake_extract,
    )


def test_session_service_create_and_resolve_session():
    service = _make_service()

    session_id, is_new = service.create_or_resume_session("alice")

    assert session_id.startswith("alice_")
    assert is_new is False or is_new is True
    assert service.is_session_owned_by_user("alice", session_id) is True


def test_session_service_rolls_session_forward():
    service = _make_service()
    original = service.build_session_id("alice")
    replacement = service.roll_session_forward(original, "alice")

    assert replacement.startswith("alice_")
    assert service.resolve_session_id(original) == replacement


def test_session_service_reset_session_returns_summary():
    service = _make_service()
    session_id = service.build_session_id("alice")

    result = asyncio.run(
        service.reset_session(
            session_id=session_id,
            userid="alice",
            preserve_memory=True,
            model=None,
        )
    )

    assert result["status"] == "success"
    assert result["messages_cleared"] == 2
