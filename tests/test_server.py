import os
import sys
import asyncio
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.adapters.api_adapter import PreparedAgentRun
from agent.runtime.service import RuntimeServiceResult
from agent.runtime.types import TurnResult


def _import_server(monkeypatch):
    async def noop():
        return None

    fake_scheduler = SimpleNamespace(start=noop, stop=noop)
    monkeypatch.setitem(sys.modules, "services.scheduler", fake_scheduler)
    import server

    return server


def _prepared(session_id: str = "alice_1") -> PreparedAgentRun:
    return PreparedAgentRun(
        config={"configurable": {"thread_id": session_id}},
        context=SimpleNamespace(),
        messages=[{"role": "user", "content": "hi"}],
        effective_session_id=session_id,
    )


def test_invoke_endpoint_uses_runtime_service(monkeypatch):
    server = _import_server(monkeypatch)

    async def fake_invoke_request(req):
        prepared = _prepared("alice_1")
        turn = TurnResult(
            response_messages=[{"role": "assistant", "content": "hello"}],
            final_response="hello",
            chunk_count=1,
        )
        return RuntimeServiceResult(prepared=prepared, turn=turn)

    monkeypatch.setattr(server.runtime_service, "invoke_request", fake_invoke_request)
    monkeypatch.setattr(server, "_touch_session", lambda session_id, userid: None)
    req = server.CliriuxRequest(
        userid="alice",
        messages=[server.Message(role="user", content="hi")],
    )

    response = asyncio.run(server.invoke(req))

    assert response.final_response == "hello"
    assert response.messages[0]["role"] == "assistant"


def test_stream_endpoint_emits_session_chunk_and_done(monkeypatch):
    server = _import_server(monkeypatch)

    async def fake_prepare_request(req):
        return _prepared("alice_1")

    async def fake_stream_prepared_request(req, *, prepared, on_text_fn):
        await on_text_fn("hello")
        return RuntimeServiceResult(
            prepared=prepared,
            turn=TurnResult(final_response="hello", chunk_count=1),
        )

    monkeypatch.setattr(server.runtime_service, "prepare_request", fake_prepare_request)
    monkeypatch.setattr(server.runtime_service, "stream_prepared_request", fake_stream_prepared_request)
    monkeypatch.setattr(server, "_touch_session", lambda session_id, userid: None)
    req = server.CliriuxRequest(
        userid="alice",
        messages=[server.Message(role="user", content="hi")],
    )

    async def collect():
        return [event async for event in server._sse_generator(req, request_id="test")]

    body = "".join(asyncio.run(collect()))
    assert '"type": "session"' in body
    assert '"type": "chunk"' in body
    assert '"type": "done"' in body
    assert "hello" in body


def test_sessions_endpoint_delegates_to_session_service(monkeypatch):
    server = _import_server(monkeypatch)

    monkeypatch.setattr(
        server.session_service,
        "create_or_resume_session",
        lambda userid, session_id=None: ("alice_1", True),
    )
    req = server.SessionCreateRequest(userid="alice")

    response = asyncio.run(server.create_session(req))

    assert response.session_id == "alice_1"
    assert response.userid == "alice"
    assert response.is_new is True


def test_reset_session_endpoint_delegates_to_session_service(monkeypatch):
    server = _import_server(monkeypatch)

    async def fake_reset_session(**kwargs):
        return {
            "session_id": kwargs["session_id"],
            "effective_session_id": kwargs["session_id"],
            "replacement_session_id": "alice_2",
            "messages_cleared": 2,
            "preferences_extracted": 1,
            "memory_preserved": True,
        }

    monkeypatch.setattr(server.session_service, "reset_session", fake_reset_session)

    response = asyncio.run(server.reset_short_term_session(session_id="alice_1", userid="alice"))

    assert response["replacement_session_id"] == "alice_2"
