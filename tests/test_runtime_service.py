import asyncio
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.adapters.api_adapter import PreparedAgentRun
from agent.runtime.service import AgentRuntimeService
from agent.runtime.types import TurnResult


def _make_service(fake_runtime):
    class DummyContext:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    async def fake_prepare(req, **kwargs):
        return PreparedAgentRun(
            config={"configurable": {"thread_id": "t1"}},
            context=SimpleNamespace(),
            messages=[{"role": "user", "content": "hello"}],
            effective_session_id="alice_1",
        )

    return AgentRuntimeService(
        agent_runtime=fake_runtime,
        app="app",
        context_cls=DummyContext,
        logger=SimpleNamespace(),
        prepare_agent_run_fn=fake_prepare,
        invoke_graph_fn=lambda *args, **kwargs: None,
        extract_text_fn=lambda chunk: "text",
        persist_transcript_fn=lambda **kwargs: None,
        transcript_store_factory=lambda: "store",
        resolve_session_id=lambda session_id: session_id,
        is_session_owned_by_user=lambda userid, session_id: True,
        session_config=lambda session_id: {"configurable": {"thread_id": session_id}},
        get_session_state=lambda session_id: None,
        has_session_messages=lambda state: False,
        is_memory_enabled_fn=lambda: False,
        get_memory_manager_fn=lambda: None,
        build_session_recall_block_fn=lambda **kwargs: "",
    )


def test_runtime_service_invoke_request_uses_prepared_request():
    calls = []

    class FakeRuntime:
        async def run_turn(self, **kwargs):
            calls.append(kwargs)
            return TurnResult(final_response="done", chunk_count=2)

    service = _make_service(FakeRuntime())
    req = SimpleNamespace(userid="alice", messages=[SimpleNamespace(role="user", content="hello")])

    result = asyncio.run(service.invoke_request(req))

    assert result.prepared.effective_session_id == "alice_1"
    assert result.turn.final_response == "done"
    assert calls[0]["session_id"] == "alice_1"


def test_runtime_service_stream_prepared_request_reuses_prepared_state():
    calls = []

    class FakeRuntime:
        async def run_stream_turn(self, **kwargs):
            calls.append(kwargs)
            await kwargs["on_text_fn"]("chunk")
            return TurnResult(final_response="done", chunk_count=1)

    service = _make_service(FakeRuntime())
    req = SimpleNamespace(userid="alice", messages=[SimpleNamespace(role="user", content="hello")])
    prepared = PreparedAgentRun(
        config={"configurable": {"thread_id": "t1"}},
        context=SimpleNamespace(),
        messages=[{"role": "user", "content": "hello"}],
        effective_session_id="alice_1",
    )
    seen = []

    async def on_text(text: str):
        seen.append(text)

    result = asyncio.run(service.stream_prepared_request(req, prepared=prepared, on_text_fn=on_text))

    assert seen == ["chunk"]
    assert result.prepared is prepared
    assert calls[0]["session_id"] == "alice_1"


def test_runtime_service_invoke_chat_request_uses_runtime_turn():
    calls = []

    class FakeRuntime:
        async def run_turn(self, **kwargs):
            calls.append(kwargs)
            return TurnResult(final_response="chat done", chunk_count=1)

    service = _make_service(FakeRuntime())
    turn = asyncio.run(
        service.invoke_chat_request(
            message="hello",
            user_id="temp_1",
            system_prompt="You are helpful",
            model="anthropic/claude-sonnet-4-5-20250929",
        )
    )

    assert turn.final_response == "chat done"
    assert calls[0]["session_id"] == "temp_1"
