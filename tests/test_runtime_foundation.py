import sys
import os
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


def test_agent_runtime_inspects_messages_and_routes_task_type():
    from agent.runtime import AgentRuntime

    runtime = AgentRuntime()
    inspection = runtime.inspect_messages(
        [
            HumanMessage(content="帮我发一封邮件给导师"),
            ToolMessage(
                content='{"ok": true, "tool": "search_emails", "summary": "找到相关邮件", "data": {"count": 2}}',
                tool_call_id="tool-1",
                name="search_emails",
            ),
        ],
        default_model="anthropic/claude-sonnet-4-5-20250929",
    )

    assert inspection.latest_user_text == "帮我发一封邮件给导师"
    assert inspection.task_type == "email"
    assert inspection.tool_artifacts[0]["data"]["count"] == 2


def test_agent_runtime_detects_pending_confirmation_from_model_response():
    from agent.runtime import AgentRuntime

    runtime = AgentRuntime()
    response = AIMessage(
        content="我可以帮你发出这封邮件。",
        tool_calls=[
            {
                "id": "call-1",
                "name": "send_email",
                "args": {"to": "advisor@example.com", "subject": "Meeting"},
                "type": "tool_call",
            }
        ],
    )

    inspection = runtime.inspect_model_response(response)

    assert inspection.pending_confirmation is not None
    assert inspection.pending_confirmation.preview.startswith("send_email(")


def test_agent_runtime_run_turn_executes_and_persists():
    from agent.runtime import AgentRuntime

    runtime = AgentRuntime()
    persisted = []

    async def fake_invoke_graph(app, *, messages, config, context):
        return ([{"role": "assistant", "content": "完成了", "tool_calls": None}], "完成了", 3)

    def fake_persist(**kwargs):
        persisted.append(kwargs)

    turn = asyncio.run(
        runtime.run_turn(
            app=object(),
            messages=[{"role": "user", "content": "hello"}],
            config={"configurable": {"thread_id": "t1"}},
            context=object(),
            request_messages=[HumanMessage(content="hello")],
            user_id="alice",
            session_id="alice_1",
            invoke_graph_fn=fake_invoke_graph,
            transcript_store=object(),
            persist_transcript_fn=fake_persist,
        )
    )

    assert turn.final_response == "完成了"
    assert turn.chunk_count == 3
    assert persisted[0]["user_id"] == "alice"


def test_agent_runtime_run_stream_turn_emits_chunks_and_persists():
    from agent.runtime import AgentRuntime

    runtime = AgentRuntime()
    seen = []
    persisted = []

    class FakeApp:
        async def astream(self, payload, config=None, context=None):
            yield {"call_model": {"messages": [AIMessage(content="第一段")]}}
            yield {"call_model": {"messages": [AIMessage(content="最终答案")]}}

    async def on_text(text: str):
        seen.append(text)

    def fake_persist(**kwargs):
        persisted.append(kwargs)

    turn = asyncio.run(
        runtime.run_stream_turn(
            app=FakeApp(),
            messages=[{"role": "user", "content": "hello"}],
            config={"configurable": {"thread_id": "t1"}},
            context=object(),
            request_messages=[HumanMessage(content="hello")],
            user_id="alice",
            session_id="alice_1",
            extract_text_fn=lambda chunk: chunk["call_model"]["messages"][0].content,
            transcript_store=object(),
            persist_transcript_fn=fake_persist,
            on_text_fn=on_text,
        )
    )

    assert seen == ["第一段", "最终答案"]
    assert turn.final_response == "最终答案"
    assert turn.chunk_count == 2
    assert persisted[0]["final_response"] == "最终答案"
