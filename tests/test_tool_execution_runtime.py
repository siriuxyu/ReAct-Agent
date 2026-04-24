import os
import sys
import asyncio
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_core.messages import AIMessage, ToolMessage


def test_resolve_runtime_tools_delegates_to_factory():
    from agent.runtime.tool_execution import resolve_runtime_tools

    tools = resolve_runtime_tools("alice", lambda user_id: [user_id, "tool"])

    assert tools == ["alice", "tool"]


def test_execute_tool_phase_intercepts_confirmation():
    from agent.runtime.tool_execution import execute_tool_phase

    result = asyncio.run(
        execute_tool_phase(
            messages=[
                AIMessage(
                    content="send it",
                    tool_calls=[{"id": "1", "name": "send_email", "args": {"to": "a@example.com"}}],
                )
            ],
            all_tools=[],
            confirmed_tool_calls=[],
        )
    )

    assert result["pending_confirmation"]["requested_tools"] == ["send_email"]
    assert result["messages"][0].name == "send_email"


def test_execute_tool_phase_runs_confirmed_tool_calls():
    from agent.runtime.tool_execution import execute_tool_phase

    class FakeToolNode:
        def __init__(self, all_tools):
            self.all_tools = all_tools

        async def ainvoke(self, state_dict):
            return {
                "messages": [
                    ToolMessage(
                        content="echo:hello",
                        tool_call_id="1",
                        name="echo_tool",
                    )
                ]
            }

    with patch("agent.runtime.tool_execution.ToolNode", FakeToolNode):
        result = asyncio.run(
            execute_tool_phase(
                messages=[],
                all_tools=["echo_tool"],
                confirmed_tool_calls=[
                    {"id": "1", "name": "echo_tool", "args": {"text": "hello"}}
                ],
            )
        )

    assert result["confirmed_tool_calls"] == []
    assert any(getattr(message, "name", "") == "echo_tool" for message in result["messages"])
