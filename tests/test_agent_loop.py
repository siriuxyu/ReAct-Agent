import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage


def test_agent_loop_runs_framework_neutral_cycle():
    from agent.runtime.loop import AgentLoop
    from agent.runtime.types import TurnResult

    seen = []
    loop = AgentLoop()

    async def fake_action(decision):
        seen.append(decision.to_payload())
        return TurnResult(final_response=f"model={decision.selected_model}", chunk_count=1)

    result = asyncio.run(
        loop.run_once(
            [HumanMessage(content="hello")],
            default_model="openai/gpt-4o-mini",
            action_fn=fake_action,
        )
    )

    assert result.final_response == "model=openai/gpt-4o-mini"
    assert seen[0]["action"] == "respond"
    assert seen[0]["workspace"]["decision_trace"][0]["kind"] == "model_route"


def test_agent_loop_waits_when_workspace_has_pending_action():
    from agent.runtime.loop import AgentLoop

    loop = AgentLoop()
    inspection = loop.observe(
        [HumanMessage(content="可以发邮件吗")],
        default_model="openai/gpt-4o-mini",
        pending_confirmation={
            "tool_calls": [{"id": "1", "name": "send_email", "args": {"to": "a@example.com"}}],
            "preview": "send_email(to=a@example.com)",
            "highest_side_effect": "external_send",
        },
    )
    decision = loop.decide(inspection)

    assert decision.action == "await_confirmation"
    assert decision.reason == "workspace_pending_action"
