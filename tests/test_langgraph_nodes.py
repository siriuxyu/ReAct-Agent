import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import AIMessage, HumanMessage


def test_route_after_planner_prefers_confirmed_tools():
    from agent.runtime.langgraph_nodes import route_after_planner
    from agent.state import State

    state = State(messages=[], confirmed_tool_calls=[{"name": "send_email"}])
    assert route_after_planner(state) == "tools"


def test_route_after_planner_handles_cancelled_confirmation():
    from agent.runtime.langgraph_nodes import route_after_planner
    from agent.state import State

    state = State(messages=[], confirmation_resolution="cancelled")
    assert route_after_planner(state) == "confirmation_cancelled"


def test_model_output_route_ends_without_tool_calls_when_extraction_disabled():
    from agent.runtime.langgraph_nodes import create_model_output_route
    from agent.state import State

    route_model_output = create_model_output_route(logger=type("L", (), {"debug": lambda *a, **k: None, "info": lambda *a, **k: None})())
    state = State(
        messages=[HumanMessage(content="hi"), AIMessage(content="hello")],
        enable_preference_extraction=False,
    )

    assert route_model_output(state) == "__end__"


def test_model_output_route_sends_tool_calls_to_tools():
    from agent.runtime.langgraph_nodes import create_model_output_route
    from agent.state import State

    route_model_output = create_model_output_route(logger=type("L", (), {"debug": lambda *a, **k: None, "info": lambda *a, **k: None})())
    state = State(
        messages=[
            HumanMessage(content="查邮件"),
            AIMessage(
                content="",
                tool_calls=[{"id": "call-1", "name": "search_emails", "args": {}}],
            ),
        ]
    )

    assert route_model_output(state) == "tools"
