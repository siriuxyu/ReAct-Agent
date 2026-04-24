import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_select_model_for_step_prefers_fast_model_for_simple_chat(monkeypatch):
    monkeypatch.setenv("ROUTER_FAST_MODEL", "openai/gpt-4o-mini")
    from agent.model_router import select_model_for_step

    chosen = select_model_for_step(
        "anthropic/claude-sonnet-4-5-20250929",
        task_type="chat",
        step_name="assistant",
        latest_user_text="hi",
        has_tool_results=False,
    )
    assert chosen == "openai/gpt-4o-mini"


def test_select_model_for_step_prefers_complex_model_for_tool_work(monkeypatch):
    monkeypatch.setenv("ROUTER_COMPLEX_MODEL", "anthropic/claude-sonnet-4-5-20250929")
    from agent.model_router import select_model_for_step

    chosen = select_model_for_step(
        "openai/gpt-4o-mini",
        task_type="calendar",
        step_name="assistant",
        latest_user_text="帮我安排下周的面试并比较空闲时间",
        has_tool_results=True,
    )
    assert chosen == "anthropic/claude-sonnet-4-5-20250929"


def test_explain_model_route_returns_auditable_reason(monkeypatch):
    monkeypatch.setenv("ROUTER_COMPLEX_MODEL", "anthropic/claude-sonnet-4-5-20250929")
    from agent.runtime.router import explain_model_route

    decision = explain_model_route(
        "openai/gpt-4o-mini",
        task_type="email",
        latest_user_text="总结邮件并安排后续计划",
        has_tool_results=True,
    )

    assert decision.selected_model == "anthropic/claude-sonnet-4-5-20250929"
    assert decision.reason == "complex_or_tool_task"
    assert decision.signals["has_tool_results"] is True
