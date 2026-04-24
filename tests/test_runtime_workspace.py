import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_runtime_workspace_records_observations_without_plan_steps():
    from agent.runtime.workspace import build_runtime_workspace

    workspace = build_runtime_workspace(
        goal="帮我看一下最近的邮件",
        task_type="mail",
        artifacts=[
            {
                "tool": "search_emails",
                "ok": True,
                "summary": "Found 3 messages from Bob",
                "data": {"count": 3},
            }
        ],
    )
    payload = workspace.to_payload()

    assert payload["goal"] == "帮我看一下最近的邮件"
    assert payload["task_type"] == "mail"
    assert payload["observations"][0]["source"] == "search_emails"
    assert "plan_steps" not in payload


def test_runtime_workspace_structures_pending_action():
    from agent.runtime.workspace import build_runtime_workspace

    workspace = build_runtime_workspace(
        goal="发邮件给 Alice",
        task_type="mail",
        pending_confirmation={
            "tool_calls": [{"id": "call-1", "name": "send_email", "args": {"to": "a@example.com"}}],
            "preview": "send email to Alice",
            "highest_side_effect": "external_send",
        },
    )
    payload = workspace.to_payload()

    assert payload["pending_action"]["risk_level"] == "external_send"
    assert payload["pending_action"]["tool_calls"][0]["name"] == "send_email"
    assert "await_user_confirmation:external_send" in payload["constraints"]


def test_agent_runtime_inspection_includes_workspace():
    from langchain_core.messages import HumanMessage, ToolMessage

    from agent.runtime import AgentRuntime

    runtime = AgentRuntime()
    inspection = runtime.inspect_messages(
        [
            HumanMessage(content="总结刚才搜索到的结果"),
            ToolMessage(
                content='{"ok": true, "tool": "web_searcher", "summary": "Found source material"}',
                name="web_searcher",
                tool_call_id="call-1",
            ),
        ],
        default_model="anthropic/claude-sonnet-4-5-20250929",
    )

    assert inspection.workspace is not None
    payload = inspection.workspace.to_payload()
    assert payload["observations"][0]["source"] == "web_searcher"
    assert payload["decision_trace"][0]["kind"] == "model_route"
    assert inspection.route_decision is not None
