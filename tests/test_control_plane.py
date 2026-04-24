import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage, ToolMessage


def test_latest_user_text_ignores_generated_summary():
    from agent.control import latest_user_text

    messages = [
        HumanMessage(content="真实用户消息"),
        HumanMessage(
            content="[CONTEXT SUMMARY]: generated",
            additional_kwargs={"summary_generated": True},
        ),
    ]

    assert latest_user_text(messages) == "真实用户消息"


def test_confirmation_classifier_handles_yes_and_no():
    from agent.control import classify_confirmation_response

    assert classify_confirmation_response("yes, go ahead") == "approve"
    assert classify_confirmation_response("先别做这个") == "reject"


def test_requires_confirmation_for_side_effect_tools():
    from agent.control import requires_confirmation

    assert requires_confirmation([{"name": "send_email", "args": {}}]) is True
    assert requires_confirmation([{"name": "search_emails", "args": {}}]) is False


def test_collect_tool_artifacts_parses_json_payload():
    from agent.control import collect_tool_artifacts

    messages = [
        ToolMessage(
            content='{"ok": true, "tool": "send_email", "summary": "邮件已发送", "data": {"message_id": "m1"}}',
            tool_call_id="call-1",
            name="send_email",
        )
    ]

    artifacts = collect_tool_artifacts(messages)
    assert artifacts[0]["tool"] == "send_email"
    assert artifacts[0]["data"]["message_id"] == "m1"
