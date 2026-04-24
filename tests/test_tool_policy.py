import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_evaluate_tool_calls_reports_highest_side_effect():
    from agent.policy.tool_policy import evaluate_tool_calls

    decision = evaluate_tool_calls(
        [
            {"name": "search_emails", "args": {}},
            {"name": "delete_task", "args": {"task_id": "1"}},
        ]
    )

    assert decision.requires_confirmation is True
    assert decision.highest_side_effect == "destructive"
    assert "mail" in decision.capabilities
    assert "tasks" in decision.capabilities
    assert decision.timeout_budget_seconds > 0
    assert "delete_task" in decision.dry_run_candidates or decision.dry_run_candidates == []


def test_build_confirmation_request_includes_metadata():
    from agent.policy import build_confirmation_request

    payload = build_confirmation_request(
        [{"id": "1", "name": "send_email", "args": {"to": "a@example.com"}, "type": "tool_call"}]
    )

    assert payload["highest_side_effect"] == "external_send"
    assert "send_email" in payload["dry_run_candidates"]
    assert payload["tool_calls"][0]["supports_dry_run"] is True
    assert payload["preview_payloads"][0]["dry_run_handler"] == "preview_email"
    assert payload["requires_explicit_confirmation"] is True


def test_build_confirmation_request_redacts_message_body():
    from agent.policy import build_confirmation_request

    payload = build_confirmation_request(
        [
            {
                "id": "1",
                "name": "send_email",
                "args": {"to": "a@example.com", "body": "private note"},
                "type": "tool_call",
            }
        ]
    )

    assert payload["tool_calls"][0]["args"]["body"] == "[redacted:12 chars]"
    assert payload["preview_payloads"][0]["args"]["body"] == "[redacted:12 chars]"


def test_tool_metadata_exposes_execution_contract():
    from tools.metadata import get_tool_metadata

    metadata = get_tool_metadata("send_email")
    payload = metadata.to_payload()

    assert payload["capability"] == "mail"
    assert payload["timeout_seconds"] > 0
    assert payload["dry_run_handler"] == "preview_email"
    assert payload["output_schema"]["required"] == ["ok", "tool", "summary"]
