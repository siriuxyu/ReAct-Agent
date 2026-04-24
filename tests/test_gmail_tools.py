import sys, os, base64
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch, MagicMock


def _make_gmail_service(messages=None, message_detail=None, sent_id="sent-1"):
    """构造 mock Gmail service。"""
    svc = MagicMock()
    svc.users().messages().list().execute.return_value = {
        "messages": messages or []
    }
    body_data = base64.urlsafe_b64encode(b"Hello, this is the email body.").decode()
    svc.users().messages().get().execute.return_value = message_detail or {
        "id": "msg-1",
        "threadId": "thread-1",
        "snippet": "Test snippet",
        "payload": {
            "mimeType": "text/plain",
            "headers": [
                {"name": "From", "value": "sender@example.com"},
                {"name": "Subject", "value": "Test Subject"},
                {"name": "Date", "value": "Mon, 22 Apr 2026 10:00:00 +0800"},
                {"name": "Message-ID", "value": "<msg-1@example.com>"},
            ],
            "body": {"data": body_data},
        },
    }
    svc.users().messages().send().execute.return_value = {"id": sent_id}
    return svc


@patch("tools.gmail.get_credentials")
@patch("tools.gmail.build")
def test_list_emails_returns_message_info(mock_build, mock_creds):
    mock_build.return_value = _make_gmail_service(
        messages=[{"id": "msg-1"}]
    )
    from tools.gmail import list_emails
    result = list_emails.invoke({"max_results": 5, "label_ids": []})
    assert "msg-1" in result
    assert "sender@example.com" in result or "Test Subject" in result


@patch("tools.gmail.get_credentials")
@patch("tools.gmail.build")
def test_list_emails_empty(mock_build, mock_creds):
    mock_build.return_value = _make_gmail_service(messages=[])
    from tools.gmail import list_emails
    result = list_emails.invoke({"max_results": 5, "label_ids": []})
    assert "没有" in result


@patch("tools.gmail.get_credentials")
@patch("tools.gmail.build")
def test_search_emails_returns_results(mock_build, mock_creds):
    mock_build.return_value = _make_gmail_service(messages=[{"id": "msg-1"}])
    from tools.gmail import search_emails
    result = search_emails.invoke({"query": "is:unread", "max_results": 5})
    assert "msg-1" in result


@patch("tools.gmail.get_credentials")
@patch("tools.gmail.build")
def test_search_emails_no_results(mock_build, mock_creds):
    mock_build.return_value = _make_gmail_service(messages=[])
    from tools.gmail import search_emails
    result = search_emails.invoke({"query": "from:nobody", "max_results": 5})
    assert "没有找到" in result


@patch("tools.gmail.get_credentials")
@patch("tools.gmail.build")
def test_read_email_returns_body(mock_build, mock_creds):
    mock_build.return_value = _make_gmail_service()
    from tools.gmail import read_email
    result = read_email.invoke({"message_id": "msg-1"})
    assert "Hello, this is the email body." in result
    assert "Test Subject" in result


@patch("tools.gmail.get_credentials")
@patch("tools.gmail.build")
def test_send_email_new(mock_build, mock_creds):
    mock_build.return_value = _make_gmail_service()
    from tools.gmail import send_email
    result = send_email.invoke({
        "to": "recipient@example.com",
        "subject": "Hello",
        "body": "Test body",
        "reply_to_message_id": None,
    })
    assert "sent-1" in result or "发送" in result


@patch("tools.gmail.get_credentials")
@patch("tools.gmail.build")
def test_send_email_reply(mock_build, mock_creds):
    mock_build.return_value = _make_gmail_service()
    from tools.gmail import send_email
    result = send_email.invoke({
        "to": "sender@example.com",
        "subject": "Re: Test Subject",
        "body": "Reply body",
        "reply_to_message_id": "msg-1",
    })
    assert "sent-1" in result or "发送" in result


@patch("tools.gmail.get_credentials")
def test_no_auth_returns_message(mock_creds):
    from auth.google_oauth import NeedsAuthorizationError
    mock_creds.side_effect = NeedsAuthorizationError("no token")
    from tools.gmail import list_emails
    result = list_emails.invoke({"max_results": 5, "label_ids": []})
    assert "授权" in result or "auth" in result.lower()


@patch("tools.gmail.get_credentials")
@patch("tools.gmail.build")
def test_read_email_nested_multipart(mock_build, mock_creds):
    """验证嵌套 multipart 结构中能正确提取 text/plain。"""
    body_data = base64.urlsafe_b64encode(b"Nested plain text body.").decode()
    nested_detail = {
        "id": "msg-2",
        "threadId": "thread-2",
        "snippet": "snippet only",
        "payload": {
            "mimeType": "multipart/mixed",
            "headers": [
                {"name": "From", "value": "a@example.com"},
                {"name": "Subject", "value": "Nested"},
                {"name": "Date", "value": "Tue, 22 Apr 2026 10:00:00 +0800"},
            ],
            "body": {},
            "parts": [
                {
                    "mimeType": "multipart/alternative",
                    "body": {},
                    "parts": [
                        {
                            "mimeType": "text/plain",
                            "body": {"data": body_data},
                        },
                        {
                            "mimeType": "text/html",
                            "body": {"data": base64.urlsafe_b64encode(b"<p>HTML</p>").decode()},
                        },
                    ],
                }
            ],
        },
    }
    svc = MagicMock()
    svc.users().messages().get().execute.return_value = nested_detail
    mock_build.return_value = svc
    from tools.gmail import read_email
    result = read_email.invoke({"message_id": "msg-2"})
    assert "Nested plain text body." in result
    assert "snippet only" not in result
