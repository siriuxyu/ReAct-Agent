from __future__ import annotations

import base64
import re
from email.mime.text import MIMEText
from typing import Optional

try:
    from googleapiclient.discovery import build
except ImportError:
    build = None
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from auth.google_oauth import get_credentials, NeedsAuthorizationError
from .result_utils import tool_error, tool_ok

_AUTH_MSG = "需要授权 Google，请访问 /auth/google 完成授权。"


def _service():
    creds = get_credentials()
    if build is None:
        raise RuntimeError("googleapiclient is not installed")
    return build("gmail", "v1", credentials=creds)


def _decode_body(payload: dict) -> str:
    """从 Gmail message payload 递归提取纯文本正文。优先 text/plain，fallback 剥离 HTML。"""
    def _extract(part: dict, target_mime: str) -> str:
        mime = part.get("mimeType", "")
        if mime == target_mime:
            data = part.get("body", {}).get("data", "")
            if data:
                return base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="replace")
        for child in part.get("parts", []):
            result = _extract(child, target_mime)
            if result:
                return result
        return ""

    plain = _extract(payload, "text/plain")
    if plain:
        return plain

    html = _extract(payload, "text/html")
    if html:
        return re.sub(r"<[^>]+>", "", html).strip()

    return ""


# ── List ──────────────────────────────────────────────────────────────────────

class ListEmailsInput(BaseModel):
    max_results: int = Field(5, description="最多返回条数，默认 5")
    label_ids: list[str] = Field(default_factory=list, description="标签过滤，如 ['INBOX', 'UNREAD']")


@tool("list_emails", args_schema=ListEmailsInput)
def list_emails(max_results: int = 5, label_ids: list[str] = None) -> str:
    """列出最近的 Gmail 邮件，返回 ID、发件人、主题、时间和摘要。"""
    try:
        svc = _service()
    except NeedsAuthorizationError:
        return _AUTH_MSG

    params: dict = {"userId": "me", "maxResults": max_results}
    if label_ids:
        params["labelIds"] = label_ids

    result = svc.users().messages().list(**params).execute()
    messages = result.get("messages", [])
    if not messages:
        return tool_ok("list_emails", "没有找到邮件。", items=[])

    items = []
    lines = []
    for msg in messages:
        detail = svc.users().messages().get(
            userId="me", id=msg["id"], format="metadata",
            metadataHeaders=["From", "Subject", "Date"],
        ).execute()
        headers = {h["name"]: h["value"] for h in detail.get("payload", {}).get("headers", [])}
        snippet = detail.get("snippet", "")[:100]
        item = {
            "id": msg["id"],
            "from": headers.get("From", ""),
            "subject": headers.get("Subject", ""),
            "date": headers.get("Date", ""),
            "snippet": snippet,
        }
        items.append(item)
        lines.append(
            f"[{msg['id']}] {headers.get('Date', '')} | "
            f"From: {headers.get('From', '')} | "
            f"Subject: {headers.get('Subject', '')} | {snippet}"
        )
    return tool_ok("list_emails", "\n".join(lines), items=items)


# ── Search ─────────────────────────────────────────────────────────────────────

class SearchEmailsInput(BaseModel):
    query: str = Field(..., description="Gmail 搜索语法，如 'from:boss@example.com is:unread'")
    max_results: int = Field(5, description="最多返回条数，默认 5")


@tool("search_emails", args_schema=SearchEmailsInput)
def search_emails(query: str, max_results: int = 5) -> str:
    """使用 Gmail 搜索语法搜索邮件（支持 from:、subject:、is:unread 等）。"""
    try:
        svc = _service()
    except NeedsAuthorizationError:
        return _AUTH_MSG

    result = svc.users().messages().list(userId="me", q=query, maxResults=max_results).execute()
    messages = result.get("messages", [])
    if not messages:
        return tool_ok("search_emails", f"没有找到匹配「{query}」的邮件。", query=query, items=[])

    items = []
    lines = []
    for msg in messages:
        detail = svc.users().messages().get(
            userId="me", id=msg["id"], format="metadata",
            metadataHeaders=["From", "Subject", "Date"],
        ).execute()
        headers = {h["name"]: h["value"] for h in detail.get("payload", {}).get("headers", [])}
        snippet = detail.get("snippet", "")[:100]
        item = {
            "id": msg["id"],
            "from": headers.get("From", ""),
            "subject": headers.get("Subject", ""),
            "date": headers.get("Date", ""),
            "snippet": snippet,
        }
        items.append(item)
        lines.append(
            f"[{msg['id']}] {headers.get('Date', '')} | "
            f"From: {headers.get('From', '')} | "
            f"Subject: {headers.get('Subject', '')} | {snippet}"
        )
    return tool_ok("search_emails", "\n".join(lines), query=query, items=items)


# ── Read ───────────────────────────────────────────────────────────────────────

class ReadEmailInput(BaseModel):
    message_id: str = Field(..., description="邮件 ID，由 list_emails 或 search_emails 返回")


@tool("read_email", args_schema=ReadEmailInput)
def read_email(message_id: str) -> str:
    """读取指定邮件的完整正文（优先纯文本，fallback 剥离 HTML）。"""
    try:
        svc = _service()
    except NeedsAuthorizationError:
        return _AUTH_MSG

    try:
        msg = svc.users().messages().get(userId="me", id=message_id, format="full").execute()
    except Exception:
        return tool_error(
            "read_email",
            f"无法读取邮件 {message_id}，可能已被删除或 ID 无效。",
            message_id=message_id,
        )

    payload = msg.get("payload", {})
    headers = {h["name"]: h["value"] for h in payload.get("headers", [])}
    body = _decode_body(payload) or msg.get("snippet", "(正文为空)")

    return tool_ok(
        "read_email",
        (
            f"From: {headers.get('From', '')}\n"
            f"Subject: {headers.get('Subject', '')}\n"
            f"Date: {headers.get('Date', '')}\n"
            f"---\n{body}"
        ),
        message_id=message_id,
        from_address=headers.get("From", ""),
        subject=headers.get("Subject", ""),
        date=headers.get("Date", ""),
        body=body,
    )


# ── Send ───────────────────────────────────────────────────────────────────────

class SendEmailInput(BaseModel):
    to: str = Field(..., description="收件人邮箱地址")
    subject: str = Field(..., description="邮件主题")
    body: str = Field(..., description="邮件正文（纯文本）")
    reply_to_message_id: Optional[str] = Field(None, description="若为回复，填写原邮件 ID；留空则发新邮件")


@tool("send_email", args_schema=SendEmailInput)
def send_email(to: str, subject: str, body: str,
               reply_to_message_id: Optional[str] = None) -> str:
    """发送新邮件，或回复已有邮件线程（传 reply_to_message_id 时自动带线程 header）。"""
    try:
        svc = _service()
    except NeedsAuthorizationError:
        return _AUTH_MSG

    mime_msg = MIMEText(body, "plain", "utf-8")
    mime_msg["To"] = to
    mime_msg["Subject"] = subject
    mime_msg["From"] = "me"

    thread_id = None
    if reply_to_message_id:
        try:
            original = svc.users().messages().get(
                userId="me", id=reply_to_message_id, format="metadata",
                metadataHeaders=["Message-ID", "References"],
            ).execute()
            hdrs = {h["name"]: h["value"]
                    for h in original.get("payload", {}).get("headers", [])}
            if "Message-ID" in hdrs:
                mime_msg["In-Reply-To"] = hdrs["Message-ID"]
                refs = hdrs.get("References", "").strip()
                mime_msg["References"] = (refs + " " + hdrs["Message-ID"]).strip()
            thread_id = original.get("threadId")
        except Exception:
            pass  # 降级为发新邮件

    raw = base64.urlsafe_b64encode(mime_msg.as_bytes()).decode()
    body_dict: dict = {"raw": raw}
    if thread_id:
        body_dict["threadId"] = thread_id

    sent = svc.users().messages().send(userId="me", body=body_dict).execute()
    return tool_ok(
        "send_email",
        f"邮件已发送 (id: {sent['id']})",
        message_id=sent["id"],
        to=to,
        subject=subject,
        reply_to_message_id=reply_to_message_id,
        thread_id=thread_id,
    )
