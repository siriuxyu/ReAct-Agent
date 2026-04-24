# Gmail + Google Tasks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 Cliriux agent 增加 Gmail 邮件管理（读/搜索/发送/回复）和 Google Tasks 任务管理（多列表增删改查）能力，复用现有 Google OAuth token，合并所有 scope。

**Architecture:** 单体扩展，不增加新进程。修改 `auth/google_oauth.py` 合并 scope，新建 `tools/gmail.py`（4 工具）和 `tools/tasks.py`（6 工具），注册进 `tools/__init__.py`。用户需删除旧 token 重新授权一次。

**Tech Stack:** `google-api-python-client`（已安装），`langchain_core.tools.tool`，Pydantic BaseModel，`email.mime`（标准库），`base64`（标准库）

---

## 文件变更总览

| 文件 | 操作 | 说明 |
|------|------|------|
| `auth/google_oauth.py` | 修改第 11 行 | 扩展 SCOPES 列表 |
| `tools/gmail.py` | 新建 | 4 个 Gmail 工具 |
| `tools/tasks.py` | 新建 | 6 个 Google Tasks 工具 |
| `tools/__init__.py` | 修改 | 注册 10 个新工具 |
| `tests/test_oauth_scopes.py` | 新建 | 验证 scope 列表 |
| `tests/test_gmail_tools.py` | 新建 | Gmail 工具测试 |
| `tests/test_tasks_tools.py` | 新建 | Tasks 工具测试 |

---

## Task 1: 扩展 OAuth Scope

**Files:**
- Modify: `auth/google_oauth.py:11`
- Test: `tests/test_oauth_scopes.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_oauth_scopes.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_scopes_include_gmail_readonly():
    from auth.google_oauth import SCOPES
    assert "https://www.googleapis.com/auth/gmail.readonly" in SCOPES


def test_scopes_include_gmail_send():
    from auth.google_oauth import SCOPES
    assert "https://www.googleapis.com/auth/gmail.send" in SCOPES


def test_scopes_include_tasks():
    from auth.google_oauth import SCOPES
    assert "https://www.googleapis.com/auth/tasks" in SCOPES


def test_scopes_still_include_calendar():
    from auth.google_oauth import SCOPES
    assert "https://www.googleapis.com/auth/calendar" in SCOPES
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
conda run -n agent python -m pytest Cliriux/tests/test_oauth_scopes.py -v
```

预期：前三个 FAIL（scope 不存在），第四个 PASS。

- [ ] **Step 3: 修改 `auth/google_oauth.py`，替换第 11 行**

将：
```python
SCOPES = ["https://www.googleapis.com/auth/calendar"]
```

替换为：
```python
SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/tasks",
]
```

- [ ] **Step 4: 运行测试，确认全部通过**

```bash
conda run -n agent python -m pytest Cliriux/tests/test_oauth_scopes.py -v
```

预期：4 passed。

- [ ] **Step 5: 提交**

```bash
git -C Cliriux add auth/google_oauth.py tests/test_oauth_scopes.py
git -C Cliriux commit -m "[chore] expand Google OAuth scopes for Gmail and Tasks"
```

---

## Task 2: Gmail 工具

**Files:**
- Create: `tools/gmail.py`
- Test: `tests/test_gmail_tools.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_gmail_tools.py
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
```

- [ ] **Step 2: 运行测试，确认全部失败**

```bash
conda run -n agent python -m pytest Cliriux/tests/test_gmail_tools.py -v
```

预期：8 个 ERROR（ImportError: cannot import `tools.gmail`）。

- [ ] **Step 3: 创建 `tools/gmail.py`**

```python
# tools/gmail.py
from __future__ import annotations

import base64
import re
from email.mime.text import MIMEText
from typing import Optional

from googleapiclient.discovery import build
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from auth.google_oauth import get_credentials, NeedsAuthorizationError

_AUTH_MSG = "需要授权 Google，请访问 /auth/google 完成授权。"


def _service():
    return build("gmail", "v1", credentials=get_credentials())


def _decode_body(payload: dict) -> str:
    """从 Gmail message payload 提取纯文本正文。"""
    if payload.get("mimeType") == "text/plain":
        data = payload.get("body", {}).get("data", "")
        if data:
            return base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="replace")

    for part in payload.get("parts", []):
        if part.get("mimeType") == "text/plain":
            data = part.get("body", {}).get("data", "")
            if data:
                return base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="replace")

    for part in payload.get("parts", []):
        if part.get("mimeType") == "text/html":
            data = part.get("body", {}).get("data", "")
            if data:
                html = base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="replace")
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
        return "没有找到邮件。"

    lines = []
    for msg in messages:
        detail = svc.users().messages().get(
            userId="me", id=msg["id"], format="metadata",
            metadataHeaders=["From", "Subject", "Date"],
        ).execute()
        headers = {h["name"]: h["value"] for h in detail.get("payload", {}).get("headers", [])}
        snippet = detail.get("snippet", "")[:100]
        lines.append(
            f"[{msg['id']}] {headers.get('Date', '')} | "
            f"From: {headers.get('From', '')} | "
            f"Subject: {headers.get('Subject', '')} | {snippet}"
        )
    return "\n".join(lines)


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
        return f"没有找到匹配「{query}」的邮件。"

    lines = []
    for msg in messages:
        detail = svc.users().messages().get(
            userId="me", id=msg["id"], format="metadata",
            metadataHeaders=["From", "Subject", "Date"],
        ).execute()
        headers = {h["name"]: h["value"] for h in detail.get("payload", {}).get("headers", [])}
        snippet = detail.get("snippet", "")[:100]
        lines.append(
            f"[{msg['id']}] {headers.get('Date', '')} | "
            f"From: {headers.get('From', '')} | "
            f"Subject: {headers.get('Subject', '')} | {snippet}"
        )
    return "\n".join(lines)


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
        return f"无法读取邮件 {message_id}，可能已被删除或 ID 无效。"

    payload = msg.get("payload", {})
    headers = {h["name"]: h["value"] for h in payload.get("headers", [])}
    body = _decode_body(payload) or msg.get("snippet", "(正文为空)")

    return (
        f"From: {headers.get('From', '')}\n"
        f"Subject: {headers.get('Subject', '')}\n"
        f"Date: {headers.get('Date', '')}\n"
        f"---\n{body}"
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
    return f"邮件已发送 (id: {sent['id']})"
```

- [ ] **Step 4: 运行测试，确认全部通过**

```bash
conda run -n agent python -m pytest Cliriux/tests/test_gmail_tools.py -v
```

预期：8 passed。

- [ ] **Step 5: 提交**

```bash
git -C Cliriux add tools/gmail.py tests/test_gmail_tools.py
git -C Cliriux commit -m "[feat] add Gmail tools (list, search, read, send)"
```

---

## Task 3: Google Tasks 工具

**Files:**
- Create: `tools/tasks.py`
- Test: `tests/test_tasks_tools.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_tasks_tools.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch, MagicMock


def _make_tasks_service(task_lists=None, tasks=None):
    """构造 mock Tasks service。"""
    svc = MagicMock()
    svc.tasklists().list().execute.return_value = {
        "items": task_lists or [{"id": "list-1", "title": "工作"}]
    }
    svc.tasklists().insert().execute.return_value = {"id": "list-2", "title": "新列表"}
    svc.tasks().list().execute.return_value = {
        "items": tasks or [
            {"id": "task-1", "title": "写报告", "status": "needsAction"}
        ]
    }
    svc.tasks().insert().execute.return_value = {"id": "task-2", "title": "新任务"}
    svc.tasks().patch().execute.return_value = {"id": "task-1", "title": "写报告", "status": "completed"}
    svc.tasks().delete().execute.return_value = None
    return svc


@patch("tools.tasks.get_credentials")
@patch("tools.tasks.build")
def test_list_task_lists(mock_build, mock_creds):
    mock_build.return_value = _make_tasks_service()
    from tools.tasks import list_task_lists
    result = list_task_lists.invoke({})
    assert "list-1" in result
    assert "工作" in result


@patch("tools.tasks.get_credentials")
@patch("tools.tasks.build")
def test_list_task_lists_empty(mock_build, mock_creds):
    mock_build.return_value = _make_tasks_service(task_lists=[])
    from tools.tasks import list_task_lists
    result = list_task_lists.invoke({})
    assert "没有" in result


@patch("tools.tasks.get_credentials")
@patch("tools.tasks.build")
def test_create_task_list(mock_build, mock_creds):
    mock_build.return_value = _make_tasks_service()
    from tools.tasks import create_task_list
    result = create_task_list.invoke({"title": "新列表"})
    assert "list-2" in result or "新列表" in result


@patch("tools.tasks.get_credentials")
@patch("tools.tasks.build")
def test_list_tasks(mock_build, mock_creds):
    mock_build.return_value = _make_tasks_service()
    from tools.tasks import list_tasks
    result = list_tasks.invoke({"task_list_id": "list-1", "show_completed": False})
    assert "task-1" in result
    assert "写报告" in result


@patch("tools.tasks.get_credentials")
@patch("tools.tasks.build")
def test_list_tasks_empty(mock_build, mock_creds):
    mock_build.return_value = _make_tasks_service(tasks=[])
    from tools.tasks import list_tasks
    result = list_tasks.invoke({"task_list_id": "list-1", "show_completed": False})
    assert "没有" in result


@patch("tools.tasks.get_credentials")
@patch("tools.tasks.build")
def test_create_task(mock_build, mock_creds):
    mock_build.return_value = _make_tasks_service()
    from tools.tasks import create_task
    result = create_task.invoke({
        "task_list_id": "list-1",
        "title": "新任务",
        "notes": None,
        "due": None,
    })
    assert "task-2" in result or "新任务" in result


@patch("tools.tasks.get_credentials")
@patch("tools.tasks.build")
def test_create_task_with_due(mock_build, mock_creds):
    mock_build.return_value = _make_tasks_service()
    from tools.tasks import create_task
    result = create_task.invoke({
        "task_list_id": "list-1",
        "title": "截止任务",
        "notes": "备注",
        "due": "2026-04-25",
    })
    assert "task-2" in result or "截止任务" in result or "新任务" in result


@patch("tools.tasks.get_credentials")
@patch("tools.tasks.build")
def test_complete_task(mock_build, mock_creds):
    mock_build.return_value = _make_tasks_service()
    from tools.tasks import complete_task
    result = complete_task.invoke({"task_list_id": "list-1", "task_id": "task-1"})
    assert "完成" in result or "completed" in result.lower()


@patch("tools.tasks.get_credentials")
@patch("tools.tasks.build")
def test_delete_task(mock_build, mock_creds):
    mock_build.return_value = _make_tasks_service()
    from tools.tasks import delete_task
    result = delete_task.invoke({"task_list_id": "list-1", "task_id": "task-1"})
    assert "删除" in result or "task-1" in result


@patch("tools.tasks.get_credentials")
def test_no_auth_returns_message(mock_creds):
    from auth.google_oauth import NeedsAuthorizationError
    mock_creds.side_effect = NeedsAuthorizationError("no token")
    from tools.tasks import list_task_lists
    result = list_task_lists.invoke({})
    assert "授权" in result or "auth" in result.lower()
```

- [ ] **Step 2: 运行测试，确认全部失败**

```bash
conda run -n agent python -m pytest Cliriux/tests/test_tasks_tools.py -v
```

预期：10 个 ERROR（ImportError: cannot import `tools.tasks`）。

- [ ] **Step 3: 创建 `tools/tasks.py`**

```python
# tools/tasks.py
from __future__ import annotations

from typing import Optional

from googleapiclient.discovery import build
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from auth.google_oauth import get_credentials, NeedsAuthorizationError

_AUTH_MSG = "需要授权 Google，请访问 /auth/google 完成授权。"


def _service():
    return build("tasks", "v1", credentials=get_credentials())


# ── List Task Lists ────────────────────────────────────────────────────────────

@tool("list_task_lists")
def list_task_lists() -> str:
    """列出所有 Google Tasks 任务列表，返回 ID 和标题。"""
    try:
        svc = _service()
    except NeedsAuthorizationError:
        return _AUTH_MSG

    result = svc.tasklists().list().execute()
    items = result.get("items", [])
    if not items:
        return "没有任务列表。"
    return "\n".join(f"[{tl['id']}] {tl['title']}" for tl in items)


# ── Create Task List ────────────────────────────────────────────────────────────

class CreateTaskListInput(BaseModel):
    title: str = Field(..., description="新任务列表的标题")


@tool("create_task_list", args_schema=CreateTaskListInput)
def create_task_list(title: str) -> str:
    """新建一个 Google Tasks 任务列表。"""
    try:
        svc = _service()
    except NeedsAuthorizationError:
        return _AUTH_MSG

    result = svc.tasklists().insert(body={"title": title}).execute()
    return f"已创建任务列表「{result['title']}」(id: {result['id']})"


# ── List Tasks ──────────────────────────────────────────────────────────────────

class ListTasksInput(BaseModel):
    task_list_id: str = Field(..., description="任务列表 ID，由 list_task_lists 返回")
    show_completed: bool = Field(False, description="是否显示已完成任务，默认 False")


@tool("list_tasks", args_schema=ListTasksInput)
def list_tasks(task_list_id: str, show_completed: bool = False) -> str:
    """列出指定任务列表中的任务。"""
    try:
        svc = _service()
    except NeedsAuthorizationError:
        return _AUTH_MSG

    result = svc.tasks().list(
        tasklist=task_list_id,
        showCompleted=show_completed,
        showHidden=show_completed,
    ).execute()
    items = result.get("items", [])
    if not items:
        return "该列表中没有任务。"

    lines = []
    for t in items:
        status = "✓" if t.get("status") == "completed" else "○"
        due = f" [截止: {t['due'][:10]}]" if t.get("due") else ""
        notes = f" — {t['notes']}" if t.get("notes") else ""
        lines.append(f"{status} [{t['id']}] {t['title']}{due}{notes}")
    return "\n".join(lines)


# ── Create Task ─────────────────────────────────────────────────────────────────

class CreateTaskInput(BaseModel):
    task_list_id: str = Field(..., description="任务列表 ID")
    title: str = Field(..., description="任务标题")
    notes: Optional[str] = Field(None, description="任务备注（可选）")
    due: Optional[str] = Field(None, description="截止日期，格式 YYYY-MM-DD（不含时间，为 Tasks API 限制）")


@tool("create_task", args_schema=CreateTaskInput)
def create_task(task_list_id: str, title: str,
                notes: Optional[str] = None, due: Optional[str] = None) -> str:
    """在指定列表中创建新任务。精确时间提醒请用 set_reminder 而非 due。"""
    try:
        svc = _service()
    except NeedsAuthorizationError:
        return _AUTH_MSG

    body: dict = {"title": title}
    if notes:
        body["notes"] = notes
    if due:
        body["due"] = f"{due}T00:00:00.000Z"

    result = svc.tasks().insert(tasklist=task_list_id, body=body).execute()
    return f"已创建任务「{result['title']}」(id: {result['id']})"


# ── Complete / Delete Task ──────────────────────────────────────────────────────

class TaskActionInput(BaseModel):
    task_list_id: str = Field(..., description="任务列表 ID")
    task_id: str = Field(..., description="任务 ID，由 list_tasks 返回")


@tool("complete_task", args_schema=TaskActionInput)
def complete_task(task_list_id: str, task_id: str) -> str:
    """将任务标记为已完成。"""
    try:
        svc = _service()
    except NeedsAuthorizationError:
        return _AUTH_MSG

    result = svc.tasks().patch(
        tasklist=task_list_id, task=task_id,
        body={"status": "completed"},
    ).execute()
    return f"任务「{result['title']}」已标记为完成。"


@tool("delete_task", args_schema=TaskActionInput)
def delete_task(task_list_id: str, task_id: str) -> str:
    """永久删除指定任务。"""
    try:
        svc = _service()
    except NeedsAuthorizationError:
        return _AUTH_MSG

    svc.tasks().delete(tasklist=task_list_id, task=task_id).execute()
    return f"任务 {task_id} 已删除。"
```

- [ ] **Step 4: 运行测试，确认全部通过**

```bash
conda run -n agent python -m pytest Cliriux/tests/test_tasks_tools.py -v
```

预期：10 passed。

- [ ] **Step 5: 提交**

```bash
git -C Cliriux add tools/tasks.py tests/test_tasks_tools.py
git -C Cliriux commit -m "[feat] add Google Tasks tools (list-lists, create-list, list, create, complete, delete)"
```

---

## Task 4: 注册工具

**Files:**
- Modify: `tools/__init__.py`

- [ ] **Step 1: 写失败测试（内联在 test_oauth_scopes.py 追加）**

在 `tests/test_oauth_scopes.py` 末尾追加：

```python
def test_gmail_tools_registered():
    from tools import TOOLS
    names = [t.name for t in TOOLS]
    for name in ["list_emails", "search_emails", "read_email", "send_email"]:
        assert name in names, f"工具 {name} 未注册"


def test_tasks_tools_registered():
    from tools import TOOLS
    names = [t.name for t in TOOLS]
    for name in ["list_task_lists", "create_task_list", "list_tasks",
                 "create_task", "complete_task", "delete_task"]:
        assert name in names, f"工具 {name} 未注册"
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
conda run -n agent python -m pytest Cliriux/tests/test_oauth_scopes.py::test_gmail_tools_registered Cliriux/tests/test_oauth_scopes.py::test_tasks_tools_registered -v
```

预期：2 FAIL（工具未注册）。

- [ ] **Step 3: 修改 `tools/__init__.py`**

将：
```python
from typing import Any, Callable

from .calculator import calculator
from .get_weather import get_weather
from .translator import translator
from .web_reader import web_reader
from .web_searcher import web_searcher
from .file_system_search import file_system_search
from .save_preference import make_save_preference
from .calendar import (
    create_calendar_event,
    list_calendar_events,
    update_calendar_event,
    delete_calendar_event,
    find_free_slots,
)
from .reminder import set_reminder, list_reminders, delete_reminder

TOOLS: list[Callable[..., Any]] = [
    calculator,
    get_weather,
    translator,
    web_reader,
    web_searcher,
    file_system_search,
    create_calendar_event,
    list_calendar_events,
    update_calendar_event,
    delete_calendar_event,
    find_free_slots,
    set_reminder,
    list_reminders,
    delete_reminder,
]
```

替换为：
```python
from typing import Any, Callable

from .calculator import calculator
from .get_weather import get_weather
from .translator import translator
from .web_reader import web_reader
from .web_searcher import web_searcher
from .file_system_search import file_system_search
from .save_preference import make_save_preference
from .calendar import (
    create_calendar_event,
    list_calendar_events,
    update_calendar_event,
    delete_calendar_event,
    find_free_slots,
)
from .reminder import set_reminder, list_reminders, delete_reminder
from .gmail import list_emails, search_emails, read_email, send_email
from .tasks import (
    list_task_lists,
    create_task_list,
    list_tasks,
    create_task,
    complete_task,
    delete_task,
)

TOOLS: list[Callable[..., Any]] = [
    calculator,
    get_weather,
    translator,
    web_reader,
    web_searcher,
    file_system_search,
    create_calendar_event,
    list_calendar_events,
    update_calendar_event,
    delete_calendar_event,
    find_free_slots,
    set_reminder,
    list_reminders,
    delete_reminder,
    list_emails,
    search_emails,
    read_email,
    send_email,
    list_task_lists,
    create_task_list,
    list_tasks,
    create_task,
    complete_task,
    delete_task,
]
```

- [ ] **Step 4: 运行全部测试，确认无回归**

```bash
conda run -n agent python -m pytest Cliriux/tests/test_oauth_scopes.py Cliriux/tests/test_gmail_tools.py Cliriux/tests/test_tasks_tools.py Cliriux/tests/test_calendar_tools.py Cliriux/tests/test_reminder_tools.py -v
```

预期：全部通过（6 + 8 + 10 + 6 + 3 = 33 tests passed）。

- [ ] **Step 5: 提交**

```bash
git -C Cliriux add tools/__init__.py tests/test_oauth_scopes.py
git -C Cliriux commit -m "[feat] register Gmail and Tasks tools in TOOLS list"
```

---

## 授权说明（实施后用户操作）

新工具实装后，需要重新授权以获取 Gmail + Tasks scope：

```bash
# 删除旧 token（scope 已变更，旧 token 无法用于新 API）
rm Cliriux/auth/token.json

# 访问授权链接（确保 server 运行中）
# http://localhost:8000/auth/google
# 在浏览器完成授权后，新 token 自动保存
```
