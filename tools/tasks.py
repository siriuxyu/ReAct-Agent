from __future__ import annotations

from typing import Optional

try:
    from googleapiclient.discovery import build
except ImportError:
    build = None
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from auth.google_oauth import get_credentials, NeedsAuthorizationError
from .result_utils import tool_ok

_AUTH_MSG = "需要授权 Google，请访问 /auth/google 完成授权。"


def _service():
    creds = get_credentials()
    if build is None:
        raise RuntimeError("googleapiclient is not installed")
    return build("tasks", "v1", credentials=creds)


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
        return tool_ok("list_task_lists", "没有任务列表。", items=[])
    summary = "\n".join(f"[{tl['id']}] {tl['title']}" for tl in items)
    return tool_ok("list_task_lists", summary, items=items)


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
    return tool_ok(
        "create_task_list",
        f"已创建任务列表「{result['title']}」(id: {result['id']})",
        task_list=result,
    )


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
        return tool_ok("list_tasks", "该列表中没有任务。", items=[], task_list_id=task_list_id)

    lines = []
    for t in items:
        status = "✓" if t.get("status") == "completed" else "○"
        due = f" [截止: {t['due'][:10]}]" if t.get("due") else ""
        notes = f" — {t['notes']}" if t.get("notes") else ""
        lines.append(f"{status} [{t['id']}] {t['title']}{due}{notes}")
    return tool_ok(
        "list_tasks",
        "\n".join(lines),
        task_list_id=task_list_id,
        items=items,
    )


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
    return tool_ok(
        "create_task",
        f"已创建任务「{result['title']}」(id: {result['id']})",
        task_list_id=task_list_id,
        task=result,
    )


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
    return tool_ok(
        "complete_task",
        f"任务「{result['title']}」已标记为完成。",
        task_list_id=task_list_id,
        task=result,
    )


@tool("delete_task", args_schema=TaskActionInput)
def delete_task(task_list_id: str, task_id: str) -> str:
    """永久删除指定任务。"""
    try:
        svc = _service()
    except NeedsAuthorizationError:
        return _AUTH_MSG

    svc.tasks().delete(tasklist=task_list_id, task=task_id).execute()
    return tool_ok(
        "delete_task",
        f"任务 {task_id} 已删除。",
        task_list_id=task_list_id,
        task_id=task_id,
    )
