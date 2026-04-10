from __future__ import annotations
from datetime import datetime, timezone

from pydantic import BaseModel, Field
from langchain_core.tools import tool

try:
    import services.scheduler as scheduler
except ImportError:
    class _MissingScheduler:
        def add_job(self, *args, **kwargs):
            raise RuntimeError("scheduler dependencies are not installed")

        def list_jobs(self):
            return []

        def remove_job(self, job_id: str):
            return False

    scheduler = _MissingScheduler()

try:
    from services.notifier import send
except ImportError:
    async def send(*args, **kwargs):
        raise RuntimeError("notifier dependencies are not installed")
from .result_utils import tool_error, tool_ok


# ── Set Reminder ──────────────────────────────────────────────────────────────

class SetReminderInput(BaseModel):
    message: str = Field(..., description="提醒内容")
    remind_at: str = Field(..., description="提醒时间，ISO 8601 格式，含时区，如 2026-04-22T15:00:00+08:00")
    channels: list[str] = Field(
        default_factory=lambda: ["telegram"],
        description="通知渠道列表，可选值：telegram、email、all",
    )


@tool("set_reminder", args_schema=SetReminderInput)
def set_reminder(message: str, remind_at: str, channels: list[str] = None) -> str:
    """设定一个定时提醒，到点通过指定渠道推送消息。"""
    channels = channels or ["telegram"]
    try:
        run_date = datetime.fromisoformat(remind_at)
        if run_date.tzinfo is None:
            from zoneinfo import ZoneInfo
            import os
            run_date = run_date.replace(tzinfo=ZoneInfo(os.environ.get("TIMEZONE", "Asia/Shanghai")))
    except ValueError as e:
        return tool_error("set_reminder", f"时间格式错误：{e}", remind_at=remind_at)

    if run_date <= datetime.now(timezone.utc):
        return tool_error(
            "set_reminder",
            "提醒时间不能是过去的时间，请指定未来的时间。",
            remind_at=remind_at,
        )

    job_id = scheduler.add_job(
        send,
        run_date,
        args=[message, channels],
    )
    summary = f"提醒已设定（id: {job_id}）：将在 {run_date.strftime('%Y-%m-%d %H:%M %Z')} 提醒你「{message}」"
    return tool_ok(
        "set_reminder",
        summary,
        job_id=job_id,
        message=message,
        remind_at=run_date.isoformat(),
        channels=channels,
    )


# ── List Reminders ────────────────────────────────────────────────────────────

class ListRemindersInput(BaseModel):
    pass


@tool("list_reminders", args_schema=ListRemindersInput)
def list_reminders() -> str:
    """列出所有尚未触发的提醒。"""
    jobs = scheduler.list_jobs()
    if not jobs:
        return tool_ok("list_reminders", "当前没有待触发的提醒。", items=[])
    items = [
        {"id": j["id"], "next_run_time": j["next_run_time"]}
        for j in jobs
    ]
    lines = [f"- [{item['id']}] 下次触发：{item['next_run_time']}" for item in items]
    return tool_ok(
        "list_reminders",
        "待触发的提醒：\n" + "\n".join(lines),
        items=items,
    )


# ── Delete Reminder ───────────────────────────────────────────────────────────

class DeleteReminderInput(BaseModel):
    job_id: str = Field(..., description="要取消的提醒 ID")


@tool("delete_reminder", args_schema=DeleteReminderInput)
def delete_reminder(job_id: str) -> str:
    """取消一个尚未触发的提醒。"""
    ok = scheduler.remove_job(job_id)
    if ok:
        return tool_ok("delete_reminder", f"提醒 {job_id} 已删除。", job_id=job_id)
    return tool_error(
        "delete_reminder",
        f"找不到提醒 {job_id}，可能已触发或 ID 不正确。",
        job_id=job_id,
    )
