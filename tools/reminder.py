from __future__ import annotations
from datetime import datetime, timezone

from pydantic import BaseModel, Field
from langchain_core.tools import tool

import services.scheduler as scheduler
from services.notifier import send


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
        return f"时间格式错误：{e}"

    if run_date <= datetime.now(timezone.utc):
        return "提醒时间不能是过去的时间，请指定未来的时间。"

    job_id = scheduler.add_job(
        send,
        run_date,
        args=[message, channels],
    )
    return f"提醒已设定（id: {job_id}）：将在 {run_date.strftime('%Y-%m-%d %H:%M %Z')} 提醒你「{message}」"


# ── List Reminders ────────────────────────────────────────────────────────────

class ListRemindersInput(BaseModel):
    pass


@tool("list_reminders", args_schema=ListRemindersInput)
def list_reminders() -> str:
    """列出所有尚未触发的提醒。"""
    jobs = scheduler.list_jobs()
    if not jobs:
        return "当前没有待触发的提醒。"
    lines = [f"- [{j['id']}] 下次触发：{j['next_run_time']}" for j in jobs]
    return "待触发的提醒：\n" + "\n".join(lines)


# ── Delete Reminder ───────────────────────────────────────────────────────────

class DeleteReminderInput(BaseModel):
    job_id: str = Field(..., description="要取消的提醒 ID")


@tool("delete_reminder", args_schema=DeleteReminderInput)
def delete_reminder(job_id: str) -> str:
    """取消一个尚未触发的提醒。"""
    ok = scheduler.remove_job(job_id)
    if ok:
        return f"提醒 {job_id} 已删除。"
    return f"找不到提醒 {job_id}，可能已触发或 ID 不正确。"
