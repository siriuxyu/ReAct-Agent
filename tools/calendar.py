from __future__ import annotations
import os
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

try:
    from googleapiclient.discovery import build
except ImportError:
    build = None
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from auth.google_oauth import get_credentials, NeedsAuthorizationError
from .result_utils import tool_ok

_AUTH_MSG = "需要授权 Google Calendar，请访问 /auth/google 完成授权。"
_CALENDAR_ID = lambda: os.environ.get("GOOGLE_CALENDAR_ID", "primary")
_TZ = lambda: ZoneInfo(os.environ.get("TIMEZONE", "Asia/Shanghai"))


def _service():
    creds = get_credentials()
    if build is None:
        raise RuntimeError("googleapiclient is not installed")
    return build("calendar", "v3", credentials=creds)


# ── Create ────────────────────────────────────────────────────────────────────

class CreateEventInput(BaseModel):
    title: str = Field(..., description="事件标题")
    start: str = Field(..., description="开始时间，ISO 8601，含时区，如 2026-04-22T14:00:00+08:00")
    end: str = Field(..., description="结束时间，ISO 8601，含时区")
    description: str = Field("", description="事件描述")
    attendees: list[str] = Field(default_factory=list, description="与会者邮箱列表")
    recurrence: str = Field("", description="重复规则 RRULE 字符串，如 RRULE:FREQ=WEEKLY;BYDAY=MO，留空表示不重复")


@tool("create_calendar_event", args_schema=CreateEventInput)
def create_calendar_event(title: str, start: str, end: str,
                          description: str = "", attendees: list[str] = None,
                          recurrence: str = "") -> str:
    """在 Google Calendar 创建日程事件，支持重复规则和与会者邀请。"""
    try:
        svc = _service()
    except NeedsAuthorizationError:
        return _AUTH_MSG
    body: dict = {
        "summary": title,
        "description": description,
        "start": {"dateTime": start},
        "end": {"dateTime": end},
        "reminders": {"useDefault": False, "overrides": [
            {"method": "email", "minutes": 30},
            {"method": "popup", "minutes": 10},
        ]},
    }
    if attendees:
        body["attendees"] = [{"email": e} for e in attendees]
    if recurrence:
        body["recurrence"] = [recurrence]
    event = svc.events().insert(calendarId=_CALENDAR_ID(), body=body).execute()
    return tool_ok(
        "create_calendar_event",
        f"已创建事件「{event['summary']}」(id: {event['id']})，链接: {event.get('htmlLink', '')}",
        event=event,
    )


# ── List ──────────────────────────────────────────────────────────────────────

class ListEventsInput(BaseModel):
    time_min: str = Field(..., description="查询开始时间，ISO 8601")
    time_max: str = Field(..., description="查询结束时间，ISO 8601")
    max_results: int = Field(10, description="最多返回条数，默认 10")


@tool("list_calendar_events", args_schema=ListEventsInput)
def list_calendar_events(time_min: str, time_max: str, max_results: int = 10) -> str:
    """查询 Google Calendar 中指定时间段内的日程列表。"""
    try:
        svc = _service()
    except NeedsAuthorizationError:
        return _AUTH_MSG
    result = svc.events().list(
        calendarId=_CALENDAR_ID(),
        timeMin=time_min,
        timeMax=time_max,
        maxResults=max_results,
        singleEvents=True,
        orderBy="startTime",
    ).execute()
    items = result.get("items", [])
    if not items:
        return tool_ok(
            "list_calendar_events",
            "该时间段内没有日程。",
            items=[],
            time_min=time_min,
            time_max=time_max,
        )
    lines = []
    for ev in items:
        start = ev["start"].get("dateTime", ev["start"].get("date", ""))
        lines.append(f"- [{ev['id']}] {ev.get('summary', '无标题')} @ {start}")
    return tool_ok(
        "list_calendar_events",
        "\n".join(lines),
        items=items,
        time_min=time_min,
        time_max=time_max,
    )


# ── Update ────────────────────────────────────────────────────────────────────

class UpdateEventInput(BaseModel):
    event_id: str = Field(..., description="事件 ID")
    title: Optional[str] = Field(None, description="新标题（留空不修改）")
    start: Optional[str] = Field(None, description="新开始时间，ISO 8601（留空不修改）")
    end: Optional[str] = Field(None, description="新结束时间，ISO 8601（留空不修改）")
    description: Optional[str] = Field(None, description="新描述（留空不修改）")


@tool("update_calendar_event", args_schema=UpdateEventInput)
def update_calendar_event(event_id: str, title: Optional[str] = None, start: Optional[str] = None,
                          end: Optional[str] = None, description: Optional[str] = None) -> str:
    """修改 Google Calendar 中已有事件的标题、时间或描述。"""
    try:
        svc = _service()
    except NeedsAuthorizationError:
        return _AUTH_MSG
    patch_body: dict = {}
    if title is not None:
        patch_body["summary"] = title
    if start is not None:
        patch_body["start"] = {"dateTime": start}
    if end is not None:
        patch_body["end"] = {"dateTime": end}
    if description is not None:
        patch_body["description"] = description
    if not patch_body:
        return tool_ok("update_calendar_event", "没有要修改的内容。", event_id=event_id)
    event = svc.events().patch(
        calendarId=_CALENDAR_ID(), eventId=event_id, body=patch_body
    ).execute()
    return tool_ok(
        "update_calendar_event",
        f"事件 {event['id']} 已更新：{event.get('summary', '')}",
        event=event,
    )


# ── Delete ────────────────────────────────────────────────────────────────────

class DeleteEventInput(BaseModel):
    event_id: str = Field(..., description="要删除的事件 ID")


@tool("delete_calendar_event", args_schema=DeleteEventInput)
def delete_calendar_event(event_id: str) -> str:
    """删除 Google Calendar 中的指定事件。"""
    try:
        svc = _service()
    except NeedsAuthorizationError:
        return _AUTH_MSG
    svc.events().delete(calendarId=_CALENDAR_ID(), eventId=event_id).execute()
    return tool_ok("delete_calendar_event", f"事件 {event_id} 已删除。", event_id=event_id)


# ── Find Free Slots ───────────────────────────────────────────────────────────

class FindFreeSlotsInput(BaseModel):
    date: str = Field(..., description="查询日期，格式 YYYY-MM-DD")
    duration_minutes: int = Field(..., description="所需空闲时长（分钟）")


@tool("find_free_slots", args_schema=FindFreeSlotsInput)
def find_free_slots(date: str, duration_minutes: int) -> str:
    """查询指定日期中有哪些连续空闲时段可安排事件。"""
    try:
        svc = _service()
    except NeedsAuthorizationError:
        return _AUTH_MSG
    tz = _TZ()
    day = datetime.fromisoformat(date).replace(tzinfo=tz)
    time_min = day.replace(hour=0, minute=0, second=0).isoformat()
    time_max = day.replace(hour=23, minute=59, second=59).isoformat()

    fb = svc.freebusy().query(body={
        "timeMin": time_min,
        "timeMax": time_max,
        "timeZone": str(tz),
        "items": [{"id": _CALENDAR_ID()}],
    }).execute()
    busy = fb["calendars"][_CALENDAR_ID()]["busy"]

    work_start = day.replace(hour=8, minute=0, second=0)
    work_end = day.replace(hour=22, minute=0, second=0)
    busy_periods = [
        (datetime.fromisoformat(b["start"]).astimezone(tz),
         datetime.fromisoformat(b["end"]).astimezone(tz))
        for b in busy
    ]
    busy_periods.sort(key=lambda x: x[0])

    free_slots = []
    cursor = work_start
    for bs, be in busy_periods:
        if cursor < bs and (bs - cursor) >= timedelta(minutes=duration_minutes):
            free_slots.append(f"{cursor.strftime('%H:%M')} - {bs.strftime('%H:%M')}")
        cursor = max(cursor, be)
    if cursor < work_end and (work_end - cursor) >= timedelta(minutes=duration_minutes):
        free_slots.append(f"{cursor.strftime('%H:%M')} - {work_end.strftime('%H:%M')}")

    if not free_slots:
        return tool_ok(
            "find_free_slots",
            f"{date} 没有满足 {duration_minutes} 分钟的空闲时段。",
            date=date,
            duration_minutes=duration_minutes,
            free_slots=[],
        )
    summary = f"{date} 可用时段（工作时间内）：\n" + "\n".join(f"- {s}" for s in free_slots)
    return tool_ok(
        "find_free_slots",
        summary,
        date=date,
        duration_minutes=duration_minutes,
        free_slots=free_slots,
    )
