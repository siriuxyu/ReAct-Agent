import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone, timedelta


def _future_iso():
    return (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()


@patch("tools.reminder.scheduler")
def test_set_reminder_returns_job_id(mock_sched):
    mock_sched.add_job.return_value = "job-abc"
    from tools.reminder import set_reminder
    result = set_reminder.invoke({
        "message": "记得喝水",
        "remind_at": _future_iso(),
        "channels": ["telegram"],
    })
    assert "job-abc" in result or "记得喝水" in result


@patch("tools.reminder.scheduler")
def test_list_reminders(mock_sched):
    mock_sched.list_jobs.return_value = [
        {"id": "job-1", "name": "reminder", "next_run_time": _future_iso()}
    ]
    from tools.reminder import list_reminders
    result = list_reminders.invoke({})
    assert "job-1" in result


@patch("tools.reminder.scheduler")
def test_delete_reminder_success(mock_sched):
    mock_sched.remove_job.return_value = True
    from tools.reminder import delete_reminder
    result = delete_reminder.invoke({"job_id": "job-1"})
    assert "删除" in result or "job-1" in result


@patch("tools.reminder.scheduler")
def test_delete_reminder_not_found(mock_sched):
    mock_sched.remove_job.return_value = False
    from tools.reminder import delete_reminder
    result = delete_reminder.invoke({"job_id": "no-such-job"})
    assert "找不到" in result or "not found" in result.lower()


def test_set_reminder_past_time_rejected():
    """过去的时间应被拒绝。"""
    from tools.reminder import set_reminder
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    result = set_reminder.invoke({
        "message": "过期提醒",
        "remind_at": past,
        "channels": ["telegram"],
    })
    assert "过去" in result or "past" in result.lower() or "invalid" in result.lower()
