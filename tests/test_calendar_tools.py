import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch, MagicMock


def _make_service(events=None, busy=None):
    """构造 mock GCal service。"""
    service = MagicMock()
    service.events().list().execute.return_value = {
        "items": events or []
    }
    service.events().insert().execute.return_value = {
        "id": "evt-123", "summary": "Test Event",
        "htmlLink": "https://calendar.google.com/event?eid=evt-123"
    }
    service.events().patch().execute.return_value = {"id": "evt-123", "summary": "Updated"}
    service.events().delete().execute.return_value = None
    service.freebusy().query().execute.return_value = {
        "calendars": {"primary": {"busy": busy or []}}
    }
    return service


@patch("tools.calendar.get_credentials")
@patch("tools.calendar.build")
def test_create_calendar_event(mock_build, mock_creds):
    mock_build.return_value = _make_service()
    from tools.calendar import create_calendar_event
    result = create_calendar_event.invoke({
        "title": "Team Meeting",
        "start": "2026-04-22T14:00:00+08:00",
        "end": "2026-04-22T15:00:00+08:00",
        "description": "",
        "attendees": [],
        "recurrence": "",
    })
    assert "evt-123" in result or "Team Meeting" in result


@patch("tools.calendar.get_credentials")
@patch("tools.calendar.build")
def test_list_calendar_events(mock_build, mock_creds):
    mock_build.return_value = _make_service(events=[
        {"id": "e1", "summary": "Standup",
         "start": {"dateTime": "2026-04-22T09:00:00+08:00"},
         "end": {"dateTime": "2026-04-22T09:30:00+08:00"}}
    ])
    from tools.calendar import list_calendar_events
    result = list_calendar_events.invoke({
        "time_min": "2026-04-22T00:00:00+08:00",
        "time_max": "2026-04-22T23:59:59+08:00",
        "max_results": 10,
    })
    assert "Standup" in result


@patch("tools.calendar.get_credentials")
@patch("tools.calendar.build")
def test_delete_calendar_event(mock_build, mock_creds):
    mock_build.return_value = _make_service()
    from tools.calendar import delete_calendar_event
    result = delete_calendar_event.invoke({"event_id": "evt-123"})
    assert "deleted" in result.lower() or "evt-123" in result


@patch("tools.calendar.get_credentials")
@patch("tools.calendar.build")
def test_find_free_slots(mock_build, mock_creds):
    mock_build.return_value = _make_service(busy=[
        {"start": "2026-04-22T09:00:00Z", "end": "2026-04-22T10:00:00Z"}
    ])
    from tools.calendar import find_free_slots
    result = find_free_slots.invoke({"date": "2026-04-22", "duration_minutes": 30})
    assert isinstance(result, str) and len(result) > 0


@patch("tools.calendar.get_credentials")
def test_no_auth_returns_message(mock_creds):
    from auth.google_oauth import NeedsAuthorizationError
    mock_creds.side_effect = NeedsAuthorizationError("no token")
    from tools.calendar import list_calendar_events
    result = list_calendar_events.invoke({
        "time_min": "2026-04-22T00:00:00+08:00",
        "time_max": "2026-04-22T23:59:59+08:00",
        "max_results": 5,
    })
    assert "auth" in result.lower() or "授权" in result


@patch("tools.calendar.get_credentials")
@patch("tools.calendar.build")
def test_update_calendar_event(mock_build, mock_creds):
    mock_build.return_value = _make_service()
    from tools.calendar import update_calendar_event
    result = update_calendar_event.invoke({
        "event_id": "evt-123",
        "title": "Updated Meeting",
        "start": None,
        "end": None,
        "description": None,
    })
    assert "evt-123" in result or "Updated" in result
