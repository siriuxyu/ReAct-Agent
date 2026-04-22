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
