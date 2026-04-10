from typing import Any, Callable

from .metadata import ToolMetadata, get_tool_metadata, list_tool_metadata
from .registry import ToolRegistry, create_tool_registry, get_tool_registry

_TOOLS: list[Callable[..., Any]] | None = None


def _load_tools() -> list[Callable[..., Any]]:
    from .calculator import calculator
    from .get_weather import get_weather
    from .translator import translator
    from .web_reader import web_reader
    from .web_searcher import web_searcher
    from .file_system_search import file_system_search
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

    return [
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


def __getattr__(name: str) -> Any:
    if name == "TOOLS":
        global _TOOLS
        if _TOOLS is None:
            _TOOLS = _load_tools()
        return _TOOLS
    raise AttributeError(name)

__all__ = [
    "TOOLS",
    "ToolMetadata",
    "ToolRegistry",
    "create_tool_registry",
    "get_tool_metadata",
    "get_tool_registry",
    "list_tool_metadata",
]
