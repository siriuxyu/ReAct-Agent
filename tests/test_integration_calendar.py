import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch, AsyncMock


def test_calendar_tools_registered_in_tools_list():
    """calendar 和 reminder 工具应出现在 TOOLS 列表中。"""
    from tools import TOOLS
    tool_names = [t.name for t in TOOLS]
    assert "create_calendar_event" in tool_names
    assert "list_calendar_events" in tool_names
    assert "update_calendar_event" in tool_names
    assert "delete_calendar_event" in tool_names
    assert "find_free_slots" in tool_names
    assert "set_reminder" in tool_names
    assert "list_reminders" in tool_names
    assert "delete_reminder" in tool_names


@pytest.mark.asyncio
async def test_server_lifespan_starts_scheduler(tmp_path):
    """FastAPI lifespan 应启动 scheduler。"""
    with patch.dict(os.environ, {"SCHEDULER_DB_PATH": str(tmp_path / "test.db")}):
        with patch("services.scheduler.start", new_callable=AsyncMock) as mock_start, \
             patch("services.scheduler.stop", new_callable=AsyncMock) as mock_stop:
            import importlib
            import server
            importlib.reload(server)
            from fastapi.testclient import TestClient
            with TestClient(server.api):
                mock_start.assert_called_once()
            mock_stop.assert_called_once()
