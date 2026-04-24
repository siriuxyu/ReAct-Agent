import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, patch


async def _module_level_noop():
    """模块级函数，可被 APScheduler SQLite jobstore 序列化。"""
    pass


@pytest.mark.asyncio
async def test_scheduler_starts_and_stops(tmp_path):
    """start()/stop() 不抛异常，state 正确切换。"""
    with patch.dict(os.environ, {"SCHEDULER_DB_PATH": str(tmp_path / "test.db")}):
        import importlib
        import services.scheduler as mod
        importlib.reload(mod)
        await mod.start()
        assert mod.get_scheduler().running
        await mod.stop()
        assert not mod.get_scheduler().running


@pytest.mark.asyncio
async def test_add_and_remove_job(tmp_path):
    """add_job 返回 job_id，remove_job 删除它。"""
    with patch.dict(os.environ, {"SCHEDULER_DB_PATH": str(tmp_path / "test.db")}):
        import importlib
        import services.scheduler as mod
        importlib.reload(mod)
        await mod.start()

        async def _noop():
            pass

        run_at = datetime.now(timezone.utc) + timedelta(hours=1)
        job_id = mod.add_job(_noop, run_at, job_id="test-job-1")
        assert job_id == "test-job-1"

        mod.remove_job("test-job-1")
        assert mod.get_scheduler().get_job("test-job-1") is None

        await mod.stop()


@pytest.mark.asyncio
async def test_list_jobs(tmp_path):
    """list_jobs 返回包含已添加 job 的列表。"""
    with patch.dict(os.environ, {"SCHEDULER_DB_PATH": str(tmp_path / "test.db")}):
        import importlib
        import services.scheduler as mod
        importlib.reload(mod)
        await mod.start()

        async def _noop():
            pass

        run_at = datetime.now(timezone.utc) + timedelta(hours=2)
        mod.add_job(_noop, run_at, job_id="list-test-job")
        jobs = mod.list_jobs()
        ids = [j["id"] for j in jobs]
        assert "list-test-job" in ids

        await mod.stop()


@pytest.mark.asyncio
async def test_add_job_writes_to_sqlite(tmp_path):
    """使用模块级函数验证 job 确实写入 SQLite。"""
    import sqlite3
    db_path = tmp_path / "test.db"
    with patch.dict(os.environ, {"SCHEDULER_DB_PATH": str(db_path)}):
        import importlib
        import services.scheduler as mod
        importlib.reload(mod)
        await mod.start()

        run_at = datetime.now(timezone.utc) + timedelta(hours=1)
        job_id = mod.add_job(_module_level_noop, run_at, job_id="sqlite-test-job")

        # 验证 SQLite 文件存在且包含该 job
        assert db_path.exists(), "SQLite 文件应该被创建"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute("SELECT id FROM apscheduler_jobs WHERE id = ?", ("sqlite-test-job",))
        row = cursor.fetchone()
        conn.close()
        assert row is not None, "job 应写入 SQLite"

        await mod.stop()
