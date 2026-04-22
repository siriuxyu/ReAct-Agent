import asyncio
import os
import uuid
from datetime import datetime
from typing import Callable, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.executors.asyncio import AsyncIOExecutor

_scheduler: Optional[AsyncIOScheduler] = None


def _build_scheduler() -> AsyncIOScheduler:
    db_path = os.environ.get("SCHEDULER_DB_PATH", "./scheduler.db")
    return AsyncIOScheduler(
        jobstores={
            "default": SQLAlchemyJobStore(url=f"sqlite:///{db_path}"),
            "memory": MemoryJobStore(),
        },
        executors={"default": AsyncIOExecutor()},
        job_defaults={"misfire_grace_time": 60},
    )


def get_scheduler() -> AsyncIOScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = _build_scheduler()
    return _scheduler


async def start() -> None:
    global _scheduler
    _scheduler = _build_scheduler()
    _scheduler.start()


async def stop() -> None:
    sched = get_scheduler()
    if sched.running:
        sched.shutdown(wait=False)
        # Yield to event loop so APScheduler can complete shutdown
        await asyncio.sleep(0)


def add_job(func: Callable, run_at: datetime, job_id: Optional[str] = None, args: list = None) -> str:
    job_id = job_id or str(uuid.uuid4())
    scheduler = get_scheduler()
    # Try SQLite jobstore first; fall back to memory if function cannot be serialized
    try:
        scheduler.add_job(
            func,
            trigger="date",
            run_date=run_at,
            id=job_id,
            args=args or [],
            replace_existing=True,
        )
    except (ValueError, TypeError):
        scheduler.add_job(
            func,
            trigger="date",
            run_date=run_at,
            id=job_id,
            jobstore="memory",
            args=args or [],
            replace_existing=True,
        )
    return job_id


def remove_job(job_id: str) -> bool:
    try:
        get_scheduler().remove_job(job_id)
        return True
    except Exception:
        return False


def list_jobs() -> list[dict]:
    return [
        {
            "id": job.id,
            "name": job.name,
            "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
        }
        for job in get_scheduler().get_jobs()
    ]
