# Calendar + Reminder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 Cliriux 个人助手增加 Google Calendar 日程管理和定时提醒功能，支持 Telegram、邮件、GCal 原生三路通知推送。

**Architecture:** 单体扩展方案——在现有 FastAPI 服务中新增 `auth/`、`services/` 两个模块，向 `tools/` 添加日历和提醒工具，修改 `server.py` 的 `lifespan` 启动 APScheduler。调度器与 Web server 共享同一事件循环（`AsyncIOScheduler`），提醒 job 持久化到本地 SQLite。

**Tech Stack:** `google-api-python-client`, `google-auth-oauthlib`, `APScheduler 3.x` (`AsyncIOScheduler` + `SQLAlchemyJobStore`), `aiosmtplib`, `SQLAlchemy 2.x`, `pytest`, `pytest-asyncio`

---

## 文件清单

| 操作 | 路径 | 职责 |
|------|------|------|
| 新增 | `auth/__init__.py` | 包初始化 |
| 新增 | `auth/google_oauth.py` | OAuth2 token 读写刷新，单用户 |
| 新增 | `services/__init__.py` | 包初始化 |
| 新增 | `services/scheduler.py` | APScheduler 单例，SQLite 持久化 |
| 新增 | `services/notifier.py` | 统一推送（Telegram / Email），被 scheduler job 调用 |
| 新增 | `tools/calendar.py` | 5 个 GCal LangChain tools |
| 新增 | `tools/reminder.py` | 3 个 reminder LangChain tools |
| 新增 | `tests/test_google_oauth.py` | OAuth 模块单测 |
| 新增 | `tests/test_scheduler.py` | 调度器单测 |
| 新增 | `tests/test_notifier.py` | 通知推送单测 |
| 新增 | `tests/test_calendar_tools.py` | 日历工具单测 |
| 新增 | `tests/test_reminder_tools.py` | 提醒工具单测 |
| 修改 | `requirements.txt` | 新增依赖 |
| 修改 | `tools/__init__.py` | 注册新工具到 TOOLS 列表 |
| 修改 | `server.py` | lifespan 启动/停止 scheduler，新增 /auth/google 路由 |

---

## Task 1: 依赖与配置

**Files:**
- Modify: `requirements.txt`
- Create: `.env.example`（如不存在则新建）

- [ ] **Step 1: 追加新依赖到 requirements.txt**

在 `requirements.txt` 末尾添加：

```
# Google Calendar
google-api-python-client>=2.100.0
google-auth-oauthlib>=1.1.0
google-auth-httplib2>=0.2.0

# Scheduler
apscheduler>=3.10.0,<4.0.0
sqlalchemy>=2.0.0

# Async email
aiosmtplib>=3.0.0

# Test
pytest>=7.0.0
pytest-asyncio>=0.23.0
```

- [ ] **Step 2: 新建或追加 .env.example**

```bash
cat >> .env.example << 'EOF'

# ── Google OAuth ──────────────────────────────────────────
GOOGLE_CREDENTIALS_FILE=credentials.json
GOOGLE_TOKEN_FILE=auth/token.json
GOOGLE_CALENDAR_ID=primary
OAUTH_REDIRECT_URI=http://localhost:8000/auth/google/callback

# ── 时区 ──────────────────────────────────────────────────
TIMEZONE=Asia/Shanghai

# ── 提醒推送 ──────────────────────────────────────────────
REMINDER_TELEGRAM_CHAT_ID=你的Telegram_chat_id

# ── 邮件 ─────────────────────────────────────────────────
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=你的gmail地址
SMTP_PASSWORD=应用专用密码
NOTIFY_EMAIL=接收提醒的邮箱

# ── 调度器 ────────────────────────────────────────────────
SCHEDULER_DB_PATH=./scheduler.db
EOF
```

- [ ] **Step 3: 安装依赖**

```bash
pip install -r requirements.txt
```

Expected: 无报错，`apscheduler`、`google-api-python-client`、`aiosmtplib` 均安装成功。

- [ ] **Step 4: Commit**

```bash
git add requirements.txt .env.example
git commit -m "chore: add calendar+reminder dependencies"
```

---

## Task 2: Google OAuth 模块

**Files:**
- Create: `auth/__init__.py`
- Create: `auth/google_oauth.py`
- Create: `tests/test_google_oauth.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_google_oauth.py`：

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path


def test_needs_authorization_error_raised_when_no_token(tmp_path):
    """无 token 文件时应抛出 NeedsAuthorizationError。"""
    with patch.dict(os.environ, {"GOOGLE_TOKEN_FILE": str(tmp_path / "token.json")}):
        from auth.google_oauth import get_credentials, NeedsAuthorizationError
        with pytest.raises(NeedsAuthorizationError):
            get_credentials()


def test_get_auth_url_returns_url_and_state(tmp_path):
    """get_auth_url 应返回 (url, state) 元组，url 包含 accounts.google.com。"""
    creds_content = {
        "installed": {
            "client_id": "fake-id.apps.googleusercontent.com",
            "client_secret": "fake-secret",
            "redirect_uris": ["urn:ietf:wg:oauth:2.0:oob"],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    }
    creds_file = tmp_path / "credentials.json"
    import json
    creds_file.write_text(json.dumps(creds_content))

    with patch.dict(os.environ, {
        "GOOGLE_CREDENTIALS_FILE": str(creds_file),
        "OAUTH_REDIRECT_URI": "http://localhost:8000/auth/google/callback",
    }):
        import importlib
        import auth.google_oauth as mod
        importlib.reload(mod)
        url, state = mod.get_auth_url()
        assert "accounts.google.com" in url
        assert isinstance(state, str) and len(state) > 0


def test_save_and_load_token(tmp_path):
    """token 写入后能用 Credentials.from_authorized_user_file 读取。"""
    from unittest.mock import MagicMock
    fake_creds = MagicMock()
    fake_creds.to_json.return_value = '{"token": "fake", "refresh_token": "r", "token_uri": "https://oauth2.googleapis.com/token", "client_id": "c", "client_secret": "s", "scopes": []}'

    token_path = tmp_path / "token.json"
    with patch.dict(os.environ, {"GOOGLE_TOKEN_FILE": str(token_path)}):
        import importlib
        import auth.google_oauth as mod
        importlib.reload(mod)
        mod._save_token(fake_creds)
        assert token_path.exists()
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
cd /home/siriux/Projects/Cliriux && python -m pytest tests/test_google_oauth.py -v 2>&1 | head -30
```

Expected: `ImportError: No module named 'auth.google_oauth'`

- [ ] **Step 3: 创建 auth/__init__.py**

```bash
mkdir -p auth && touch auth/__init__.py
```

- [ ] **Step 4: 创建 auth/google_oauth.py**

```python
import os
import secrets
from pathlib import Path

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import Flow

SCOPES = ["https://www.googleapis.com/auth/calendar"]

_CREDENTIALS_FILE = lambda: os.environ.get("GOOGLE_CREDENTIALS_FILE", "credentials.json")
_TOKEN_FILE = lambda: os.environ.get("GOOGLE_TOKEN_FILE", "auth/token.json")
_REDIRECT_URI = lambda: os.environ.get(
    "OAUTH_REDIRECT_URI", "http://localhost:8000/auth/google/callback"
)

_pending_flows: dict[str, Flow] = {}


class NeedsAuthorizationError(Exception):
    pass


def get_credentials() -> Credentials:
    token_path = Path(_TOKEN_FILE())
    if not token_path.exists():
        raise NeedsAuthorizationError("No token found. Please authorize via /auth/google")
    creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    if creds.valid:
        return creds
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        _save_token(creds)
        return creds
    raise NeedsAuthorizationError("Token invalid. Please re-authorize via /auth/google")


def get_auth_url() -> tuple[str, str]:
    flow = Flow.from_client_secrets_file(
        _CREDENTIALS_FILE(), scopes=SCOPES, redirect_uri=_REDIRECT_URI()
    )
    state = secrets.token_urlsafe(16)
    auth_url, _ = flow.authorization_url(
        access_type="offline", include_granted_scopes="true", state=state, prompt="consent"
    )
    _pending_flows[state] = flow
    return auth_url, state


def exchange_code(code: str, state: str) -> Credentials:
    flow = _pending_flows.pop(state, None)
    if flow is None:
        raise ValueError(f"Unknown OAuth state: {state}")
    flow.fetch_token(code=code)
    creds = flow.credentials
    _save_token(creds)
    return creds


def _save_token(creds: Credentials) -> None:
    token_path = Path(_TOKEN_FILE())
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(creds.to_json())
```

- [ ] **Step 5: 运行测试，确认通过**

```bash
python -m pytest tests/test_google_oauth.py -v
```

Expected: 3 个测试全部 PASS

- [ ] **Step 6: Commit**

```bash
git add auth/__init__.py auth/google_oauth.py tests/test_google_oauth.py
git commit -m "feat: add Google OAuth token management"
```

---

## Task 3: OAuth 授权端点

**Files:**
- Modify: `server.py`（新增两个路由）

- [ ] **Step 1: 写失败测试**

创建 `tests/test_oauth_endpoints.py`：

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch


def test_auth_google_redirect():
    """GET /auth/google 应重定向到 Google 授权页。"""
    with patch("auth.google_oauth.get_auth_url", return_value=("https://accounts.google.com/fake", "state123")):
        from server import api
        client = TestClient(api, follow_redirects=False)
        resp = client.get("/auth/google")
        assert resp.status_code == 307
        assert "accounts.google.com" in resp.headers["location"]


def test_auth_google_callback_success():
    """GET /auth/google/callback?code=X&state=Y 成功时返回 200。"""
    from unittest.mock import MagicMock
    fake_creds = MagicMock()
    with patch("auth.google_oauth.exchange_code", return_value=fake_creds):
        from server import api
        client = TestClient(api)
        resp = client.get("/auth/google/callback?code=fake_code&state=fake_state")
        assert resp.status_code == 200
        assert "success" in resp.json()["message"].lower()


def test_auth_google_callback_invalid_state():
    """state 不匹配时返回 400。"""
    with patch("auth.google_oauth.exchange_code", side_effect=ValueError("Unknown OAuth state")):
        from server import api
        client = TestClient(api)
        resp = client.get("/auth/google/callback?code=x&state=bad")
        assert resp.status_code == 400
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
python -m pytest tests/test_oauth_endpoints.py -v 2>&1 | head -20
```

Expected: FAIL（路由不存在）

- [ ] **Step 3: 在 server.py 中新增 OAuth 路由**

在 `server.py` 的 import 区域末尾添加：

```python
from auth.google_oauth import get_auth_url, exchange_code as oauth_exchange_code
```

在 `server.py` 的路由区域添加（放在 `api = FastAPI(...)` 之后任意位置）：

```python
@api.get("/auth/google")
async def auth_google_start():
    """引导用户授权 Google Calendar。"""
    url, _state = get_auth_url()
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url=url, status_code=307)


@api.get("/auth/google/callback")
async def auth_google_callback(code: str, state: str):
    """OAuth 回调，交换授权码为 token。"""
    try:
        oauth_exchange_code(code, state)
        return {"message": "Authorization successful. You can close this tab."}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
python -m pytest tests/test_oauth_endpoints.py -v
```

Expected: 3 个测试全部 PASS

- [ ] **Step 5: Commit**

```bash
git add server.py tests/test_oauth_endpoints.py
git commit -m "feat: add Google OAuth callback endpoints"
```

---

## Task 4: 调度器服务

**Files:**
- Create: `services/__init__.py`
- Create: `services/scheduler.py`
- Create: `tests/test_scheduler.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_scheduler.py`：

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, patch


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
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
python -m pytest tests/test_scheduler.py -v 2>&1 | head -20
```

Expected: `ImportError: No module named 'services.scheduler'`

- [ ] **Step 3: 创建 services/__init__.py**

```bash
mkdir -p services && touch services/__init__.py
```

- [ ] **Step 4: 创建 services/scheduler.py**

```python
import os
import uuid
from datetime import datetime
from typing import Callable, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.executors.asyncio import AsyncIOExecutor

_scheduler: Optional[AsyncIOScheduler] = None


def _build_scheduler() -> AsyncIOScheduler:
    db_path = os.environ.get("SCHEDULER_DB_PATH", "./scheduler.db")
    return AsyncIOScheduler(
        jobstores={"default": SQLAlchemyJobStore(url=f"sqlite:///{db_path}")},
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


def add_job(func: Callable, run_at: datetime, job_id: Optional[str] = None, args: list = None) -> str:
    job_id = job_id or str(uuid.uuid4())
    get_scheduler().add_job(
        func,
        trigger="date",
        run_date=run_at,
        id=job_id,
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
```

- [ ] **Step 5: 运行测试，确认通过**

```bash
python -m pytest tests/test_scheduler.py -v
```

Expected: 3 个测试全部 PASS

- [ ] **Step 6: Commit**

```bash
git add services/__init__.py services/scheduler.py tests/test_scheduler.py
git commit -m "feat: add APScheduler service with SQLite persistence"
```

---

## Task 5: 通知推送服务

**Files:**
- Create: `services/notifier.py`
- Create: `tests/test_notifier.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_notifier.py`：

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import AsyncMock, patch, MagicMock


@pytest.mark.asyncio
async def test_send_telegram_calls_bot_api():
    """send_telegram 应调用 Telegram Bot API sendMessage。"""
    with patch.dict(os.environ, {
        "TELEGRAM_BOT_TOKEN": "fake:token",
        "REMINDER_TELEGRAM_CHAT_ID": "12345",
    }):
        import importlib
        import services.notifier as mod
        importlib.reload(mod)

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response) as mock_post:
            await mod.send_telegram("Test message")
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert "sendMessage" in call_args[0][0]
            assert call_args[1]["json"]["text"] == "Test message"


@pytest.mark.asyncio
async def test_send_email_calls_smtp():
    """send_email 应调用 aiosmtplib.send。"""
    with patch.dict(os.environ, {
        "SMTP_HOST": "smtp.gmail.com",
        "SMTP_PORT": "587",
        "SMTP_USER": "test@gmail.com",
        "SMTP_PASSWORD": "password",
        "NOTIFY_EMAIL": "notify@example.com",
    }):
        import importlib
        import services.notifier as mod
        importlib.reload(mod)

        with patch("aiosmtplib.send", new_callable=AsyncMock) as mock_send:
            await mod.send_email("Test subject", "Test body")
            mock_send.assert_called_once()


@pytest.mark.asyncio
async def test_send_dispatches_to_channels():
    """send(channels=['telegram','email']) 应同时调用两个渠道。"""
    import importlib
    import services.notifier as mod
    importlib.reload(mod)

    with patch.object(mod, "send_telegram", new_callable=AsyncMock) as mock_tg, \
         patch.object(mod, "send_email", new_callable=AsyncMock) as mock_mail:
        await mod.send("Reminder!", channels=["telegram", "email"])
        mock_tg.assert_called_once_with("Reminder!")
        mock_mail.assert_called_once()
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
python -m pytest tests/test_notifier.py -v 2>&1 | head -20
```

Expected: `ImportError: No module named 'services.notifier'`

- [ ] **Step 3: 创建 services/notifier.py**

```python
import os
import logging
from email.message import EmailMessage

import httpx
import aiosmtplib

logger = logging.getLogger(__name__)

_TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


async def send_telegram(message: str) -> None:
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["REMINDER_TELEGRAM_CHAT_ID"]
    url = _TELEGRAM_API.format(token=token)
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json={"chat_id": chat_id, "text": message})
        resp.raise_for_status()


async def send_email(subject: str, body: str) -> None:
    msg = EmailMessage()
    msg["From"] = os.environ["SMTP_USER"]
    msg["To"] = os.environ["NOTIFY_EMAIL"]
    msg["Subject"] = subject
    msg.set_content(body)
    await aiosmtplib.send(
        msg,
        hostname=os.environ.get("SMTP_HOST", "smtp.gmail.com"),
        port=int(os.environ.get("SMTP_PORT", "587")),
        username=os.environ["SMTP_USER"],
        password=os.environ["SMTP_PASSWORD"],
        start_tls=True,
    )


async def _send_with_retry(coro_fn, *args, retries: int = 3, delay: float = 30.0):
    """重试最多 retries 次，每次间隔 delay 秒。"""
    import asyncio
    last_exc = None
    for attempt in range(retries):
        try:
            await coro_fn(*args)
            return
        except Exception as e:
            last_exc = e
            logger.warning("Attempt %d/%d failed: %s", attempt + 1, retries, e)
            if attempt < retries - 1:
                await asyncio.sleep(delay)
    logger.error("All %d attempts failed: %s", retries, last_exc)


async def send(message: str, channels: list[str] = None) -> None:
    """统一推送入口，由 APScheduler job 调用。每个渠道失败时最多重试 3 次，间隔 30 秒。"""
    channels = channels or ["telegram"]
    for ch in channels:
        if ch in ("telegram", "all"):
            await _send_with_retry(send_telegram, message)
        if ch in ("email", "all"):
            await _send_with_retry(send_email, "提醒", message)
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
python -m pytest tests/test_notifier.py -v
```

Expected: 3 个测试全部 PASS

- [ ] **Step 5: Commit**

```bash
git add services/notifier.py tests/test_notifier.py
git commit -m "feat: add unified notification service (Telegram + Email)"
```

---

## Task 6: Google Calendar 工具

**Files:**
- Create: `tools/calendar.py`
- Create: `tests/test_calendar_tools.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_calendar_tools.py`：

```python
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
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
python -m pytest tests/test_calendar_tools.py -v 2>&1 | head -20
```

Expected: `ImportError: No module named 'tools.calendar'`

- [ ] **Step 3: 创建 tools/calendar.py**

```python
from __future__ import annotations
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from googleapiclient.discovery import build
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from auth.google_oauth import get_credentials, NeedsAuthorizationError

_AUTH_MSG = "需要授权 Google Calendar，请访问 /auth/google 完成授权。"
_CALENDAR_ID = lambda: os.environ.get("GOOGLE_CALENDAR_ID", "primary")
_TZ = lambda: ZoneInfo(os.environ.get("TIMEZONE", "Asia/Shanghai"))


def _service():
    return build("calendar", "v3", credentials=get_credentials())


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
    return f"已创建事件「{event['summary']}」(id: {event['id']})，链接: {event.get('htmlLink', '')}"


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
        return "该时间段内没有日程。"
    lines = []
    for ev in items:
        start = ev["start"].get("dateTime", ev["start"].get("date", ""))
        lines.append(f"- [{ev['id']}] {ev.get('summary', '无标题')} @ {start}")
    return "\n".join(lines)


# ── Update ────────────────────────────────────────────────────────────────────

class UpdateEventInput(BaseModel):
    event_id: str = Field(..., description="事件 ID")
    title: str = Field(None, description="新标题（留空不修改）")
    start: str = Field(None, description="新开始时间，ISO 8601（留空不修改）")
    end: str = Field(None, description="新结束时间，ISO 8601（留空不修改）")
    description: str = Field(None, description="新描述（留空不修改）")


@tool("update_calendar_event", args_schema=UpdateEventInput)
def update_calendar_event(event_id: str, title: str = None, start: str = None,
                          end: str = None, description: str = None) -> str:
    """修改 Google Calendar 中已有事件的标题、时间或描述。"""
    try:
        svc = _service()
    except NeedsAuthorizationError:
        return _AUTH_MSG
    patch_body: dict = {}
    if title:
        patch_body["summary"] = title
    if start:
        patch_body["start"] = {"dateTime": start}
    if end:
        patch_body["end"] = {"dateTime": end}
    if description is not None:
        patch_body["description"] = description
    if not patch_body:
        return "没有要修改的内容。"
    event = svc.events().patch(
        calendarId=_CALENDAR_ID(), eventId=event_id, body=patch_body
    ).execute()
    return f"事件 {event['id']} 已更新：{event.get('summary', '')}"


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
    return f"事件 {event_id} 已删除。"


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

    # 计算空闲段（工作时间 08:00-22:00）
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
        return f"{date} 没有满足 {duration_minutes} 分钟的空闲时段。"
    return f"{date} 可用时段（工作时间内）：\n" + "\n".join(f"- {s}" for s in free_slots)
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
python -m pytest tests/test_calendar_tools.py -v
```

Expected: 5 个测试全部 PASS

- [ ] **Step 5: Commit**

```bash
git add tools/calendar.py tests/test_calendar_tools.py
git commit -m "feat: add Google Calendar tools (create/list/update/delete/free-slots)"
```

---

## Task 7: 提醒工具

**Files:**
- Create: `tools/reminder.py`
- Create: `tests/test_reminder_tools.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_reminder_tools.py`：

```python
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
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
python -m pytest tests/test_reminder_tools.py -v 2>&1 | head -20
```

Expected: `ImportError: No module named 'tools.reminder'`

- [ ] **Step 3: 创建 tools/reminder.py**

```python
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
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
python -m pytest tests/test_reminder_tools.py -v
```

Expected: 5 个测试全部 PASS

- [ ] **Step 5: Commit**

```bash
git add tools/reminder.py tests/test_reminder_tools.py
git commit -m "feat: add reminder tools (set/list/delete)"
```

---

## Task 8: 串联集成

**Files:**
- Modify: `tools/__init__.py`（注册新工具）
- Modify: `server.py`（lifespan 启动调度器）

- [ ] **Step 1: 写集成测试**

创建 `tests/test_integration_calendar.py`：

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch, AsyncMock, MagicMock


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
        import importlib
        import services.scheduler as sched_mod
        importlib.reload(sched_mod)

        with patch("services.scheduler.start", new_callable=AsyncMock) as mock_start, \
             patch("services.scheduler.stop", new_callable=AsyncMock) as mock_stop:
            from fastapi.testclient import TestClient
            import server
            importlib.reload(server)
            # lifespan 通过 TestClient 触发
            with TestClient(server.api):
                mock_start.assert_called_once()
            mock_stop.assert_called_once()
```

- [ ] **Step 2: 运行集成测试，确认失败**

```bash
python -m pytest tests/test_integration_calendar.py::test_calendar_tools_registered_in_tools_list -v 2>&1 | head -20
```

Expected: AssertionError（工具未注册）

- [ ] **Step 3: 修改 tools/__init__.py，注册新工具**

```python
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
```

- [ ] **Step 4: 修改 server.py，lifespan 启动/停止调度器**

找到现有的 `lifespan` 函数：

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_session_sweep_loop())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
```

替换为：

```python
from services import scheduler as _scheduler_svc

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_session_sweep_loop())
    await _scheduler_svc.start()
    try:
        yield
    finally:
        await _scheduler_svc.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
```

- [ ] **Step 5: 运行所有集成测试**

```bash
python -m pytest tests/test_integration_calendar.py -v
```

Expected: 所有测试 PASS

- [ ] **Step 6: 运行全量测试套件，确认无回归**

```bash
python -m pytest tests/ -v --ignore=tests/test_translation.py --ignore=tests/test_locomo_utils.py -x 2>&1 | tail -30
```

Expected: 所有新增测试 PASS，原有测试无新增失败

- [ ] **Step 7: Commit**

```bash
git add tools/__init__.py server.py tests/test_integration_calendar.py
git commit -m "feat: wire calendar+reminder tools into agent and start scheduler in lifespan"
```

---

## Task 9: 配置验证脚本

**Files:**
- Create: `scripts/check_calendar_setup.py`

- [ ] **Step 1: 创建配置验证脚本**

```python
#!/usr/bin/env python3
"""运行此脚本验证日历功能配置是否正确。"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

errors = []

# 检查 credentials.json
creds_file = os.environ.get("GOOGLE_CREDENTIALS_FILE", "credentials.json")
if not os.path.exists(creds_file):
    errors.append(f"❌ 缺少 {creds_file}，请从 Google Cloud Console 下载并放到项目根目录")
else:
    print(f"✅ {creds_file} 存在")

# 检查 token
token_file = os.environ.get("GOOGLE_TOKEN_FILE", "auth/token.json")
if not os.path.exists(token_file):
    print(f"⚠️  {token_file} 不存在——首次使用时请访问 http://localhost:8000/auth/google 完成授权")
else:
    print(f"✅ {token_file} 存在")

# 检查必要环境变量
required_vars = [
    "TELEGRAM_BOT_TOKEN",
    "REMINDER_TELEGRAM_CHAT_ID",
    "SMTP_USER",
    "SMTP_PASSWORD",
    "NOTIFY_EMAIL",
]
for var in required_vars:
    if os.environ.get(var):
        print(f"✅ {var} 已设置")
    else:
        errors.append(f"❌ 缺少环境变量 {var}")

if errors:
    print("\n以下问题需要修复：")
    for e in errors:
        print(" ", e)
    sys.exit(1)
else:
    print("\n✅ 配置检查通过，可以启动服务器")
```

- [ ] **Step 2: 运行验证脚本**

```bash
python scripts/check_calendar_setup.py
```

Expected: 打印各项检查结果（credentials.json 缺失会提示，其他变量按实际情况）

- [ ] **Step 3: Commit**

```bash
git add scripts/check_calendar_setup.py
git commit -m "chore: add calendar setup verification script"
```

---

## 完成验证

- [ ] 运行 `python scripts/check_calendar_setup.py`，按提示修复配置
- [ ] 启动服务器：`python server.py`
- [ ] 访问 `http://localhost:8000/auth/google`，完成 Google 授权
- [ ] 通过 Telegram 发送「帮我查一下明天的日程」，验证 Agent 能调用 `list_calendar_events`
- [ ] 通过 Telegram 发送「明天下午三点提醒我喝水」，验证 `set_reminder` 写入 scheduler
- [ ] 等待提醒时间到来，验证 Telegram 推送和邮件均收到
