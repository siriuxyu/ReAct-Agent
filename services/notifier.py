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
