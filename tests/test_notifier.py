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
