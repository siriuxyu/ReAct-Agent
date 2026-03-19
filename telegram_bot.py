"""Telegram Bot front-end for the ReAct Agent server.

Each Telegram chat_id is mapped to a unique userid so memory is isolated
per user. Supports text, photos, and PDF documents via the agent's
multimodal API. Streams responses via SSE for a live-typing feel.

Required env vars:
  TELEGRAM_BOT_TOKEN   — from @BotFather
  AGENT_BASE_URL       — defaults to http://localhost:8000
"""

import asyncio
import base64
import json
import logging
import os

import httpx
from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

load_dotenv()

TELEGRAM_TOKEN: str = os.environ["TELEGRAM_BOT_TOKEN"]
AGENT_BASE_URL: str = os.environ.get("AGENT_BASE_URL", "http://localhost:8000")

# Minimum chars of new content before editing the in-progress Telegram message
_STREAM_EDIT_THRESHOLD = 80

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _userid(update: Update) -> str:
    """Stable userid derived from the Telegram chat id."""
    return f"tg_{update.effective_chat.id}"


async def _download_b64(file_id: str, context: ContextTypes.DEFAULT_TYPE) -> str:
    file = await context.bot.get_file(file_id)
    data = await file.download_as_bytearray()
    return base64.b64encode(data).decode()


async def _build_content(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Return a str or multimodal content-block list for the incoming message."""
    msg = update.message

    if msg.text:
        return msg.text

    if msg.photo:
        b64 = await _download_b64(msg.photo[-1].file_id, context)
        blocks = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
            {"type": "text", "text": msg.caption or "请描述这张图片。"},
        ]
        return blocks

    if msg.document:
        mime = msg.document.mime_type or ""
        b64 = await _download_b64(msg.document.file_id, context)
        caption = msg.caption or ""

        if mime == "application/pdf":
            return [
                {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": b64}},
                {"type": "text", "text": caption or "请总结这份文档的内容。"},
            ]
        if mime.startswith("image/"):
            return [
                {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}},
                {"type": "text", "text": caption or "请描述这张图片。"},
            ]
        return f"收到文件《{msg.document.file_name}》，但暂不支持 {mime} 格式。"

    return None


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "你好！我是你的个人 AI 助理 👋\n\n"
        "直接发消息和我对话，我会记住我们聊过的内容。\n"
        "也可以发图片或 PDF，我会帮你分析。\n\n"
        "可用命令：\n"
        "/reset — 开始新对话（保留长期记忆）\n"
        "/memory [关键词] — 查看记住了什么\n"
        "/forget — 清除所有记忆\n"
        "/help — 显示帮助"
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "命令说明：\n\n"
        "/reset — 清除当前对话，开始新会话（长期记忆保留）\n"
        "/memory [关键词] — 搜索记忆内容，不加关键词则列出最近记忆\n"
        "/forget — 清除所有长期记忆（不可恢复）\n\n"
        "支持发送：文字、图片、PDF 文件"
    )


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    userid = _userid(update)
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            r = await client.post(
                f"{AGENT_BASE_URL}/reset/{userid}",
                params={"preserve_memory": True},
            )
            r.raise_for_status()
            await update.message.reply_text("✓ 对话已重置，长期记忆已保留。")
        except Exception as e:
            logger.error("Reset failed for %s: %s", userid, e)
            await update.message.reply_text(f"重置失败：{e}")


async def cmd_memory(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    userid = _userid(update)
    query = " ".join(context.args) if context.args else "用户偏好"
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            r = await client.post(
                f"{AGENT_BASE_URL}/memory/{userid}/search",
                json={"query": query, "limit": 10},
            )
            r.raise_for_status()
            results = r.json().get("results", [])
            if not results:
                await update.message.reply_text("暂无相关记忆。")
                return
            lines = [f"找到 {len(results)} 条记忆：\n"]
            for i, item in enumerate(results, 1):
                snippet = (item.get("content") or "")[:200].replace("\n", " ")
                lines.append(f"{i}. {snippet}")
            await update.message.reply_text("\n".join(lines))
        except Exception as e:
            logger.error("Memory search failed for %s: %s", userid, e)
            await update.message.reply_text(f"查询失败：{e}")


async def cmd_forget(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    userid = _userid(update)
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            r = await client.delete(f"{AGENT_BASE_URL}/memory/{userid}")
            r.raise_for_status()
            await update.message.reply_text("✓ 所有记忆已清除。")
        except Exception as e:
            logger.error("Forget failed for %s: %s", userid, e)
            await update.message.reply_text(f"清除失败：{e}")


# ---------------------------------------------------------------------------
# Message handler
# ---------------------------------------------------------------------------

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    userid = _userid(update)

    content = await _build_content(update, context)
    if content is None:
        await update.message.reply_text("暂不支持此消息类型。")
        return

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action=ChatAction.TYPING,
    )

    placeholder = await update.message.reply_text("▌")
    accumulated = ""
    last_edited_len = 0

    try:
        async with httpx.AsyncClient(timeout=180) as client:
            async with client.stream(
                "POST",
                f"{AGENT_BASE_URL}/stream",
                json={
                    "userid": userid,
                    "messages": [{"role": "user", "content": content}],
                },
                headers={"Accept": "text/event-stream"},
            ) as resp:
                resp.raise_for_status()

                async for raw_line in resp.aiter_lines():
                    if not raw_line.startswith("data:"):
                        continue
                    try:
                        event = json.loads(raw_line[5:].strip())
                    except json.JSONDecodeError:
                        continue

                    etype = event.get("type")

                    if etype == "chunk":
                        accumulated = event.get("content", "")
                        new_chars = len(accumulated) - last_edited_len
                        if new_chars >= _STREAM_EDIT_THRESHOLD:
                            try:
                                await placeholder.edit_text(accumulated + " ▌")
                                last_edited_len = len(accumulated)
                            except Exception:
                                pass

                    elif etype == "done":
                        final = event.get("final_response") or accumulated
                        try:
                            await placeholder.edit_text(final or "(无回复)")
                        except Exception:
                            pass
                        break

                    elif etype == "error":
                        await placeholder.edit_text(
                            f"出错了：{event.get('message', '未知错误')}"
                        )
                        break

    except httpx.TimeoutException:
        await placeholder.edit_text("请求超时，请稍后重试。")
    except Exception as e:
        logger.error("Stream error for %s: %s", userid, e, exc_info=True)
        await placeholder.edit_text(f"出错了：{e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(CommandHandler("memory", cmd_memory))
    app.add_handler(CommandHandler("forget", cmd_forget))
    app.add_handler(
        MessageHandler(
            filters.TEXT | filters.PHOTO | filters.Document.ALL,
            handle_message,
        )
    )

    logger.info("Telegram bot polling (agent: %s)", AGENT_BASE_URL)
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
