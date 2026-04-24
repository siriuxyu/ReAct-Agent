# 日历 + 提醒功能设计文档

日期：2026-04-21

## 目标

为 Cliriux 个人助手增加 Google Calendar 集成与定时提醒能力，使 Agent 能够管理日程、设定提醒，并通过 Telegram、邮件、Google Calendar 原生通知三个渠道推送。

---

## 架构方案

**方案：单体扩展（方案一）**

在现有 FastAPI 服务中直接扩展，不增加新进程，保持 docker-compose 不变。调度器随 FastAPI lifespan 启动，与 Web server 共享同一进程。

---

## 新增文件结构

```
Cliriux/
├── tools/
│   ├── calendar.py        # 5 个 GCal 工具，注册进 Agent
│   └── reminder.py        # 3 个提醒工具，写入 APScheduler SQLite
├── services/
│   ├── __init__.py
│   ├── scheduler.py       # APScheduler 单例 + SQLite jobstore
│   └── notifier.py        # 统一推送（Telegram / GCal / Email）
├── auth/
│   ├── __init__.py
│   └── google_oauth.py    # OAuth2 token 管理，单用户（个人助手）
└── server.py              # 修改：注册 lifespan 启动/停止 scheduler
```

---

## 数据流

```
用户: 「明天下午三点提醒我开会」
  → Agent 调用 create_calendar_event()
      → GCal API 创建事件（含邮件 + GCal 原生提醒）
  → Agent 调用 set_reminder()
      → 写入 APScheduler SQLite jobstore
  → 到点时 APScheduler 触发 notifier.send()
      → Telegram 推送 + 邮件发送
```

---

## Google OAuth 授权

- 使用 `google-auth-oauthlib` 实现 OAuth 2.0 授权码流程
- 凭据文件：项目根目录 `credentials.json`（从 Google Cloud Console 下载）
- Token 存储：`auth/token.json`（本地文件，包含 access_token + refresh_token）
- 所有 Telegram 用户共用同一个 Google 账号（个人助手场景）
- Token 过期：`google-auth` 自动 refresh，无感知
- Refresh token 失效（超 6 个月未用）：Agent 回复授权链接，用户重新走一次 OAuth 流程

**用户首次使用流程：**
1. 用户发送任意日历指令
2. Agent 检测到无有效 token，回复授权 URL
3. 用户点击链接，浏览器完成授权
4. Token 保存，后续自动刷新

---

## 工具接口

### 日历工具（`tools/calendar.py`）

| 工具名 | 关键参数 | 说明 |
|--------|----------|------|
| `create_calendar_event` | title, start, end, description, attendees, recurrence | 创建事件，支持重复规则（RRULE） |
| `list_calendar_events` | time_min, time_max, max_results | 查询时间段内日程列表 |
| `update_calendar_event` | event_id, **kwargs | 修改标题/时间/描述/与会者等 |
| `delete_calendar_event` | event_id | 删除事件 |
| `find_free_slots` | date, duration_minutes | 查找指定日期的空闲时段 |

- 时间参数统一使用 ISO 8601 格式，时区依据 `TIMEZONE` 环境变量
- `recurrence` 参数接受 RRULE 字符串，如 `RRULE:FREQ=WEEKLY;BYDAY=MO`
- 修改重复事件时，Agent 须询问用户「只改这次」还是「改所有」

### 提醒工具（`tools/reminder.py`）

| 工具名 | 关键参数 | 说明 |
|--------|----------|------|
| `set_reminder` | message, remind_at, user_id, channels | 设定提醒；channels 可选 `telegram`/`email`/`all` |
| `list_reminders` | user_id | 列出该用户所有未触发的提醒 |
| `delete_reminder` | job_id | 取消指定提醒 |

- `remind_at` 为 ISO 8601 时间字符串
- Agent 若只收到「明天提醒我」无具体时间，须追问，不得猜测

---

## 调度器（`services/scheduler.py`）

- 使用 `APScheduler` >= 3.10，`SQLAlchemyJobStore`，数据库文件 `scheduler.db`
- 随 FastAPI `lifespan` 启动和关闭
- 服务重启后从 SQLite 恢复所有 job
- 已过期 job 在恢复时立即触发一次，推送「错过的提醒」通知

---

## 通知推送（`services/notifier.py`）

三个渠道：

| 渠道 | 实现 |
|------|------|
| Telegram | 调用 Bot API `sendMessage`，使用现有 `TELEGRAM_BOT_TOKEN` |
| 邮件 | `aiosmtplib` 异步 SMTP，支持 Gmail 应用专用密码 |
| GCal 原生提醒 | 在 `create_calendar_event` 时写入 `reminders` 字段（popup + email） |

**失败重试：** Telegram/邮件发送失败最多重试 3 次，间隔 30 秒。所有渠道均失败则记录日志。

---

## 环境变量

```env
# Google Calendar
GOOGLE_CALENDAR_ID=primary
TIMEZONE=Asia/Shanghai

# 提醒推送目标
REMINDER_TELEGRAM_CHAT_ID=你的Telegram_chat_id

# 邮件
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=你的gmail地址
SMTP_PASSWORD=应用专用密码
NOTIFY_EMAIL=接收提醒的邮箱

# 调度器
SCHEDULER_DB_PATH=./scheduler.db
```

---

## 错误处理

| 场景 | 处理方式 |
|------|----------|
| GCal API 限额超出 | 返回错误给 Agent，Agent 告知用户稍后重试 |
| Token refresh 失败 | Agent 回复授权链接，引导重新授权 |
| 提醒发送失败 | 重试 3 次，间隔 30 秒，失败记录日志 |
| 时区未配置 | 默认 `Asia/Shanghai` |
| 提醒时间模糊 | Agent 追问具体时间，不猜测 |
| 重复事件修改 | Agent 询问「只改这次」或「改所有」 |

---

## 新增依赖

```
google-api-python-client>=2.100.0
google-auth-oauthlib>=1.1.0
google-auth-httplib2>=0.2.0
apscheduler>=3.10.0
aiosmtplib>=3.0.0
sqlalchemy>=2.0.0
```

---

## 不在本次范围内

- 多用户各自绑定不同 Google 账号
- CalDAV / Apple Calendar 支持
- 提醒的 snooze（延迟）功能
- Celery 迁移
