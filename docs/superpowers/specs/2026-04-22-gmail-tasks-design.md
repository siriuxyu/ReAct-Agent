# Gmail + Google Tasks 功能设计文档

日期：2026-04-22

## 目标

为 Cliriux 个人助手增加 Gmail 邮件管理和 Google Tasks 任务管理能力，复用现有 Google OAuth 基础设施，统一扩展 token scope，一次重授权覆盖全部 Google 服务。

---

## 架构方案

**单体扩展（沿用方案一）**

在现有 FastAPI 服务中直接扩展，不增加新进程。复用 `auth/google_oauth.py` 的 OAuth token 管理，合并所有 Google scope 到同一 `token.json`。

---

## 改动文件结构

```
Cliriux/
├── auth/
│   └── google_oauth.py    # 修改：扩展 SCOPES，加入 Gmail + Tasks
├── tools/
│   ├── gmail.py           # 新建：4 个 Gmail 工具
│   ├── tasks.py           # 新建：6 个 Google Tasks 工具
│   └── __init__.py        # 修改：注册 10 个新工具
└── tests/
    ├── test_gmail_tools.py     # 新建
    ├── test_tasks_tools.py     # 新建
    └── test_oauth_scopes.py    # 新建
```

---

## OAuth Scope 变更

`auth/google_oauth.py` 中 `SCOPES` 从：

```python
SCOPES = ["https://www.googleapis.com/auth/calendar"]
```

扩展为：

```python
SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/tasks",
]
```

**重授权流程：** 删除 `auth/token.json`，用户访问 `/auth/google` 重新走一次 OAuth 流程。授权后所有 Google 工具共享同一 token，自动刷新不感知。

---

## 工具接口

### Gmail 工具（`tools/gmail.py`）

| 工具名 | 关键参数 | 说明 |
|--------|----------|------|
| `list_emails` | `max_results`, `label_ids` | 列出最近邮件，返回发件人/主题/时间/摘要 |
| `search_emails` | `query`, `max_results` | 支持 Gmail 搜索语法（`from:`, `subject:`, `is:unread` 等） |
| `read_email` | `message_id` | 读取完整正文，优先 `text/plain`，fallback 剥离 HTML 标签 |
| `send_email` | `to`, `subject`, `body`, `reply_to_message_id?` | 发送新邮件或回复；传 `reply_to_message_id` 时自动带线程 header |

- `message_id` 由 `list_emails` / `search_emails` 返回，agent 先查再读/回复
- 正文解析：优先取 `text/plain` part，无则取 `text/html` 并用正则剥离标签
- 所有工具：`try: svc = _service() except NeedsAuthorizationError: return _AUTH_MSG`（与 calendar.py 模式一致）

### Google Tasks 工具（`tools/tasks.py`）

| 工具名 | 关键参数 | 说明 |
|--------|----------|------|
| `list_task_lists` | — | 列出所有任务列表，返回 id + 标题 |
| `create_task_list` | `title` | 新建任务列表 |
| `list_tasks` | `task_list_id`, `show_completed` | 列出任务，`show_completed` 默认 `False` |
| `create_task` | `task_list_id`, `title`, `notes?`, `due?` | 创建任务；`due` 为 ISO 8601 日期（无时间部分，Tasks API 限制） |
| `complete_task` | `task_list_id`, `task_id` | 标记任务为已完成（`status: "completed"`） |
| `delete_task` | `task_list_id`, `task_id` | 永久删除任务 |

- `task_id` 由 `list_tasks` 返回
- `due` 只支持日期，精确时间提醒请使用 `set_reminder`
- 工具均使用 `args_schema` Pydantic BaseModel（与 calendar.py 模式一致）

---

## 数据流示例

```
用户: 「帮我看看有没有未读邮件」
  → Agent 调用 search_emails(query="is:unread", max_results=5)
      → 返回邮件列表（id, from, subject, snippet）
  → Agent 调用 read_email(message_id="...")
      → 返回完整正文
  → Agent 回复摘要

用户: 「回复说我明天可以」
  → Agent 调用 send_email(to="...", subject="Re:...", body="...", reply_to_message_id="...")
      → Gmail API 发送，带线程 In-Reply-To header
```

```
用户: 「在工作列表里加一个任务：周五前交报告」
  → Agent 调用 list_task_lists()
      → 找到"工作"列表的 task_list_id
  → Agent 调用 create_task(task_list_id="...", title="周五前交报告", due="2026-04-24")
      → 返回创建结果
```

---

## 错误处理

| 场景 | 处理方式 |
|------|----------|
| token 无效 / scope 不足 | 返回授权链接，引导用户重新授权（复用 `NeedsAuthorizationError`） |
| Gmail API 限额超出 | 返回错误信息，告知稍后重试 |
| 邮件正文解码失败 | 返回原始 snippet，不抛异常 |
| `reply_to_message_id` 不存在 | 降级为发送新邮件 |
| Tasks API 错误 | 返回中文错误描述 |

---

## 测试计划

| 测试文件 | 覆盖内容 |
|----------|----------|
| `tests/test_gmail_tools.py` | mock Gmail API，覆盖 4 个工具的正常路径和错误路径 |
| `tests/test_tasks_tools.py` | mock Tasks API，覆盖 6 个工具的正常路径和错误路径 |
| `tests/test_oauth_scopes.py` | 验证 `SCOPES` 包含全部 4 个必要权限 |

---

## 新增依赖

无。`google-api-python-client` 已包含 Gmail 和 Tasks API 支持。

---

## 不在本次范围内

- 邮件标记已读 / 归档（B → C 升级，后续可加）
- Gmail 标签管理
- Google Tasks 子任务
- 多个 Google 账号支持
