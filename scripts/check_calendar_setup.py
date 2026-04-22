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
