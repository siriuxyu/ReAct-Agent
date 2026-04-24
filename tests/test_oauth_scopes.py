import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_scopes_include_gmail_readonly():
    from auth.google_oauth import SCOPES
    assert "https://www.googleapis.com/auth/gmail.readonly" in SCOPES


def test_scopes_include_gmail_send():
    from auth.google_oauth import SCOPES
    assert "https://www.googleapis.com/auth/gmail.send" in SCOPES


def test_scopes_include_tasks():
    from auth.google_oauth import SCOPES
    assert "https://www.googleapis.com/auth/tasks" in SCOPES


def test_scopes_still_include_calendar():
    from auth.google_oauth import SCOPES
    assert "https://www.googleapis.com/auth/calendar" in SCOPES


def test_gmail_tools_registered():
    from tools import TOOLS
    names = [t.name for t in TOOLS]
    for name in ["list_emails", "search_emails", "read_email", "send_email"]:
        assert name in names, f"工具 {name} 未注册"


def test_tasks_tools_registered():
    from tools import TOOLS
    names = [t.name for t in TOOLS]
    for name in ["list_task_lists", "create_task_list", "list_tasks",
                 "create_task", "complete_task", "delete_task"]:
        assert name in names, f"工具 {name} 未注册"
