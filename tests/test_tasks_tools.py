import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch, MagicMock


def _make_tasks_service(task_lists=None, tasks=None):
    """构造 mock Tasks service。"""
    svc = MagicMock()

    # 设置 tasklists 的 list 方法
    tasklists_mock = MagicMock()
    list_mock = MagicMock()
    list_mock.execute.return_value = {
        "items": task_lists if task_lists is not None else [{"id": "list-1", "title": "工作"}]
    }
    tasklists_mock.list.return_value = list_mock

    # 设置 tasklists 的 insert 方法
    insert_mock = MagicMock()
    insert_mock.execute.return_value = {"id": "list-2", "title": "新列表"}
    tasklists_mock.insert.return_value = insert_mock

    svc.tasklists.return_value = tasklists_mock

    # 设置 tasks 的 list 方法
    tasks_list_mock = MagicMock()
    tasks_list_execute_mock = MagicMock()
    tasks_list_execute_mock.execute.return_value = {
        "items": tasks if tasks is not None else [
            {"id": "task-1", "title": "写报告", "status": "needsAction"}
        ]
    }
    tasks_list_mock.list.return_value = tasks_list_execute_mock

    # 设置 tasks 的 insert 方法
    tasks_insert_mock = MagicMock()
    tasks_insert_execute_mock = MagicMock()
    tasks_insert_execute_mock.execute.return_value = {"id": "task-2", "title": "新任务"}
    tasks_insert_mock.insert.return_value = tasks_insert_execute_mock

    # 设置 tasks 的 patch 方法
    tasks_patch_mock = MagicMock()
    tasks_patch_execute_mock = MagicMock()
    tasks_patch_execute_mock.execute.return_value = {"id": "task-1", "title": "写报告", "status": "completed"}
    tasks_patch_mock.patch.return_value = tasks_patch_execute_mock

    # 设置 tasks 的 delete 方法
    tasks_delete_mock = MagicMock()
    tasks_delete_execute_mock = MagicMock()
    tasks_delete_execute_mock.execute.return_value = None
    tasks_delete_mock.delete.return_value = tasks_delete_execute_mock

    # 将所有方法合并到一个 tasks 对象
    tasks_obj = MagicMock()
    tasks_obj.list = tasks_list_mock.list
    tasks_obj.insert = tasks_insert_mock.insert
    tasks_obj.patch = tasks_patch_mock.patch
    tasks_obj.delete = tasks_delete_mock.delete

    svc.tasks.return_value = tasks_obj

    return svc


@patch("tools.tasks.get_credentials")
@patch("tools.tasks.build")
def test_list_task_lists(mock_build, mock_creds):
    mock_build.return_value = _make_tasks_service()
    from tools.tasks import list_task_lists
    result = list_task_lists.invoke({})
    assert "list-1" in result
    assert "工作" in result


@patch("tools.tasks.get_credentials")
@patch("tools.tasks.build")
def test_list_task_lists_empty(mock_build, mock_creds):
    mock_build.return_value = _make_tasks_service(task_lists=[])
    from tools.tasks import list_task_lists
    result = list_task_lists.invoke({})
    assert "没有" in result


@patch("tools.tasks.get_credentials")
@patch("tools.tasks.build")
def test_create_task_list(mock_build, mock_creds):
    mock_build.return_value = _make_tasks_service()
    from tools.tasks import create_task_list
    result = create_task_list.invoke({"title": "新列表"})
    assert "list-2" in result or "新列表" in result


@patch("tools.tasks.get_credentials")
@patch("tools.tasks.build")
def test_list_tasks(mock_build, mock_creds):
    mock_build.return_value = _make_tasks_service()
    from tools.tasks import list_tasks
    result = list_tasks.invoke({"task_list_id": "list-1", "show_completed": False})
    assert "task-1" in result
    assert "写报告" in result


@patch("tools.tasks.get_credentials")
@patch("tools.tasks.build")
def test_list_tasks_empty(mock_build, mock_creds):
    mock_build.return_value = _make_tasks_service(tasks=[])
    from tools.tasks import list_tasks
    result = list_tasks.invoke({"task_list_id": "list-1", "show_completed": False})
    assert "没有" in result


@patch("tools.tasks.get_credentials")
@patch("tools.tasks.build")
def test_create_task(mock_build, mock_creds):
    mock_build.return_value = _make_tasks_service()
    from tools.tasks import create_task
    result = create_task.invoke({
        "task_list_id": "list-1",
        "title": "新任务",
        "notes": None,
        "due": None,
    })
    assert "task-2" in result or "新任务" in result


@patch("tools.tasks.get_credentials")
@patch("tools.tasks.build")
def test_create_task_with_due(mock_build, mock_creds):
    mock_build.return_value = _make_tasks_service()
    from tools.tasks import create_task
    result = create_task.invoke({
        "task_list_id": "list-1",
        "title": "截止任务",
        "notes": "备注",
        "due": "2026-04-25",
    })
    assert "task-2" in result or "截止任务" in result or "新任务" in result


@patch("tools.tasks.get_credentials")
@patch("tools.tasks.build")
def test_complete_task(mock_build, mock_creds):
    mock_build.return_value = _make_tasks_service()
    from tools.tasks import complete_task
    result = complete_task.invoke({"task_list_id": "list-1", "task_id": "task-1"})
    assert "完成" in result or "completed" in result.lower()


@patch("tools.tasks.get_credentials")
@patch("tools.tasks.build")
def test_delete_task(mock_build, mock_creds):
    mock_build.return_value = _make_tasks_service()
    from tools.tasks import delete_task
    result = delete_task.invoke({"task_list_id": "list-1", "task_id": "task-1"})
    assert "删除" in result or "task-1" in result


@patch("tools.tasks.get_credentials")
def test_no_auth_returns_message(mock_creds):
    # Import from tools.tasks so the exception class matches what the
    # except clause catches, even if auth.google_oauth was reloaded.
    from tools.tasks import NeedsAuthorizationError
    mock_creds.side_effect = NeedsAuthorizationError("no token")
    from tools.tasks import list_task_lists
    result = list_task_lists.invoke({})
    assert "授权" in result or "auth" in result.lower()
