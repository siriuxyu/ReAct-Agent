import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_tool_registry_indexes_base_tools():
    from tools import get_tool_registry

    registry = get_tool_registry()

    assert registry.get_tool("send_email") is not None
    assert "create_task" in registry.list_tool_names()


def test_tool_registry_build_runtime_tools_extends_dynamic_tools():
    from tools.registry import create_tool_registry

    def fake_tool():
        return None

    fake_tool.name = "fake_tool"

    registry = create_tool_registry([fake_tool])
    runtime_tools = registry.build_runtime_tools(
        user_id="alice",
        dynamic_tool_builder=lambda user_id: [f"dynamic:{user_id}"],
    )

    assert runtime_tools[0] is fake_tool
    assert runtime_tools[1] == "dynamic:alice"


def test_tool_registry_groups_capabilities():
    from tools import get_tool_registry

    registry = get_tool_registry()
    capabilities = registry.list_capabilities()

    assert "mail" in capabilities
    assert "send_email" in capabilities["mail"]
    assert "tasks" in capabilities
