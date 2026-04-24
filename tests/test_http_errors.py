import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_is_graph_recursion_error_matches_message_and_type_name():
    from agent.adapters.http_errors import is_graph_recursion_error

    class FakeGraphRecursionError(Exception):
        pass

    assert is_graph_recursion_error(FakeGraphRecursionError("boom"), FakeGraphRecursionError) is True
    assert is_graph_recursion_error(RuntimeError("hit recursion limit"), None) is True


def test_recursion_error_message_mentions_tool_limit():
    from agent.adapters.http_errors import RECURSION_ERROR_MESSAGE

    assert "maximum tool call limit" in RECURSION_ERROR_MESSAGE
