import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import AIMessage, ToolMessage


def test_collect_response_messages_skips_tool_payloads():
    from agent.adapters.langgraph_adapter import collect_response_messages

    chunk = {
        "tools": {
            "messages": [
                ToolMessage(content="{}", tool_call_id="tool-1", name="search"),
                AIMessage(content="最终回复"),
            ]
        }
    }

    messages = collect_response_messages(chunk)

    assert len(messages) == 1
    assert messages[0]["content"] == "最终回复"


def test_extract_text_from_chunk_handles_text_blocks():
    from agent.adapters.langgraph_adapter import extract_text_from_chunk

    chunk = {
        "call_model": {
            "messages": [
                AIMessage(
                    content=[
                        {"type": "server_tool_use", "name": "web_search"},
                        {"type": "text", "text": "这里是最后答案"},
                    ]
                )
            ]
        }
    }

    assert extract_text_from_chunk(chunk) == "这里是最后答案"
