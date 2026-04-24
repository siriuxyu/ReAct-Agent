import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage, SystemMessage


def test_filter_messages_skips_generated_summary_messages():
    from agent.preference import filter_messages_for_extraction

    messages = [
        HumanMessage(content="我喜欢咖啡"),
        SystemMessage(
            content="[CONTEXT SUMMARY]: User likes coffee",
            additional_kwargs={"summary_generated": True},
        ),
    ]

    filtered = filter_messages_for_extraction(messages)
    assert len(filtered) == 1
    assert filtered[0].content == "我喜欢咖啡"
