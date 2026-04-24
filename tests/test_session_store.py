import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_session_store_persists_and_recalls_messages(tmp_path):
    from agent.memory.session_store import SessionStore

    store = SessionStore(tmp_path / "sessions.db")
    store.add_message(
        user_id="alice",
        session_id="alice_old",
        role="user",
        content="We decided to send the internship update on Friday.",
    )
    store.add_message(
        user_id="alice",
        session_id="alice_old",
        role="assistant",
        content="Reminder noted: send the internship update on Friday afternoon.",
    )

    results = store.search_messages(user_id="alice", query="internship update friday", limit=2)

    assert len(results) == 2
    assert results[0]["score"] >= results[1]["score"]
    assert "Friday" in results[0]["content"]


def test_session_recall_block_formats_results(tmp_path):
    from agent.memory.recall import build_session_recall_block
    from agent.memory.session_store import SessionStore

    store = SessionStore(tmp_path / "sessions.db")
    store.add_message(
        user_id="bob",
        session_id="bob_old",
        role="assistant",
        content="You prefer concise weekly summaries instead of daily reports.",
    )

    block = build_session_recall_block(
        user_id="bob",
        query="weekly summaries",
        store=store,
    )

    assert block.startswith("## Recalled session context")
    assert "bob_old" in block
    assert "concise weekly summaries" in block
