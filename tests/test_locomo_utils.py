import asyncio

import pytest

from benchmark.locomo_storage_utils import (
    build_conversation_memories_chunked,
    build_persona_map,
    normalize_locomo_evidence,
)
from scripts.download_benchmarks import convert_locomo_sample
from scripts.measure_recall import ensure_chroma_ready


def test_normalize_locomo_evidence_splits_dirty_values():
    assert normalize_locomo_evidence(["D8:6; D9:17", "D8:6", None, "junk D3:1"]) == [
        "D8:6",
        "D9:17",
        "D3:1",
    ]


def test_convert_locomo_sample_preserves_multimodal_and_normalizes_evidence():
    sample = {
        "sample_id": "conv-26",
        "conversation": {
            "speaker_a": "Caroline",
            "speaker_b": "Melanie",
            "session_1_date_time": "1:56 pm on 8 May, 2023",
            "session_1": [
                {
                    "speaker": "Melanie",
                    "dia_id": "D1:1",
                    "text": "Here is our latest work.",
                    "img_url": ["https://example.com/img.jpg"],
                    "blip_caption": "a painting of a sunset",
                    "query": "sunset painting",
                    "re-download": True,
                }
            ],
        },
        "qa": [
            {
                "question": "What did Melanie paint recently?",
                "evidence": ["D8:6; D9:17"],
                "category": 1,
            }
        ],
    }

    converted = convert_locomo_sample(sample)

    assert converted["sample_id"] == "conv-26"
    assert converted["test_cases"][0]["session_date"] == "2023-05-08"
    assert converted["test_cases"][0]["session_datetime_raw"] == "1:56 pm on 8 May, 2023"
    turn = converted["test_cases"][0]["conversation"][0]
    assert turn["blip_caption"] == "a painting of a sunset"
    assert turn["query"] == "sunset painting"
    assert turn["img_url"] == ["https://example.com/img.jpg"]
    assert turn["re_download"] is True
    assert converted["qa"][0]["evidence"] == ["D8:6", "D9:17"]
    assert converted["qa"][0]["evidence_raw"] == ["D8:6; D9:17"]


def test_build_persona_map_and_chunk_builder_keep_dates_and_captions():
    data = {
        "test_cases": [
            {
                "id": "conv-26_session_1",
                "session_date": "2023-05-08",
                "session_datetime_raw": "1:56 pm on 8 May, 2023",
                "conversation": [
                    {
                        "speaker": "Melanie",
                        "content": "Here is our latest work.",
                        "dia_id": "D1:1",
                        "blip_caption": "a painting of a sunset",
                        "query": "sunset painting",
                        "img_url": ["https://example.com/img.jpg"],
                    },
                    {
                        "speaker": "Caroline",
                        "content": "Looks great.",
                        "dia_id": "D1:2",
                    },
                ],
            }
        ],
        "qa": [
            {
                "question": "What did Melanie paint recently?",
                "evidence": ["D1:1; D1:2"],
                "category": 1,
            }
        ],
    }

    personas_convos, session_dates, session_datetimes_raw, personas_qa = build_persona_map(data)
    assert session_dates["conv-26"] == ["2023-05-08"]
    assert session_datetimes_raw["conv-26"] == ["1:56 pm on 8 May, 2023"]
    assert personas_qa["conv-26"][0]["evidence"] == ["D1:1", "D1:2"]

    memories = build_conversation_memories_chunked(
        conversations=personas_convos["conv-26"],
        persona="conv-26",
        turns_per_chunk=3,
        max_chunk_chars=1500,
        session_dates=session_dates["conv-26"],
        session_datetimes_raw=session_datetimes_raw["conv-26"],
    )

    assert len(memories) == 1
    assert memories[0]["session_date"] == "2023-05-08"
    assert memories[0]["session_datetime_raw"] == "1:56 pm on 8 May, 2023"
    assert "[Session 1 - 2023-05-08]" in memories[0]["content"]
    assert "[D1:1] Melanie: Here is our latest work." in memories[0]["content"]
    assert "[image_caption] a painting of a sunset" in memories[0]["content"]
    assert memories[0]["chunk_queries"] == ["sunset painting"]
    assert memories[0]["chunk_img_urls"] == ["https://example.com/img.jpg"]
    assert memories[0]["chunk_blip_captions"] == ["a painting of a sunset"]


def test_measure_recall_refuses_rebuild_existing_user_without_explicit_flag(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "scripts.measure_recall.find_existing_user_docs",
        lambda chroma_path, collection_name, user_id: 3,
    )

    with pytest.raises(RuntimeError, match="allow-rebuild-existing-user"):
        asyncio.run(
            ensure_chroma_ready(
                chroma_path=tmp_path / "chroma",
                collection_name="agent_memories",
                user_id="memory_test_conv-26",
                converted_path=tmp_path / "missing.json",
                turns_per_chunk=3,
                max_chunk_chars=1500,
                auto_bootstrap=True,
                force_bootstrap=True,
                allow_rebuild_existing_user=False,
            )
        )
