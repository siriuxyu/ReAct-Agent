#!/usr/bin/env python3
"""Download benchmark datasets and generate the local LoCoMo files.

This script keeps large/plain benchmark JSON files out of git while preserving
the exact on-disk layout expected by the repo.
"""

from __future__ import annotations

import argparse
import json
import ssl
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional
from urllib.request import urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.locomo_storage_utils import normalize_locomo_evidence

OFFICIAL_LOCOMO_URL = (
    "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
)


def download_json(url: str) -> Any:
    context = ssl.create_default_context()
    with urlopen(url, context=context) as response:
        return json.load(response)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def iter_session_numbers(conversation: Dict[str, Any]) -> Iterable[int]:
    seen = set()
    for key in conversation:
        if not key.startswith("session_") or key.endswith("_date_time"):
            continue
        try:
            seen.add(int(key.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(seen)


def parse_session_date(date_str: str) -> Optional[str]:
    """Parse LoCoMo date string like '4:10 pm on 26 October, 2023' -> '2023-10-26'."""
    import re
    m = re.search(r"(\d+)\s+(\w+),\s+(\d{4})", date_str)
    if not m:
        return None
    try:
        return datetime.strptime(
            f"{m.group(1)} {m.group(2)} {m.group(3)}", "%d %B %Y"
        ).strftime("%Y-%m-%d")
    except ValueError:
        return None


def convert_locomo_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    conversation = sample["conversation"]
    speaker_a = conversation["speaker_a"]
    speaker_b = conversation["speaker_b"]
    sample_id = sample["sample_id"]

    test_cases: List[Dict[str, Any]] = []
    for session_num in iter_session_numbers(conversation):
        turns = conversation.get(f"session_{session_num}") or []
        if not isinstance(turns, list) or not turns:
            continue

        converted_turns = []
        for turn in turns:
            speaker = turn.get("speaker", "")
            if speaker == speaker_a:
                role = "user"
            elif speaker == speaker_b:
                role = "agent_expected"
            else:
                role = "user"

            converted_turns.append(
                {
                    "role": role,
                    "speaker": speaker,
                    "content": turn.get("text", ""),
                    "dia_id": turn.get("dia_id"),
                    "img_url": turn.get("img_url"),
                    "blip_caption": turn.get("blip_caption"),
                    "query": turn.get("query"),
                    "re_download": turn.get("re-download"),
                }
            )

        raw_date = conversation.get(f"session_{session_num}_date_time", "")
        session_date = parse_session_date(raw_date) if raw_date else None

        test_cases.append(
            {
                "id": f"{sample_id}_session_{session_num}",
                "type": "extra_session_context",
                "expected_tools": [],
                "session_date": session_date,
                "session_datetime_raw": raw_date or None,
                "conversation": converted_turns,
            }
        )

    qa_items = []
    for qa in sample.get("qa", []):
        qa_copy = dict(qa)
        raw_evidence = qa_copy.get("evidence") or []
        normalized_evidence = normalize_locomo_evidence(raw_evidence)
        qa_copy["evidence"] = normalized_evidence
        if normalized_evidence != raw_evidence:
            qa_copy["evidence_raw"] = raw_evidence
        qa_items.append(qa_copy)

    return {
        "sample_id": sample_id,
        "test_cases": test_cases,
        "qa": qa_items,
    }


def find_sample(samples: Any, sample_id: str) -> Dict[str, Any]:
    if isinstance(samples, dict):
        if samples.get("sample_id") == sample_id:
            return samples
        raise ValueError(f"Expected a list of samples, got dict without {sample_id}")

    if not isinstance(samples, list):
        raise ValueError("Official LoCoMo payload has unexpected schema")

    for sample in samples:
        if isinstance(sample, dict) and sample.get("sample_id") == sample_id:
            return sample

    raise ValueError(f"Sample {sample_id!r} not found in official LoCoMo data")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download benchmark files and regenerate local LoCoMo fixtures."
    )
    parser.add_argument(
        "--benchmark-dir",
        default="benchmark",
        help="Directory to populate (default: ./benchmark)",
    )
    parser.add_argument(
        "--locomo-sample-id",
        default="conv-26",
        help="Official LoCoMo sample_id to extract into locomo1.json",
    )
    args = parser.parse_args(argv)

    benchmark_dir = Path(args.benchmark_dir).resolve()

    print(f"Writing benchmark files into {benchmark_dir}")

    official_samples = download_json(OFFICIAL_LOCOMO_URL)
    locomo_sample = find_sample(official_samples, args.locomo_sample_id)
    write_json(benchmark_dir / "locomo1.json", locomo_sample)
    write_json(
        benchmark_dir / "locomo1_converted.json",
        convert_locomo_sample(locomo_sample),
    )
    print(f"Downloaded official LoCoMo sample {args.locomo_sample_id} -> locomo1.json")
    print("Generated locomo1_converted.json from the official sample")

    return 0


if __name__ == "__main__":
    sys.exit(main())
