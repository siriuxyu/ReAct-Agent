"""
Measure retrieval metrics for the LoCoMo memory benchmark.

For each QA question:
  1. Query ChromaDB with top-k
  2. Compare retrieved chunks against gold evidence dia_ids (e.g. [D1:3])
  3. Report:
       - Recall@k: average evidence coverage
       - AnyHit@k: fraction of questions with any gold dia_id found
       - AllHit@k: fraction of questions with all gold dia_ids found

Usage:
    python measure_recall.py [--k 5 10 25 50] [--user-id memory_test_conv-26]
"""

import argparse
import json
import os
import sys
import asyncio
import re
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv()
from benchmark.locomo_storage_utils import (
    bootstrap_persona_to_chroma,
    build_persona_map,
    infer_persona_from_user_id,
    load_json,
    normalize_locomo_evidence,
)

DEFAULT_CHROMA_PATH = Path(
    os.environ.get("CHROMA_PERSIST_PATH", str(REPO_ROOT / "chroma_db_data"))
).expanduser()
DEFAULT_LOCOMO_PATH = REPO_ROOT / "benchmark" / "locomo1.json"
DEFAULT_CONVERTED_PATH = REPO_ROOT / "benchmark" / "locomo1_converted.json"
COLLECTION_NAME = "agent_memories"
CATEGORY_NAMES = {
    1: "Identity/Profile",
    2: "Temporal",
    3: "Multi Hop",
    4: "Open Domain",
    5: "Adversarial",
}


def load_qa(locomo_path: Path):
    with open(locomo_path, encoding="utf-8") as f:
        data = json.load(f)
    qa_items = []
    for qa in data["qa"]:
        qa_copy = dict(qa)
        qa_copy["evidence"] = normalize_locomo_evidence(qa_copy.get("evidence"))
        qa_items.append(qa_copy)
    return qa_items


def get_collection(chroma_path: Path, collection_name: str):
    import chromadb

    client = chromadb.PersistentClient(path=str(chroma_path))
    return client.get_collection(collection_name)


def find_existing_user_docs(chroma_path: Path, collection_name: str, user_id: str) -> int:
    if not chroma_path.exists():
        return 0

    import chromadb

    client = chromadb.PersistentClient(path=str(chroma_path))
    try:
        collection = client.get_collection(collection_name)
    except Exception:
        return 0

    try:
        results = collection.get(where={"user_id": user_id}, limit=1, include=[])
        return len(results.get("ids", []))
    except Exception:
        return 0


async def ensure_chroma_ready(
    *,
    chroma_path: Path,
    collection_name: str,
    user_id: str,
    converted_path: Path,
    turns_per_chunk: int,
    max_chunk_chars: int,
    auto_bootstrap: bool,
    force_bootstrap: bool,
    allow_rebuild_existing_user: bool,
) -> None:
    existing_docs = find_existing_user_docs(chroma_path, collection_name, user_id)
    if existing_docs > 0 and not force_bootstrap:
        return
    if existing_docs > 0 and force_bootstrap and not allow_rebuild_existing_user:
        raise RuntimeError(
            "Refusing to rebuild an existing user_id namespace without explicit confirmation. "
            "Pass --allow-rebuild-existing-user if you really want to overwrite stored memories "
            f"for user_id={user_id}."
        )

    if not auto_bootstrap:
        raise FileNotFoundError(
            f"No stored memories found for user_id={user_id} in {chroma_path}. "
            "Remove --no-auto-bootstrap or create the Chroma data first."
        )

    if not converted_path.exists():
        raise FileNotFoundError(f"LoCoMo converted file not found: {converted_path}")

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required to bootstrap Chroma because chunk embeddings must be created."
        )

    converted_data = load_json(converted_path)
    personas_convos, personas_session_dates, personas_session_datetimes_raw, _ = build_persona_map(
        converted_data
    )
    personas = sorted(personas_convos.keys())
    persona = infer_persona_from_user_id(user_id, personas)
    if not persona:
        raise ValueError(
            f"Could not infer persona from user_id={user_id}. Available personas: {', '.join(personas)}"
        )

    print(
        f"[bootstrap] {'Rebuilding' if existing_docs > 0 else 'Building'} Chroma data "
        f"for {user_id}, persona {persona}..."
    )
    stats = await bootstrap_persona_to_chroma(
        persona=persona,
        userid=user_id,
        conversations=personas_convos[persona],
        session_dates=personas_session_dates.get(persona),
        session_datetimes_raw=personas_session_datetimes_raw.get(persona),
        persist_path=chroma_path,
        collection_name=collection_name,
        openai_api_key=api_key,
        turns_per_chunk=turns_per_chunk,
        max_chunk_chars=max_chunk_chars,
        clear_existing=True,
    )
    print(
        f"[bootstrap] Stored {stats['stored_count']} chunks "
        f"({stats['total_chars']} chars) into {chroma_path}"
    )


async def embed_query(text: str) -> list:
    """Embed a query using the same embedding service as storage."""
    from agent.storage.embedding_service import OpenAIEmbeddingService
    api_key = os.environ.get("OPENAI_API_KEY", "")
    svc = OpenAIEmbeddingService(api_key=api_key, model="text-embedding-3-small")
    return await svc.embed_text(text)


def extract_dia_ids(retrieved_docs: list) -> list[set[str]]:
    """Extract dia_ids from retrieved docs, where stored form is like [D1:3]."""
    pattern = re.compile(r"\[(D\d+:\d+)\]")
    return [set(pattern.findall(doc)) for doc in retrieved_docs]


async def measure_recall(k_values: list, user_id: str, chroma_path: Path, locomo_path: Path):
    if not locomo_path.exists():
        raise FileNotFoundError(f"LoCoMo file not found: {locomo_path}")

    qa_pairs = load_qa(locomo_path)
    col = get_collection(chroma_path, COLLECTION_NAME)

    print(f"Total QA pairs: {len(qa_pairs)}")
    print(f"ChromaDB docs: {col.count()}")
    print(f"User ID filter: {user_id}")
    print()

    # Results per k and per category
    results = {
        k: {
            "recall_sum": 0.0,
            "any_hits": 0,
            "all_hits": 0,
            "total": 0,
            "by_cat": defaultdict(
                lambda: {"recall_sum": 0.0, "any_hits": 0, "all_hits": 0, "total": 0}
            ),
        }
        for k in k_values
    }
    question_details = []

    for i, qa in enumerate(qa_pairs):
        question = qa.get("question", "")
        evidence = qa.get("evidence", [])
        category = qa.get("category", 0)

        # Skip questions with no evidence (adversarial Cat 5)
        if not evidence:
            print(f"  [{i+1:3d}] Cat {category} - no evidence, skipping")
            continue

        # Embed query
        try:
            query_embedding = await embed_query(question)
        except Exception as e:
            print(f"  [{i+1:3d}] Embedding failed: {e}")
            continue

        # Query at max k, then slice
        max_k = max(k_values)
        try:
            result = col.query(
                query_embeddings=[query_embedding],
                n_results=min(max_k, col.count()),
                where={"user_id": user_id},
                include=["documents"]
            )
            retrieved = result["documents"][0]  # list of doc texts
        except Exception as e:
            print(f"  [{i+1:3d}] Query failed: {e}")
            continue

        extracted = extract_dia_ids(retrieved)
        retrieved_ids = result["ids"][0] if result.get("ids") else []
        gold = set(evidence)

        # Check hit at each k
        hit_info = {}
        per_k_metrics = {}
        for k in k_values:
            top_hits = set().union(*extracted[:k]) if extracted[:k] else set()
            matched = gold & top_hits
            recall = len(matched) / len(gold) if gold else 0.0
            any_hit = bool(matched)
            all_hit = matched == gold and bool(gold)
            per_k_metrics[k] = {
                "matched": sorted(matched),
                "recall": recall,
                "any_hit": any_hit,
                "all_hit": all_hit,
            }
            results[k]["total"] += 1
            results[k]["by_cat"][category]["total"] += 1
            results[k]["recall_sum"] += recall
            results[k]["by_cat"][category]["recall_sum"] += recall
            if any_hit:
                results[k]["any_hits"] += 1
                results[k]["by_cat"][category]["any_hits"] += 1
            if all_hit:
                results[k]["all_hits"] += 1
                results[k]["by_cat"][category]["all_hits"] += 1
            hit_info[k] = f"R={recall:.2f}/Any={'Y' if any_hit else 'n'}/All={'Y' if all_hit else 'n'}"

        evidence_str = ", ".join(evidence[:3])
        hit_str = "  ".join(f"@{k}:{hit_info[k]}" for k in k_values)
        print(f"  [{i+1:3d}] Cat {category}  {hit_str}  evidence=[{evidence_str}]")
        question_details.append(
            {
                "question_index": i + 1,
                "category": category,
                "question": question,
                "answer": qa.get("answer"),
                "evidence": evidence,
                "retrieved_ids": retrieved_ids,
                "metrics_by_k": per_k_metrics,
            }
        )

    # Summary
    print()
    print("=" * 60)
    print(f"{'k':<8} {'Recall@k':<12} {'AnyHit@k':<18} {'AllHit@k':<18} {'Total'}")
    print("-" * 60)
    for k in k_values:
        r = results[k]
        total = r["total"]
        recall = r["recall_sum"] / total if total > 0 else 0
        any_rate = r["any_hits"] / total if total > 0 else 0
        all_rate = r["all_hits"] / total if total > 0 else 0
        print(
            f"@{k:<7} {recall:.1%}       "
            f"{r['any_hits']}/{total} ({any_rate:.1%})   "
            f"{r['all_hits']}/{total} ({all_rate:.1%})   "
            f"{total}"
        )

    print()
    print("By category (@10 if present, else first k):")
    k10 = 10 if 10 in k_values else k_values[0]
    for cat in sorted(results[k10]["by_cat"].keys()):
        c = results[k10]["by_cat"][cat]
        total = c["total"]
        recall = c["recall_sum"] / total if total > 0 else 0
        any_rate = c["any_hits"] / total if total > 0 else 0
        all_rate = c["all_hits"] / total if total > 0 else 0
        name = CATEGORY_NAMES.get(cat, f"Cat {cat}")
        print(
            f"  Cat {cat} ({name}): "
            f"Recall={recall:.1%}  AnyHit={any_rate:.1%}  AllHit={all_rate:.1%}  (n={total})"
        )

    print()
    print(f"Cat 1 miss cases by Recall@{k10} (top 10):")
    cat1_cases = [q for q in question_details if q["category"] == 1]
    cat1_cases.sort(
        key=lambda q: (
            q["metrics_by_k"][k10]["recall"],
            q["metrics_by_k"][k10]["any_hit"],
            q["question_index"],
        )
    )
    for item in cat1_cases[:10]:
        metrics = item["metrics_by_k"][k10]
        print(
            f"  [{item['question_index']:3d}] "
            f"Recall={metrics['recall']:.2f} "
            f"Any={'Y' if metrics['any_hit'] else 'n'} "
            f"All={'Y' if metrics['all_hit'] else 'n'} "
            f"matched={metrics['matched']}/{item['evidence']}"
        )
        print(f"        Q: {item['question']}")
        print(f"        retrieved: {item['retrieved_ids'][:3]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", nargs="+", type=int, default=[5, 10, 25, 50])
    parser.add_argument("--user-id", default="memory_test_conv-26")
    parser.add_argument("--chroma-path", default=str(DEFAULT_CHROMA_PATH))
    parser.add_argument("--locomo-path", default=str(DEFAULT_LOCOMO_PATH))
    parser.add_argument("--converted-path", default=str(DEFAULT_CONVERTED_PATH))
    parser.add_argument(
        "--no-auto-bootstrap",
        action="store_true",
        help="Do not auto-create Chroma data when the collection or user memories are missing.",
    )
    parser.add_argument(
        "--force-bootstrap",
        action="store_true",
        help="Rebuild Chroma memories for this user_id even if stored docs already exist.",
    )
    parser.add_argument(
        "--allow-rebuild-existing-user",
        action="store_true",
        help="Required with --force-bootstrap when the target user_id already has stored docs.",
    )
    parser.add_argument("--turns-per-chunk", type=int, default=3)
    parser.add_argument("--max-chunk-chars", type=int, default=1500)
    args = parser.parse_args()

    chroma_path = Path(args.chroma_path).expanduser()
    locomo_path = Path(args.locomo_path).expanduser()
    converted_path = Path(args.converted_path).expanduser()

    asyncio.run(
        ensure_chroma_ready(
            chroma_path=chroma_path,
            collection_name=COLLECTION_NAME,
            user_id=args.user_id,
            converted_path=converted_path,
            turns_per_chunk=args.turns_per_chunk,
            max_chunk_chars=args.max_chunk_chars,
            auto_bootstrap=not args.no_auto_bootstrap,
            force_bootstrap=args.force_bootstrap,
            allow_rebuild_existing_user=args.allow_rebuild_existing_user,
        )
    )
    asyncio.run(measure_recall(args.k, args.user_id, chroma_path, locomo_path))
