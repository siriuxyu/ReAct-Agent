"""
Measure Recall@k for the LoCoMo memory benchmark.

For each QA question:
  1. Query ChromaDB with top-k
  2. Check if any retrieved chunk contains the evidence dia_ids (e.g. [D1:3])
  3. Recall@k = fraction of questions where evidence was found

Usage:
    python measure_recall.py [--k 5 10 25 50] [--user-id memory_test_conv-26]
"""

import argparse
import json
import os
import sys
import asyncio
from collections import defaultdict

# Use the agent env's python - imports
sys.path.insert(0, '/home/siriux/Projects/ReAct')

from dotenv import load_dotenv
load_dotenv()

import chromadb

CHROMA_PATH = "/home/siriux/Projects/ReAct/chroma_db_data"
LOCOMO_PATH = "/home/siriux/Projects/ReAct/benchmark/locomo1.json"
COLLECTION_NAME = "langmem_memories"


def load_qa(locomo_path: str):
    with open(locomo_path) as f:
        data = json.load(f)
    return data["qa"]


def get_collection(chroma_path: str, collection_name: str):
    client = chromadb.PersistentClient(path=chroma_path)
    return client.get_collection(collection_name)


async def embed_query(text: str) -> list:
    """Embed a query using the same embedding service as storage."""
    from agent.storage.embedding_service import OpenAIEmbeddingService
    api_key = os.environ.get("OPENAI_API_KEY", "")
    svc = OpenAIEmbeddingService(api_key=api_key, model="text-embedding-3-small")
    return await svc.embed_text(text)


def check_hit(retrieved_docs: list, evidence: list) -> bool:
    """Check if any retrieved doc contains any of the evidence dia_ids."""
    if not evidence:
        return False
    for doc in retrieved_docs:
        for dia_id in evidence:
            # dia_id is like "D1:3" - stored in chunks as "[D1:3]"
            if f"[{dia_id}]" in doc:
                return True
    return False


async def measure_recall(k_values: list, user_id: str):
    qa_pairs = load_qa(LOCOMO_PATH)
    col = get_collection(CHROMA_PATH, COLLECTION_NAME)

    print(f"Total QA pairs: {len(qa_pairs)}")
    print(f"ChromaDB docs: {col.count()}")
    print(f"User ID filter: {user_id}")
    print()

    # Results per k and per category
    results = {k: {"hits": 0, "total": 0, "by_cat": defaultdict(lambda: {"hits": 0, "total": 0})} for k in k_values}

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

        # Check hit at each k
        hit_info = {}
        for k in k_values:
            top_k_docs = retrieved[:k]
            hit = check_hit(top_k_docs, evidence)
            results[k]["total"] += 1
            results[k]["by_cat"][category]["total"] += 1
            if hit:
                results[k]["hits"] += 1
                results[k]["by_cat"][category]["hits"] += 1
            hit_info[k] = "HIT" if hit else "miss"

        evidence_str = ", ".join(evidence[:3])
        hit_str = "  ".join(f"@{k}:{hit_info[k]}" for k in k_values)
        print(f"  [{i+1:3d}] Cat {category}  {hit_str}  evidence=[{evidence_str}]")

    # Summary
    print()
    print("=" * 60)
    print(f"{'k':<8} {'Recall@k':<12} {'Hits':<8} {'Total'}")
    print("-" * 60)
    for k in k_values:
        r = results[k]
        total = r["total"]
        hits = r["hits"]
        recall = hits / total if total > 0 else 0
        print(f"@{k:<7} {recall:.1%}       {hits:<8} {total}")

    print()
    print("By category (Recall@10):")
    cat_names = {1: "Single Hop", 2: "Temporal", 3: "Multi Hop", 4: "Open Domain", 5: "Adversarial"}
    k10 = 10 if 10 in k_values else k_values[0]
    for cat in sorted(results[k10]["by_cat"].keys()):
        c = results[k10]["by_cat"][cat]
        total = c["total"]
        hits = c["hits"]
        recall = hits / total if total > 0 else 0
        name = cat_names.get(cat, f"Cat {cat}")
        print(f"  Cat {cat} ({name}): {recall:.1%}  ({hits}/{total})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", nargs="+", type=int, default=[5, 10, 25, 50])
    parser.add_argument("--user-id", default="memory_test_conv-26")
    args = parser.parse_args()

    asyncio.run(measure_recall(args.k, args.user_id))
