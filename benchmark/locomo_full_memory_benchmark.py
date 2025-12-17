#!/usr/bin/env python
"""
LoCoMo full memory-system benchmark runner.

This is the "store -> reset -> retrieve -> answer" benchmark path:
1. Store all conversation history in long-term memory chunks
2. Optionally extract per-session facts
3. Reset short-term session state while preserving long-term memory
4. Ask QA questions that must be answered via memory retrieval

Use this script when you want the end-to-end memory benchmark.
For the other LoCoMo scripts in this repo:
- benchmark_runner.py: evidence-fed QA, not a true memory benchmark
- locomo_simplified_memory_benchmark.py: simplified "preload transcript then ask" benchmark
- locomo_full_memory_benchmark.py: full long-term memory benchmark
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
import time
import uuid
import requests
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional, Set

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.extraction.fact_extraction import extract_session_observations
from benchmark_runner import (
    load_json,
    normalize_text,
    call_agent,
    reset_short_term_session,
)
from locomo_storage_utils import (
    build_conversation_memories_chunked,
    build_persona_map,
)

# Chunking settings
DEFAULT_TURNS_PER_CHUNK = 3  # Number of conversation turns per memory chunk
DEFAULT_MAX_CHUNK_CHARS = 1500  # Max chars per chunk (~375 tokens)

# QA batch settings (for Phase 2)
DEFAULT_MAX_QA_CHARS = 2000  # ~500 tokens per QA batch
DEFAULT_MAX_QUESTIONS_PER_BATCH = 3  # Questions per API call

# Rate limiting
DEFAULT_DELAY_SEC = 30.0  # Increased to avoid rate limiting (agent may call search_memory multiple times)
DEFAULT_MEMORY_STORE_DELAY = 1.0  # Shorter delay for memory storage (no LLM call)
DEFAULT_MAX_RETRIES = 2  # Retries for failed batches

# Checkpoint settings
DEFAULT_CHECKPOINT_FILE = "locomo_checkpoint.json"


# =============================================================================
# Checkpoint Management
# =============================================================================

def load_checkpoint(checkpoint_file: str) -> Optional[Dict[str, Any]]:
    """Load checkpoint from file if it exists."""
    if not os.path.exists(checkpoint_file):
        return None
    try:
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)
        print(f"[CHECKPOINT] Loaded checkpoint from {checkpoint_file}")
        print(f"  Phase 1 completed: {len(checkpoint.get('phase1_completed', []))} personas")
        print(f"  Phase 2 completed: {len(checkpoint.get('phase2_completed', []))} personas")
        return checkpoint
    except Exception as e:
        print(f"[WARN] Failed to load checkpoint: {e}")
        return None


def save_checkpoint(
    checkpoint_file: str,
    phase1_completed: List[str],
    phase2_completed: List[str],
    storage_stats: Dict[str, Any],
    results: List[Dict[str, Any]],
    config: Dict[str, Any],
    current_persona_progress: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save current progress to checkpoint file.
    
    Args:
        current_persona_progress: Partial progress for persona currently being processed
            {"persona": str, "qa_results": List[...]}
    """
    checkpoint = {
        "timestamp": datetime.now().isoformat(),
        "phase1_completed": phase1_completed,
        "phase2_completed": phase2_completed,
        "storage_stats": storage_stats,
        "results": results,
        "config": config,
    }
    if current_persona_progress:
        checkpoint["current_persona_progress"] = current_persona_progress
    
    try:
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] Failed to save checkpoint: {e}")


def delete_checkpoint(checkpoint_file: str, backup: bool = True) -> None:
    """
    Delete checkpoint file after successful completion.
    
    Args:
        checkpoint_file: Path to checkpoint file
        backup: If True, create a backup copy before deleting
    """
    if os.path.exists(checkpoint_file):
        try:
            if backup:
                # Create backup with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_file = checkpoint_file.replace(".json", f"_final_{timestamp}.json")
                shutil.copy2(checkpoint_file, backup_file)
                print(f"[CHECKPOINT] Backed up to: {backup_file}")
            
            os.remove(checkpoint_file)
            print(f"[CHECKPOINT] Deleted checkpoint file: {checkpoint_file}")
        except Exception as e:
            print(f"[WARN] Failed to delete checkpoint: {e}")


# =============================================================================
# Utility Functions
# =============================================================================

def estimate_tokens(text: str) -> int:
    """Estimate token count (roughly 4 chars per token)."""
    return len(text) // 4


def throttle(delay_sec: float) -> None:
    if delay_sec > 0:
        time.sleep(delay_sec)


# =============================================================================
# Memory Storage Functions
# =============================================================================

def store_memory(
    server_url: str,
    userid: str,
    key: str,
    content: str,
    timeout: float,
    document_type: Optional[str] = None,
) -> bool:
    """Store a memory via the /memory API."""
    url = f"{server_url.rstrip('/')}/memory/{userid}/store"
    payload: Dict[str, Any] = {"key": key, "content": content}
    if document_type:
        payload["document_type"] = document_type
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        return resp.status_code == 200
    except Exception as e:
        print(f"[WARN] Failed to store memory: {e}")
        return False


def clear_user_memories(
    server_url: str,
    userid: str,
    timeout: float,
) -> bool:
    """Clear all memories for a user via DELETE /memory/{userid}."""
    url = f"{server_url.rstrip('/')}/memory/{userid}"
    try:
        resp = requests.delete(url, timeout=timeout)
        return resp.status_code == 200
    except Exception as e:
        print(f"[WARN] Failed to clear memories: {e}")
        return False


def list_memories(
    server_url: str,
    userid: str,
    timeout: float,
) -> List[Dict[str, Any]]:
    """List all memories for a user."""
    url = f"{server_url.rstrip('/')}/memory/{userid}"
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200:
            return resp.json().get("memories", [])
    except Exception as e:
        print(f"[WARN] Failed to list memories: {e}")
    return []


def search_memories(
    server_url: str,
    userid: str,
    query: str,
    limit: int,
    timeout: float,
) -> List[Dict[str, Any]]:
    """Search memories for a user via POST /memory/{userid}/search."""
    url = f"{server_url.rstrip('/')}/memory/{userid}/search"
    payload = {"query": query, "limit": limit}
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", [])
    except Exception as e:
        print(f"[WARN] Failed to search memories: {e}")
        return []


async def store_all_persona_memories(
    persona: str,
    conversations: List[List[Dict[str, Any]]],
    server_url: str,
    userid: str,
    timeout: float,
    memory_store_delay: float,
    turns_per_chunk: int,
    max_chunk_chars: int,
    skip_clear: bool = False,
    session_dates: Optional[List[str]] = None,
    session_datetimes_raw: Optional[List[str]] = None,
    model: str = "anthropic/claude-haiku-4-5-20251001",
    extract_observations: bool = True,
) -> Tuple[int, int]:
    """
    Store all conversation memories for a persona (no truncation).

    Args:
        skip_clear: If True, don't clear existing memories (for resume)
        session_dates: Optional ISO date strings per session for chunk headers
        model: Model used for observation extraction
        extract_observations: If True, also extract and store per-session facts

    Returns:
        (stored_count, total_chars)
    """
    # Clear existing memories (unless resuming with existing memories)
    if not skip_clear:
        clear_user_memories(server_url, userid, timeout)
        throttle(memory_store_delay)

    # Build chunked memories
    memories = build_conversation_memories_chunked(
        conversations, persona, turns_per_chunk, max_chunk_chars,
        session_dates=session_dates,
        session_datetimes_raw=session_datetimes_raw,
    )

    stored_count = 0
    total_chars = 0
    total_memories = len(memories)
    print(f"  Storing {total_memories} chunks (1s delay each, ~{total_memories}s)...")

    for chunk_idx, mem in enumerate(memories, start=1):
        content_chars = len(mem["content"])
        total_chars += content_chars

        success = store_memory(
            server_url=server_url,
            userid=userid,
            key=mem["key"],
            content=mem["content"],
            timeout=timeout,
        )
        if success:
            stored_count += 1

        if chunk_idx % 10 == 0 or chunk_idx == total_memories:
            print(f"  [{chunk_idx}/{total_memories}] chunks stored", flush=True)

        throttle(memory_store_delay)

    # Phase 1b: extract and store per-session observations
    if extract_observations:
        print(f"  Extracting observations for {len(conversations)} sessions...")
        for session_idx, session_turns in enumerate(conversations, start=1):
            date_str = ""
            if session_dates and session_idx <= len(session_dates):
                date_str = f" - {session_dates[session_idx - 1]}"

            # Build plain session text for LLM
            lines = [f"[Session {session_idx}{date_str}]"]
            for turn in session_turns:
                speaker = turn.get("speaker") or turn.get("role") or "Unknown"
                text = turn.get("content") or turn.get("text", "")
                if text:
                    lines.append(f"{speaker}: {text}")
                blip_caption = turn.get("blip_caption")
                if blip_caption:
                    lines.append(f"[image_caption] {blip_caption}")
            session_text = "\n".join(lines)

            try:
                observations = await extract_session_observations(session_text, model)
                if observations:
                    obs_key = f"{persona}_obs_s{session_idx}"
                    obs_content = f"[Session {session_idx}{date_str} - Facts]\n{observations}"
                    success = store_memory(
                        server_url=server_url,
                        userid=userid,
                        key=obs_key,
                        content=obs_content,
                        timeout=timeout,
                        document_type="extracted_fact",
                    )
                    if success:
                        stored_count += 1
                        total_chars += len(obs_content)
                    print(f"    Session {session_idx}: {len(observations.splitlines())} facts extracted")
            except Exception as e:
                print(f"    [WARN] Session {session_idx} observation extraction failed: {e}")

            throttle(memory_store_delay)

    return stored_count, total_chars


# =============================================================================
# QA Functions
# =============================================================================

def build_qa_batch_prompt(
    qa_items: List[Dict[str, Any]],
    start_idx: int = 1,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Build QA prompt for a batch of questions.

    Returns:
        (prompt_text, payload_with_expected_answers)
    """
    header = (
        "You have access to stored memories about past conversations. "
        "Search your memory thoroughly to answer each question.\n\n"
        "Rules:\n"
        "- Answer concisely (under 10 words when possible)\n"
        "- Use absolute dates/years (e.g. '2022', 'May 2023'), never relative terms like 'last year' or 'recently'\n"
        "- For multi-part questions, search memory multiple times with different queries\n"
        "- If you truly cannot find the answer after searching, write 'Unknown'\n"
        "- Return ONLY JSON with schema: {\"answers\": [{\"question_id\": \"...\", \"answer\": \"...\"}]}\n\n"
        "Questions:"
    )

    instructions = [header]
    payload: List[Dict[str, Any]] = []

    for idx, qa in enumerate(qa_items, start=start_idx):
        qid = qa.get("id") or f"q{idx}"
        question = qa.get("question", "")
        category = qa.get("category", 0)

        # Add category-specific hint
        hint = ""
        if category == 3:
            hint = " [Hint: requires combining info from multiple memories — search multiple times]"
        elif category == 2:
            hint = " [Hint: answer with exact year or date, not relative time]"

        question_text = f"{idx}. question_id={qid}\n   Question: {question}{hint}"
        instructions.append(question_text)
        payload.append({
            "id": qid,
            "question": question,
            "expected": qa.get("answer"),
            "category": category,
        })

    prompt = "\n".join(instructions)
    return prompt, payload


def chunk_qa_items(
    qa_items: List[Dict[str, Any]],
    max_chars: int = DEFAULT_MAX_QA_CHARS,
    max_questions: int = DEFAULT_MAX_QUESTIONS_PER_BATCH,
) -> List[List[Dict[str, Any]]]:
    """
    Split QA items into batches that fit within token limits.
    
    Args:
        qa_items: All QA items for a persona
        max_chars: Max characters per batch prompt
        max_questions: Max questions per batch
    
    Returns:
        List of QA item batches
    """
    batches = []
    current_batch = []
    current_chars = 300  # Header overhead
    
    for qa in qa_items:
        question = qa.get("question", "")
        question_chars = len(question) + 50  # Overhead for formatting
        
        # Check if we need to start a new batch
        should_break = (
            len(current_batch) >= max_questions or
            (current_chars + question_chars > max_chars and current_batch)
        )
        
        if should_break:
            batches.append(current_batch)
            current_batch = []
            current_chars = 300
        
        current_batch.append(qa)
        current_chars += question_chars
    
    # Don't forget the last batch
    if current_batch:
        batches.append(current_batch)
    
    return batches


def parse_json_answers(raw: str) -> Dict[str, str]:
    """Parse JSON answers from agent response, handling markdown code blocks."""
    import re
    
    clean_raw = raw.strip()
    
    # Handle ```json ... ``` or ``` ... ```
    code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', clean_raw)
    if code_block_match:
        clean_raw = code_block_match.group(1).strip()
    
    # Try to extract JSON object
    try:
        data = json.loads(clean_raw)
    except json.JSONDecodeError:
        # Try to find JSON object in raw text
        json_match = re.search(r'\{[\s\S]*\}', clean_raw)
        if json_match:
            try:
                data = json.loads(json_match.group())
            except json.JSONDecodeError:
                return {}
        else:
            return {}
    
    if isinstance(data, dict) and "answers" in data:
        answers = data["answers"]
    else:
        answers = data
    
    result: Dict[str, str] = {}
    if isinstance(answers, dict):
        for key, value in answers.items():
            result[str(key)] = str(value)
    elif isinstance(answers, list):
        for entry in answers:
            if isinstance(entry, dict):
                qid = entry.get("question_id") or entry.get("id")
                ans = entry.get("answer")
                if qid and ans:
                    result[str(qid)] = str(ans)
    return result


def _normalize_dates(text: str) -> str:
    """
    Normalize date expressions to a canonical form for flexible matching.

    Converts formats like "7 May 2023", "May 7, 2023", "May 7 2023" all to
    "7 may 2023" so substring matching works across format variants.
    """
    import re
    months = {
        "january": "1", "february": "2", "march": "3", "april": "4",
        "may": "5", "june": "6", "july": "7", "august": "8",
        "september": "9", "october": "10", "november": "11", "december": "12",
        "jan": "1", "feb": "2", "mar": "3", "apr": "4",
        "jun": "6", "jul": "7", "aug": "8", "sep": "9", "oct": "10",
        "nov": "11", "dec": "12",
    }
    result = text.lower()
    # "May 7, 2023" or "May 7 2023" → "7 may 2023"
    result = re.sub(
        r'\b(' + '|'.join(months.keys()) + r')[,\s]+(\d{1,2})[,\s]+(\d{4})\b',
        lambda m: f"{m.group(2)} {m.group(1)} {m.group(3)}",
        result,
    )
    # Remove commas left in dates
    result = re.sub(r'(\d),(\s)', r'\1\2', result)
    return result


def _is_correct(expected: str, answer: str) -> bool:
    """
    Flexible correctness check:
    1. Exact substring match (normalized)
    2. Date-normalized substring match
    3. Token-F1 >= 0.5 (soft match for paraphrases)
    """
    from benchmark_runner import compute_token_f1

    exp_norm = normalize_text(expected)
    ans_norm = normalize_text(answer)

    if not exp_norm:
        return False

    # 1. Strict substring
    if exp_norm in ans_norm:
        return True

    # 2. Date-normalized substring
    if _normalize_dates(exp_norm) in _normalize_dates(ans_norm):
        return True

    # 3. Token-F1 soft match
    if compute_token_f1(ans_norm, exp_norm) >= 0.5:
        return True

    return False


def evaluate_batch_answers(
    raw: str,
    qa_payload: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Evaluate parsed answers against expected answers."""
    parsed = parse_json_answers(raw)
    results: List[Dict[str, Any]] = []

    for entry in qa_payload:
        qid = entry["id"]
        expected = entry.get("expected")
        answer_text = parsed.get(qid)

        if answer_text is None:
            # Fallback: check if expected appears anywhere in raw response
            correct = bool(expected) and _is_correct(expected, raw)
            answer_text = (
                f"[Found in response: {expected}]" if correct
                else "[Answer not found in response]"
            )
        else:
            correct = bool(expected) and _is_correct(expected, answer_text)

        results.append({
            "question_id": qid,
            "question": entry["question"],
            "expected_answer": expected,
            "model_answer": answer_text,
            "answer_correct": correct,
            "category": entry.get("category"),
        })

    return results


def _find_first_dia_hit_rank(
    retrieved_results: List[Dict[str, Any]],
    evidence: List[str],
) -> Optional[int]:
    """
    Return the 1-based rank of the first retrieved result that contains any gold dia_id.

    A hit is defined as a retrieved chunk whose content contains a gold evidence id,
    stored in memories as text like "[D18:5] ...".
    """
    if not evidence:
        return None

    for rank, result in enumerate(retrieved_results, start=1):
        content = str(result.get("content", ""))
        for dia_id in evidence:
            if f"[{dia_id}]" in content:
                return rank
    return None


def _extract_dia_ids_from_results(retrieved_results: List[Dict[str, Any]]) -> List[Set[str]]:
    """Extract dia_ids like D18:5 from retrieved chunk contents."""
    pattern = re.compile(r"\[(D\d+:\d+)\]")
    extracted: List[Set[str]] = []
    for result in retrieved_results:
        content = str(result.get("content", ""))
        extracted.append(set(pattern.findall(content)))
    return extracted


def evaluate_retrieval_for_questions(
    qa_payload: List[Dict[str, Any]],
    *,
    userid: str,
    server_url: str,
    timeout: float,
    k_values: List[int],
) -> Dict[str, Dict[str, Any]]:
    """
    Measure retrieval metrics using the raw question text as the search query.

    Returns a mapping keyed by question_id. Each value contains:
      - query_text
      - query_source
      - evidence
      - first_hit_rank
      - reciprocal_rank
      - recall_by_k
      - any_hit_by_k
      - all_hit_by_k
      - retrieved_keys (top max_k result keys)

    This intentionally evaluates the retriever directly, not the agent's actual
    emitted search_memory queries.
    """
    if not qa_payload or not k_values:
        return {}

    max_k = max(k_values)
    retrieval_by_qid: Dict[str, Dict[str, Any]] = {}

    for entry in qa_payload:
        qid = entry["id"]
        question = entry.get("question", "")
        evidence = list(entry.get("evidence") or [])
        category = entry.get("category")

        if not evidence:
            retrieval_by_qid[qid] = {
                "query_source": "question_text",
                "query_text": question,
                "evidence": evidence,
                "category": category,
                "skipped": True,
                "skip_reason": "no_evidence",
                "first_hit_rank": None,
                "reciprocal_rank": 0.0,
                "recall_by_k": {str(k): 0.0 for k in k_values},
                "any_hit_by_k": {str(k): False for k in k_values},
                "all_hit_by_k": {str(k): False for k in k_values},
                "dia_id_hit_by_k": {str(k): False for k in k_values},
                "retrieved_keys": [],
            }
            continue

        retrieved_results = search_memories(
            server_url=server_url,
            userid=userid,
            query=question,
            limit=max_k,
            timeout=timeout,
        )
        # Recall is measured only against raw chunks (which carry dia_ids).
        # Extracted-fact documents are intentionally excluded here — they are a
        # general-purpose index layer and should not be required to contain dia_ids.
        raw_results = [
            r for r in retrieved_results
            if r.get("type") != "extracted_fact"
        ]
        first_hit_rank = _find_first_dia_hit_rank(raw_results, evidence)
        reciprocal_rank = 1.0 / first_hit_rank if first_hit_rank else 0.0
        extracted_dia_ids = _extract_dia_ids_from_results(raw_results)
        gold_evidence = set(evidence)
        recall_by_k: Dict[str, float] = {}
        any_hit_by_k: Dict[str, bool] = {}
        all_hit_by_k: Dict[str, bool] = {}
        for k in k_values:
            top_hits = set().union(*extracted_dia_ids[:k]) if extracted_dia_ids[:k] else set()
            matched = gold_evidence & top_hits
            recall_by_k[str(k)] = len(matched) / len(gold_evidence) if gold_evidence else 0.0
            any_hit_by_k[str(k)] = bool(matched)
            all_hit_by_k[str(k)] = matched == gold_evidence and bool(gold_evidence)

        retrieval_by_qid[qid] = {
            "query_source": "question_text",
            "query_text": question,
            "evidence": evidence,
            "category": category,
            "skipped": False,
            "first_hit_rank": first_hit_rank,
            "reciprocal_rank": reciprocal_rank,
            "recall_by_k": recall_by_k,
            "any_hit_by_k": any_hit_by_k,
            "all_hit_by_k": all_hit_by_k,
            "dia_id_hit_by_k": any_hit_by_k,
            "retrieved_keys": [str(r.get("key", "")) for r in retrieved_results],
        }

    return retrieval_by_qid


def ask_qa_batches(
    userid: str,
    qa_items: List[Dict[str, Any]],
    server_url: str,
    system_prompt: Optional[str],
    model: Optional[str],
    max_search_results: Optional[int],
    timeout: float,
    delay_sec: float,
    max_qa_chars: int,
    max_questions_per_batch: int,
    *,
    retrieval_k_values: Optional[List[int]] = None,
    existing_results: Optional[List[Dict[str, Any]]] = None,
    on_batch_complete: Optional[callable] = None,
    max_retries: int = 2,
) -> List[Dict[str, Any]]:
    """
    Ask QA questions in batches, respecting token limits.
    
    Args:
        existing_results: Results from previous run (for resume)
        on_batch_complete: Callback called after each batch with (batch_idx, all_results)
        max_retries: Number of retries for failed batches
    
    Returns:
        List of all QA results
    """
    if not qa_items:
        return []
    
    batches = chunk_qa_items(
        qa_items,
        max_chars=max_qa_chars,
        max_questions=max_questions_per_batch,
    )
    
    # Resume from existing results
    all_results = list(existing_results) if existing_results else []
    completed_questions = {r["question_id"] for r in all_results}
    
    question_idx = 1
    
    for batch_idx, batch in enumerate(batches, start=1):
        # Check if this batch was already completed
        batch_question_ids = {qa.get("id") or f"q{question_idx + i}" for i, qa in enumerate(batch)}
        if batch_question_ids.issubset(completed_questions):
            print(f"    Batch {batch_idx}/{len(batches)}: Already completed (skipping)")
            question_idx += len(batch)
            continue
        
        qa_prompt, qa_payload = build_qa_batch_prompt(batch, start_idx=question_idx)
        prompt_chars = len(qa_prompt)
        prompt_tokens = estimate_tokens(qa_prompt)
        
        print(f"    Batch {batch_idx}/{len(batches)}: {len(batch)} questions "
              f"({prompt_chars} chars, ~{prompt_tokens} tokens)", flush=True)

        for qa, payload in zip(batch, qa_payload):
            payload["evidence"] = list(qa.get("evidence") or [])

        retrieval_by_qid = evaluate_retrieval_for_questions(
            qa_payload,
            userid=userid,
            server_url=server_url,
            timeout=timeout,
            k_values=retrieval_k_values or [1, 5, 10],
        )
        
        # Retry logic
        batch_results = None
        last_error = None
        
        # Generate unique thread_id for this batch to avoid message history pollution
        # userid is kept the same for memory search (search_memory tool uses userid)
        batch_thread_id = f"{userid}_batch_{batch_idx}_{uuid.uuid4().hex[:8]}"
        
        for retry in range(max_retries + 1):
            if retry > 0:
                import requests as _requests
                # Only back off for rate-limit errors; timeout/connection errors retry immediately
                is_rate_limit = isinstance(last_error, _requests.HTTPError) and \
                    getattr(last_error.response, "status_code", None) == 429
                retry_delay = delay_sec * (retry + 1) if is_rate_limit else 1.0
                print(f"      Retry {retry}/{max_retries} after {retry_delay}s...")
                throttle(retry_delay)
                # Generate new thread_id for retry to ensure clean state
                batch_thread_id = f"{userid}_batch_{batch_idx}_retry{retry}_{uuid.uuid4().hex[:8]}"
            else:
                throttle(delay_sec)

            try:
                qa_response, _ = call_agent(
                    [{"role": "user", "content": qa_prompt}],
                    server_url=server_url,
                    userid=userid,  # For memory search
                    system_prompt=system_prompt,
                    model=model,
                    max_search_results=max_search_results,
                    timeout=timeout,
                    thread_id=batch_thread_id,  # Unique per batch to avoid message pollution
                    enable_web_search=False,  # Memory benchmark: no web search needed
                    enable_preference_extraction=False,  # Memory benchmark: skip extraction to save tokens
                )

                batch_results = evaluate_batch_answers(qa_response, qa_payload)
                for result in batch_results:
                    qid = result["question_id"]
                    result["retrieval"] = retrieval_by_qid.get(qid)

                # Fallback: retry individual questions that returned "not found"
                not_found = [
                    r for r in batch_results
                    if "[Answer not found" in str(r.get("model_answer", ""))
                    or r.get("model_answer", "").strip() == ""
                ]
                if not_found:
                    print(f"      [FALLBACK] Retrying {len(not_found)} unanswered questions individually...")
                    throttle(delay_sec)
                    for nf in not_found:
                        q_text = nf["question"]
                        fallback_prompt = (
                            f"Search your memory for information about: {q_text}\n"
                            f"Answer concisely in under 10 words using absolute dates. "
                            f'Return JSON: {{"answers": [{{"question_id": "{nf["question_id"]}", "answer": "..."}}]}}'
                        )
                        fb_thread_id = f"{userid}_fallback_{nf['question_id']}_{uuid.uuid4().hex[:8]}"
                        try:
                            fb_response, _ = call_agent(
                                [{"role": "user", "content": fallback_prompt}],
                                server_url=server_url,
                                userid=userid,
                                system_prompt=system_prompt,
                                model=model,
                                max_search_results=max_search_results,
                                timeout=timeout,
                                thread_id=fb_thread_id,
                                enable_web_search=False,
                                enable_preference_extraction=False,
                            )
                            fb_payload = [{
                                "id": nf["question_id"],
                                "question": q_text,
                                "expected": nf.get("expected_answer"),
                                "category": nf.get("category"),
                            }]
                            fb_results = evaluate_batch_answers(fb_response, fb_payload)
                            if fb_results and "[Answer not found" not in str(fb_results[0].get("model_answer", "")):
                                fb_results[0]["retrieval"] = retrieval_by_qid.get(nf["question_id"])
                                # Replace the not-found result with fallback result
                                for i, r in enumerate(batch_results):
                                    if r["question_id"] == nf["question_id"]:
                                        batch_results[i] = fb_results[0]
                                        break
                        except Exception as fb_e:
                            print(f"      [FALLBACK ERROR] {nf['question_id']}: {fb_e}")
                        throttle(delay_sec)

                break  # Success

            except Exception as e:
                last_error = e
                print(f"      [ERROR] Attempt {retry + 1} failed: {e}")

        if batch_results is not None:
            all_results.extend(batch_results)
            correct = sum(1 for r in batch_results if r["answer_correct"])
            print(f"      -> Batch accuracy: {correct}/{len(batch_results)}")
        else:
            # All retries failed
            print(f"    [ERROR] Batch {batch_idx} failed after {max_retries + 1} attempts: {last_error}")
            for qa in qa_payload:
                all_results.append({
                    "question_id": qa["id"],
                    "question": qa["question"],
                    "expected_answer": qa.get("expected"),
                    "model_answer": f"[Error after {max_retries + 1} attempts: {last_error}]",
                    "answer_correct": False,
                    "category": qa.get("category"),
                    "retrieval": retrieval_by_qid.get(qa["id"]),
                })
        
        # Call the callback for checkpoint saving
        if on_batch_complete:
            on_batch_complete(batch_idx, all_results)
        
        question_idx += len(batch)
    
    return all_results


# =============================================================================
# Main Entry Point
# =============================================================================

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="LoCoMo full memory benchmark: store chunks, reset session, retrieve, answer, and score."
    )
    parser.add_argument(
        "--benchmark-dir",
        default=os.path.dirname(os.path.abspath(__file__)),
    )
    parser.add_argument(
        "--locomo-file",
        default=None,
        help="Path to locomo1_converted.json generated by scripts/download_benchmarks.py",
    )
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful AI assistant with access to long-term memory. "
                "Use your memory tools to recall past conversations when answering questions.",
    )
    parser.add_argument("--max-search-results", type=int, default=10)
    parser.add_argument("--userid-prefix", default="memory_test")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--delay-sec",
        type=float,
        default=DEFAULT_DELAY_SEC,
        help=f"Delay between LLM API calls (default: {DEFAULT_DELAY_SEC}s)",
    )
    parser.add_argument(
        "--memory-store-delay",
        type=float,
        default=DEFAULT_MEMORY_STORE_DELAY,
        help=f"Delay between memory store calls (default: {DEFAULT_MEMORY_STORE_DELAY}s)",
    )
    parser.add_argument("--limit-personas", type=int, default=None)
    parser.add_argument(
        "--output",
        default="locomo_full_memory_results.json",
        help="Output file for results",
    )
    
    # Chunking arguments
    parser.add_argument(
        "--turns-per-chunk",
        type=int,
        default=DEFAULT_TURNS_PER_CHUNK,
        help=f"Number of conversation turns per memory chunk (default: {DEFAULT_TURNS_PER_CHUNK})",
    )
    parser.add_argument(
        "--max-chunk-chars",
        type=int,
        default=DEFAULT_MAX_CHUNK_CHARS,
        help=f"Max chars per memory chunk (default: {DEFAULT_MAX_CHUNK_CHARS})",
    )
    
    # QA batch arguments
    parser.add_argument(
        "--max-qa-chars",
        type=int,
        default=DEFAULT_MAX_QA_CHARS,
        help=f"Max chars per QA batch prompt (default: {DEFAULT_MAX_QA_CHARS})",
    )
    parser.add_argument(
        "--max-questions-per-batch",
        type=int,
        default=DEFAULT_MAX_QUESTIONS_PER_BATCH,
        help=f"Max questions per QA batch (default: {DEFAULT_MAX_QUESTIONS_PER_BATCH})",
    )
    
    parser.add_argument(
        "--skip-observations",
        action="store_true",
        help="Skip per-session observation extraction (faster, lower cost)",
    )

    # Resume arguments
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint",
    )
    parser.add_argument(
        "--checkpoint-file",
        default=DEFAULT_CHECKPOINT_FILE,
        help=f"Checkpoint file path (default: {DEFAULT_CHECKPOINT_FILE})",
    )
    parser.add_argument(
        "--force-restart",
        action="store_true",
        help="Force restart, ignoring any existing checkpoint",
    )
    parser.add_argument(
        "--skip-phase1",
        action="store_true",
        help="Skip Phase 1 (memory storage) and go straight to Phase 2 QA. "
             "Use when memories are already in ChromaDB from a previous run.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Max retries for failed QA batches (default: {DEFAULT_MAX_RETRIES})",
    )
    parser.add_argument(
        "--quick-test",
        type=int,
        default=0,
        metavar="N",
        help="Quick test: sample N questions per category (e.g. --quick-test 3 runs 15 questions total)",
    )
    parser.add_argument(
        "--retrieval-k",
        nargs="+",
        type=int,
        default=[1, 5, 10],
        help="k values for retrieval Recall@k / MRR reporting (default: 1 5 10)",
    )

    args = parser.parse_args(argv)
    locomo_file = args.locomo_file or os.path.join(args.benchmark_dir, "locomo1_converted.json")
    
    # Configuration for checkpoint
    config = {
        "locomo_file": locomo_file,
        "server_url": args.server_url,
        "userid_prefix": args.userid_prefix,
        "turns_per_chunk": args.turns_per_chunk,
        "max_chunk_chars": args.max_chunk_chars,
        "max_qa_chars": args.max_qa_chars,
        "max_questions_per_batch": args.max_questions_per_batch,
        "retrieval_k": args.retrieval_k,
    }
    
    # Load checkpoint if resuming
    checkpoint = None
    phase1_completed: List[str] = []
    phase2_completed: List[str] = []
    storage_stats: Dict[str, Any] = {}
    all_results: List[Dict[str, Any]] = []
    current_persona_progress: Optional[Dict[str, Any]] = None
    
    if args.force_restart:
        print("[CHECKPOINT] Force restart - ignoring any existing checkpoint")
        delete_checkpoint(args.checkpoint_file)
    elif args.resume:
        checkpoint = load_checkpoint(args.checkpoint_file)
        if checkpoint:
            phase1_completed = checkpoint.get("phase1_completed", [])
            phase2_completed = checkpoint.get("phase2_completed", [])
            storage_stats = checkpoint.get("storage_stats", {})
            all_results = checkpoint.get("results", [])
            current_persona_progress = checkpoint.get("current_persona_progress")
            print(f"[CHECKPOINT] Resuming with {len(phase1_completed)} phase1, {len(phase2_completed)} phase2 completed")
            if current_persona_progress:
                persona = current_persona_progress.get("persona")
                partial_results = current_persona_progress.get("qa_results", [])
                print(f"  Partial progress for {persona}: {len(partial_results)} questions answered")
        else:
            print("[CHECKPOINT] No checkpoint found, starting fresh")
    elif os.path.exists(args.checkpoint_file):
        print(f"[CHECKPOINT] Found existing checkpoint: {args.checkpoint_file}")
        print("  Use --resume to continue from checkpoint")
        print("  Use --force-restart to start fresh")
        print("  Proceeding with fresh start...")
    
    print(f"Loading benchmark from: {locomo_file}")
    data = load_json(locomo_file)

    (
        personas_convos,
        personas_session_dates,
        personas_session_datetimes_raw,
        personas_qa,
    ) = build_persona_map(data)
    personas = list(personas_convos.keys())
    
    if args.limit_personas is not None:
        personas = personas[: args.limit_personas]
    
    phase1_completed_set: Set[str] = set(phase1_completed)
    phase2_completed_set: Set[str] = set(phase2_completed)
    
    print(f"\n{'='*60}")
    print("LoCoMo Memory Benchmark Runner v2 (Full Coverage + Resume)")
    print(f"{'='*60}")
    print(f"Personas to test: {len(personas)}")
    print(f"Server: {args.server_url}")
    print(f"\nChunking settings:")
    print(f"  Turns per chunk: {args.turns_per_chunk}")
    print(f"  Max chunk chars: {args.max_chunk_chars} (~{args.max_chunk_chars//4} tokens)")
    print(f"\nQA batch settings:")
    print(f"  Max QA chars per batch: {args.max_qa_chars} (~{args.max_qa_chars//4} tokens)")
    print(f"  Max questions per batch: {args.max_questions_per_batch}")
    print(f"\nRate limiting:")
    print(f"  LLM call delay: {args.delay_sec}s")
    print(f"  Memory store delay: {args.memory_store_delay}s")
    print(f"\nResume status:")
    print(f"  Checkpoint file: {args.checkpoint_file}")
    print(f"  Phase 1 already completed: {len(phase1_completed_set)} personas")
    print(f"  Phase 2 already completed: {len(phase2_completed_set)} personas")
    print(f"{'='*60}")
    
    # =========================================================================
    # PHASE 1: Store all conversations for all personas
    # =========================================================================
    print(f"\n{'='*60}")
    print("PHASE 1: Storing all conversations")
    print(f"{'='*60}")

    if args.skip_phase1:
        print("Skipping Phase 1 (--skip-phase1): assuming memories already in ChromaDB")
        for persona in personas:
            phase1_completed_set.add(persona)

    total_chunks_stored = sum(s.get("chunks_stored", 0) for s in storage_stats.values())
    total_chars_stored = sum(s.get("chars_stored", 0) for s in storage_stats.values())

    for persona in personas:
        # Skip if already completed in phase 1
        if persona in phase1_completed_set:
            print(f"\n[{persona}] Already stored (skipping)")
            continue
        
        userid = f"{args.userid_prefix}_{persona}"
        conversations = personas_convos[persona]
        
        # Count total turns
        total_turns = sum(len(c) for c in conversations)
        print(f"\n[{persona}] {len(conversations)} sessions, {total_turns} turns")
        
        try:
            stored_count, total_chars = asyncio.run(store_all_persona_memories(
                persona=persona,
                conversations=conversations,
                server_url=args.server_url,
                userid=userid,
                timeout=args.timeout,
                memory_store_delay=args.memory_store_delay,
                turns_per_chunk=args.turns_per_chunk,
                max_chunk_chars=args.max_chunk_chars,
                session_dates=personas_session_dates.get(persona),
                session_datetimes_raw=personas_session_datetimes_raw.get(persona),
                model=args.model or "anthropic/claude-haiku-4-5-20251001",
                extract_observations=not args.skip_observations,
            ))
            
            # Verify storage
            memories = list_memories(args.server_url, userid, args.timeout)
            
            print(f"  -> Stored {stored_count} chunks ({total_chars} chars, ~{total_chars//4} tokens)")
            print(f"  -> Verified {len(memories)} memories in storage")
            
            storage_stats[persona] = {
                "sessions": len(conversations),
                "total_turns": total_turns,
                "chunks_stored": stored_count,
                "chars_stored": total_chars,
                "verified_memories": len(memories),
            }
            
            total_chunks_stored += stored_count
            total_chars_stored += total_chars
            
            # Mark as completed and save checkpoint
            phase1_completed.append(persona)
            phase1_completed_set.add(persona)
            
            save_checkpoint(
                args.checkpoint_file,
                phase1_completed,
                phase2_completed,
                storage_stats,
                all_results,
                config,
            )
            print(f"  [CHECKPOINT] Saved (Phase 1: {len(phase1_completed)}/{len(personas)})")
            
        except Exception as e:
            print(f"  [ERROR] Failed to store memories for {persona}: {e}")
            print(f"  [INFO] Run with --resume to retry this persona")
            # Don't mark as completed, will retry on resume
            continue
    
    print(f"\n{'='*60}")
    print(f"Phase 1 Complete: {total_chunks_stored} total chunks, "
          f"{total_chars_stored} chars (~{total_chars_stored//4} tokens)")
    print(f"{'='*60}")
    
    # =========================================================================
    # PHASE 2: Ask QA questions in batches
    # =========================================================================
    print(f"\n{'='*60}")
    print("PHASE 2: Asking QA questions")
    print(f"{'='*60}")
    
    total_correct = sum(
        sum(1 for r in res.get("qa_results", []) if r.get("answer_correct"))
        for res in all_results
    )
    total_questions = sum(len(res.get("qa_results", [])) for res in all_results)
    
    for persona in personas:
        # Skip if already completed in phase 2
        if persona in phase2_completed_set:
            print(f"\n[{persona}] QA already completed (skipping)")
            continue
        
        # Skip if phase 1 not completed for this persona
        if persona not in phase1_completed_set:
            print(f"\n[{persona}] Phase 1 not completed, skipping QA")
            continue
        
        userid = f"{args.userid_prefix}_{persona}"
        qa_items = personas_qa.get(persona, [])

        # Quick test: sample N questions per category
        if args.quick_test > 0:
            import random
            from collections import defaultdict as _defaultdict
            by_cat: dict = _defaultdict(list)
            for q in qa_items:
                by_cat[q.get("category", 0)].append(q)
            sampled = []
            for cat_qs in by_cat.values():
                sampled.extend(random.sample(cat_qs, min(args.quick_test, len(cat_qs))))
            qa_items = sampled
            print(f"  [QUICK TEST] Sampled {len(qa_items)} questions ({args.quick_test} per category)")

        if not qa_items:
            print(f"\n[{persona}] No QA questions, skipping")
            all_results.append({
                "persona": persona,
                "storage": storage_stats.get(persona, {}),
                "qa_results": [],
            })
            phase2_completed.append(persona)
            phase2_completed_set.add(persona)
            continue
        
        print(f"\n[{persona}] {len(qa_items)} QA questions")
        
        # Check for partial progress from previous run
        existing_qa_results: List[Dict[str, Any]] = []
        if current_persona_progress and current_persona_progress.get("persona") == persona:
            existing_qa_results = current_persona_progress.get("qa_results", [])
            print(f"  Resuming from {len(existing_qa_results)} previously answered questions")
            current_persona_progress = None  # Clear after using
        
        try:
            # Reset short-term session (keep memory) - only if not resuming
            if not existing_qa_results:
                throttle(args.delay_sec)
                try:
                    reset_short_term_session(
                        args.server_url, userid, args.timeout,
                        preserve_memory=True, model=args.model,
                    )
                    print(f"  Reset short-term session (memory preserved)")
                except Exception as exc:
                    print(f"  [WARN] Failed to reset session: {exc}")
            
            # Create callback for batch-level checkpoint saving
            def on_batch_complete(batch_idx: int, batch_results: List[Dict[str, Any]]):
                save_checkpoint(
                    args.checkpoint_file,
                    phase1_completed,
                    phase2_completed,
                    storage_stats,
                    all_results,
                    config,
                    current_persona_progress={
                        "persona": persona,
                        "qa_results": batch_results,
                    },
                )
                print(f"      [CHECKPOINT] Batch {batch_idx} saved ({len(batch_results)} total)")
            
            # Ask questions in batches
            qa_results = ask_qa_batches(
                userid=userid,
                qa_items=qa_items,
                server_url=args.server_url,
                system_prompt=args.system_prompt,
                model=args.model,
                max_search_results=args.max_search_results,
                timeout=args.timeout,
                delay_sec=args.delay_sec,
                max_qa_chars=args.max_qa_chars,
                max_questions_per_batch=args.max_questions_per_batch,
                retrieval_k_values=args.retrieval_k,
                existing_results=existing_qa_results,
                on_batch_complete=on_batch_complete,
                max_retries=args.max_retries,
            )
            
            correct = sum(1 for r in qa_results if r["answer_correct"])
            total = len(qa_results)
            total_correct += correct
            total_questions += total
            
            print(f"  -> Total accuracy: {correct}/{total} ({correct/total*100:.1f}%)" if total > 0 else "  -> No results")
            
            all_results.append({
                "persona": persona,
                "storage": storage_stats.get(persona, {}),
                "qa_results": qa_results,
            })
            
            # Mark as completed and save checkpoint
            phase2_completed.append(persona)
            phase2_completed_set.add(persona)
            
            save_checkpoint(
                args.checkpoint_file,
                phase1_completed,
                phase2_completed,
                storage_stats,
                all_results,
                config,
            )
            print(f"  [CHECKPOINT] Saved (Phase 2: {len(phase2_completed)}/{len(personas)})")
            
        except Exception as e:
            print(f"  [ERROR] QA failed for {persona}: {e}")
            print(f"  [INFO] Run with --resume to retry this persona")
            continue

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Personas tested: {len(all_results)}")
    print(f"Total memory chunks stored: {total_chunks_stored}")
    print(f"Total questions: {total_questions}")
    print(f"Correct answers: {total_correct}")
    if total_questions > 0:
        print(f"Overall accuracy: {total_correct/total_questions*100:.1f}%")
    
    # Category breakdown
    category_totals: Dict[Any, int] = defaultdict(int)
    category_correct: Dict[Any, int] = defaultdict(int)
    retrieval_totals: Dict[int, int] = {k: 0 for k in args.retrieval_k}
    retrieval_recall_sum: Dict[int, float] = {k: 0.0 for k in args.retrieval_k}
    retrieval_any_hits: Dict[int, int] = {k: 0 for k in args.retrieval_k}
    retrieval_all_hits: Dict[int, int] = {k: 0 for k in args.retrieval_k}
    retrieval_rr_sum = 0.0
    retrieval_rr_count = 0
    retrieval_by_category: Dict[Any, Dict[str, Any]] = defaultdict(
        lambda: {
            "total": 0,
            "reciprocal_rank_sum": 0.0,
            "recall_sum_by_k": {k: 0.0 for k in args.retrieval_k},
            "any_hits_by_k": {k: 0 for k in args.retrieval_k},
            "all_hits_by_k": {k: 0 for k in args.retrieval_k},
        }
    )
    for result in all_results:
        for qa in result.get("qa_results", []):
            cat = qa.get("category")
            category_totals[cat] += 1
            if qa.get("answer_correct"):
                category_correct[cat] += 1
            retrieval = qa.get("retrieval") or {}
            if retrieval.get("skipped"):
                continue
            recall_by_k = retrieval.get("recall_by_k") or {}
            any_hit_by_k = retrieval.get("any_hit_by_k") or {}
            all_hit_by_k = retrieval.get("all_hit_by_k") or {}
            retrieval_rr_sum += float(retrieval.get("reciprocal_rank") or 0.0)
            retrieval_rr_count += 1
            retrieval_by_category[cat]["total"] += 1
            retrieval_by_category[cat]["reciprocal_rank_sum"] += float(
                retrieval.get("reciprocal_rank") or 0.0
            )
            for k in args.retrieval_k:
                retrieval_totals[k] += 1
                retrieval_recall_sum[k] += float(recall_by_k.get(str(k), 0.0))
                retrieval_by_category[cat]["recall_sum_by_k"][k] += float(recall_by_k.get(str(k), 0.0))
                if any_hit_by_k.get(str(k)):
                    retrieval_any_hits[k] += 1
                    retrieval_by_category[cat]["any_hits_by_k"][k] += 1
                if all_hit_by_k.get(str(k)):
                    retrieval_all_hits[k] += 1
                    retrieval_by_category[cat]["all_hits_by_k"][k] += 1
    
    if category_totals:
        print("\nPer-category accuracy:")
        for cat in sorted(category_totals.keys(), key=lambda x: (x is None, x)):
            tot = category_totals[cat]
            corr = category_correct[cat]
            pct = corr / tot * 100 if tot > 0 else 0
            print(f"  Category {cat}: {corr}/{tot} ({pct:.1f}%)")

    if retrieval_rr_count > 0:
        print("\nRetrieval metrics (question-as-query):")
        for k in args.retrieval_k:
            total = retrieval_totals[k]
            recall = retrieval_recall_sum[k] / total if total > 0 else 0.0
            any_hit = retrieval_any_hits[k] / total if total > 0 else 0.0
            all_hit = retrieval_all_hits[k] / total if total > 0 else 0.0
            print(f"  Recall@{k}: {recall:.1%}")
            print(f"  AnyHit@{k}: {retrieval_any_hits[k]}/{total} ({any_hit:.1%})")
            print(f"  AllHit@{k}: {retrieval_all_hits[k]}/{total} ({all_hit:.1%})")
        print(f"  MRR: {retrieval_rr_sum / retrieval_rr_count:.3f}")

        print("\nPer-category retrieval:")
        for cat in sorted(retrieval_by_category.keys(), key=lambda x: (x is None, x)):
            stats = retrieval_by_category[cat]
            total = stats["total"]
            if total <= 0:
                continue
            per_k = ", ".join(
                f"R@{k}={stats['recall_sum_by_k'][k] / total:.1%}"
                f"/Any@{k}={stats['any_hits_by_k'][k] / total:.1%}"
                f"/All@{k}={stats['all_hits_by_k'][k] / total:.1%}"
                for k in args.retrieval_k
            )
            mrr = stats["reciprocal_rank_sum"] / total
            print(f"  Category {cat}: {per_k}, MRR={mrr:.3f}")

    retrieval_summary = {
        "k_values": args.retrieval_k,
        "query_source": "question_text",
        "questions_evaluated": retrieval_rr_count,
        "mrr": (retrieval_rr_sum / retrieval_rr_count) if retrieval_rr_count > 0 else 0.0,
        "recall_by_k": {
            str(k): (retrieval_recall_sum[k] / retrieval_totals[k]) if retrieval_totals[k] > 0 else 0.0
            for k in args.retrieval_k
        },
        "any_hit_by_k": {
            str(k): (retrieval_any_hits[k] / retrieval_totals[k]) if retrieval_totals[k] > 0 else 0.0
            for k in args.retrieval_k
        },
        "all_hit_by_k": {
            str(k): (retrieval_all_hits[k] / retrieval_totals[k]) if retrieval_totals[k] > 0 else 0.0
            for k in args.retrieval_k
        },
        "any_hits_count_by_k": {str(k): retrieval_any_hits[k] for k in args.retrieval_k},
        "all_hits_count_by_k": {str(k): retrieval_all_hits[k] for k in args.retrieval_k},
        "totals_by_k": {str(k): retrieval_totals[k] for k in args.retrieval_k},
        "by_category": {
            str(cat): {
                "total": stats["total"],
                "mrr": (stats["reciprocal_rank_sum"] / stats["total"]) if stats["total"] > 0 else 0.0,
                "recall_by_k": {
                    str(k): (stats["recall_sum_by_k"][k] / stats["total"]) if stats["total"] > 0 else 0.0
                    for k in args.retrieval_k
                },
                "any_hit_by_k": {
                    str(k): (stats["any_hits_by_k"][k] / stats["total"]) if stats["total"] > 0 else 0.0
                    for k in args.retrieval_k
                },
                "all_hit_by_k": {
                    str(k): (stats["all_hits_by_k"][k] / stats["total"]) if stats["total"] > 0 else 0.0
                    for k in args.retrieval_k
                },
            }
            for cat, stats in retrieval_by_category.items()
        },
    }

    summary = {
        "personas_run": len(all_results),
        "total_chunks_stored": total_chunks_stored,
        "total_chars_stored": total_chars_stored,
        "qa_accuracy_total": total_correct,
        "qa_questions_total": total_questions,
        "accuracy_percent": (total_correct / total_questions * 100) if total_questions > 0 else 0,
        "category_totals": dict(category_totals),
        "category_correct": dict(category_correct),
        "retrieval": retrieval_summary,
        "details": all_results,
    }

    output_path = os.path.join(os.getcwd(), args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nResults written to: {output_path}")
    
    # Check if all personas completed successfully
    all_completed = (
        len(phase1_completed_set) == len(personas) and
        len(phase2_completed_set) == len(personas)
    )
    
    if all_completed:
        delete_checkpoint(args.checkpoint_file)
        print("\n[SUCCESS] All personas completed successfully!")
    else:
        print(f"\n[INCOMPLETE] Some personas not completed:")
        print(f"  Phase 1 incomplete: {set(personas) - phase1_completed_set}")
        print(f"  Phase 2 incomplete: {set(personas) - phase2_completed_set}")
        print(f"  Run with --resume to continue")


if __name__ == "__main__":
    main()
