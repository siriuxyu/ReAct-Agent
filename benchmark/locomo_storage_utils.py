from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from agent.interfaces import StorageDocument, StorageType
from agent.storage.vector_storage import VectorStorageBackend


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_locomo_evidence(evidence: Any) -> List[str]:
    """
    Normalize LoCoMo evidence ids into a flat dia_id list.

    Some official entries contain values like "D8:6; D9:17" inside a single
    list element. We split those into ["D8:6", "D9:17"] while preserving order.
    """
    if not evidence:
        return []

    values = evidence if isinstance(evidence, list) else [evidence]
    normalized: List[str] = []
    seen = set()

    for value in values:
        if value is None:
            continue
        for match in re.findall(r"D\d+:\d+", str(value)):
            if match not in seen:
                seen.add(match)
                normalized.append(match)

    return normalized


def build_persona_map(
    data: Dict[str, Any],
) -> Tuple[
    Dict[str, List[List[Dict[str, Any]]]],
    Dict[str, List[str]],
    Dict[str, List[str]],
    Dict[str, List[Dict[str, Any]]],
]:
    """Build mapping from persona to conversations, session dates, and QA items."""
    personas_conversations: Dict[str, List[List[Dict[str, Any]]]] = defaultdict(list)
    personas_session_dates: Dict[str, List[str]] = defaultdict(list)
    personas_session_datetimes_raw: Dict[str, List[str]] = defaultdict(list)
    personas_qa: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    dia_persona: Dict[str, str] = {}

    for tc in data.get("test_cases", []):
        persona = (tc.get("id") or "persona_unknown").split("_session")[0]
        personas_conversations[persona].append(tc.get("conversation") or [])
        personas_session_dates[persona].append(tc.get("session_date") or "")
        personas_session_datetimes_raw[persona].append(tc.get("session_datetime_raw") or "")

        for turn in tc.get("conversation") or []:
            dia_id = turn.get("dia_id")
            if dia_id:
                dia_persona[dia_id] = persona

    for qa in data.get("qa", []):
        persona = None
        qa["evidence"] = normalize_locomo_evidence(qa.get("evidence"))
        for ev in qa.get("evidence") or []:
            persona = dia_persona.get(ev)
            if persona:
                break
        if persona:
            personas_qa[persona].append(qa)

    return (
        personas_conversations,
        personas_session_dates,
        personas_session_datetimes_raw,
        personas_qa,
    )


def build_conversation_memories_chunked(
    conversations: List[List[Dict[str, Any]]],
    persona: str,
    turns_per_chunk: int,
    max_chunk_chars: int,
    session_dates: Optional[List[str]] = None,
    session_datetimes_raw: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Convert conversations into chunked memory items.

    Each chunk keeps original `dia_id` markers in the content so downstream
    retrieval evaluation can match against gold LoCoMo evidence ids.
    """
    memories: List[Dict[str, Any]] = []

    for session_idx, convo in enumerate(conversations, start=1):
        if not convo:
            continue

        current_chunk_lines: List[str] = []
        current_chunk_chars = 0
        turns_in_chunk = 0
        chunk_in_session = 0

        date_str = ""
        if session_dates and session_idx <= len(session_dates):
            raw_date = session_dates[session_idx - 1]
            if raw_date:
                date_str = f" - {raw_date}"
        session_header = f"[Session {session_idx}{date_str}]"
        session_datetime_raw = ""
        if session_datetimes_raw and session_idx <= len(session_datetimes_raw):
            session_datetime_raw = session_datetimes_raw[session_idx - 1] or ""

        current_chunk_queries: List[str] = []
        current_chunk_img_urls: List[str] = []
        current_chunk_blip_captions: List[str] = []

        for turn in convo:
            speaker = turn.get("speaker") or turn.get("role") or "unknown"
            content = turn.get("content", "")
            dia_id = turn.get("dia_id", "")
            blip_caption = turn.get("blip_caption", "")
            query = turn.get("query", "")
            img_urls = turn.get("img_url") or []

            if not content and not blip_caption:
                continue

            line_parts: List[str] = []
            if content:
                if dia_id:
                    line_parts.append(f"[{dia_id}] {speaker}: {content}")
                else:
                    line_parts.append(f"{speaker}: {content}")
            if blip_caption:
                line_parts.append(f"[image_caption] {blip_caption}")

            line_block = "\n".join(line_parts)
            line_chars = len(line_block) + 1
            should_break = (
                turns_in_chunk >= turns_per_chunk
                or (current_chunk_chars + line_chars > max_chunk_chars and turns_in_chunk > 0)
            )

            if should_break and current_chunk_lines:
                memories.append(
                    {
                        "key": f"{persona}_s{session_idx}_c{chunk_in_session}",
                        "content": session_header + "\n" + "\n".join(current_chunk_lines),
                        "session_index": session_idx,
                        "chunk_index": chunk_in_session,
                        "session_date": session_dates[session_idx - 1] if session_dates and session_idx <= len(session_dates) else "",
                        "session_datetime_raw": session_datetime_raw,
                        "chunk_queries": list(current_chunk_queries),
                        "chunk_img_urls": list(current_chunk_img_urls),
                        "chunk_blip_captions": list(current_chunk_blip_captions),
                    }
                )
                current_chunk_lines = []
                current_chunk_chars = 0
                turns_in_chunk = 0
                chunk_in_session += 1
                current_chunk_queries = []
                current_chunk_img_urls = []
                current_chunk_blip_captions = []

            current_chunk_lines.append(line_block)
            current_chunk_chars += line_chars
            turns_in_chunk += 1
            if query and query not in current_chunk_queries:
                current_chunk_queries.append(query)
            if blip_caption and blip_caption not in current_chunk_blip_captions:
                current_chunk_blip_captions.append(blip_caption)
            for img_url in img_urls:
                if img_url and img_url not in current_chunk_img_urls:
                    current_chunk_img_urls.append(img_url)

        if current_chunk_lines:
            memories.append(
                {
                    "key": f"{persona}_s{session_idx}_c{chunk_in_session}",
                    "content": session_header + "\n" + "\n".join(current_chunk_lines),
                    "session_index": session_idx,
                    "chunk_index": chunk_in_session,
                    "session_date": session_dates[session_idx - 1] if session_dates and session_idx <= len(session_dates) else "",
                    "session_datetime_raw": session_datetime_raw,
                    "chunk_queries": list(current_chunk_queries),
                    "chunk_img_urls": list(current_chunk_img_urls),
                    "chunk_blip_captions": list(current_chunk_blip_captions),
                }
            )

    return memories


def infer_persona_from_user_id(user_id: str, personas: Sequence[str]) -> Optional[str]:
    """Infer a LoCoMo persona from a user_id like `memory_test_conv-26`."""
    if user_id in personas:
        return user_id

    suffix_matches = [persona for persona in personas if user_id.endswith(f"_{persona}")]
    if suffix_matches:
        return max(suffix_matches, key=len)

    if len(personas) == 1:
        return personas[0]

    return None


async def bootstrap_persona_to_chroma(
    *,
    persona: str,
    userid: str,
    conversations: List[List[Dict[str, Any]]],
    session_dates: Optional[List[str]],
    session_datetimes_raw: Optional[List[str]],
    persist_path: Path,
    collection_name: str,
    openai_api_key: str,
    turns_per_chunk: int,
    max_chunk_chars: int,
    clear_existing: bool = True,
) -> Dict[str, Any]:
    """
    Store chunked LoCoMo conversation memories directly into Chroma.

    This bypasses the agent and server entirely. It only performs embedding +
    vector storage, which is enough for retrieval-only evaluation.
    """
    backend = VectorStorageBackend(
        collection_name=collection_name,
        persist_path=str(persist_path),
        embedding_provider="openai",
        openai_api_key=openai_api_key,
    )
    await backend.initialize()

    if clear_existing:
        existing_docs = await backend.get_user_contexts(user_id=userid, limit=5000)
        for doc in existing_docs:
            await backend.delete_document(doc.id, userid)

    memories = build_conversation_memories_chunked(
        conversations=conversations,
        persona=persona,
        turns_per_chunk=turns_per_chunk,
        max_chunk_chars=max_chunk_chars,
        session_dates=session_dates,
        session_datetimes_raw=session_datetimes_raw,
    )

    now = datetime.now()
    documents: List[StorageDocument] = []
    total_chars = 0

    for mem in memories:
        total_chars += len(mem["content"])
        documents.append(
            StorageDocument(
                id=mem["key"],
                user_id=userid,
                session_id=f"{persona}_session_{mem['session_index']}",
                document_type=StorageType.LONG_TERM_CONTEXT,
                content=mem["content"],
                embedding=None,
                metadata={
                    "persona": persona,
                    "session_index": mem["session_index"],
                    "chunk_index": mem["chunk_index"],
                    "session_date": mem.get("session_date", ""),
                    "session_datetime_raw": mem.get("session_datetime_raw", ""),
                    "has_images": bool(mem.get("chunk_img_urls")),
                    "chunk_queries_json": json.dumps(mem.get("chunk_queries", []), ensure_ascii=False),
                    "chunk_img_urls_json": json.dumps(mem.get("chunk_img_urls", []), ensure_ascii=False),
                    "chunk_blip_captions_json": json.dumps(mem.get("chunk_blip_captions", []), ensure_ascii=False),
                    "source": "locomo_measure_recall",
                },
                created_at=now,
                updated_at=now,
            )
        )

    if documents:
        await backend.store_documents_batch(documents, generate_embeddings=True)

    return {
        "persona": persona,
        "user_id": userid,
        "stored_count": len(documents),
        "total_chars": total_chars,
    }
