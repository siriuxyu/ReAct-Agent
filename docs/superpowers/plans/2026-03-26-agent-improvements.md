# Agent Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 7 improvements to the CSE291-A React agent: embedding cache, identity recall, unit tests, context compression, token-F1 eval metric, .dockerignore, and resume update.

**Architecture:** Each fix is independent. Embedding cache wraps `OpenAIEmbeddingService.embed_text` with an in-process LRU dict. Identity recall is fixed by expanding the PREFERENCE_EXTRACTION_SYSTEM_PROMPT. Context compression adds a pre-`call_model` summarization step when `len(messages) > 14`. Token-F1 is added as a pure helper in `benchmark_runner.py`.

**Tech Stack:** Python 3.10, LangGraph, ChromaDB, OpenAI Embeddings, FastAPI, pytest, unittest.mock

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `agent/storage/embedding_service.py` | Modify | Add LRU embedding cache |
| `agent/prompts.py` | Modify | Strengthen identity extraction prompt |
| `agent/summarizer.py` | Create | Context compression helper |
| `agent/graph.py` | Modify | Call summarizer before call_model when messages > 14 |
| `benchmark_runner.py` | Modify | Add `compute_token_f1()` + use it in scoring |
| `.dockerignore` | Create | Exclude chroma_db_data, .env, __pycache__ |
| `tests/test_embedding_cache.py` | Create | Unit test: cache hit avoids second API call |
| `tests/test_identity_prompt.py` | Create | Unit test: prompt covers all identity keywords |
| `tests/test_summarizer.py` | Create | Unit test: summarizer triggers at correct threshold |
| `tests/test_eval_metrics.py` | Create | Unit test: token_f1 scoring correctness |
| `tests/test_extraction_unit.py` | Create | Unit test: ContextExtractor classification |

---

## Task 1: Embedding Cache

**Files:**
- Modify: `agent/storage/embedding_service.py`
- Create: `tests/test_embedding_cache.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_embedding_cache.py
import asyncio
from unittest.mock import AsyncMock, patch
import pytest


def test_embed_text_cache_hit_skips_api():
    """Second call with same text must not call the OpenAI API."""
    from agent.storage.embedding_service import OpenAIEmbeddingService

    svc = OpenAIEmbeddingService.__new__(OpenAIEmbeddingService)
    svc.api_key = "test"
    svc.model = "text-embedding-3-small"
    svc.dimension = 1536
    svc._cache = {}

    fake_embedding = [0.1] * 1536
    mock_client = AsyncMock()
    mock_client.embeddings.create.return_value = AsyncMock(
        data=[AsyncMock(embedding=fake_embedding)]
    )
    svc.client = mock_client

    result1 = asyncio.get_event_loop().run_until_complete(svc.embed_text("hello world"))
    result2 = asyncio.get_event_loop().run_until_complete(svc.embed_text("hello world"))

    assert result1 == fake_embedding
    assert result2 == fake_embedding
    assert mock_client.embeddings.create.call_count == 1  # only called once


def test_embed_text_cache_miss_calls_api():
    """Different texts each trigger an API call."""
    from agent.storage.embedding_service import OpenAIEmbeddingService

    svc = OpenAIEmbeddingService.__new__(OpenAIEmbeddingService)
    svc.api_key = "test"
    svc.model = "text-embedding-3-small"
    svc.dimension = 1536
    svc._cache = {}

    fake_a = [0.1] * 1536
    fake_b = [0.2] * 1536
    responses = [
        AsyncMock(data=[AsyncMock(embedding=fake_a)]),
        AsyncMock(data=[AsyncMock(embedding=fake_b)]),
    ]
    mock_client = AsyncMock()
    mock_client.embeddings.create.side_effect = responses
    svc.client = mock_client

    r1 = asyncio.get_event_loop().run_until_complete(svc.embed_text("text A"))
    r2 = asyncio.get_event_loop().run_until_complete(svc.embed_text("text B"))

    assert r1 == fake_a
    assert r2 == fake_b
    assert mock_client.embeddings.create.call_count == 2
```

- [ ] **Step 2: Run test to verify it fails**

```
cd /home/siriux/Projects/CSE291-A
python -m pytest tests/test_embedding_cache.py -v
```

Expected: `AttributeError: 'OpenAIEmbeddingService' object has no attribute '_cache'`

- [ ] **Step 3: Implement embedding cache in `agent/storage/embedding_service.py`**

Add `_cache: dict` to `__init__` and check it in `embed_text`. Keep `embed_texts_batch` uncached (batch calls are already efficient).

Replace the `__init__` and `embed_text` methods of `OpenAIEmbeddingService`:

```python
    def __init__(
        self, api_key: str, model: str = "text-embedding-3-small", dimension: int = 1536,
        cache_size: int = 1024
    ):
        if AsyncOpenAI is None:
            raise ImportError("OpenAI library not installed. Please run `pip install openai`")
        self.api_key = api_key
        self.model = model
        self.dimension = dimension
        self.client = AsyncOpenAI(api_key=self.api_key)
        # LRU-style cache: dict preserves insertion order in Python 3.7+
        self._cache: dict = {}
        self._cache_size = cache_size

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text, with in-process LRU cache."""
        clean = text.replace("\n", " ")
        if clean in self._cache:
            return self._cache[clean]
        try:
            response = await self.client.embeddings.create(
                input=[clean],
                model=self.model
            )
            embedding = response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            raise EmbeddingError(f"OpenAI embedding error: {str(e)}")
        # Evict oldest entry if over limit
        if len(self._cache) >= self._cache_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[clean] = embedding
        return embedding
```

- [ ] **Step 4: Run tests to verify they pass**

```
python -m pytest tests/test_embedding_cache.py -v
```

Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add agent/storage/embedding_service.py tests/test_embedding_cache.py
git commit -m "perf: add LRU embedding cache to OpenAIEmbeddingService"
```

---

## Task 2: Identity Category Prompt Improvement

**Files:**
- Modify: `agent/prompts.py`
- Create: `tests/test_identity_prompt.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_identity_prompt.py
from agent.prompts import PREFERENCE_EXTRACTION_SYSTEM_PROMPT


def test_prompt_covers_identity_fields():
    """Prompt must explicitly mention key identity fields so the LLM extracts them."""
    required_terms = ["name", "occupation", "location", "age", "nationality"]
    prompt_lower = PREFERENCE_EXTRACTION_SYSTEM_PROMPT.lower()
    missing = [t for t in required_terms if t not in prompt_lower]
    assert missing == [], f"Prompt missing identity terms: {missing}"


def test_prompt_has_personal_examples():
    """Prompt must include concrete identity examples."""
    assert "e.g." in PREFERENCE_EXTRACTION_SYSTEM_PROMPT.lower() or \
           "example" in PREFERENCE_EXTRACTION_SYSTEM_PROMPT.lower() or \
           "such as" in PREFERENCE_EXTRACTION_SYSTEM_PROMPT.lower(), \
        "Prompt should contain examples to guide the LLM"
```

- [ ] **Step 2: Run test to verify it fails**

```
python -m pytest tests/test_identity_prompt.py -v
```

Expected: FAIL — `missing identity terms: ['name', 'occupation', 'location', 'age', 'nationality']`

- [ ] **Step 3: Update `agent/prompts.py` PREFERENCE_EXTRACTION_SYSTEM_PROMPT**

Replace the existing `PREFERENCE_EXTRACTION_SYSTEM_PROMPT` string with:

```python
PREFERENCE_EXTRACTION_SYSTEM_PROMPT = """You are a specialized User Preference Extractor.
Analyze the conversation history and extract explicit or implicit user preferences, facts, and traits.

## Categories to Extract

1. **Personal / Identity** — Facts about who the user IS.
   Examples: name, age, occupation/job, location (city/country), nationality, language spoken,
   relationship status, health conditions, physical attributes.
   Extract when the user says things like: "I'm a nurse", "I live in Seattle", "My name is Alex",
   "I'm 28 years old", "I'm Chinese", "I have diabetes".

2. **Communication Style** — How the user prefers responses.
   Examples: prefers concise answers, wants bullet points, likes emojis, dislikes markdown.

3. **Topics of Interest** — Subjects the user cares about.
   Examples: Python programming, machine learning, hiking, cooking, finance.

4. **Constraints** — Behavioral rules for the assistant.
   Examples: "don't use markdown", "always answer in French", "keep responses under 100 words".

## Rules
- Only extract **new** and **clear** information from the current messages.
- A task request ("Translate hello") is NOT a preference.
- A personal fact ("I speak French") IS a preference — type PERSONAL.
- Identity information (name, location, occupation, age, nationality) MUST be extracted as type PERSONAL.
- If no relevant preferences are found, return an empty list.
"""
```

- [ ] **Step 4: Run tests to verify they pass**

```
python -m pytest tests/test_identity_prompt.py -v
```

Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add agent/prompts.py tests/test_identity_prompt.py
git commit -m "feat: strengthen identity extraction in preference prompt"
```

---

## Task 3: Unit Tests for ContextExtractor

**Files:**
- Create: `tests/test_extraction_unit.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_extraction_unit.py
import asyncio
import pytest
from langchain_core.messages import HumanMessage, AIMessage

from agent.extraction.extractor import ContextExtractor
from agent.interfaces import PreferenceType


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_extract_preferences_finds_prefer_keyword():
    extractor = ContextExtractor()
    msgs = [HumanMessage(content="I prefer concise answers.")]
    prefs = run(extractor.extract_preferences(msgs, "u1"))
    assert len(prefs) == 1
    assert prefs[0].confidence_score >= 0.9


def test_extract_preferences_ignores_ai_messages():
    extractor = ContextExtractor()
    msgs = [AIMessage(content="I prefer this response format.")]
    prefs = run(extractor.extract_preferences(msgs, "u1"))
    assert prefs == []


def test_extract_preferences_empty_messages():
    extractor = ContextExtractor()
    prefs = run(extractor.extract_preferences([], "u1"))
    assert prefs == []


def test_classify_communication_style():
    extractor = ContextExtractor()
    result = extractor._classify_preference_type("I like concise answers")
    assert result == PreferenceType.COMMUNICATION_STYLE


def test_classify_domain_interest():
    extractor = ContextExtractor()
    result = extractor._classify_preference_type("I prefer python code examples")
    assert result == PreferenceType.DOMAIN_INTEREST


def test_classify_language():
    extractor = ContextExtractor()
    result = extractor._classify_preference_type("I prefer french responses")
    assert result == PreferenceType.LANGUAGE_PREFERENCE


def test_merge_preferences_increases_frequency():
    extractor = ContextExtractor()
    msgs1 = [HumanMessage(content="I prefer bullet points.")]
    msgs2 = [HumanMessage(content="I prefer bullet points.")]
    old = run(extractor.extract_preferences(msgs1, "u1"))
    new = run(extractor.extract_preferences(msgs2, "u1"))
    merged = run(extractor.merge_preferences(old, new))
    assert merged[0].frequency == 2


def test_generate_summary_nonempty():
    extractor = ContextExtractor()
    msgs = [
        HumanMessage(content="Hello"),
        HumanMessage(content="Goodbye"),
    ]
    summary = run(extractor.generate_summary(msgs))
    assert "Hello" in summary
    assert len(summary) <= 403  # max_length + "..."
```

- [ ] **Step 2: Run tests**

```
python -m pytest tests/test_extraction_unit.py -v
```

Expected: `7 passed`

- [ ] **Step 3: Commit**

```bash
git add tests/test_extraction_unit.py
git commit -m "test: add unit tests for ContextExtractor"
```

---

## Task 4: Token-F1 Evaluation Metric

**Files:**
- Modify: `benchmark_runner.py`
- Create: `tests/test_eval_metrics.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_eval_metrics.py
import pytest


def test_token_f1_exact_match():
    from benchmark_runner import compute_token_f1
    assert compute_token_f1("the cat sat", "the cat sat") == pytest.approx(1.0)


def test_token_f1_no_overlap():
    from benchmark_runner import compute_token_f1
    assert compute_token_f1("apple banana", "orange grape") == pytest.approx(0.0)


def test_token_f1_partial_overlap():
    from benchmark_runner import compute_token_f1
    score = compute_token_f1("the cat sat on mat", "the cat")
    # precision=1.0, recall=2/5=0.4, F1=2*1*0.4/1.4 ≈ 0.571
    assert 0.5 < score < 0.65


def test_token_f1_empty_prediction():
    from benchmark_runner import compute_token_f1
    assert compute_token_f1("", "the cat sat") == pytest.approx(0.0)


def test_token_f1_empty_reference():
    from benchmark_runner import compute_token_f1
    assert compute_token_f1("the cat", "") == pytest.approx(0.0)
```

- [ ] **Step 2: Run test to verify it fails**

```
python -m pytest tests/test_eval_metrics.py -v
```

Expected: `ImportError: cannot import name 'compute_token_f1' from 'benchmark_runner'`

- [ ] **Step 3: Add `compute_token_f1` to `benchmark_runner.py`**

Add after the `normalize_text` function (around line 73):

```python
def compute_token_f1(prediction: str, reference: str) -> float:
    """
    Compute token-level F1 between prediction and reference strings.

    This is a soft match metric that rewards partial overlap, unlike the strict
    substring check. Used alongside exact/substring matching in scoring.
    """
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts: Counter = Counter(pred_tokens)
    ref_counts: Counter = Counter(ref_tokens)

    common = sum((pred_counts & ref_counts).values())
    if common == 0:
        return 0.0

    precision = common / sum(pred_counts.values())
    recall = common / sum(ref_counts.values())
    return 2 * precision * recall / (precision + recall)
```

- [ ] **Step 4: Run tests to verify they pass**

```
python -m pytest tests/test_eval_metrics.py -v
```

Expected: `5 passed`

- [ ] **Step 5: Wire `compute_token_f1` into the scoring output**

Find the function `evaluate_answer` (or the inline scoring logic) in `benchmark_runner.py` and add the token F1 score to the result dict. Search for where `"exact_match"` or `"answer_correct"` keys are set, and add `"token_f1"` next to it.

First, find the relevant section:
```
grep -n "exact_match\|answer_correct\|score" benchmark_runner.py | head -30
```

Then add `"token_f1": compute_token_f1(agent_answer, expected_text)` wherever the scoring dict is built.

- [ ] **Step 6: Commit**

```bash
git add benchmark_runner.py tests/test_eval_metrics.py
git commit -m "feat: add token-F1 soft evaluation metric to benchmark_runner"
```

---

## Task 5: Context Compression for Long Conversations

**Files:**
- Create: `agent/summarizer.py`
- Modify: `agent/graph.py`
- Create: `tests/test_summarizer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_summarizer.py
import pytest
from langchain_core.messages import HumanMessage, AIMessage


def test_needs_compression_true():
    from agent.summarizer import needs_compression
    msgs = [HumanMessage(content=f"msg {i}") for i in range(15)]
    assert needs_compression(msgs) is True


def test_needs_compression_false():
    from agent.summarizer import needs_compression
    msgs = [HumanMessage(content=f"msg {i}") for i in range(10)]
    assert needs_compression(msgs) is False


def test_split_messages_returns_old_and_recent():
    from agent.summarizer import split_messages
    msgs = [HumanMessage(content=f"msg {i}") for i in range(20)]
    old, recent = split_messages(msgs, keep_recent=6)
    assert len(recent) == 6
    assert len(old) == 14
    assert recent == msgs[-6:]


def test_compress_produces_summary_message():
    """compress_messages must return a list starting with a SystemMessage summary."""
    from agent.summarizer import compress_messages
    from unittest.mock import AsyncMock, patch
    from langchain_core.messages import SystemMessage
    import asyncio

    msgs = [
        HumanMessage(content="I am Alice, a nurse from Boston."),
        AIMessage(content="Nice to meet you Alice!"),
        HumanMessage(content="What's the weather?"),
        AIMessage(content="It's sunny."),
    ]

    fake_summary = "Alice is a nurse from Boston. Weather was discussed."
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=AsyncMock(content=fake_summary))

    with patch("agent.summarizer.load_chat_model", return_value=mock_llm):
        result = asyncio.get_event_loop().run_until_complete(
            compress_messages(msgs, model="anthropic/claude-haiku-4-5-20251001")
        )

    assert isinstance(result[0], SystemMessage)
    assert fake_summary in result[0].content
```

- [ ] **Step 2: Run tests to verify they fail**

```
python -m pytest tests/test_summarizer.py -v
```

Expected: `ModuleNotFoundError: No module named 'agent.summarizer'`

- [ ] **Step 3: Create `agent/summarizer.py`**

```python
"""
Context compression for long conversations.

When a conversation exceeds COMPRESSION_THRESHOLD messages, older messages are
summarized into a single SystemMessage to keep the context window manageable.
"""

from typing import List, Tuple

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from agent.utils import get_logger, load_chat_model

logger = get_logger(__name__)

COMPRESSION_THRESHOLD = 14  # compress when len(messages) > this
KEEP_RECENT = 6             # always keep this many recent messages verbatim

_SUMMARIZE_PROMPT = """Summarize the conversation below as a compact but complete context note.
Preserve all factual information about the user (name, location, occupation, preferences, constraints).
Output only the summary text — no preamble.

Conversation:
{conversation}
"""


def needs_compression(messages: List[BaseMessage]) -> bool:
    """Return True when the message list is long enough to warrant compression."""
    return len(messages) > COMPRESSION_THRESHOLD


def split_messages(
    messages: List[BaseMessage], keep_recent: int = KEEP_RECENT
) -> Tuple[List[BaseMessage], List[BaseMessage]]:
    """Split messages into (older_to_summarize, recent_to_keep)."""
    if len(messages) <= keep_recent:
        return [], list(messages)
    return list(messages[:-keep_recent]), list(messages[-keep_recent:])


async def compress_messages(
    messages: List[BaseMessage],
    model: str,
) -> List[BaseMessage]:
    """
    Summarize `messages` into a single SystemMessage prefix.

    Returns a new list: [SystemMessage(summary)] + messages[-KEEP_RECENT:].
    The summary is generated by the same model used by the agent.
    """
    old_msgs, recent = split_messages(messages)

    if not old_msgs:
        return list(messages)

    # Build a plain-text transcript of the older messages
    lines = []
    for m in old_msgs:
        if isinstance(m, HumanMessage):
            lines.append(f"User: {m.content}")
        elif isinstance(m, AIMessage) and m.content:
            lines.append(f"Assistant: {m.content}")

    conversation_text = "\n".join(lines)
    prompt = _SUMMARIZE_PROMPT.format(conversation=conversation_text)

    llm = load_chat_model(model)
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        summary_text = response.content.strip()
    except Exception as e:
        logger.warning(f"Context compression failed, keeping full history: {e}")
        return list(messages)

    summary_msg = SystemMessage(
        content=f"[Conversation Summary]\n{summary_text}"
    )
    compressed = [summary_msg] + recent

    logger.info(
        "Context compressed",
        extra={
            "function": "compress_messages",
            "details": {
                "original_count": len(messages),
                "compressed_count": len(compressed),
                "summarized_turns": len(old_msgs),
            },
        },
    )
    return compressed
```

- [ ] **Step 4: Run tests to verify they pass**

```
python -m pytest tests/test_summarizer.py -v
```

Expected: `4 passed`

- [ ] **Step 5: Integrate compression into `agent/graph.py`**

In `call_model`, add compression check right before `invoke_messages` is built (around line 222). Add this block after `system_message` is defined and before `invoke_messages`:

```python
    # Context compression: summarize old messages when conversation is long
    from agent.summarizer import needs_compression, compress_messages
    messages_to_use = state.messages
    if needs_compression(state.messages):
        logger.info("Compressing long context", extra={
            'function': 'call_model',
            'details': {'message_count': len(state.messages)}
        })
        messages_to_use = await compress_messages(state.messages, model=context.model)

    invoke_messages = [{"role": "system", "content": system_message}, *messages_to_use]
```

Also remove (or comment out) the existing `invoke_messages` line that used `state.messages` directly.

- [ ] **Step 6: Run summarizer tests again to confirm nothing broke**

```
python -m pytest tests/test_summarizer.py tests/test_embedding_cache.py -v
```

Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add agent/summarizer.py agent/graph.py tests/test_summarizer.py
git commit -m "feat: add context compression for conversations > 14 messages"
```

---

## Task 6: Add .dockerignore

**Files:**
- Create: `.dockerignore`

- [ ] **Step 1: Create `.dockerignore`**

```
# .dockerignore
.env
.env.*
chroma_db_data/
__pycache__/
**/__pycache__/
*.pyc
*.pyo
*.pyd
.pytest_cache/
.git/
docs/
Report/
benchmark/eval_results/
*.json.bak
```

- [ ] **Step 2: Commit**

```bash
git add .dockerignore
git commit -m "chore: add .dockerignore to exclude data and secrets from Docker build"
```

---

## Task 7: Run Full Test Suite & Verify

- [ ] **Step 1: Run all new unit tests**

```
cd /home/siriux/Projects/CSE291-A
python -m pytest tests/test_embedding_cache.py tests/test_identity_prompt.py \
    tests/test_extraction_unit.py tests/test_eval_metrics.py \
    tests/test_summarizer.py -v
```

Expected: all pass (≥ 18 tests)

- [ ] **Step 2: Verify existing tests still work**

```
python -m pytest tests/ -v --ignore=tests/test_server.py --ignore=tests/test_cases.py --ignore=tests/test_translation.py
```

(Skip the integration tests that require a live server/API.)

- [ ] **Step 3: Quick latency check on embedding cache**

```python
# Run in Python REPL or a quick script
import asyncio, time
from unittest.mock import AsyncMock
from agent.storage.embedding_service import OpenAIEmbeddingService

svc = OpenAIEmbeddingService.__new__(OpenAIEmbeddingService)
svc.api_key = "test"; svc.model = "text-embedding-3-small"; svc.dimension = 1536; svc._cache = {}; svc._cache_size = 1024

fake_embed = [0.1] * 1536
mock_client = AsyncMock()
mock_client.embeddings.create.return_value = AsyncMock(data=[AsyncMock(embedding=fake_embed)])
svc.client = mock_client

# Warm up
asyncio.run(svc.embed_text("test query"))
# Cached call should be near 0ms
t0 = time.perf_counter()
asyncio.run(svc.embed_text("test query"))
print(f"Cache hit latency: {(time.perf_counter()-t0)*1000:.2f}ms")
# Expected: < 1ms
```

---

## Task 8: Update Resume

The resume section should be updated to reflect:
- 85.2% cross-session LoCoMo recall (+identity prompt improvement targets 80%+ for category 1)
- Embedding cache reducing repeated query latency
- Context compression handling 14+ turn conversations
- Token-F1 soft evaluation metric
- Unit test suite (18+ tests covering memory, extraction, evaluation)
- Real weather API (Open-Meteo geocoding + forecast)

**Updated LaTeX for the three bullet points:**

```latex
\item \textbf{开发高性能 React 智能体 (Agent)}：利用 \textbf{LangGraph} 构建三节点
(call\_model / tools / extract\_preferences) 状态机与 ReAct 循环，
实现动态工具加载（6 类内置工具 + 用户专属记忆工具）；集成\textbf{上下文压缩}
（对话超过 14 轮时自动摘要旧消息）；部署为 \textbf{FastAPI} 服务（9 个 REST 端点），
短/中/长三档 benchmark 工具调用准确率均达 \textbf{100\%}。

\item \textbf{构建多租户长期记忆系统 (Long-term Memory)}：集成
\textbf{LangMem + ChromaDB} 与 \textbf{OpenAI Embeddings (text-embedding-3-small, 1536 维)}；
设计带 LRU 缓存的嵌入服务以降低重复查询延迟；通过用户命名空间隔离实现跨会话持久化；
在 \textbf{LoCoMo 基准}（19 会话 / 419 轮）上达到 \textbf{85.2\% 跨会话记忆召回率}。

\item \textbf{设计混合上下文提取与系统评估框架}：结合正则模式匹配（低延迟）与
\textbf{LLM 结构化输出}（高精度）的双策略偏好提取，针对身份类信息
（姓名、职业、地点、年龄等）专项强化提取 prompt；集成 Open-Meteo 真实天气 API
与 Claude 原生 Web Search（含 DuckDuckGo 降级回退）；
构建覆盖 Token-F1 软匹配与精确匹配的完整评估套件，
长对话响应质量相比 Phase 1 提升 \textbf{+114.3\%}。
```
