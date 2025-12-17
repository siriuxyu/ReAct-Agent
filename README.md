# ReAct Agent Server

FastAPI server and benchmark harness for a LangGraph-based agent with:
- tool use
- long-term memory backed by ChromaDB
- benchmark runners for tool usage and LoCoMo

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Set environment variables:

```bash
export ANTHROPIC_API_KEY="..."
export OPENAI_API_KEY="..."
export CHROMA_PERSIST_PATH="./chroma_db_data"  # optional
```

Run the server:

```bash
python server.py
```

The server listens on `http://localhost:8000`.

Run the CLI agent:

```bash
python run_agent.py
```

## Core Endpoints

Invoke the agent:

```bash
curl -X POST "http://localhost:8000/invoke" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello"}],
    "userid": "test_user"
  }'
```

Reset short-term session, keep long-term memory:

```bash
curl -X POST "http://localhost:8000/reset/test_user?preserve_memory=true"
```

Clear a user's long-term memory:

```bash
curl -X DELETE "http://localhost:8000/memory/test_user"
```

Search a user's stored memory:

```bash
curl -X POST "http://localhost:8000/memory/test_user/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "favorite food", "limit": 10}'
```

## Memory Model

- Short-term state is session/thread history managed by LangGraph.
- Long-term memory is stored in ChromaDB under a per-user namespace.
- The agent currently gets `search_memory`; direct memory writes are mainly done through the API and benchmark utilities.
- `POST /reset/...` clears short-term session state. With `preserve_memory=true`, long-term memory stays intact.

## Benchmarks

Tool-usage benchmarks:

```bash
python benchmark/benchmark_runner.py --dataset short
python benchmark/benchmark_runner.py --dataset medium
python benchmark/benchmark_runner.py --dataset long
```

LoCoMo benchmark files can be regenerated with:

```bash
python scripts/download_benchmarks.py
```

LoCoMo has three distinct entrypoints:

### 1. Evidence-Fed QA

This is not a true memory benchmark. It injects gold evidence turns into the prompt.

```bash
python benchmark/locomo_evidence_qa_benchmark.py --dataset locomo
```

### 2. Simplified Memory Benchmark

Cheap approximation:
- preload transcript
- reset short-term session
- ask QA in one batch

```bash
python benchmark/locomo_simplified_memory_benchmark.py --limit-personas 2
```

Default output:

```text
locomo_simplified_memory_results.json
```

### 3. Full Memory Benchmark

End-to-end memory benchmark:
- chunk all sessions into long-term memory
- optionally extract facts
- reset short-term session
- answer QA via memory retrieval

```bash
python benchmark/locomo_full_memory_benchmark.py --limit-personas 2
```

Default output:

```text
locomo_full_memory_results.json
```

## LoCoMo Data Notes

The repo keeps two LoCoMo forms:

- `benchmark/locomo1.json`
  Original single-sample LoCoMo payload
- `benchmark/locomo1_converted.json`
  Runner-friendly converted format

The converted file now preserves:
- `session_date`
- `session_datetime_raw`
- turn-level `img_url`
- turn-level `blip_caption`
- turn-level `query`
- normalized `qa.evidence`

For memory storage and retrieval:
- `blip_caption` is injected into chunk text
- raw datetime and image/query info are retained in metadata

## Retrieval-Only Evaluation

If you only want retrieval metrics without running agent QA:

```bash
python scripts/measure_recall.py --k 1 5 10 --user-id memory_test_conv-26
```

If no Chroma data exists for that `user_id`, the script can bootstrap chunk storage automatically.

Useful flags:

```bash
--force-bootstrap
--allow-rebuild-existing-user
--chroma-path /path/to/chroma_db_data
--converted-path benchmark/locomo1_converted.json
```

## Project Layout

```text
agent/                         Core agent logic
tools/                         Tool implementations
server.py                      FastAPI server
run_agent.py                   CLI agent runner
benchmark/benchmark_runner.py  Tool benchmarks + evidence-fed LoCoMo QA
benchmark/locomo_evidence_qa_benchmark.py
benchmark/locomo_simplified_memory_benchmark.py
benchmark/locomo_full_memory_benchmark.py
benchmark/locomo_storage_utils.py
scripts/download_benchmarks.py
scripts/measure_recall.py
benchmark/locomo1.json
benchmark/locomo1_converted.json
```

## Notes

- `OPENAI_API_KEY` is required for embeddings.
- Redis is optional. Without `REDIS_URL`, checkpointing and embedding cache fall back to in-memory behavior.
- For isolated benchmark runs, prefer changing `--userid-prefix` or `CHROMA_PERSIST_PATH` instead of reusing the same memory namespace.
