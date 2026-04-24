# Cliriux Agent Server

FastAPI server and benchmark harness for a personal assistant agent with:
- a framework-agnostic runtime layer
- long-term memory backed by ChromaDB
- cross-session transcript recall in SQLite
- tool registry, policy checks, and confirmation gates
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

## Architecture

The server now follows a layered runtime instead of putting business logic directly in the LangGraph app:

```text
server.py
  -> agent/runtime/service.py
  -> agent/runtime/loop.py
  -> agent/runtime/agent_runtime.py
  -> agent/runtime/workspace.py
  -> agent/memory + agent/policy + tools/registry.py
  -> agent/adapters/langgraph_adapter.py
  -> agent/graph.py
```

Key modules:

- `agent/runtime/`
  Shared request preparation, observe/decide/act loop primitives, per-turn execution, streaming, session services, tool execution helpers, and the runtime workspace.
- `agent/policy/`
  Approval and tool-risk evaluation for side-effectful actions.
- `agent/memory/`
  Split memory layers for profile memory, session recall, and per-turn task scratchpad assembly.
- `tools/registry.py`
  Central metadata, capability grouping, runtime tool construction, and tool execution contracts.
- `agent/adapters/`
  Thin LangGraph and API adapters so framework details do not leak everywhere.

LangGraph still powers orchestration, but it is treated as a backend runtime rather than the center of the application architecture.

The runtime workspace is explicit agent working memory, not a fixed workflow. It records current goal, observations, artifacts, pending action, policy constraints, and decision traces while leaving the next action to the agent/runtime decision layer.

`agent/runtime/loop.py` exposes a framework-neutral `observe -> decide -> act` seam. The current LangGraph graph is one execution backend, not the agent architecture itself.

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

- `profile memory`
  Durable user preferences and identity facts stored in ChromaDB under a per-user namespace.
- `session recall`
  Cross-session transcript snippets stored in SQLite and injected on demand for recall-heavy prompts.
- `task scratchpad`
  Ephemeral per-turn context describing the active user goal, session, and task type. This is assembled at runtime and is not persisted as long-term memory.
- `POST /reset/...` clears short-term session state. With `preserve_memory=true`, durable profile memory remains intact.

At request time, these layers are merged by `agent/memory/runtime_recall.py` and appended to the system prompt for new turns.

## Tooling Model

- Tools are registered through `tools/registry.py` and annotated in `tools/metadata.py`.
- Side-effectful operations such as email sends, calendar writes, task creation, and reminder deletion can be routed through approval checks before execution.
- Confirmation payloads include structured, redacted previews plus dry-run metadata, so external sends and destructive actions are auditable before execution.
- Tool results are normalized into structured payloads so the runtime can reason over summaries and artifacts instead of reparsing display strings.

## Model Routing

`agent/runtime/router.py` returns auditable route decisions for each runtime step:

- `selected_model`
- `reason`
- task and complexity signals
- whether tool results were present

These decisions are written into the runtime workspace decision trace and can be benchmarked without calling external APIs.

## Benchmarks

Runtime control-plane micro-benchmarks do not call external APIs:

```bash
python benchmark/runtime_micro_benchmark.py --iterations 1000
python benchmark/runtime_micro_benchmark.py --iterations 1000 --json
```

The JSON output includes a run manifest with commit hash, Python version, platform, timestamp, and iteration count.

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

Local retrieval-only runs have measured `Recall@10 = 89.8%` for the current memory pipeline. Treat this as a reproducible local benchmark number: keep the exact dataset path, Chroma path, and command output with the run when reporting it externally.

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
agent/runtime/                 Runtime service, turn execution, streaming, sessions
agent/runtime/loop.py          Framework-neutral observe/decide/act loop
agent/runtime/workspace.py     Structured agent working memory
agent/policy/                  Approval and tool policy
agent/memory/                  Profile memory, session recall, scratchpad assembly
agent/adapters/                API and LangGraph adapters
agent/graph.py                 LangGraph wiring
tools/                         Tool implementations + registry/metadata
server.py                      FastAPI server
run_agent.py                   CLI agent runner
benchmark/benchmark_runner.py  Tool benchmarks + evidence-fed LoCoMo QA
benchmark/runtime_micro_benchmark.py
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
