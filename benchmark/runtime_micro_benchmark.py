"""Micro-benchmarks for Cliriux runtime control-plane overhead."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.policy.tool_policy import evaluate_tool_calls

_WORKSPACE_SPEC = spec_from_file_location(
    "cliriux_runtime_workspace",
    REPO_ROOT / "agent" / "runtime" / "workspace.py",
)
if _WORKSPACE_SPEC is None or _WORKSPACE_SPEC.loader is None:
    raise RuntimeError("Could not load runtime workspace module")
_WORKSPACE_MODULE = module_from_spec(_WORKSPACE_SPEC)
sys.modules[_WORKSPACE_SPEC.name] = _WORKSPACE_MODULE
_WORKSPACE_SPEC.loader.exec_module(_WORKSPACE_MODULE)
build_runtime_workspace = _WORKSPACE_MODULE.build_runtime_workspace

_ROUTER_SPEC = spec_from_file_location(
    "cliriux_runtime_router",
    REPO_ROOT / "agent" / "runtime" / "router.py",
)
if _ROUTER_SPEC is None or _ROUTER_SPEC.loader is None:
    raise RuntimeError("Could not load runtime router module")
_ROUTER_MODULE = module_from_spec(_ROUTER_SPEC)
sys.modules[_ROUTER_SPEC.name] = _ROUTER_MODULE
_ROUTER_SPEC.loader.exec_module(_ROUTER_MODULE)
explain_model_route = _ROUTER_MODULE.explain_model_route


@dataclass(frozen=True)
class BenchmarkResult:
    """One micro-benchmark timing result."""

    name: str
    iterations: int
    total_ms: float
    mean_us: float
    p95_us: float

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "iterations": self.iterations,
            "total_ms": round(self.total_ms, 3),
            "mean_us": round(self.mean_us, 3),
            "p95_us": round(self.p95_us, 3),
        }


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _manifest(iterations: int) -> dict[str, Any]:
    return {
        "benchmark": "runtime_micro",
        "iterations": iterations,
        "commit": _git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def _measure(name: str, iterations: int, fn) -> BenchmarkResult:
    samples: list[float] = []
    start = time.perf_counter()
    for _ in range(iterations):
        op_start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - op_start) * 1_000_000)
    total_ms = (time.perf_counter() - start) * 1_000
    return BenchmarkResult(
        name=name,
        iterations=iterations,
        total_ms=total_ms,
        mean_us=statistics.fmean(samples),
        p95_us=statistics.quantiles(samples, n=20)[18] if len(samples) >= 20 else max(samples),
    )


def run_runtime_microbenchmarks(iterations: int = 1000) -> dict[str, Any]:
    """Run local runtime micro-benchmarks without invoking LLMs or network tools."""
    artifacts = [
        {
            "tool": "search_emails",
            "ok": True,
            "summary": "Found messages from Alice",
            "data": {"count": 3},
        },
        {
            "tool": "find_free_slots",
            "ok": True,
            "summary": "Found two open slots",
            "data": {"slots": ["2026-04-24T14:00:00", "2026-04-24T15:30:00"]},
        },
    ]
    tool_calls = [
        {"name": "search_emails", "args": {"query": "from:alice"}},
        {"name": "send_email", "args": {"to": "alice@example.com", "subject": "Follow-up"}},
        {"name": "create_task", "args": {"title": "Send notes"}},
    ]

    results = [
        _measure(
            "runtime_workspace_build",
            iterations,
            lambda: build_runtime_workspace(
                goal="Coordinate with Alice and send a follow-up",
                task_type="mail",
                artifacts=artifacts,
                pending_confirmation={
                    "tool_calls": [tool_calls[1]],
                    "preview": "send follow-up email",
                    "highest_side_effect": "external_send",
                },
            ),
        ),
        _measure(
            "tool_policy_evaluate",
            iterations,
            lambda: evaluate_tool_calls(tool_calls),
        ),
        _measure(
            "model_route_explain",
            iterations,
            lambda: explain_model_route(
                "anthropic/claude-sonnet-4-5-20250929",
                task_type="mail",
                latest_user_text="Summarize Alice's email and draft a follow-up plan",
                has_tool_results=True,
            ),
        ),
    ]
    return {
        "manifest": _manifest(iterations),
        "iterations": iterations,
        "benchmarks": [result.to_payload() for result in results],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Cliriux runtime micro-benchmarks.")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    payload = run_runtime_microbenchmarks(iterations=args.iterations)
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    print("Runtime micro-benchmarks")
    for result in payload["benchmarks"]:
        print(
            f"- {result['name']}: mean={result['mean_us']}us "
            f"p95={result['p95_us']}us total={result['total_ms']}ms"
        )


if __name__ == "__main__":
    main()
