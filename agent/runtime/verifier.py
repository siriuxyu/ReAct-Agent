"""Lightweight verification helpers for structured tool artifacts."""

from __future__ import annotations

from typing import Any, Iterable


def artifact_has_error(artifact: dict[str, Any]) -> bool:
    """Return True when an artifact explicitly signals failure."""
    if artifact.get("ok") is False:
        return True
    summary = str(artifact.get("summary", "")).lower()
    return summary.startswith("error")


def split_artifacts(artifacts: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split artifacts into successful and failed groups."""
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for artifact in artifacts:
        if artifact_has_error(artifact):
            failures.append(artifact)
        else:
            successes.append(artifact)
    return successes, failures
