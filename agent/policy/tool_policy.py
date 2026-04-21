"""Tool policy evaluation based on centralized metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from tools.metadata import ToolMetadata, get_tool_metadata

_SEVERITY_ORDER = {
    "read": 0,
    "write": 1,
    "external_send": 2,
    "destructive": 3,
}


@dataclass(frozen=True)
class ToolDecision:
    """Centralized policy outcome for one or more tool calls."""

    requires_confirmation: bool
    highest_side_effect: str = "read"
    confirmation_required: list[dict[str, Any]] = field(default_factory=list)
    dry_run_candidates: list[str] = field(default_factory=list)
    metadata: dict[str, ToolMetadata] = field(default_factory=dict)
    capabilities: list[str] = field(default_factory=list)
    timeout_budget_seconds: int = 0


def evaluate_tool_calls(tool_calls: Iterable[dict[str, Any]]) -> ToolDecision:
    """Evaluate tool calls against metadata-driven policy rules."""
    required: list[dict[str, Any]] = []
    dry_run_candidates: list[str] = []
    metadata_map: dict[str, ToolMetadata] = {}
    capabilities: set[str] = set()
    timeout_budget = 0
    highest = "read"

    for tool_call in tool_calls:
        name = tool_call.get("name", "")
        metadata = get_tool_metadata(name)
        metadata_map[name] = metadata
        capabilities.add(metadata.capability)
        timeout_budget += metadata.timeout_seconds
        if _SEVERITY_ORDER[metadata.side_effect] > _SEVERITY_ORDER[highest]:
            highest = metadata.side_effect
        if metadata.requires_confirmation:
            required.append(tool_call)
        if metadata.supports_dry_run:
            dry_run_candidates.append(name)

    return ToolDecision(
        requires_confirmation=bool(required),
        highest_side_effect=highest,
        confirmation_required=required,
        dry_run_candidates=dry_run_candidates,
        metadata=metadata_map,
        capabilities=sorted(capabilities),
        timeout_budget_seconds=timeout_budget,
    )
