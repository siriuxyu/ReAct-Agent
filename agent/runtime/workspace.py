"""Structured agent working memory that is not a fixed workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RuntimeObservation:
    """One observed fact from the conversation or a tool artifact."""

    source: str
    summary: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "summary": self.summary,
            "data": dict(self.data),
        }


@dataclass(frozen=True)
class PendingAction:
    """A proposed action waiting on user approval or runtime policy."""

    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    preview: str = ""
    risk_level: str = "read"

    def to_payload(self) -> dict[str, Any]:
        return {
            "tool_calls": [dict(tool_call) for tool_call in self.tool_calls],
            "preview": self.preview,
            "risk_level": self.risk_level,
        }


@dataclass(frozen=True)
class RuntimeDecisionTrace:
    """One auditable decision made by the runtime control plane."""

    kind: str
    decision: str
    reason: str = ""
    signals: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "decision": self.decision,
            "reason": self.reason,
            "signals": dict(self.signals),
        }


@dataclass(frozen=True)
class RuntimeWorkspace:
    """Per-turn working memory for agent decisions.

    This records what the agent knows and what policy is constraining. It does
    not prescribe a sequence of steps for the agent to execute.
    """

    goal: str = ""
    task_type: str = "chat"
    observations: list[RuntimeObservation] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    pending_action: PendingAction | None = None
    constraints: list[str] = field(default_factory=list)
    decision_trace: list[RuntimeDecisionTrace] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "task_type": self.task_type,
            "observations": [observation.to_payload() for observation in self.observations],
            "artifacts": [dict(artifact) for artifact in self.artifacts],
            "pending_action": (
                self.pending_action.to_payload() if self.pending_action is not None else None
            ),
            "constraints": list(self.constraints),
            "decision_trace": [trace.to_payload() for trace in self.decision_trace],
        }


def _artifact_observation(artifact: dict[str, Any]) -> RuntimeObservation:
    tool = str(artifact.get("tool") or "tool")
    summary = str(artifact.get("summary") or "").strip()
    if not summary:
        ok = artifact.get("ok")
        summary = f"{tool} completed" if ok is not False else f"{tool} failed"
    return RuntimeObservation(
        source=tool,
        summary=summary,
        data=dict(artifact.get("data") or {}),
    )


def _pending_action_from_confirmation(
    pending_confirmation: Any,
) -> PendingAction | None:
    if pending_confirmation is None:
        return None
    if hasattr(pending_confirmation, "to_payload"):
        payload = pending_confirmation.to_payload()
    else:
        payload = dict(pending_confirmation)
    tool_calls = [dict(tool_call) for tool_call in payload.get("tool_calls", [])]
    if not tool_calls:
        return None
    return PendingAction(
        tool_calls=tool_calls,
        preview=str(payload.get("preview") or ""),
        risk_level=str(payload.get("highest_side_effect") or "write"),
    )


def build_runtime_workspace(
    *,
    goal: str,
    task_type: str,
    artifacts: list[dict[str, Any]] | None = None,
    pending_confirmation: Any = None,
    confirmation_resolution: str = "",
    decision_trace: list[RuntimeDecisionTrace] | None = None,
) -> RuntimeWorkspace:
    """Build working memory from current runtime facts without choosing a workflow."""
    artifact_list = [dict(artifact) for artifact in artifacts or []]
    pending_action = _pending_action_from_confirmation(pending_confirmation)
    constraints: list[str] = []
    if pending_action is not None:
        constraints.append(f"await_user_confirmation:{pending_action.risk_level}")
    if confirmation_resolution:
        constraints.append(f"confirmation:{confirmation_resolution}")
    if any(artifact.get("ok") is False for artifact in artifact_list):
        constraints.append("tool_failure_observed")

    return RuntimeWorkspace(
        goal=goal.strip(),
        task_type=task_type or "chat",
        observations=[_artifact_observation(artifact) for artifact in artifact_list],
        artifacts=artifact_list,
        pending_action=pending_action,
        constraints=constraints,
        decision_trace=list(decision_trace or []),
    )
