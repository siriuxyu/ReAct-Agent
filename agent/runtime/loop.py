"""Framework-neutral observe/decide/act loop primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Iterable, Literal

from langchain_core.messages import BaseMessage

from .agent_runtime import AgentRuntime
from .types import RuntimeInspection, TurnResult

LoopAction = Literal["respond", "await_confirmation"]


@dataclass(frozen=True)
class AgentLoopDecision:
    """Decision produced after observing the current conversation state."""

    action: LoopAction
    selected_model: str
    reason: str
    inspection: RuntimeInspection

    def to_payload(self) -> dict:
        return {
            "action": self.action,
            "selected_model": self.selected_model,
            "reason": self.reason,
            "task_type": self.inspection.task_type,
            "workspace": (
                self.inspection.workspace.to_payload()
                if self.inspection.workspace is not None
                else None
            ),
        }


class AgentLoop:
    """Small control loop that does not depend on LangGraph or any backend."""

    def __init__(self, runtime: AgentRuntime | None = None) -> None:
        self.runtime = runtime or AgentRuntime()

    def observe(
        self,
        messages: Iterable[BaseMessage],
        *,
        default_model: str,
        selected_model: str = "",
        pending_confirmation: dict | None = None,
    ) -> RuntimeInspection:
        """Convert conversation state into runtime facts."""
        return self.runtime.inspect_messages(
            messages,
            default_model=default_model,
            selected_model=selected_model,
            pending_confirmation=pending_confirmation,
        )

    def decide(self, inspection: RuntimeInspection) -> AgentLoopDecision:
        """Choose the next high-level action without choosing a framework path."""
        if inspection.pending_confirmation is not None:
            return AgentLoopDecision(
                action="await_confirmation",
                selected_model=inspection.selected_model,
                reason="pending_confirmation",
                inspection=inspection,
            )
        if inspection.workspace is not None and inspection.workspace.pending_action is not None:
            return AgentLoopDecision(
                action="await_confirmation",
                selected_model=inspection.selected_model,
                reason="workspace_pending_action",
                inspection=inspection,
            )
        return AgentLoopDecision(
            action="respond",
            selected_model=inspection.selected_model,
            reason=inspection.route_decision.reason if inspection.route_decision else "default",
            inspection=inspection,
        )

    async def act(
        self,
        decision: AgentLoopDecision,
        action_fn: Callable[[AgentLoopDecision], Awaitable[TurnResult]],
    ) -> TurnResult:
        """Delegate execution to any backend-specific action function."""
        return await action_fn(decision)

    async def run_once(
        self,
        messages: Iterable[BaseMessage],
        *,
        default_model: str,
        action_fn: Callable[[AgentLoopDecision], Awaitable[TurnResult]],
        selected_model: str = "",
        pending_confirmation: dict | None = None,
    ) -> TurnResult:
        """Run one observe/decide/act cycle against a caller-provided backend."""
        inspection = self.observe(
            messages,
            default_model=default_model,
            selected_model=selected_model,
            pending_confirmation=pending_confirmation,
        )
        decision = self.decide(inspection)
        return await self.act(decision, action_fn)
