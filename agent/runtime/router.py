"""Task and model routing helpers for the runtime layer."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any

_CALENDAR_KEYWORDS = ("calendar", "schedule", "meeting", "event", "日程", "会议", "安排", "空闲")
_EMAIL_KEYWORDS = ("email", "mail", "gmail", "邮件", "收件箱", "发邮件")
_TASK_KEYWORDS = ("task", "todo", "checklist", "任务", "待办", "清单")
_REMINDER_KEYWORDS = ("remind", "reminder", "提醒", "闹钟")
_MEMORY_KEYWORDS = ("remember", "memory", "preference", "偏好", "记住", "记忆")
_SEARCH_KEYWORDS = ("search", "find", "look up", "查找", "搜索", "查询", "看看")


@dataclass(frozen=True)
class ModelRouteDecision:
    """Auditable model routing decision for one runtime step."""

    selected_model: str
    default_model: str
    step_name: str = "assistant"
    task_type: str = "chat"
    reason: str = "default"
    signals: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        """Return a serializable routing trace."""
        return {
            "selected_model": self.selected_model,
            "default_model": self.default_model,
            "step_name": self.step_name,
            "task_type": self.task_type,
            "reason": self.reason,
            "signals": dict(self.signals),
        }


def classify_task_type(text: str) -> str:
    """Infer a coarse task type for routing and control decisions."""
    normalized = text.strip().lower()
    if not normalized:
        return "chat"
    if any(token in normalized for token in _CALENDAR_KEYWORDS):
        return "calendar"
    if any(token in normalized for token in _EMAIL_KEYWORDS):
        return "email"
    if any(token in normalized for token in _TASK_KEYWORDS):
        return "tasks"
    if any(token in normalized for token in _REMINDER_KEYWORDS):
        return "reminder"
    if any(token in normalized for token in _MEMORY_KEYWORDS):
        return "memory"
    if any(token in normalized for token in _SEARCH_KEYWORDS):
        return "search"
    return "chat"


def select_model_for_step(
    default_model_spec: str,
    *,
    task_type: str = "chat",
    step_name: str = "assistant",
    latest_user_text: str = "",
    has_tool_results: bool = False,
) -> str:
    """Choose a primary model for a specific step before fallback applies."""
    return explain_model_route(
        default_model_spec,
        task_type=task_type,
        step_name=step_name,
        latest_user_text=latest_user_text,
        has_tool_results=has_tool_results,
    ).selected_model


def explain_model_route(
    default_model_spec: str,
    *,
    task_type: str = "chat",
    step_name: str = "assistant",
    latest_user_text: str = "",
    has_tool_results: bool = False,
) -> ModelRouteDecision:
    """Choose a model and return the signals that drove the decision."""
    fast_model = os.environ.get("ROUTER_FAST_MODEL", "").strip()
    planner_model = os.environ.get("ROUTER_PLANNER_MODEL", "").strip() or fast_model
    tool_model = os.environ.get("ROUTER_TOOL_MODEL", "").strip()
    complex_model = os.environ.get("ROUTER_COMPLEX_MODEL", "").strip()
    extraction_model = os.environ.get("ROUTER_EXTRACTION_MODEL", "").strip() or fast_model
    summary_model = os.environ.get("ROUTER_SUMMARY_MODEL", "").strip() or fast_model

    normalized_task = (task_type or "chat").lower()
    normalized_step = (step_name or "assistant").lower()
    user_text = latest_user_text or ""
    complex_keywords = ("multi", "compare", "summarize", "analyze", "plan", "设计", "总结", "比较", "分析")
    is_complex = len(user_text) > 400 or any(token in user_text.lower() for token in complex_keywords)
    task_prefers_tools = normalized_task in {"calendar", "email", "tasks", "reminder", "search", "memory"}
    signals = {
        "fast_model_configured": bool(fast_model),
        "planner_model_configured": bool(planner_model),
        "tool_model_configured": bool(tool_model),
        "complex_model_configured": bool(complex_model),
        "has_tool_results": has_tool_results,
        "is_complex": is_complex,
        "task_prefers_tools": task_prefers_tools,
        "user_text_chars": len(user_text),
    }

    if normalized_step == "planner":
        selected = planner_model or default_model_spec
        return ModelRouteDecision(
            selected_model=selected,
            default_model=default_model_spec,
            step_name=normalized_step,
            task_type=normalized_task,
            reason="planner_step" if planner_model else "planner_step_default",
            signals=signals,
        )
    if normalized_step == "preference_extraction":
        selected = extraction_model or default_model_spec
        return ModelRouteDecision(
            selected_model=selected,
            default_model=default_model_spec,
            step_name=normalized_step,
            task_type=normalized_task,
            reason="preference_extraction_step" if extraction_model else "preference_extraction_default",
            signals=signals,
        )
    if normalized_step == "summarizer":
        selected = summary_model or default_model_spec
        return ModelRouteDecision(
            selected_model=selected,
            default_model=default_model_spec,
            step_name=normalized_step,
            task_type=normalized_task,
            reason="summarizer_step" if summary_model else "summarizer_default",
            signals=signals,
        )
    if normalized_step == "tool_execution" and tool_model:
        return ModelRouteDecision(
            selected_model=tool_model,
            default_model=default_model_spec,
            step_name=normalized_step,
            task_type=normalized_task,
            reason="tool_execution_step",
            signals=signals,
        )

    if (is_complex or has_tool_results or task_prefers_tools) and complex_model:
        return ModelRouteDecision(
            selected_model=complex_model,
            default_model=default_model_spec,
            step_name=normalized_step,
            task_type=normalized_task,
            reason="complex_or_tool_task",
            signals=signals,
        )
    if not is_complex and not has_tool_results and fast_model and normalized_task == "chat":
        return ModelRouteDecision(
            selected_model=fast_model,
            default_model=default_model_spec,
            step_name=normalized_step,
            task_type=normalized_task,
            reason="simple_chat_fast_path",
            signals=signals,
        )
    return ModelRouteDecision(
        selected_model=default_model_spec,
        default_model=default_model_spec,
        step_name=normalized_step,
        task_type=normalized_task,
        reason="default",
        signals=signals,
    )
