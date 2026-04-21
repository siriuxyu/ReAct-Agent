"""HTTP-facing runtime error helpers."""

from __future__ import annotations


RECURSION_ERROR_MESSAGE = (
    "Error: Agent reached maximum tool call limit (25 iterations). "
    "The question may be too complex or require information not available in memory."
)


def is_graph_recursion_error(exc: Exception, graph_recursion_error_cls) -> bool:
    """Return True when an exception represents a LangGraph recursion limit failure."""
    return (
        graph_recursion_error_cls is not None and isinstance(exc, graph_recursion_error_cls)
    ) or "GraphRecursionError" in type(exc).__name__ or "recursion limit" in str(exc).lower()
