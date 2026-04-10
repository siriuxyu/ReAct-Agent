"""Registry for built-in tools plus runtime-scoped tool assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from .metadata import ToolMetadata, get_tool_metadata, list_tool_metadata


@dataclass
class ToolRegistry:
    """Central registry for built-in tools and their metadata."""

    _base_tools: dict[str, Callable[..., Any]]

    def get_base_tools(self) -> list[Callable[..., Any]]:
        """Return built-in tools in registration order."""
        return list(self._base_tools.values())

    def get_tool(self, name: str) -> Callable[..., Any] | None:
        """Return one registered tool by name, if present."""
        return self._base_tools.get(name)

    def list_tool_names(self) -> list[str]:
        """Return registered tool names."""
        return list(self._base_tools.keys())

    def get_metadata(self, name: str) -> ToolMetadata:
        """Return metadata for one tool."""
        return get_tool_metadata(name)

    def list_metadata(self) -> dict[str, ToolMetadata]:
        """Return a snapshot of tool metadata."""
        return list_tool_metadata()

    def list_capabilities(self) -> dict[str, list[str]]:
        """Group registered tool names by declared capability."""
        capabilities: dict[str, list[str]] = {}
        for name in self.list_tool_names():
            metadata = self.get_metadata(name)
            capabilities.setdefault(metadata.capability, []).append(name)
        return capabilities

    def build_runtime_tools(
        self,
        *,
        user_id: str | None = None,
        dynamic_tool_builder: Callable[[str | None], Iterable[Callable[..., Any]]] | None = None,
    ) -> list[Callable[..., Any]]:
        """Build the full runtime tool list including optional dynamic tools."""
        tools = self.get_base_tools()
        if dynamic_tool_builder is not None:
            tools.extend(dynamic_tool_builder(user_id))
        return tools


_default_registry: ToolRegistry | None = None


def create_tool_registry(tools: Iterable[Callable[..., Any]]) -> ToolRegistry:
    """Create a registry from a flat iterable of tool callables."""
    indexed: dict[str, Callable[..., Any]] = {}
    for tool in tools:
        name = getattr(tool, "name", None) or getattr(tool, "__name__", str(tool))
        indexed[name] = tool
    return ToolRegistry(_base_tools=indexed)


def get_tool_registry() -> ToolRegistry:
    """Return the process-wide default tool registry."""
    global _default_registry
    if _default_registry is None:
        from . import TOOLS

        _default_registry = create_tool_registry(TOOLS)
    return _default_registry
