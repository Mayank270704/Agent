"""A minimal name -> Tool registry.

This is intentionally small: it only stores and retrieves `Tool` instances by
name. It has no idea what any tool needs as input, how to build that input,
or what to do with a tool's result — that knowledge stays with the individual
tools and the orchestrator (see app/agent/orchestrator.py). The registry's
only job is `name -> tool` lookup.
"""
from __future__ import annotations

from app.tools.base import Tool


class ToolRegistrationError(ValueError):
    """Raised when registering a tool name that's already taken."""


class ToolNotFoundError(LookupError):
    """Raised when looking up a tool name that isn't registered."""


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register `tool` under its own `tool.name`.

        Rejects duplicate names outright rather than silently replacing an
        existing tool — a second registration under the same name is almost
        always a bug (e.g. two tools accidentally sharing a name), and
        silently overwriting would hide it.
        """
        if tool.name in self._tools:
            raise ToolRegistrationError(
                f"A tool named '{tool.name}' is already registered."
            )
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolNotFoundError(f"No tool registered under the name '{name}'.") from exc

    def has(self, name: str) -> bool:
        return name in self._tools

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())
