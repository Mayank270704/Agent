"""A minimal name -> Tool registry.

This is intentionally small: it only stores and retrieves `Tool` instances by
name, and can describe them. It has no idea what any tool needs as input,
how to build that input, or what to do with a tool's result — that knowledge
stays with the individual tools and the orchestrator (see
app/agent/orchestrator.py). It also has no idea what a user *wants* — it
never classifies a request or picks a tool for one; that stays
LLMDecisionMaker's job (app/agent/decision_maker.py), advised only by
Router's deterministic hint (app/agent/router.py). The registry's job is
`name -> tool` lookup, plus turning registered tools into structured,
read-only `ToolDescriptor`s (see `describe`/`describe_all` below) — the
foundation for an agent that can answer "what capabilities are available to
me?" and describe them consistently, e.g. in an LLM prompt later.
"""
from __future__ import annotations

from app.tools.base import RiskLevel, Tool, ToolCapability, ToolDescriptor


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

    def describe(self, name: str) -> ToolDescriptor:
        """Structured metadata for one registered tool. Raises
        ToolNotFoundError (same as `get`) if `name` isn't registered."""
        return self._to_descriptor(self.get(name))

    def describe_all(self) -> list[ToolDescriptor]:
        """Structured metadata for every registered tool — answers "what
        capabilities are currently available?" without exposing the tool
        objects themselves. Order matches `list_tools()`."""
        return [self._to_descriptor(tool) for tool in self.list_tools()]

    @staticmethod
    def _to_descriptor(tool: Tool) -> ToolDescriptor:
        # name/description/input_schema are required by the Tool contract, so
        # they're read directly rather than defaulted — a tool missing one of
        # these has a real bug worth surfacing, not one worth hiding.
        return ToolDescriptor(
            name=tool.name,
            description=tool.description,
            input_schema=dict(tool.input_schema),
            # output_description/permissions are genuinely optional (see
            # ToolDescriptor's docstring) — most tools today don't define
            # them, so they're read defensively rather than required.
            output_description=getattr(tool, "output_description", ""),
            permissions=tuple(getattr(tool, "permissions", ())),
            # Milestone 18: capability/risk_level/requires_confirmation are
            # read the identical defensive way, defaulting to
            # ToolDescriptor's own (least-alarming) defaults for a tool
            # that declares none of them. This is the ONE place these
            # three values cross from a `Tool` object into the
            # authoritative `ToolDescriptor` a PermissionPolicy consults —
            # they are read from the TOOL, never from a decision, a
            # prompt, or any model output.
            capability=getattr(tool, "capability", ToolCapability.READ),
            risk_level=getattr(tool, "risk_level", RiskLevel.LOW),
            requires_confirmation=getattr(tool, "requires_confirmation", False),
        )
