"""Common contract shared by every agent tool.

Each tool exposes a `name`, a `description`, and a single `execute(input)`
method that returns a `ToolResult`. This lets callers (today: the
orchestrator's hardcoded per-route dispatch; later: a generic tool registry
and agent loop) treat every tool the same way regardless of what it does
internally, instead of each tool returning a differently-shaped value
(WebSearchTool returned `list[dict]`, TimeTool/DateTool returned `dict`).

Error contract (deliberately NOT "catch everything and set success=False"):
- Invalid input (empty query, unparseable date, missing required config) is a
  precondition failure. Tools raise `ValueError` for it, exactly as before
  this abstraction existed — callers that already handle `ValueError` keep
  working unchanged.
- An expected *operational* failure of the tool's underlying work (a network
  error, a bad API response, malformed JSON from a remote service) is
  reported as `ToolResult(success=False, error=...)`, not raised. This is the
  case the orchestrator is designed to recover from gracefully.
- An unexpected bug (anything not covered above) is not caught by the tool at
  all and propagates normally, exactly as it would today.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ToolResult:
    """Uniform envelope every tool returns from `execute()`.

    - success=True  -> `data` holds the tool's normal output. Its shape is
      still tool-specific (a dict for TimeTool/DateTool, a list[dict] of
      sources for WebSearchTool) — this envelope standardizes the *outer*
      shape (success/data/error), not the tool's domain data.
    - success=False -> `error` holds a human-readable message describing an
      expected, recoverable operational failure. `data` is None.
    """

    success: bool
    data: Any = None
    error: str | None = None

    @classmethod
    def ok(cls, data: Any) -> "ToolResult":
        return cls(success=True, data=data, error=None)

    @classmethod
    def fail(cls, error: str) -> "ToolResult":
        return cls(success=False, data=None, error=error)


@runtime_checkable
class Tool(Protocol):
    """Structural contract every tool implements.

    This is a Protocol, not an ABC, on purpose: tools (and test fakes) don't
    need to inherit from anything — they just need the right shape. That
    keeps each tool file simple and lets tests keep using plain duck-typed
    fake classes, same as before this abstraction was introduced.

    `execute` takes a single optional string. Each tool decides what that
    string means (a search query for WebSearchTool, a date string for
    DateTool, nothing at all for TimeTool) and is responsible for validating
    it itself — the orchestrator never needs tool-specific knowledge to call
    any of them.
    """

    name: str
    description: str

    def execute(self, input: str | None = None) -> ToolResult:
        ...
