"""Common contract shared by every agent tool.

Each tool exposes a `name`, a `description`, an `input_schema`, and a single
`execute(input)` method that returns a `ToolResult`. This lets callers
(today: ToolRegistry + AgentLoop; earlier: the orchestrator's now-retired
per-route dispatch) treat every tool the same way regardless of what it does
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

Structured metadata (`input_schema`, `ToolDescriptor`): this is foundation
work for a general-purpose agent that can eventually answer "what
capabilities are available to me?" and describe them to an LLM in a
consistent shape. `ToolRegistry.describe()`/`describe_all()`
(app/agent/tool_registry.py) build `ToolDescriptor`s from registered tools.
Nothing here performs tool selection, domain routing, or intent
classification — a tool describes its own capability; deciding which one to
use remains entirely LLMDecisionMaker's job.
"""
from __future__ import annotations

from dataclasses import dataclass, field
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

    `input_schema` describes, at a glance, what that single string means for
    this tool — e.g. `{"query": "string"}` for WebSearchTool, or `{}` for a
    tool (like TimeTool) that takes no input. It is intentionally a plain
    `dict[str, str]` (parameter name -> short type/kind), not a JSON-Schema
    document — enough structure to be useful in a prompt later, without
    pulling in a schema library this project doesn't otherwise need.
    """

    name: str
    description: str
    input_schema: dict[str, str]

    def execute(self, input: str | None = None) -> ToolResult:
        ...


@dataclass(frozen=True)
class ToolDescriptor:
    """Structured, read-only metadata describing one tool's capability.

    This is what "what capabilities are currently available to me?" resolves
    to: a plain data snapshot suitable for handing to an LLM prompt, logging,
    or a future capability-discovery UI. It never holds a reference to the
    tool itself and can't execute anything.

    `output_description` and `permissions` are genuinely optional — most
    tools today don't define `permissions` (there's no permission/security
    system in this codebase yet) and it isn't invented here; a tool that
    doesn't define `output_description`/`permissions` simply gets the
    defaults below (see ToolRegistry._to_descriptor for how these are read).
    """

    name: str
    description: str
    input_schema: dict[str, str] = field(default_factory=dict)
    output_description: str = ""
    permissions: tuple[str, ...] = ()
