"""Application-owned tool authorization (Milestone 18).

    ToolDescriptor (app/tools/base.py)  +  ExecutionContext
                    |
                    v
    PermissionPolicy.evaluate(...)      <-- pure, injected, LLM-independent
                    |
                    v
    PermissionDecision: ALLOW | DENY | CONFIRM

This module answers exactly one question: "is the application willing to
let THIS tool run, right now?" It has no knowledge of the LLM, the agent
loop, tool execution, or ToolResult — see app/agent/tool_execution.py for
the component that actually USES a verdict from here to gate a real
`tool.execute()` call.

--------------------------------------------------------------------------
Core principle: the LLM is not an authority
--------------------------------------------------------------------------
Nothing in this module ever reads model output. `evaluate()`'s two
arguments are a `ToolDescriptor` (built by `ToolRegistry` from a tool's
OWN Python attributes — see app/tools/base.py) and an `ExecutionContext`
(trusted, application-populated state — see below). A model's raw JSON
can contain an `"authorized": true` or `"permission": "admin"` field; that
JSON is parsed by `LLMDecisionMaker._parse_tool_decision`
(app/agent/decision_maker.py), which reads ONLY `tool_name` and
`tool_input` — such extra fields never survive into an `AgentDecision`
and are therefore structurally unreachable from this module. There is no
code path here, or anywhere downstream, that reads a field named
`authorized`, `permission`, `capability`, `risk_level`, or `confirmed`
off anything model-derived.

--------------------------------------------------------------------------
Why ALLOW/DENY/CONFIRM, not True/False
--------------------------------------------------------------------------
A boolean would conflate two different questions: "is this tool ever
usable" and "is this specific attempt authorized RIGHT NOW". CONFIRM
represents an action the application is willing to allow only with an
additional trusted approval it does not currently have (Milestone 18
design §15) — a UI/approval WORKFLOW for producing that approval is
explicitly out of scope; only the backend decision point exists here.

--------------------------------------------------------------------------
ExecutionContext: trusted application state, not user/model/memory text
--------------------------------------------------------------------------
`ExecutionContext` carries the only two things this milestone's default
policy needs, both application-populated and neither derived from
anything the model, the user, or a memory layer produced:

- `session_id`: descriptive metadata only. Deliberately NOT consulted by
  `AllowlistPermissionPolicy.evaluate()` — a session identifier must not
  itself grant privilege (Milestone 18 design §11). It exists so a
  FUTURE, more elaborate policy could scope decisions per session without
  this dataclass needing to change.
- `confirmed_tools`: the set of tool names a TRUSTED application caller
  has, for this one request, already confirmed. This is never populated
  from conversation history, semantic memory, or any text a model
  produced — see `is_confirmed()`.

This is deliberately NOT ConversationMemory, NOT authentication, NOT a
persistent permission store. It is a small, request-scoped, throwaway
value object — the "current execution/request context" the Milestone 18
design calls for, nothing more. It is never written to any memory layer
(ConversationMemory, EpisodicMemory, SemanticMemory) and never read from
one.

--------------------------------------------------------------------------
Why the allow-set lives on the POLICY, not the CONTEXT
--------------------------------------------------------------------------
"Which tools exist and are approved for this deployment" is a STATIC,
application-configuration fact — decided once, by whoever constructs
`AllowlistPermissionPolicy`, not something that varies per request. Only
the DYNAMIC, per-request fact ("has trusted confirmation already been
given for tool X, on this particular request") lives on
`ExecutionContext`. Splitting it this way is also what keeps "allow all
registered tools" impossible by construction (Milestone 18 design §9):
`AllowlistPermissionPolicy` has no code path that reads `ToolRegistry` at
all, so a newly registered tool is never automatically approved — the
allow-set must be extended explicitly, in application code, independent
of registration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

from app.tools.base import ToolDescriptor


class PermissionDecision(Enum):
    """The only three outcomes a PermissionPolicy may return. A closed,
    typed vocabulary — never a string a caller might fuzzy-match against
    (e.g. `if "not allowed" in message`), which is exactly the failure
    mode Milestone 18 was scoped to eliminate."""

    ALLOW = "allow"
    DENY = "deny"
    CONFIRM = "confirm"


@dataclass(frozen=True)
class ExecutionContext:
    """Trusted, application-populated state for one execution/request.

    See the module docstring for what belongs here and why `session_id`
    is descriptive-only. `confirmed_tools` is a frozenset specifically so
    a context, once built, cannot be mutated in place by anything
    downstream (including, deliberately, by anything derived from model
    output) — a NEW context must be constructed to change what is
    confirmed, and only application code ever does that construction.
    """

    session_id: str | None = None
    confirmed_tools: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Reject anything that is not a collection of tool-name strings,
        and normalize what remains to a frozenset (Milestone 18-C).

        This mirrors, deliberately and exactly, the guard
        `AllowlistPermissionPolicy.__init__` already applies to
        `allowed_tools` — including its explicit `str` rejection. Without
        it, `confirmed_tools="delete_file"` (a plausible caller typo: a
        bare string where a set was meant) is accepted silently, and
        `is_confirmed()`'s `in` operator then degrades from SET MEMBERSHIP
        to SUBSTRING matching: every tool whose name is a substring of
        that string — `"delete"`, `"file"`, even `"e"` — reports as
        trusted-confirmed. Confirmation is the last gate in front of a
        HIGH-risk tool, so it must fail CLOSED on a malformed value rather
        than silently widen to match names nobody confirmed.

        Normalizing to `frozenset` (rather than merely validating) is what
        keeps the "a context cannot be mutated in place once built"
        guarantee in the class docstring true for a caller who passed a
        list or set: the mutable original is not retained.
        """
        confirmed = self.confirmed_tools
        if isinstance(confirmed, str) or not isinstance(confirmed, (frozenset, set, list, tuple)):
            raise ValueError("confirmed_tools must be a set/frozenset/list/tuple of tool name strings.")
        for name in confirmed:
            if not isinstance(name, str) or not name.strip():
                raise ValueError("every entry in confirmed_tools must be a non-empty string.")
        object.__setattr__(self, "confirmed_tools", frozenset(confirmed))

    def is_confirmed(self, tool_name: str) -> bool:
        """True only if `tool_name` is in the TRUSTED confirmed set this
        context was constructed with. There is no other way for a tool to
        appear confirmed — not conversation history, not semantic memory,
        not a `"confirmed": true` field anywhere in a model's output."""
        return tool_name in self.confirmed_tools


@runtime_checkable
class PermissionPolicy(Protocol):
    """Whatever decides whether one tool may run, given trusted context.

    A Protocol, matching every other injectable collaborator in this
    codebase (DecisionMaker, CorrectionPolicy, Tool, ...). `evaluate` MUST
    be:
    - deterministic: the same descriptor + context always yields the same
      decision.
    - pure: no mutation of `descriptor` or `context`.
    - free of LLM calls and free of network calls — an authorization
      decision must never depend on reaching an external service.
    - non-self-escalating: nothing about calling `evaluate()` can ever
      cause a LATER call, for the same tool and the same context, to
      return a more permissive answer. (`AllowlistPermissionPolicy` holds
      this trivially — it reads only its own construction-time state and
      the two arguments; it stores no mutable history of past calls.)
    """

    def evaluate(self, descriptor: ToolDescriptor, context: ExecutionContext) -> PermissionDecision:
        ...


class AllowlistPermissionPolicy:
    """The one PermissionPolicy implementation: an explicit, application-
    owned allow-set, with confirmation layered on top for any allowed
    tool that declares `requires_confirmation=True`.

    --------------------------------------------------------------------
    Deny by default, explicitly (Milestone 18 design §9)
    --------------------------------------------------------------------
    A tool name not in `allowed_tools` is DENIED — there is no "allow
    everything registered" fallback anywhere in this class, and this
    class never even holds a reference to a `ToolRegistry`. Registering a
    new tool therefore has ZERO effect on what this policy allows;
    authorization is extended only by explicitly adding a name to
    `allowed_tools`, in application code, as a deliberate act separate
    from registration.

    --------------------------------------------------------------------
    Confirmation, layered on top of ALLOW
    --------------------------------------------------------------------
    A tool must clear the allow-set FIRST. Only then does
    `descriptor.requires_confirmation` matter: if it is `True` and
    `context.is_confirmed(descriptor.name)` is `False`, the verdict is
    CONFIRM rather than ALLOW. A tool that is not on the allow-set is
    DENIED regardless of `requires_confirmation` or the context — CONFIRM
    is never used to soften a DENY.

    --------------------------------------------------------------------
    The model cannot downgrade risk or grant itself access
    --------------------------------------------------------------------
    `evaluate()` never reads anything off the model's decision — its only
    inputs are `descriptor` (built entirely from the TOOL's own Python
    attributes by `ToolRegistry`, before this policy is ever consulted)
    and `context` (trusted application state). A descriptor claiming
    `capability=DESTRUCTIVE` cannot be overridden by anything the model
    said, because the model never had a channel to say it in the first
    place — see the module docstring.
    """

    def __init__(self, allowed_tools: frozenset[str] | set[str] | list[str] | tuple[str, ...]):
        if not isinstance(allowed_tools, (frozenset, set, list, tuple)) or isinstance(allowed_tools, str):
            raise ValueError("allowed_tools must be a set/frozenset/list/tuple of tool name strings.")
        for name in allowed_tools:
            if not isinstance(name, str) or not name.strip():
                raise ValueError("every entry in allowed_tools must be a non-empty string.")

        self.allowed_tools: frozenset[str] = frozenset(allowed_tools)

    def evaluate(self, descriptor: ToolDescriptor, context: ExecutionContext) -> PermissionDecision:
        if not isinstance(descriptor, ToolDescriptor):
            raise ValueError("descriptor must be a ToolDescriptor.")
        if not isinstance(context, ExecutionContext):
            raise ValueError("context must be an ExecutionContext.")

        if descriptor.name not in self.allowed_tools:
            return PermissionDecision.DENY

        if descriptor.requires_confirmation and not context.is_confirmed(descriptor.name):
            return PermissionDecision.CONFIRM

        return PermissionDecision.ALLOW
