"""The single authoritative tool-execution boundary (Milestone 18).

    tool_name, tool_input, ExecutionContext
                |
                v
    ToolExecutionGate.execute(...)
                |
        1. resolve   (ToolRegistry — unchanged, still authoritative)
                |
        2. authorize (PermissionPolicy — app/agent/permissions.py)
                |
        3. validate  (the tool's own optional `validate()`, if it has one)
                |
        4. confirm   (block here if the policy said CONFIRM)
                |
        5. execute   (Tool.execute — unchanged contract, returns ToolResult)

This is the ONE place in the codebase that turns a proposed
(tool_name, tool_input) pair into an actual `tool.execute()` call, once a
`ToolExecutionGate` is wired into `AgentLoop` (see app/agent/loop.py;
wiring is opt-in — `AgentLoop(tool_execution_gate=None)`, the default,
means no gate exists and this module is never consulted, preserving every
pre-Milestone-18 caller's behavior exactly).

--------------------------------------------------------------------------
What this class does NOT do
--------------------------------------------------------------------------
- Call the LLM, or ask it anything. `execute()`'s three arguments come
  from the caller (AgentLoop); nothing here reaches out for more.
- Trust model-provided authorization or confirmation. Nothing in this
  module reads an `AgentDecision`, raw model JSON, or any field named
  `authorized`/`permission`/`confirmed` — its only inputs are a tool
  name/input pair and a trusted `ExecutionContext` (app/agent/permissions.py).
- Mutate the `PermissionPolicy` it was given, or the `ToolRegistry`.
- Invent a tool. `self.tools.get(tool_name)` is the SAME `ToolRegistry`
  lookup every other caller already uses — this class holds no separate
  tool table of its own (Milestone 18 design §1: "DO NOT create a second
  ToolRegistry").
- Bypass input validation. A tool's own `ValueError` contract
  (app/tools/base.py) is unchanged and still the authority on "is this
  input usable"; see `_maybe_validate` for the one small, backward-
  compatible addition this class makes to when that check can run.
- Bypass Milestone 17. This class knows nothing about `CorrectionPolicy`,
  `FailureCategory`, or `AgentState` — it raises typed exceptions
  (`PermissionDeniedError`, `ConfirmationRequiredError`) and returns
  `ToolResult`, and `AgentLoop` is the one place that turns those into
  Milestone 17 failure categories (see app/agent/loop.py).

--------------------------------------------------------------------------
Order, and why validation sits between authorization and confirmation
--------------------------------------------------------------------------
Resolve -> authorize -> validate -> confirm -> execute is the order the
Milestone 18 design specifies. DENY is checked first among the
authorization outcomes: a denied tool never reaches validation, so an
attacker cannot use "helpfully" well-formed input to get further than a
badly-formed one would — DENY always wins regardless of input quality.

Validation for a tool that does not define its own `validate()` (every
tool in this codebase today) still happens — inside `tool.execute()`
itself, exactly as before Milestone 18, which is why such a tool's
validation is only reachable once BOTH authorization and confirmation
have already cleared. That is not a weaker guarantee: no destructive
work happens before a `ValueError` would be raised in any tool that
exists today (see app/tools/date.py, app/tools/web_search.py), so
"validated inside execute()" and "validated before execute()" are
equivalent in outcome for the current tool set. `validate()` is an
OPTIONAL, additive hook (read via `getattr`, exactly like
`ToolDescriptor`'s other optional metadata) for a future tool that wants
its input checked before a caller is asked to confirm it — nothing in the
`Tool` Protocol changed, so every existing tool and every existing
duck-typed test fake still satisfies it unchanged.
"""
from __future__ import annotations

import logging

from app.agent.permissions import ExecutionContext, PermissionDecision, PermissionPolicy
from app.agent.tool_registry import ToolNotFoundError, ToolRegistry
from app.tools.base import ToolResult

logger = logging.getLogger(__name__)


class PermissionDeniedError(Exception):
    """Raised by `ToolExecutionGate.execute()` when the injected
    `PermissionPolicy` returns `PermissionDecision.DENY` for a resolved,
    registered tool.

    Deliberately its own type, not `ValueError` (which already means
    "invalid tool input" in this codebase — see app/tools/base.py) and not
    `ToolNotFoundError` (a different failure: the tool exists and is
    known, it is simply not authorized). `AgentLoop` catches this
    specifically and — unlike every other tool-execution failure — never
    offers it to an injected Milestone 17 `CorrectionPolicy` (see
    app/agent/loop.py): a denial must not become a self-correction
    opportunity, or "propose the same privileged action again" becomes a
    structurally available retry loop. That is enforced in the loop, not
    here; this exception only carries the fact that denial happened.
    """


class ConfirmationRequiredError(Exception):
    """Raised by `ToolExecutionGate.execute()` when the injected
    `PermissionPolicy` returns `PermissionDecision.CONFIRM` — the tool is
    on the allow-list, but the trusted `ExecutionContext` does not (yet)
    show it as confirmed.

    Like `PermissionDeniedError`, `AgentLoop` treats this as terminal for
    the current execution and never offers it to a `CorrectionPolicy`:
    nothing the model can do WITHIN one execution changes the trusted
    `ExecutionContext` it was given (that context is constructed once, by
    the application, before `AgentLoop.run()` starts) — see
    app/agent/permissions.py's module docstring. Trusted confirmation
    arriving is a NEW request with a NEW context, not a correction of this
    one.
    """


class ToolExecutionGate:
    """The authoritative resolve -> authorize -> validate -> confirm ->
    execute boundary. See the module docstring for the full pipeline and
    what this class deliberately does not do.
    """

    def __init__(self, tool_registry: ToolRegistry, permission_policy: PermissionPolicy):
        if not isinstance(tool_registry, ToolRegistry):
            raise ValueError("tool_registry must be a ToolRegistry.")
        if not isinstance(permission_policy, PermissionPolicy):
            raise ValueError("permission_policy must implement the PermissionPolicy protocol.")

        self.tools = tool_registry
        self.permission_policy = permission_policy

    def execute(self, tool_name: str, tool_input: str | None, context: ExecutionContext) -> ToolResult:
        """Run one proposed tool action through the full authorization
        pipeline, or raise a typed failure explaining why it did not run.

        Raises:
            ToolNotFoundError: `tool_name` is not registered (identical to
                `ToolRegistry.get`'s own contract — this method adds no
                new unknown-tool semantics).
            PermissionDeniedError: the policy returned DENY.
            ConfirmationRequiredError: the policy returned CONFIRM and
                `context` does not show the tool as trusted-confirmed.
            ValueError: the tool (or its optional `validate()`) rejected
                `tool_input` — identical contract to calling
                `tool.execute()` directly, just possibly raised one step
                earlier (see `_maybe_validate`).

        Returns:
            Whatever `tool.execute(tool_input)` returns — this method
            never wraps, alters, or re-interprets a successful or
            operationally-failed `ToolResult`.
        """
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ValueError("tool_name must be a non-empty string.")
        if not isinstance(context, ExecutionContext):
            raise ValueError("context must be an ExecutionContext.")

        # 1. RESOLVE — the existing ToolRegistry remains the sole source of
        # truth for tool existence. No separate table, no fallback lookup.
        tool = self.tools.get(tool_name)  # raises ToolNotFoundError, unchanged
        descriptor = self.tools.describe(tool_name)

        # 1b. BIND IDENTITY (Milestone 18-C) — the thing authorized must BE
        # the thing executed.
        #
        # `ToolRegistry` stores a tool under `tool.name` as it was at
        # REGISTRATION time, but `_to_descriptor` re-reads `tool.name` at
        # DESCRIBE time, and `PermissionPolicy.evaluate` keys both its
        # allow-list check and its `context.is_confirmed()` check off
        # `descriptor.name`. If a tool's self-declared `name` ever diverges
        # from the registry key it is stored under, those two names are
        # different identities: the policy would rule on `descriptor.name`
        # while `tool` — resolved by the CALLER-supplied `tool_name` — is
        # what actually runs. A tool registered as `evil_tool` that later
        # sets `self.name = "time"` would then execute under `time`'s
        # authorization, and a confirmation the application granted for
        # `time` would satisfy `evil_tool`'s confirmation requirement.
        #
        # Nothing model-derived can cause that divergence today (tool
        # objects are application-authored, and `ToolRegistry.register`
        # already refuses to re-register an existing name), so this is
        # defense in depth rather than a patch for a reachable exploit.
        # It fails CLOSED, as `PermissionDeniedError` — the tool exists but
        # cannot be authorized — which `AgentLoop` already treats as
        # terminal and never offers to a `CorrectionPolicy`.
        if descriptor.name != tool_name:
            logger.error(
                "tool.identity.mismatch requested=%s descriptor=%s", tool_name, descriptor.name
            )
            raise PermissionDeniedError(
                f"Tool {tool_name!r} failed its identity check and is not authorized to execute."
            )

        # 2. AUTHORIZE — the descriptor came from the TOOL's own attributes
        # (ToolRegistry._to_descriptor), never from anything model-derived.
        decision = self.permission_policy.evaluate(descriptor, context)
        if decision is PermissionDecision.DENY:
            logger.warning("tool.authorization.denied tool=%s", tool_name)
            raise PermissionDeniedError(f"Tool {tool_name!r} is not authorized to execute.")
        logger.info("tool.authorization.allowed tool=%s decision=%s", tool_name, decision.value)

        # 3. VALIDATE — optional, additive; see the module docstring for
        # why this is a no-op for every tool that exists today.
        self._maybe_validate(tool, tool_input)

        # 4. CONFIRM — checked AFTER validation (Milestone 18 design §12),
        # and only reachable at all for a tool that cleared DENY above.
        if decision is PermissionDecision.CONFIRM:
            logger.warning("tool.confirmation.required tool=%s", tool_name)
            raise ConfirmationRequiredError(
                f"Tool {tool_name!r} requires trusted confirmation before it can run."
            )

        # 5. EXECUTE — the existing Tool contract, unchanged. A tool with
        # no separate validate() still validates its input HERE, exactly
        # as it always has; a ValueError from this call is not caught here
        # and propagates with its existing meaning.
        logger.info("tool.execution.started tool=%s", tool_name)
        result = tool.execute(tool_input)
        if result.success:
            logger.info("tool.execution.completed tool=%s", tool_name)
        else:
            logger.warning("tool.execution.failed tool=%s", tool_name)
        return result

    def _maybe_validate(self, tool: object, tool_input: str | None) -> None:
        """Call `tool.validate(tool_input)` if, and only if, the tool
        defines one. Absent on every tool in this codebase today — this
        is a pure extension point, read defensively via `getattr` exactly
        like `ToolDescriptor`'s other optional fields (app/tools/base.py),
        so no existing `Tool` implementation or duck-typed test fake needs
        to change.

        A `validate` that raises `ValueError` is logged as a rejected
        input and re-raised unchanged — the SAME exception type and
        meaning `tool.execute()` already uses for the identical purpose,
        so `AgentLoop` needs no new exception handling to receive it.
        """
        validate = getattr(tool, "validate", None)
        if validate is None:
            return
        try:
            validate(tool_input)
        except ValueError:
            logger.warning("tool.input.rejected tool=%s", getattr(tool, "name", "<unknown>"))
            raise
