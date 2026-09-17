"""Self-correction taxonomy and policy (Step 17).

    AgentLoop  (unchanged: single loop, sole AgentState mutator)
        |
        |  at one of its existing failure sites, classifies what happened
        v
    Failure(category, tool_name, detail)
        |
        v
    CorrectionPolicy.evaluate(state, failure)   <-- pure, injected, no I/O
        |
        v
    CorrectionVerdict(action, safe_message, signature)
        |
        +-- CORRECT   -> AgentLoop records a CorrectionNote and continues
        |                to its NEXT NORMAL iteration (no new loop, no
        |                retry branch — see app/agent/loop.py)
        +-- TERMINATE -> AgentLoop fails the state exactly as it always
                          has, byte for byte, with correction_policy=None
                          or omitted entirely.

--------------------------------------------------------------------------
What this module is NOT
--------------------------------------------------------------------------
This module contains NO execution, NO state mutation, and NO LLM calls.
`CorrectionPolicy.evaluate` is a pure function of `(state, failure)`: given
the same inputs it returns the same verdict, it reads `state.corrections`
but never writes to `state`, and it never touches a tool, a decision
maker, or the network. AgentLoop remains the ONLY component that mutates
AgentState and the ONLY execution loop in this codebase — see its module
docstring for why a second loop was rejected.

--------------------------------------------------------------------------
Default OFF, and what that guarantees
--------------------------------------------------------------------------
`AgentLoop(correction_policy=None)` — the default — makes every one of its
five classification sites behave EXACTLY as before this module existed:
`AgentLoop._apply_correction_or_fail` returns `False` immediately when no
policy is injected, before this module's types are even constructed. This
is what makes Milestone 17 additive rather than a behavior change to any
existing caller (ChatService, AgentOrchestrator's default) or test.

--------------------------------------------------------------------------
Safety: no bytes originating outside the application, by default
--------------------------------------------------------------------------
`Failure.tool_name` is populated ONLY where it is a value AgentLoop already
validated against the registry BEFORE this module ever sees it (an
INVALID_TOOL_INPUT or TOOL_EXECUTION_FAILED failure can only occur after
`ToolRegistry.get(tool_name)` already succeeded). For UNKNOWN_TOOL, the
name the model invented is NEVER carried into a `Failure` — it is
untrusted text with no reason to exist inside a correction message, a log
line, or a repetition signature. See `BudgetedCorrectionPolicy._signature`.

`Failure.detail` may carry tool-authored text (a `ValueError` message, or
`ToolResult.error`) — this text can interpolate whatever the model passed
as tool input, so it is treated as untrusted. `BudgetedCorrectionPolicy`
never renders it into `safe_message` unless the caller explicitly opts in
via `include_tool_error_text=True` (default `False` — see the Step 17
design's approved decision 2), and even then it is length-bounded.

`safe_message` itself is drawn from a FIXED, closed vocabulary
(`_SAFE_MESSAGES`) — one application-authored template per
`FailureCategory`. No category's message ever interpolates the model's own
prior output, an exception's raw text, or the model's own tool_name guess.

--------------------------------------------------------------------------
Termination is structural, not policy-dependent
--------------------------------------------------------------------------
Every correction is recorded as, and consumes, one AgentLoop iteration
(`state.step` is incremented at the top of every iteration, unconditionally
— see app/agent/loop.py). `max_iterations` bounds the number of iterations
regardless of what any CorrectionPolicy decides, including a hypothetical
policy that always returns CORRECT. `BudgetedCorrectionPolicy` adds a
SECOND, independent bound (`max_corrections`, shared across all
categories, never reset) plus repetition detection, but the first bound
holds even if this module were buggy or replaced entirely.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a runtime cycle
    from app.agent.state import AgentState

# Shared across every FailureCategory, never reset on a successful
# correction (Step 17 design §5: "do not optimize for maximum
# persistence" — a flat, auditable ceiling beats a policy nobody can
# reason about). This is a SECOND, independent bound on top of
# AgentLoop's own max_iterations; see the module docstring.
DEFAULT_MAX_CORRECTIONS = 2

# Bound on how much of a tool's own error text may be interpolated into a
# safe_message when include_tool_error_text=True. Mirrors
# LLMDecisionMaker's own _MAX_OBSERVATION_DATA_CHARS in spirit — this is a
# CHARACTER bound as a guard against unbounded content, not a token
# measurement.
_MAX_DETAIL_CHARS = 200


class FailureCategory(Enum):
    """The closed set of AgentLoop failures a CorrectionPolicy can be asked
    to classify. Every member corresponds to exactly one of AgentLoop's
    existing `state.fail(...)` call sites (see the Step 17 design's failure
    taxonomy) — this enum does not invent new failure modes, it names ones
    that already exist and were previously all terminal.

    Deliberately NOT included as members, because they must never be
    eligible for correction at all (see the module docstring on
    termination and app/agent/loop.py's own docstring):
    - max_iterations exhaustion — it IS the outer termination guarantee.
    - an uncaught/unexpected exception — never classified, never caught by
      this system; it propagates exactly as it always has.
    - MemorySessionIsolationError and other security/integrity failures —
      these never reach AgentLoop's decision/tool call sites at all (they
      surface from the retriever/writer, outside the loop's classification
      points), so there is nothing for this enum to name.
    """

    DECISION_PARSE = "decision_parse"
    """Categories A and B: the model's raw output could not be parsed
    into a valid AgentDecision at all (malformed JSON, missing/invalid
    action_type, a FINAL with a blank final_answer, a TOOL with a blank
    tool_name or non-string tool_input)."""

    UNKNOWN_TOOL = "unknown_tool"
    """Category C: the model named a tool that is not registered — caught
    either by LLMDecisionMaker's own validation (the common path, since it
    checks `tool_registry.has(...)` before ever building an AgentDecision)
    or, for any DecisionMaker that skips that check, by AgentLoop's own
    `ToolRegistry.get()` lookup. Both call sites map to this one category.
    """

    INVALID_TOOL_INPUT = "invalid_tool_input"
    """Category D: a registered tool rejected its input via `ValueError`
    (see app/tools/base.py's error contract). `tool_name` on the `Failure`
    is always a real, already-validated registry key here."""

    TOOL_EXECUTION_FAILED = "tool_execution_failed"
    """Category E: a registered tool ran and returned
    `ToolResult(success=False, ...)` — an expected, already-documented
    *operational* failure mode (network error, bad API response), not a
    precondition failure. `tool_name` is always a validated registry key.
    """

    PLAN_SKIPPED = "plan_skipped"
    """Category H: the model returned FINAL while its plan still had more
    than just the current step left pending (see AgentLoop._plan_blocks_
    final) — i.e. it tried to answer while skipping real, unattempted plan
    work."""

    PERMISSION_DENIED = "permission_denied"
    """Milestone 18: a resolved, registered tool was rejected by the
    injected `PermissionPolicy` (app/agent/permissions.py) via
    `PermissionDecision.DENY`. Distinct from UNKNOWN_TOOL (the tool does
    not exist) — here the tool exists and is known, it is simply not
    authorized.

    UNLIKE every category above, `AgentLoop` never offers this one to a
    CorrectionPolicy at all — not even to `BudgetedCorrectionPolicy`, which
    also refuses it independently as defense in depth (see its own
    `evaluate()`). This is deliberately a STRUCTURAL guarantee, not a
    well-behaved-default-policy convention: "the LLM must never be able
    to grant itself permission" (Milestone 18's core principle) must hold
    even if a caller injects a custom, permissive, or buggy
    CorrectionPolicy. Allowing a denial to reach ANY policy's `evaluate()`
    would open exactly the loop Milestone 18 forbids: permission denied ->
    correction feedback -> the model tries the SAME privileged request
    again -> denied again -> ... A session's trusted authorization context
    is fixed for the lifetime of one `AgentLoop.run()` call; nothing that
    happens during that call can change it, so no in-request retry could
    ever succeed regardless — the correct response is to fail the request,
    not spend budget discovering that repeatedly."""

    CONFIRMATION_REQUIRED = "confirmation_required"
    """Milestone 18: a resolved, allow-listed tool was rejected because
    the injected `PermissionPolicy` returned `PermissionDecision.CONFIRM`
    and the trusted `ExecutionContext` for this execution did not show it
    as confirmed.

    Treated identically to PERMISSION_DENIED for correction purposes, and
    for the same reason: the trusted confirmation state is fixed for the
    duration of one execution (constructed by the application before
    `AgentLoop.run()` starts), so nothing the model does within that
    execution can cause it to become confirmed. Trusted confirmation
    arriving is a NEW request with a NEW context, not a self-correction of
    this one (see app/agent/tool_execution.py's `ConfirmationRequiredError`
    docstring)."""


@dataclass(frozen=True)
class Failure(object):
    """One classified failure at a single AgentLoop iteration, handed to a
    CorrectionPolicy for a verdict.

    Carries only application-known or already-registry-validated data —
    see the module docstring's safety section for exactly what `tool_name`
    and `detail` are allowed to hold and why UNKNOWN_TOOL never populates
    `tool_name` with the model's invented name.
    """

    category: FailureCategory
    tool_name: str | None = None
    detail: str | None = None


class CorrectionAction(Enum):
    """What a CorrectionPolicy decided to do about one Failure."""

    CORRECT = "correct"
    TERMINATE = "terminate"


@dataclass(frozen=True)
class CorrectionVerdict:
    """A CorrectionPolicy's answer for one Failure.

    `signature` is computed here, by the POLICY — not by AgentLoop — so
    that AgentLoop stays entirely policy-agnostic (it does not know or
    care HOW failures are fingerprinted for repetition detection; it only
    stores whatever signature the policy computed, on a CORRECT verdict,
    into the resulting CorrectionNote). This is what keeps AgentLoop a
    pure mechanism with the policy as the sole source of judgment.

    `signature` is meaningless (and unused) when `action` is TERMINATE.

    `reason` is an OPTIONAL, short, log-safe label for why a TERMINATE
    verdict was reached (e.g. "repeated" or "budget_exhausted") — purely
    for observability (see AgentLoop's `correction.declined` log line). It
    is never rendered into a prompt. A policy that leaves it empty simply
    yields a less specific log line; nothing depends on it being set.
    """

    action: CorrectionAction
    safe_message: str
    signature: str = ""
    reason: str = ""


@runtime_checkable
class CorrectionPolicy(Protocol):
    """Whatever decides whether one classified Failure is worth another
    attempt.

    A Protocol, matching every other injectable collaborator in this
    codebase (DecisionMaker, Tool, EmbeddingProvider, ...): a policy needs
    no base class, only this one method. `evaluate` MUST be pure — no
    mutation of `state`, no I/O, no LLM call — because AgentLoop calls it
    synchronously inside its own control flow and trusts it to return
    quickly and deterministically for the same inputs.
    """

    def evaluate(self, state: "AgentState", failure: Failure) -> CorrectionVerdict:
        ...


# Fixed, application-authored feedback per category — never interpolates
# the model's own prior output, an exception's raw text, or an invented
# tool name. See the module docstring's safety section.
_SAFE_MESSAGES: dict[FailureCategory, str] = {
    FailureCategory.DECISION_PARSE: (
        "Your previous response could not be parsed. Respond with STRICT JSON ONLY, "
        "matching exactly one of the two shapes described above."
    ),
    FailureCategory.UNKNOWN_TOOL: (
        "The tool you requested is not registered. Choose only from the AVAILABLE "
        "TOOLS listed above, or return FINAL if no listed tool applies."
    ),
    FailureCategory.INVALID_TOOL_INPUT: (
        "The input you provided for that tool was invalid. Check the tool's input "
        "schema listed above and provide a value matching it, or choose a different "
        "action."
    ),
    FailureCategory.TOOL_EXECUTION_FAILED: (
        "The tool did not complete successfully. You may try a different approach, "
        "or return FINAL directly if you already have enough information to answer."
    ),
    FailureCategory.PLAN_SKIPPED: (
        "You returned a final answer, but the current plan still has pending steps. "
        "Continue working through the remaining plan steps before returning FINAL."
    ),
    FailureCategory.PERMISSION_DENIED: (
        "That tool is not authorized for this request. Choose a different, "
        "authorized action, or return FINAL if none applies."
    ),
    FailureCategory.CONFIRMATION_REQUIRED: (
        "That tool requires confirmation this request does not have. Choose a "
        "different action, or return FINAL if none applies."
    ),
}


class BudgetedCorrectionPolicy:
    """The one CorrectionPolicy implementation: a shared, never-reset
    correction budget plus consecutive-repetition detection.

    --------------------------------------------------------------------
    Budget (Step 17 design §5)
    --------------------------------------------------------------------
    `max_corrections` is ONE shared ceiling across every FailureCategory,
    not a separate budget per category — a separate-budget design would
    multiply the worst-case number of corrections by the number of
    categories, undermining the flat, auditable ceiling this design
    requires. It is never reset by an intervening success: `state.
    corrections` only ever grows, so `len(state.corrections)` is a
    monotonic count for the whole request.

    --------------------------------------------------------------------
    Repetition detection
    --------------------------------------------------------------------
    If the immediately PRECEDING correction has the same signature as the
    current failure, this terminates immediately — even if budget remains.
    Two consecutive identical failures mean the model is not converging;
    spending the rest of the budget on a third identical attempt would
    only delay an inevitable termination while adding nothing.

    Signatures are `category` alone for DECISION_PARSE, UNKNOWN_TOOL, and
    PLAN_SKIPPED (there is no safe, validated per-failure detail to
    fingerprint on — see the module docstring), and `category + tool_name`
    for INVALID_TOOL_INPUT / TOOL_EXECUTION_FAILED, where `tool_name` is
    always a real, already-validated registry key.

    --------------------------------------------------------------------
    include_tool_error_text (Step 17 design §3, approved decision 2)
    --------------------------------------------------------------------
    Defaults to `False`. When `False` (the default), `safe_message` is
    exactly the fixed per-category template in `_SAFE_MESSAGES` — provably
    free of any byte that did not originate in this module. When `True`,
    a bounded (`_MAX_DETAIL_CHARS`), truncated slice of `failure.detail`
    (already-existing tool/ValueError text — the SAME text that already
    reaches the model today via EXECUTION HISTORY, see the design's §3)
    is appended, for a caller who has decided that trade-off explicitly.
    """

    def __init__(
        self,
        max_corrections: int = DEFAULT_MAX_CORRECTIONS,
        *,
        include_tool_error_text: bool = False,
    ):
        if not isinstance(max_corrections, int) or isinstance(max_corrections, bool) or max_corrections < 0:
            raise ValueError("max_corrections must be an integer >= 0.")
        if not isinstance(include_tool_error_text, bool):
            raise ValueError("include_tool_error_text must be a bool.")

        self.max_corrections = max_corrections
        self.include_tool_error_text = include_tool_error_text

    def evaluate(self, state: "AgentState", failure: Failure) -> CorrectionVerdict:
        safe_message = self._safe_message(failure)

        if failure.category in (FailureCategory.PERMISSION_DENIED, FailureCategory.CONFIRMATION_REQUIRED):
            # Milestone 18: unconditional, checked before repetition/budget
            # and regardless of either. AgentLoop already never offers
            # these two categories to any policy (see FailureCategory's
            # own docstrings) -- this is DEFENSE IN DEPTH, not the primary
            # enforcement point, so that "permission denial cannot
            # self-escalate" holds even if this policy were ever consulted
            # directly, e.g. from a test or a future caller.
            return CorrectionVerdict(CorrectionAction.TERMINATE, safe_message, reason=failure.category.value)

        signature = self._signature(failure)

        if state.corrections and state.corrections[-1].signature == signature:
            # The immediately preceding correction was the SAME failure —
            # not merely "some failure occurred twice non-consecutively".
            # Consecutive-only is deliberate: a model that fails A, then B,
            # then A again is still exploring, not stuck.
            return CorrectionVerdict(CorrectionAction.TERMINATE, safe_message, reason="repeated")

        if len(state.corrections) >= self.max_corrections:
            return CorrectionVerdict(CorrectionAction.TERMINATE, safe_message, reason="budget_exhausted")

        return CorrectionVerdict(CorrectionAction.CORRECT, safe_message, signature=signature)

    def _signature(self, failure: Failure) -> str:
        if failure.category in (FailureCategory.INVALID_TOOL_INPUT, FailureCategory.TOOL_EXECUTION_FAILED):
            return f"{failure.category.value}:{failure.tool_name}"
        return failure.category.value

    def _safe_message(self, failure: Failure) -> str:
        base = _SAFE_MESSAGES[failure.category]
        if not self.include_tool_error_text or not failure.detail:
            return base

        detail = failure.detail.strip()
        if len(detail) > _MAX_DETAIL_CHARS:
            detail = detail[:_MAX_DETAIL_CHARS] + "...[truncated]"
        return f"{base} Tool detail: {detail}"
