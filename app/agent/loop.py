"""A bounded, generic agent control-flow loop.

    State
      |
    Decide  <---------------------------+
      |                                 |
      +--> final answer --> complete()  |
      |                                 |
      +--> tool action --> Tool -->     |
           Observation --> State update-+

AgentLoop is responsible ONLY for control flow: repeatedly asking a
DecisionMaker what to do next, executing at most one tool action per
iteration through a ToolRegistry, and recording everything onto an
AgentState. It has no built-in knowledge of any individual tool (it only
ever calls the generic `Tool.execute()` contract via the registry) and no
built-in knowledge of how decisions get made (that is fully injected via
DecisionMaker — a deterministic fake in tests today, something LLM-backed
later).

Iteration semantics: `AgentState.step` is 1-indexed and represents the
iteration currently being executed. It is incremented at the *start* of each
iteration, before the decision for that iteration is made. `max_iterations`
bounds how many times that can happen — the loop checks `state.step >=
max_iterations` *before* incrementing, so it can never execute more than
`max_iterations` decision cycles. If the bound is reached while the state is
still RUNNING, the state is failed with a clear, deterministic error instead
of looping forever.

As of Step 7, app/agent/orchestrator.py wires this loop together with
LLMDecisionMaker (app/agent/decision_maker.py) as the agent's execution
engine — see that module for how a request becomes an AgentState, runs
through this loop, and becomes a response.

Step 11 — plan awareness (optional, backward compatible): when
`state.plan` is set (app/agent/plan.py), this loop also advances it, using
only the smallest mechanism the existing AgentDecision/ToolResult model
supports:
- At the start of `run()`, a PENDING plan is started (-> RUNNING).
- Before each decide() call, the plan's current step (its first non-
  COMPLETED step) is started if it's still PENDING.
- A SUCCESSFUL tool action completes the plan's current step, and completes
  the plan itself once that was the last step. This is deliberately the
  only tool-side signal used — the AgentDecision JSON contract was not
  extended with a "this finishes the plan step" field, so a plan step that
  would genuinely need more than one tool call will be marked COMPLETED
  after just the first successful one. See the Step 11 report for this
  tradeoff.
- A FINAL decision may also finish the plan's current step directly, with
  no tool call at all — this is what lets a one-step plan (e.g. "Explain
  what Python is", which needs no tool) or the last step of a longer plan
  complete naturally. FINAL is only treated as *skipping* required plan
  work — and fails the state (and the plan), no retry attempted — when MORE
  than just the current step still isn't COMPLETED, i.e. the model would be
  leaving real future steps entirely unaddressed (see `_plan_blocks_final`).
- Any other execution failure (unregistered tool, invalid input, a failed
  ToolResult, hitting max_iterations) fails the plan the same way it
  already fails the state.

When `state.plan is None` (still the default), none of the above runs —
behavior is byte-for-byte identical to before Step 11.

Step 17 — self-correction, OPT-IN and DEFAULT OFF: this loop gained an
optional `correction_policy: CorrectionPolicy | None = None` constructor
argument (app/agent/reliability.py). It is consulted at exactly the five
places this loop already used to fail the state unconditionally:

    1. the decision maker raised a categorized DecisionMakerError
       (DECISION_PARSE / UNKNOWN_TOOL — see app/agent/decision_maker.py)
    2. a FINAL decision arrived while more than the current plan step was
       still pending (PLAN_SKIPPED)
    3. the decided tool is not registered (UNKNOWN_TOOL, the non-LLM path)
    4. a registered tool raised ValueError on its input (INVALID_TOOL_INPUT)
    5. a registered tool returned ToolResult(success=False, ...)
       (TOOL_EXECUTION_FAILED)

At each site, `self._apply_correction_or_fail(state, failure)`
is the ONLY thing that changed: with `correction_policy=None` (the
default, and the only behavior every caller before Step 17 exercises), it
returns `False` IMMEDIATELY — before constructing a `Failure`, before
touching `state.corrections` — so every one of the five sites fails the
state exactly as it always has, byte for byte. See
app/agent/reliability.py's module docstring for the full policy/safety
design; this loop remains the single execution engine and the only
component that mutates AgentState — a CorrectionPolicy only classifies,
it never executes, retries, or touches state itself.

A "correction" is not a special retry branch: it is simply NOT calling
`state.fail(...)` for one of the five sites above, and letting this
loop's own `while` condition carry it to its next ordinary iteration —
the SAME decide —> parse —> validate —> execute pipeline runs again,
with no bypass of tool registry validation, input validation, or any
other check. Every correction still consumes one ordinary iteration
(`state.step` is incremented, unconditionally, at the top of the loop
below), so `max_iterations` bounds the total number of corrections
regardless of what any policy decides — this is what makes runaway
self-correction structurally impossible rather than merely
policy-discouraged.

Milestone 18 — tool authorization, OPT-IN and DEFAULT OFF: this loop
gained an optional `tool_execution_gate: ToolExecutionGate | None = None`
constructor argument (app/agent/tool_execution.py), paired with an
optional `execution_context: ExecutionContext | None = None`
(app/agent/permissions.py). With no gate configured (the default, and the
only behavior every pre-Milestone-18 caller exercises), `_execute_tool_
action` resolves and executes a tool EXACTLY as it always has —
`self.tools.get(name)` then `tool.execute(input)`, with no authorization
step at all. When a gate IS configured, that same two-step sequence is
replaced by ONE call into `ToolExecutionGate.execute(...)`, which performs
resolve -> authorize -> validate -> confirm -> execute and raises one of
four typed failures (`ToolNotFoundError`, `PermissionDeniedError`,
`ConfirmationRequiredError`, `ValueError`) for every non-execution
outcome.

Two of those four are handled DIFFERENTLY from every other failure site in
this loop: `PermissionDeniedError` and `ConfirmationRequiredError` are
NEVER offered to `self.correction_policy`, unconditionally, regardless of
which policy (if any) is injected. See FailureCategory.PERMISSION_DENIED
and .CONFIRMATION_REQUIRED (app/agent/reliability.py) for the full
reasoning: a denial or an unmet confirmation must never become a
self-correction opportunity, because that is exactly the "propose the
same privileged action again" loop Milestone 18 exists to make
structurally impossible — not merely discouraged by a well-behaved
policy's default choice. `ToolNotFoundError` and `ValueError` raised
THROUGH the gate are handled identically to how they were handled before
Milestone 18 (UNKNOWN_TOOL / INVALID_TOOL_INPUT, both still eligible for
correction exactly as before) — the gate changes WHO resolves and
authorizes a tool, never how AgentLoop reacts to a stale registry entry
or bad input once it has one.

Milestone 19 — observability, OPT-IN and DEFAULT OFF: this loop gained an
optional `event_emitter: EventEmitter | None = None` constructor argument
(app/agent/telemetry.py). With no emitter configured (the default), every
`if self.event_emitter is not None:` guard below is simply never entered
— zero `AgentEvent`/`EventEmitter` objects are constructed and behavior
is byte-for-byte identical to before this milestone.

Every emission happens strictly AFTER the outcome it describes is already
final: `TOOL_PROPOSED` is emitted once the tool/input is already recorded
via `state.record_tool_call` (a fact, not a proposal-in-progress);
`TOOL_DENIED`/`TOOL_CONFIRMATION_REQUIRED`/`TOOL_INPUT_REJECTED` are
emitted from inside the `except` block for an exception `ToolExecutionGate`
already raised; `TOOL_AUTHORIZED`/`TOOL_EXECUTION_COMPLETED` are emitted
only once `_resolve_and_run_tool` has already returned a `ToolResult`.
Telemetry never reads a value back to decide anything, and nothing here
runs before, or in place of, the resolve -> authorize -> validate ->
confirm -> execute pipeline `ToolExecutionGate` already owns (see its own
module docstring) — `ToolExecutionGate` itself is deliberately UNCHANGED
by this milestone: every outcome it can produce is already distinguished
by the typed exceptions this loop catches, so wiring events here achieves
identical observability without adding a new parameter to the frozen
Milestone-18-C security boundary or its exact-signature regression test.

`CORRECTION_TRIGGERED`/`CORRECTION_DECLINED` are emitted only past the
existing `if self.correction_policy is None: return False` guard in
`_apply_correction_or_fail` — i.e. only when a real policy was actually
consulted. Emitting them unconditionally would fire a misleading
`CORRECTION_DECLINED` on every ordinary failure in the (today, default)
no-correction-policy production deployment, where there is no correction
mechanism to have declined anything.

Milestone 23 — global per-request deadline, OPT-IN and DEFAULT OFF: this
loop gained an optional `deadline: float | None = None` constructor
argument — an ABSOLUTE `time.monotonic()` cutoff, not a duration,
computed once by the caller (app/services/chat.py, at the same request
boundary `request_id` is already generated) and threaded through
AgentOrchestrator unchanged. With no deadline configured (the default),
behavior is byte-for-byte identical to before this milestone.

Individual Ollama calls already have their own client-side timeout
(app/models/llm.py, 60s) — that bounds ONE call. `deadline` bounds the
WHOLE request: a pathological multi-iteration execution (a model that
keeps requesting tools, each recovered by correction, forever approaching
but never hitting max_iterations in unlucky orderings) could otherwise
occupy a server thread far longer than any single call's timeout implies.

This is a COOPERATIVE check, not preemption: this codebase's execution is
synchronous, and there is no safe way to interrupt an arbitrary in-flight
Python call (a running LLM request, a running tool) without dangerous
thread-killing. The deadline is therefore checked at the exact same
granularity `max_iterations` already is — once, at the top of the loop,
before a new iteration starts — so the guarantee is precise and honestly
scoped: no FURTHER iteration begins once the deadline has passed. An
iteration already in flight when the deadline arrives still completes.

Exceeding the deadline fails the state exactly like `max_iterations`
exhaustion already does — the SAME terminal path, the SAME generic
failure answer at the API layer (app/main.py's existing `_failure_answer`
hardening), no new response shape. It is, like `max_iterations`,
unconditionally NEVER offered to `self.correction_policy`: correction
means "try again," which is precisely what a deadline exists to stop, so
there is no `_apply_correction_or_fail` call at this site, matching how
`max_iterations` itself has never been correctable either.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from app.agent.permissions import ExecutionContext, PermissionDecision
from app.agent.plan import Plan, PlanStatus
from app.agent.reliability import CorrectionAction, CorrectionPolicy, Failure, FailureCategory
from app.agent.state import AgentState, AgentStatus, CorrectionNote
from app.agent.telemetry import EventEmitter, EventType, elapsed_ms, monotonic_start
from app.agent.tool_execution import ConfirmationRequiredError, PermissionDeniedError, ToolExecutionGate
from app.agent.tool_registry import ToolNotFoundError, ToolRegistry

logger = logging.getLogger(__name__)


class DecisionMakerError(Exception):
    """Base class for errors a DecisionMaker implementation may raise when it
    cannot produce a usable AgentDecision (e.g. malformed LLM output — see
    DecisionParseError in app/agent/decision_maker.py, which subclasses this).

    The loop treats this as an expected, recoverable-at-this-iteration
    failure: it fails the state and stops, the same way it already handles an
    unregistered tool or invalid tool input. An unreliable decision maker
    should not be able to crash the whole execution with an uncaught
    exception. Any other exception type is NOT caught here and propagates
    normally — a genuine bug in a decision maker is not swallowed.

    `category` (Step 17) is OPTIONAL and defaults to `None`. It exists so a
    concrete DecisionMaker (LLMDecisionMaker) can tell this loop WHICH
    FailureCategory a given failure is, without this loop needing to parse
    or guess from the exception's message text. A `None` category means
    "uncategorized" — this loop treats that identically to having no
    CorrectionPolicy at all: an uncategorized failure is ALWAYS terminal,
    regardless of any injected policy. This is a deliberate safety default:
    only a failure a DecisionMaker explicitly and deliberately classified
    is ever eligible for correction, so a raw
    `DecisionMakerError("some message")` — exactly what every test fake in
    this codebase already constructs — keeps failing the state exactly as
    it always has.
    """

    def __init__(self, message: str, *, category: FailureCategory | None = None):
        super().__init__(message)
        self.category = category


class ActionType(Enum):
    """What a Decide step decided to do next."""

    FINAL = "final"
    TOOL = "tool"


@dataclass(frozen=True)
class AgentDecision:
    """The outcome of one Decide step: either a final answer, or a tool action.

    Exactly one of the two "payloads" is meaningful, selected by
    `action_type`. Invalid combinations (a FINAL decision with no answer, a
    TOOL decision with no/blank tool name) are rejected at construction.
    """

    action_type: ActionType
    tool_name: str | None = None
    tool_input: str | None = None
    final_answer: str | None = None

    def __post_init__(self) -> None:
        if self.action_type is ActionType.FINAL:
            if self.final_answer is None or not str(self.final_answer).strip():
                raise ValueError("A FINAL decision must include a non-blank final_answer.")
        elif self.action_type is ActionType.TOOL:
            if self.tool_name is None or not str(self.tool_name).strip():
                raise ValueError("A TOOL decision must include a non-blank tool_name.")
        else:
            raise ValueError(f"Unsupported action_type: {self.action_type!r}")

    @classmethod
    def final(cls, answer: str) -> "AgentDecision":
        return cls(action_type=ActionType.FINAL, final_answer=answer)

    @classmethod
    def tool(cls, name: str, input: str | None = None) -> "AgentDecision":
        return cls(action_type=ActionType.TOOL, tool_name=name, tool_input=input)


@runtime_checkable
class DecisionMaker(Protocol):
    """Whatever decides what happens next, given the current state.

    Deliberately a Protocol, matching the Tool contract in app/tools/base.py:
    a decision maker doesn't need to inherit from anything, it just needs
    this one method. The loop never assumes an LLM is involved — tests use
    fully deterministic fakes.
    """

    def decide(self, state: AgentState) -> AgentDecision:
        ...


class AgentLoop:
    """Bounded control-flow loop over a DecisionMaker and a ToolRegistry.

    See the module docstring for the diagram and the exact iteration
    semantics of `AgentState.step`.
    """

    def __init__(
        self,
        decision_maker: DecisionMaker,
        tool_registry: ToolRegistry,
        max_iterations: int = 5,
        correction_policy: CorrectionPolicy | None = None,
        tool_execution_gate: ToolExecutionGate | None = None,
        execution_context: ExecutionContext | None = None,
        event_emitter: EventEmitter | None = None,
        deadline: float | None = None,
    ):
        if max_iterations <= 0:
            raise ValueError("max_iterations must be a positive integer.")

        self.decision_maker = decision_maker
        self.tools = tool_registry
        self.max_iterations = max_iterations
        # Step 17, opt-in, default None — see the module docstring's Step
        # 17 section and app/agent/reliability.py. Never isinstance-checked
        # here, matching this loop's own existing minimalism around
        # decision_maker/tool_registry (neither is isinstance-checked
        # either); a malformed policy would simply raise from within
        # evaluate(), the same way a malformed DecisionMaker already can.
        self.correction_policy = correction_policy
        # Milestone 18, opt-in, default None — see the module docstring's
        # Milestone 18 section and app/agent/tool_execution.py. A gate
        # with no execution_context uses an empty ExecutionContext()
        # (nothing pre-confirmed) at call time — never a context that
        # trusts anything by default.
        self.tool_execution_gate = tool_execution_gate
        self.execution_context = execution_context
        # Milestone 19, opt-in, default None — see the module docstring's
        # Milestone 19 section. Never isinstance-checked, matching every
        # other optional collaborator here.
        self.event_emitter = event_emitter
        # Milestone 23, opt-in, default None — see the module docstring's
        # Milestone 23 section. An absolute `time.monotonic()` cutoff
        # (NOT a duration), computed once by the caller (app/services/
        # chat.py, at the true request boundary — the same place
        # `request_id` is already generated) and checked at the top of
        # every iteration below, alongside `max_iterations`.
        self.deadline = deadline

    def run(self, state: AgentState) -> AgentState:
        """Advance `state` until it reaches a terminal status, mutating and
        returning the same instance."""
        if state.plan is not None and state.plan.status == PlanStatus.PENDING:
            state.plan.start()

        while state.status == AgentStatus.RUNNING:
            if self.deadline is not None and time.monotonic() >= self.deadline:
                # Milestone 23: a cooperative check, deliberately at the
                # SAME granularity as max_iterations below — between
                # iterations, never mid-call. This codebase's execution is
                # synchronous (a real Ollama/tool call cannot be safely
                # preempted without dangerous thread-killing), so the
                # guarantee this makes is "no FURTHER iteration starts
                # once the deadline has passed," not "an in-flight call is
                # interrupted." Terminal, exactly like max_iterations
                # exhaustion — never offered to self.correction_policy:
                # "try again" is precisely what a deadline exists to stop.
                logger.error("execution.terminal category=deadline_exceeded step=%s", state.step)
                state.fail("Request processing exceeded the maximum allowed time.")
                self._fail_plan_if_running(state)
                break

            if state.step >= self.max_iterations:
                logger.error("execution.terminal category=max_iterations step=%s", state.step)
                state.fail(f"Maximum iterations ({self.max_iterations}) reached without a final answer.")
                self._fail_plan_if_running(state)
                break

            state.step += 1
            self._ensure_current_plan_step_running(state)
            if self.event_emitter is not None:
                self.event_emitter.emit(EventType.STEP_STARTED, step=state.step)

            try:
                decision = self.decision_maker.decide(state)
            except DecisionMakerError as exc:
                if exc.category is not None and self._apply_correction_or_fail(
                    state, Failure(category=exc.category)
                ):
                    continue
                state.fail(str(exc))
                self._fail_plan_if_running(state)
                break

            if decision.action_type is ActionType.FINAL:
                if state.plan is not None and self._plan_blocks_final(state.plan):
                    # More than just the current step is still unaddressed —
                    # the model answered before finishing required plan work.
                    # No new AgentDecision field exists to say "this step is
                    # done" (see the module docstring). Step 17: this is now
                    # ELIGIBLE for correction (category PLAN_SKIPPED) — with
                    # no policy injected, or on a TERMINATE verdict, this
                    # still fails deterministically exactly as before.
                    failure = Failure(category=FailureCategory.PLAN_SKIPPED)
                    if self._apply_correction_or_fail(state, failure):
                        continue
                    state.fail("Model returned a final answer while plan steps were still pending.")
                    self._fail_plan_if_running(state)
                    break
                if state.plan is not None:
                    # FINAL is allowed to finish off the single remaining
                    # step itself (the common case: a one-step plan needing
                    # no tool at all, or the last step of a multi-step plan
                    # once earlier steps are done) — see _plan_blocks_final.
                    self._complete_current_plan_step(state)
                self._log_recovery_if_applicable(state)
                state.complete(decision.final_answer)  # type: ignore[arg-type]
                break

            self._execute_tool_action(state, decision)

        return state

    def _apply_correction_or_fail(self, state: AgentState, failure: Failure) -> bool:
        """Consult `self.correction_policy` (if any) for `failure`.

        Returns `True` when a correction was recorded and the caller
        should CONTINUE to the next ordinary iteration instead of failing
        the state. Returns `False` when the caller must fail the state
        itself — with whatever message it already uses today — EXACTLY as
        it would with no policy at all: the no-policy-injected case is
        checked FIRST and returns `False` before constructing anything
        else, so `correction_policy=None` (the default) makes every call
        site's behavior byte-for-byte identical to before Step 17.

        This method never constructs or logs a termination message itself
        — that stays the call site's job, unchanged from before Step 17 —
        so this method's only responsibility is "should we continue, or
        not," and it never has to decide what is safe to put in a message
        (some call sites' messages contain model-derived text, e.g. an
        invented tool name; this method never touches that text).
        """
        if self.correction_policy is None:
            return False

        verdict = self.correction_policy.evaluate(state, failure)
        if verdict.action is not CorrectionAction.CORRECT:
            logger.warning(
                "correction.declined category=%s step=%s reason=%s",
                failure.category.value,
                state.step,
                verdict.reason or "policy_declined",
            )
            if self.event_emitter is not None:
                self.event_emitter.emit(
                    EventType.CORRECTION_DECLINED, step=state.step, failure_category=failure.category
                )
            return False

        state.record_correction(
            CorrectionNote(
                category=failure.category,
                safe_message=verdict.safe_message,
                step=state.step,
                signature=verdict.signature,
            )
        )
        logger.info(
            "correction.triggered category=%s step=%s attempt=%s",
            failure.category.value,
            state.step,
            len(state.corrections),
        )
        if self.event_emitter is not None:
            self.event_emitter.emit(EventType.CORRECTION_TRIGGERED, step=state.step, failure_category=failure.category)
        return True

    def _log_recovery_if_applicable(self, state: AgentState) -> None:
        """If the immediately PRECEDING iteration recorded a correction and
        THIS iteration is succeeding (a FINAL completing, or a tool call
        succeeding), log that the correction worked. Purely observational
        — records nothing on `state` and changes no control flow; a no-op
        whenever `state.corrections` is empty or the most recent entry
        does not belong to the previous step.
        """
        if not state.corrections:
            return
        last = state.corrections[-1]
        if last.step == state.step - 1:
            logger.info(
                "correction.succeeded category=%s attempts_used=%s step=%s",
                last.category.value,
                len(state.corrections),
                state.step,
            )

    def _ensure_current_plan_step_running(self, state: AgentState) -> None:
        if state.plan is None:
            return
        current = state.plan.current_step()
        if current is not None and current.status == PlanStatus.PENDING:
            state.plan.start_step(current.step_id)

    def _fail_plan_if_running(self, state: AgentState) -> None:
        """Fail the plan on a hard execution failure. If the current step
        was already RUNNING (it started this iteration, then the decision
        or its tool action failed), fail that step too via Plan.fail_step()
        — reusing its existing cascade to plan-level FAILED — rather than
        leaving a step that actually failed sitting at RUNNING forever
        ("do not silently skip failed steps"). If there is no RUNNING step
        (e.g. max_iterations was hit before this cycle's step could start),
        fail the plan directly instead: a step that never ran should not be
        fabricated as FAILED."""
        if state.plan is None or state.plan.status != PlanStatus.RUNNING:
            return
        current = state.plan.current_step()
        if current is not None and current.status == PlanStatus.RUNNING:
            state.plan.fail_step(current.step_id)
        else:
            state.plan.fail()

    def _plan_blocks_final(self, plan: Plan) -> bool:
        """True only when MORE than just the current step still isn't
        COMPLETED — i.e. FINAL would skip real future work, not just finish
        the one step already in front of the model. This is what lets a
        one-step plan (or the last step of any plan) complete naturally via
        a direct FINAL answer with no tool call at all (Part 3/8's "simple
        request" rule), while still catching the case a FINAL would skip
        steps 2, 3, ... that were never even attempted."""
        remaining = [step for step in plan.steps if step.status != PlanStatus.COMPLETED]
        return len(remaining) > 1

    def _complete_current_plan_step(self, state: AgentState) -> None:
        """Mark the plan's current step COMPLETED, and the plan itself
        COMPLETED if that was the last one. Called from two places: after a
        successful tool action (the smallest explicit signal this
        integration uses to know a tool action satisfied the current step —
        see the module docstring for the tradeoff), and when a FINAL
        decision legitimately finishes the plan's single remaining step
        with no tool call at all. No-op if there is no plan or its current
        step is already None (nothing left to complete)."""
        if state.plan is None:
            return
        current = state.plan.current_step()
        if current is None:
            return
        state.plan.complete_step(current.step_id)
        if state.plan.current_step() is None:
            state.plan.complete()

    def _resolve_and_run_tool(self, tool_name: str, tool_input: str | None):
        """Resolve and run one tool — through `self.tool_execution_gate`
        if one is configured, or directly (the exact pre-Milestone-18
        behavior) otherwise.

        Raises `ToolNotFoundError`, `PermissionDeniedError`,
        `ConfirmationRequiredError`, or `ValueError` — see
        `ToolExecutionGate.execute`'s contract (app/agent/tool_execution.py).
        With no gate configured, only `ToolNotFoundError` and `ValueError`
        are reachable, identical to every `AgentLoop` before Milestone 18;
        `PermissionDeniedError`/`ConfirmationRequiredError` structurally
        cannot occur, since nothing exists to raise them.

        When a gate IS configured, `self.tools` (this loop's own registry
        reference) is NOT consulted here at all — the gate's own
        `ToolRegistry` reference is authoritative instead. A caller should
        always construct the gate against the SAME `ToolRegistry` instance
        passed to this loop; nothing enforces that (there is no code path
        that could silently reconcile two different registries), so a
        mismatched pair is a caller bug, not a security issue — either
        registry is independently authoritative for its own set of tools,
        and only the gate's is ever reachable once one exists.
        """
        if self.tool_execution_gate is not None:
            context = self.execution_context if self.execution_context is not None else ExecutionContext()
            return self.tool_execution_gate.execute(tool_name, tool_input, context)

        tool = self.tools.get(tool_name)
        return tool.execute(tool_input)

    def _execute_tool_action(self, state: AgentState, decision: AgentDecision) -> None:
        """Run one TOOL decision: resolve/authorize/execute it, and record
        what happened. This method has no knowledge of what any specific
        tool does, and (Milestone 18) no knowledge of HOW a tool is
        authorized — that is `self.tool_execution_gate`'s job, consulted
        via `_resolve_and_run_tool` below."""
        state.record_tool_call(decision.tool_name, decision.tool_input)  # type: ignore[arg-type]
        if self.event_emitter is not None:
            self.event_emitter.emit(EventType.TOOL_PROPOSED, step=state.step, tool_name=decision.tool_name)

        # Milestone 19: measured around the WHOLE resolve->execute call, so
        # a TOOL_EXECUTION_FAILED (an operational failure that still ran
        # tool.execute(), e.g. a real network error) is timed too, not just
        # the success path. Deliberately NOT used for the exception
        # branches below (DENY/CONFIRM/rejected input) — see the module
        # docstring's Milestone 19 section on why authorization/validation
        # are not timed.
        tool_call_start = monotonic_start()
        try:
            result = self._resolve_and_run_tool(decision.tool_name, decision.tool_input)  # type: ignore[arg-type]
        except ToolNotFoundError as exc:
            # No ToolResult was ever produced, so there is nothing to record as
            # an Observation — record the error, then consult the policy
            # (Step 17, category UNKNOWN_TOOL — the non-LLM-path variant;
            # LLMDecisionMaker validates this before an AgentDecision even
            # exists, so it never reaches here in production, but any other
            # DecisionMaker that skips that check lands here). tool_name is
            # NEVER attached to this Failure — it is the model's own
            # invented, unregistered name, i.e. untrusted text (see
            # reliability.py's module docstring).
            failure = Failure(category=FailureCategory.UNKNOWN_TOOL)
            if self._apply_correction_or_fail(state, failure):
                return
            state.fail(str(exc))
            self._fail_plan_if_running(state)
            return
        except PermissionDeniedError as exc:
            # Milestone 18: PERMISSION_DENIED is NEVER offered to
            # self.correction_policy — no self._apply_correction_or_fail
            # call here at all, unlike every other failure site in this
            # method. See FailureCategory.PERMISSION_DENIED's docstring
            # (app/agent/reliability.py) for why this must be a
            # structural guarantee, not a policy default.
            if self.event_emitter is not None:
                self.event_emitter.emit(
                    EventType.TOOL_DENIED,
                    step=state.step,
                    tool_name=decision.tool_name,
                    permission_decision=PermissionDecision.DENY,
                )
            state.fail(str(exc))
            self._fail_plan_if_running(state)
            return
        except ConfirmationRequiredError as exc:
            # Milestone 18: identical treatment and identical reasoning —
            # see FailureCategory.CONFIRMATION_REQUIRED's docstring.
            if self.event_emitter is not None:
                self.event_emitter.emit(
                    EventType.TOOL_CONFIRMATION_REQUIRED,
                    step=state.step,
                    tool_name=decision.tool_name,
                    permission_decision=PermissionDecision.CONFIRM,
                )
            state.fail(str(exc))
            self._fail_plan_if_running(state)
            return
        except ValueError as exc:
            # Invalid input to the tool is a precondition failure (see
            # app.tools.base for the tool-level error contract) — again, no
            # ToolResult was produced. Any other exception type is NOT caught
            # here and propagates normally; the loop never swallows
            # unexpected programming errors. Step 17: category
            # INVALID_TOOL_INPUT — tool_name here IS a validated registry
            # key (resolution inside _resolve_and_run_tool already
            # succeeded), so it is safe to carry on the Failure; the
            # tool's own ValueError text is carried as `detail`, which
            # BudgetedCorrectionPolicy renders into feedback ONLY when
            # include_tool_error_text=True (default False — see
            # reliability.py).
            failure = Failure(
                category=FailureCategory.INVALID_TOOL_INPUT,
                tool_name=decision.tool_name,  # type: ignore[arg-type]
                detail=str(exc),
            )
            if self.event_emitter is not None:
                self.event_emitter.emit(
                    EventType.TOOL_INPUT_REJECTED, step=state.step, tool_name=decision.tool_name, success=False
                )
            if self._apply_correction_or_fail(state, failure):
                return
            state.fail(str(exc))
            self._fail_plan_if_running(state)
            return

        # Reached only when _resolve_and_run_tool returned a ToolResult
        # without raising — i.e. resolution, authorization, validation,
        # and confirmation (when a gate is configured) all already
        # cleared, and tool.execute() already ran. TOOL_AUTHORIZED is
        # therefore a statement of fact about what already happened, not
        # a decision made here.
        if self.event_emitter is not None:
            if self.tool_execution_gate is not None:
                self.event_emitter.emit(
                    EventType.TOOL_AUTHORIZED,
                    step=state.step,
                    tool_name=decision.tool_name,
                    permission_decision=PermissionDecision.ALLOW,
                )
            self.event_emitter.emit(
                EventType.TOOL_EXECUTION_COMPLETED,
                step=state.step,
                tool_name=decision.tool_name,
                success=result.success,
                duration_ms=elapsed_ms(tool_call_start),
            )

        state.add_observation(
            decision.tool_name,  # type: ignore[arg-type]
            success=result.success,
            data=result.data,
            error=result.error,
        )

        if not result.success:
            # Step 17: category TOOL_EXECUTION_FAILED — tool_name is a
            # validated registry key (see the INVALID_TOOL_INPUT note
            # above); `detail` carries the tool's own error text under the
            # identical include_tool_error_text gate.
            error_message = result.error or f"Tool '{decision.tool_name}' reported failure without a message."
            failure = Failure(
                category=FailureCategory.TOOL_EXECUTION_FAILED,
                tool_name=decision.tool_name,  # type: ignore[arg-type]
                detail=result.error,
            )
            if self._apply_correction_or_fail(state, failure):
                return
            state.fail(error_message)
            self._fail_plan_if_running(state)
            return

        self._log_recovery_if_applicable(state)
        self._complete_current_plan_step(state)
