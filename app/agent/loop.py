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
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from app.agent.plan import Plan, PlanStatus
from app.agent.state import AgentState, AgentStatus
from app.agent.tool_registry import ToolNotFoundError, ToolRegistry


class DecisionMakerError(Exception):
    """Base class for errors a DecisionMaker implementation may raise when it
    cannot produce a usable AgentDecision (e.g. malformed LLM output — see
    DecisionParseError in app/agent/decision_maker.py, which subclasses this).

    The loop treats this as an expected, recoverable-at-this-iteration
    failure: it fails the state and stops, the same way it already handles an
    unregistered tool or invalid tool input. An unreliable decision maker
    should not be able to crash the whole execution with an uncaught
    exception. Any other exception type is NOT caught here and propagates
    normally — a genuine bug in a decision maker is not swallowed."""


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
    ):
        if max_iterations <= 0:
            raise ValueError("max_iterations must be a positive integer.")

        self.decision_maker = decision_maker
        self.tools = tool_registry
        self.max_iterations = max_iterations

    def run(self, state: AgentState) -> AgentState:
        """Advance `state` until it reaches a terminal status, mutating and
        returning the same instance."""
        if state.plan is not None and state.plan.status == PlanStatus.PENDING:
            state.plan.start()

        while state.status == AgentStatus.RUNNING:
            if state.step >= self.max_iterations:
                state.fail(f"Maximum iterations ({self.max_iterations}) reached without a final answer.")
                self._fail_plan_if_running(state)
                break

            state.step += 1
            self._ensure_current_plan_step_running(state)

            try:
                decision = self.decision_maker.decide(state)
            except DecisionMakerError as exc:
                state.fail(str(exc))
                self._fail_plan_if_running(state)
                break

            if decision.action_type is ActionType.FINAL:
                if state.plan is not None and self._plan_blocks_final(state.plan):
                    # More than just the current step is still unaddressed —
                    # the model answered before finishing required plan work.
                    # No new AgentDecision field exists to say "this step is
                    # done" (see the module docstring) — rather than guess
                    # this was intentional, treat it as skipping pending
                    # plan steps and fail deterministically. No retry.
                    state.fail("Model returned a final answer while plan steps were still pending.")
                    self._fail_plan_if_running(state)
                    break
                if state.plan is not None:
                    # FINAL is allowed to finish off the single remaining
                    # step itself (the common case: a one-step plan needing
                    # no tool at all, or the last step of a multi-step plan
                    # once earlier steps are done) — see _plan_blocks_final.
                    self._complete_current_plan_step(state)
                state.complete(decision.final_answer)  # type: ignore[arg-type]
                break

            self._execute_tool_action(state, decision)

        return state

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

    def _execute_tool_action(self, state: AgentState, decision: AgentDecision) -> None:
        """Run one TOOL decision: look the tool up generically, execute it
        generically, and record what happened. This method has no knowledge
        of what any specific tool does."""
        state.record_tool_call(decision.tool_name, decision.tool_input)  # type: ignore[arg-type]

        try:
            tool = self.tools.get(decision.tool_name)  # type: ignore[arg-type]
        except ToolNotFoundError as exc:
            # No ToolResult was ever produced, so there is nothing to record as
            # an Observation — just record the error and stop.
            state.fail(str(exc))
            self._fail_plan_if_running(state)
            return

        try:
            result = tool.execute(decision.tool_input)
        except ValueError as exc:
            # Invalid input to the tool is a precondition failure (see
            # app.tools.base for the tool-level error contract) — again, no
            # ToolResult was produced. Any other exception type is NOT caught
            # here and propagates normally; the loop never swallows
            # unexpected programming errors.
            state.fail(str(exc))
            self._fail_plan_if_running(state)
            return

        state.add_observation(
            decision.tool_name,  # type: ignore[arg-type]
            success=result.success,
            data=result.data,
            error=result.error,
        )

        if not result.success:
            state.fail(result.error or f"Tool '{decision.tool_name}' reported failure without a message.")
            self._fail_plan_if_running(state)
            return

        self._complete_current_plan_step(state)
