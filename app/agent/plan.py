"""Minimal planning domain model: Plan -> PlanStep -> step execution/result.

This module is a pure data/state representation, exactly like AgentState
(app/agent/state.py) — it has no knowledge of the LLM, tools, or
orchestration, and never executes or generates anything.

What this is explicitly NOT:
- Not an LLM planner itself. This module only defines the `PlanGenerator`
  Protocol below (the contract "user_input -> Plan") — the actual LLM-backed
  implementation, `LLMPlanGenerator`, lives in app/agent/plan_generator.py,
  exactly as `Tool`'s concrete implementations live outside tools/base.py.
- As of Step 11, AgentLoop DOES read `AgentState.plan` when present (see
  app/agent/loop.py) — but only to advance/fail the plan's own steps using
  the methods defined here; nothing about how a plan gets *generated* or
  *decided upon* lives in this module. AgentOrchestrator only attaches a
  plan when explicitly given a PlanGenerator (opt-in, not the default) —
  see app/agent/orchestrator.py.

Design note on why PlanStep is frozen but Plan is mutable: this mirrors
AgentState's own pattern exactly (ToolCall/Observation/ExecutionError are
frozen records; AgentState holds mutable lists of them and appends/replaces
rather than mutating a record in place). Plan does the same: it holds a
mutable `steps` list and replaces an entry via `dataclasses.replace()`
whenever that step's status changes.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable


class PlanStatus(Enum):
    """Lifecycle status of a Plan or a single PlanStep — same style and
    vocabulary as AgentStatus (app/agent/state.py), with one addition:
    PENDING, for "created but not yet started." AgentState has no PENDING
    equivalent because it starts RUNNING immediately on creation; a Plan is
    explicitly allowed to exist before its execution begins."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class PlanStep:
    """One step of a Plan.

    Deliberately just three fields. A plan REPRESENTATION only needs to say
    what the step is and where it stands — step_id, description, status.
    Tool binding (which tool, what input) is a plan EXECUTION concern for a
    later step, not something this domain model should guess at before
    execution semantics even exist yet (see the module docstring's scope
    boundary). Adding an unused `tool_name`/`result` field now would be
    exactly the "add it because it might be useful someday" the spec asked
    to avoid.
    """

    step_id: int
    description: str
    status: PlanStatus = PlanStatus.PENDING

    def __post_init__(self) -> None:
        if self.step_id < 1:
            raise ValueError(f"step_id must be a positive integer, got {self.step_id}.")
        if self.description is None or not self.description.strip():
            raise ValueError("PlanStep description cannot be blank.")


@dataclass
class Plan:
    """An ordered, mutable sequence of PlanSteps plus overall status.

    Mutable like AgentState, for the same reason: it is meant to be updated
    as execution progresses, not replaced wholesale each time. All mutation
    goes through the explicit methods below — nothing here executes a tool
    or calls an LLM (see the module docstring).

    Step order is simply list order: `steps` is a plain list, so iterating
    it is always deterministic and matches construction order. step_id is
    an identifier for addressing a step (start_step/complete_step/fail_step
    take a step_id, not a list index), not a substitute for that ordering.
    """

    steps: list[PlanStep]
    status: PlanStatus = PlanStatus.PENDING

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError("Plan must contain at least one step.")
        step_ids = [step.step_id for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError(f"Plan step_ids must be unique, got: {step_ids}.")

    def get_step(self, step_id: int) -> PlanStep:
        """Read-only lookup, for inspection/debugging — never mutates."""
        return self._index_of(step_id)[1]

    def current_step(self) -> PlanStep | None:
        """The plan's "current" step: the first step (in list order) that is
        not yet COMPLETED — i.e. PENDING or RUNNING. Returns None once every
        step is COMPLETED.

        Step 11: deliberately derived from step statuses on every call
        rather than tracked as a separate `current_step_index` field, so it
        can never drift out of sync with the steps themselves — "prefer
        using the Plan's existing ordered steps/status values" over adding
        duplicated state that could become inconsistent.
        """
        for step in self.steps:
            if step.status in (PlanStatus.PENDING, PlanStatus.RUNNING):
                return step
        return None

    def _index_of(self, step_id: int) -> tuple[int, PlanStep]:
        for index, step in enumerate(self.steps):
            if step.step_id == step_id:
                return index, step
        raise ValueError(f"No step with step_id={step_id} in this plan.")

    def _replace_step(self, step_id: int, status: PlanStatus) -> None:
        index, step = self._index_of(step_id)
        self.steps[index] = dataclasses.replace(step, status=status)

    def start(self) -> None:
        """Transition the plan itself from PENDING to RUNNING."""
        if self.status != PlanStatus.PENDING:
            raise ValueError(f"Cannot start a plan that is already {self.status.value}.")
        self.status = PlanStatus.RUNNING

    def start_step(self, step_id: int) -> None:
        """Transition one step from PENDING to RUNNING. The plan itself
        must already be RUNNING — call start() first."""
        if self.status != PlanStatus.RUNNING:
            raise ValueError(f"Cannot start a step while the plan is {self.status.value}, not running.")
        step = self.get_step(step_id)
        if step.status != PlanStatus.PENDING:
            raise ValueError(f"Cannot start step {step_id}: it is already {step.status.value}.")
        self._replace_step(step_id, PlanStatus.RUNNING)

    def complete_step(self, step_id: int) -> None:
        """Transition one step from RUNNING to COMPLETED."""
        step = self.get_step(step_id)
        if step.status != PlanStatus.RUNNING:
            raise ValueError(f"Cannot complete step {step_id}: it is {step.status.value}, not running.")
        self._replace_step(step_id, PlanStatus.COMPLETED)

    def fail_step(self, step_id: int) -> None:
        """Transition one step from RUNNING to FAILED, and — because this
        minimal model has no retry or replanning — automatically fail the
        plan itself too: a single failed step means this plan can no longer
        reach a normal all-steps-completed success. (The plan's own status
        is guaranteed RUNNING at this point: a step can only be RUNNING if
        start_step() put it there, which itself required the plan to be
        RUNNING first, so self.fail() below cannot spuriously reject.)
        """
        step = self.get_step(step_id)
        if step.status != PlanStatus.RUNNING:
            raise ValueError(f"Cannot fail step {step_id}: it is {step.status.value}, not running.")
        self._replace_step(step_id, PlanStatus.FAILED)
        self.fail()

    def complete(self) -> None:
        """Transition the plan itself from RUNNING to COMPLETED. Only
        allowed once every step is actually COMPLETED — this is what makes
        "completing all steps" the thing that allows the plan to complete,
        without adding automatic cascade logic on the success path (unlike
        failure, success is a deliberate, explicit call)."""
        if self.status != PlanStatus.RUNNING:
            raise ValueError(f"Cannot complete a plan that is already {self.status.value}.")
        incomplete = [step.step_id for step in self.steps if step.status != PlanStatus.COMPLETED]
        if incomplete:
            raise ValueError(f"Cannot complete the plan: steps not yet completed: {incomplete}.")
        self.status = PlanStatus.COMPLETED

    def fail(self) -> None:
        """Transition the plan itself from RUNNING to FAILED directly (e.g.
        an external abort), independent of any specific step. Also the
        method fail_step() delegates to once it has recorded which step
        caused the failure."""
        if self.status != PlanStatus.RUNNING:
            raise ValueError(f"Cannot fail a plan that is already {self.status.value}.")
        self.status = PlanStatus.FAILED


@runtime_checkable
class PlanGenerator(Protocol):
    """Converts a user request into a validated Plan.

    A Protocol, not an ABC — same reasoning as Tool (app/tools/base.py) and
    DecisionMaker (app/agent/loop.py): an implementation just needs this one
    method, no inheritance required, so test fakes can stay plain classes.

    Contract:
    - `user_input` must be non-empty (blank/whitespace-only is rejected).
    - The implementation is model-agnostic from the caller's perspective —
      nothing here assumes Ollama, JSON mode, or any particular provider.
    - The return value is always a fully-validated Plan (every PlanStep and
      the Plan itself already satisfy plan.py's own invariants) — never a
      partially-formed or guessed one. A generator that cannot produce a
      valid plan raises rather than returning something invalid.
    - Implementations must NOT execute tools, mutate AgentState, call
      AgentLoop, or produce a final answer — see
      app/agent/plan_generator.py's module docstring for the full boundary.
    """

    def generate(self, user_input: str) -> Plan:
        ...
