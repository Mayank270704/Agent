"""Explicit state for a single agent execution.

This module is a pure data/state container: it has no knowledge of the LLM,
tools, routing, or orchestration, and it never calls out to any of them. It
exists to eventually support a loop shaped like:

    User Input -> State -> Decision -> Action -> Tool -> Observation ->
    State update -> Decision again -> ... -> Final Answer

That loop does not exist yet — app/agent/orchestrator.py still does a single-
shot dispatch and is not wired to this module. AgentState only records what
happened during an execution; it never causes anything to happen.

Each AgentState instance belongs to exactly one execution/task. There is no
shared or global state here — create a fresh instance per request.

Step 9: AgentState gained an optional `plan: Plan | None` field
(app/agent/plan.py). Its semantics are deliberately narrow for now:
- `plan is None` (the default) — nothing changes. This is still purely
  single-step execution; AgentLoop does not read this field at all.
- `plan is not None` — the field simply holds a Plan value for a future
  step to read. Its presence does not, by itself, change how AgentLoop or
  AgentOrchestrator behave. Plan generation and plan-aware execution are
  both explicitly out of scope for Step 9 — see app/agent/plan.py.

Step 16E-C: AgentState gained `memory_context: MemoryContext | None`,
following the exact same pattern as `plan` — a structured, already-prepared
value that the orchestrator attaches once per request and the prompt layer
(LLMDecisionMaker) renders. This does NOT couple AgentState to storage: it
holds no SemanticMemoryStore, VectorIndex, or EmbeddingProvider, only the
small immutable projection those produced (see app/agent/memory_context.py),
exactly as `messages` holds conversation turns rather than a
ConversationMemory. `MemoryContext` is imported under TYPE_CHECKING only:
`memory_context` transitively reaches the embedding layer, and a later
milestone may put a heavyweight real model there — a typing-only import
keeps that permanently off this module's runtime import path.

Step 17: AgentState gained `corrections: list[CorrectionNote]`, following
the exact same "recorded fact, not a decision" pattern as `tool_calls` /
`observations` / `errors`. A CorrectionNote is appended ONLY by AgentLoop,
ONLY when an injected CorrectionPolicy (app/agent/reliability.py) judged a
failure correctable — with no policy injected (the default), this list
stays empty forever and AgentState's behavior is unchanged from before
Step 17. AgentState still only RECORDS what happened; it does not decide
whether a failure is correctable (that is the policy's job) and it does
not retry anything (that is AgentLoop's job, by simply not failing and
continuing to its next ordinary iteration).

`corrections` intentionally does NOT duplicate `correction_attempt` /
`last_failure` as separate fields: those are one-line derivations
(`len(state.corrections)`, `state.corrections[-1]`) and storing them
separately would create two sources of truth for the same fact.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from app.agent.plan import Plan
from app.agent.reliability import FailureCategory

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from app.agent.memory_context import MemoryContext


class AgentStatus(Enum):
    """Lifecycle status of a single agent execution."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class ToolCall:
    """A single recorded attempt to invoke a tool (the "Action" step)."""

    tool_name: str
    tool_input: str | None
    step: int


@dataclass(frozen=True)
class Observation:
    """The recorded result of a single tool invocation."""

    tool_name: str
    success: bool
    data: Any = None
    error: str | None = None
    step: int = 0


@dataclass(frozen=True)
class ExecutionError:
    """A single recorded error encountered during execution."""

    message: str
    step: int


@dataclass(frozen=True)
class CorrectionNote:
    """One recorded self-correction (Step 17): a failure that an injected
    CorrectionPolicy (app/agent/reliability.py) judged correctable, so
    AgentLoop did NOT fail the state and instead continued to another
    ordinary iteration.

    `category` and `safe_message` are exactly what reached (or will reach)
    the model's prompt — `safe_message` is drawn from the policy's fixed,
    application-authored vocabulary and never contains raw model output,
    an exception's raw text, or an invented tool name (see reliability.py's
    module docstring).

    `signature` is an internal fingerprint the POLICY computed for its own
    consecutive-repetition detection. It is stored here only so a policy
    consulted again on a LATER iteration can compare against it without
    AgentLoop needing to know how fingerprinting works — it is never
    rendered into a prompt and never logged verbatim.
    """

    category: FailureCategory
    safe_message: str
    step: int
    signature: str


@dataclass
class AgentState:
    """The state of one agent execution.

    Mutable by design — unlike the terminal, single-shot result types
    elsewhere in this codebase (RouterDecision, ToolResult, AgentResult),
    this object is meant to be updated across multiple steps as an execution
    progresses. It represents facts/results of execution; it does not decide
    or perform anything itself.
    """

    user_input: str
    messages: list[dict[str, str]] = field(default_factory=list)
    step: int = 0
    tool_calls: list[ToolCall] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    errors: list[ExecutionError] = field(default_factory=list)
    final_answer: str | None = None
    status: AgentStatus = AgentStatus.RUNNING
    plan: Plan | None = None
    memory_context: MemoryContext | None = None
    corrections: list[CorrectionNote] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.user_input is None or not str(self.user_input).strip():
            raise ValueError("user_input cannot be empty.")
        if self.step < 0:
            raise ValueError("step cannot be negative.")

    def record_tool_call(self, tool_name: str, tool_input: str | None) -> None:
        """Record that a tool was invoked with the given input, at the current step."""
        self.tool_calls.append(ToolCall(tool_name=tool_name, tool_input=tool_input, step=self.step))

    def add_observation(
        self, tool_name: str, *, success: bool, data: Any = None, error: str | None = None
    ) -> None:
        """Record the result of a tool invocation, at the current step."""
        self.observations.append(
            Observation(tool_name=tool_name, success=success, data=data, error=error, step=self.step)
        )

    def record_error(self, message: str) -> None:
        """Record an error encountered during execution, without changing status."""
        if message is None or not str(message).strip():
            raise ValueError("error message cannot be empty.")
        self.errors.append(ExecutionError(message=message, step=self.step))

    def record_correction(self, note: CorrectionNote) -> None:
        """Record a self-correction (Step 17), without changing status.

        Mirrors `record_error`'s "record without judging" pattern exactly
        — called ONLY by AgentLoop, ONLY after its injected
        CorrectionPolicy already returned a CORRECT verdict. This method
        does not itself decide anything; it is pure bookkeeping.
        """
        self.corrections.append(note)

    def complete(self, final_answer: str) -> None:
        """Mark the execution as successfully finished with a final answer."""
        if self.status != AgentStatus.RUNNING:
            raise ValueError(f"Cannot complete a state that is already {self.status.value}.")
        if final_answer is None or not str(final_answer).strip():
            raise ValueError("final_answer cannot be empty.")
        self.final_answer = final_answer
        self.status = AgentStatus.COMPLETED

    def fail(self, message: str) -> None:
        """Mark the execution as failed, recording the error that caused it."""
        if self.status != AgentStatus.RUNNING:
            raise ValueError(f"Cannot fail a state that is already {self.status.value}.")
        self.record_error(message)
        self.status = AgentStatus.FAILED
