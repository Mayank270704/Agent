"""Deterministic FULL-EXECUTION evaluation (Milestone 19, Phase 5).

Extends the existing routing-decision evaluation harness
(app/agent/evaluation.py, which calls `decision_maker.decide()` once and
grades the decision alone — it never runs the loop or executes a tool) to
the whole request: a scripted LLM plus real (or fake) tools are run all
the way through `AgentOrchestrator.process()`, and every metric below is
derived from the resulting `AgentResult` plus the `AgentEvent` stream a
`ListEventSink` collected — never by adding new instrumentation, and
never by altering how the request executed.

    ExecutionCase -> run_case() -> ExecutionResult
                        |
                        v
        AgentOrchestrator(llm_client=<scripted FakeLLM>,
                           tool_registry=<caller-supplied>,
                           event_emitter=EventEmitter(ListEventSink(), ...))
                        |
                        v
              AgentResult  +  tuple[AgentEvent, ...]
                        |
                        v
              ExecutionResult (derived metrics, a verdict, nothing altered)

This module is measurement-only, matching app/agent/evaluation.py's own
module docstring: it does not decide, execute, route, or change anything
about the live agent architecture. `run_case` executes real tools (it is
NOT restricted to decision-grading the way `evaluate_case` is — a full
execution genuinely needs to run the loop to observe corrections,
completion, and latency) but every tool a case exercises is one the
CALLER constructs and passes in via `tool_registry`, so a test using this
module controls exactly what "executing a tool" means and never reaches
a real network or Ollama.

No LLM judge, no external evaluation service: final-answer correctness is
a deterministic substring predicate (`expected_answer_contains`), exactly
as declared out of scope for this milestone.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.agent.orchestrator import AgentOrchestrator
from app.agent.state import AgentStatus
from app.agent.telemetry import EventEmitter, EventType, ListEventSink
from app.agent.tool_registry import ToolRegistry


@dataclass(frozen=True)
class ExecutionCase:
    """One full-execution test case: a user input, a scripted sequence of
    raw LLM responses (one string per `decide()`/planning/extraction call
    the case is expected to trigger, in order), and what a well-behaved
    execution should look like.

    `llm_responses` is deliberately a flat, ordered list of raw strings —
    matching exactly what a `FakeLLM.generate()` fake already returns in
    every other test in this codebase (see tests/test_production_tool_
    authorization.py, tests/test_tool_authority_boundary.py, ...) — not a
    new scripting DSL. Constructing one is the caller's job; this module
    only consumes it.
    """

    name: str
    user_input: str
    llm_responses: list[str]
    expected_completion: bool
    expected_tools: tuple[str, ...] = ()
    forbidden_tools: tuple[str, ...] = ()
    expected_answer_contains: str | None = None
    category: str = ""


@dataclass(frozen=True)
class ExecutionResult:
    """The graded outcome of one `ExecutionCase`, derived entirely from an
    `AgentResult` and the `AgentEvent`s a `ListEventSink` collected during
    that one run — nothing here was computed by re-inspecting internal
    agent state or by re-running anything."""

    case: ExecutionCase
    completed: bool
    answer: str
    proposed_tools: tuple[str, ...]
    executed_tools: tuple[str, ...]
    unnecessary_tools: tuple[str, ...]
    denied_tools: tuple[str, ...]
    correction_count: int
    input_rejected_count: int
    iteration_count: int
    latency_ms: float | None
    plan_step_count: int | None
    answer_matches: bool
    passed: bool
    events: tuple[object, ...] = field(repr=False)  # tuple[AgentEvent, ...]; excluded from repr, not from access


def run_case(case: ExecutionCase, tool_registry: ToolRegistry) -> ExecutionResult:
    """Run one `ExecutionCase` through a real `AgentOrchestrator.process()`
    call, using a scripted fake LLM and the caller-supplied tool registry,
    and grade the result purely from the returned `AgentResult` and the
    collected `AgentEvent` stream.

    A fresh `ListEventSink`/`EventEmitter` is constructed per call — this
    module never reuses one across cases, for the identical reason
    app/services/chat.py never reuses an `ExecutionContext` across
    requests (see app/agent/telemetry.py's module docstring): reusing one
    would let one case's events bleed into another's correlation.
    """
    llm = _ScriptedLLM(case.llm_responses)
    sink = ListEventSink()
    emitter = EventEmitter(sink, request_id=f"eval-{case.name}")

    orchestrator = AgentOrchestrator(llm_client=llm, tool_registry=tool_registry, event_emitter=emitter)
    result = orchestrator.process(case.user_input)
    events = tuple(sink.events)

    return _grade(case, result.answer, result.status, events)


def run_all(cases: list[ExecutionCase], tool_registry: ToolRegistry) -> list[ExecutionResult]:
    return [run_case(case, tool_registry) for case in cases]


def _grade(case: ExecutionCase, answer: str, status: AgentStatus, events: tuple) -> ExecutionResult:
    proposed = tuple(e.tool_name for e in events if e.event_type is EventType.TOOL_PROPOSED)
    executed = tuple(
        e.tool_name for e in events if e.event_type is EventType.TOOL_EXECUTION_COMPLETED and e.success
    )
    denied = tuple(e.tool_name for e in events if e.event_type is EventType.TOOL_DENIED)
    correction_count = sum(1 for e in events if e.event_type is EventType.CORRECTION_TRIGGERED)
    input_rejected_count = sum(1 for e in events if e.event_type is EventType.TOOL_INPUT_REJECTED)
    iteration_count = sum(1 for e in events if e.event_type is EventType.STEP_STARTED)
    latency_ms = next(
        (e.duration_ms for e in events if e.event_type in (EventType.REQUEST_COMPLETED, EventType.REQUEST_FAILED)),
        None,
    )
    plan_step_count = next((e.plan_step_count for e in events if e.event_type is EventType.PLAN_CREATED), None)

    unnecessary = tuple(t for t in executed if t not in case.expected_tools)
    completed = status is AgentStatus.COMPLETED

    answer_matches = case.expected_answer_contains is None or (case.expected_answer_contains in (answer or ""))
    forbidden_hit = any(t in case.forbidden_tools for t in executed)
    expected_tools_satisfied = set(case.expected_tools) <= set(executed) if case.expected_tools else True

    passed = (
        completed == case.expected_completion
        and answer_matches
        and not forbidden_hit
        and (expected_tools_satisfied if case.expected_completion else True)
    )

    return ExecutionResult(
        case=case,
        completed=completed,
        answer=answer,
        proposed_tools=proposed,
        executed_tools=executed,
        unnecessary_tools=unnecessary,
        denied_tools=denied,
        correction_count=correction_count,
        input_rejected_count=input_rejected_count,
        iteration_count=iteration_count,
        latency_ms=latency_ms,
        plan_step_count=plan_step_count,
        answer_matches=answer_matches,
        passed=passed,
        events=events,
    )


@dataclass(frozen=True)
class ExecutionEvaluationSummary:
    """Aggregate metrics across a batch of `ExecutionResult`s. Rates are
    0.0 when their denominator is empty, never NaN — matching
    app/agent/evaluation.py's `EvaluationSummary` convention exactly."""

    total: int
    passed: int
    pass_rate: float
    completion_rate: float
    correction_rate: float  # fraction of cases with at least one correction
    denial_rate: float  # fraction of cases with at least one tool denial
    input_rejection_rate: float  # fraction of cases with at least one rejected input
    avg_iteration_count: float
    avg_latency_ms: float | None  # None only when no result carried a latency


def summarize(results: list[ExecutionResult]) -> ExecutionEvaluationSummary:
    total = len(results)
    if total == 0:
        return ExecutionEvaluationSummary(
            total=0,
            passed=0,
            pass_rate=0.0,
            completion_rate=0.0,
            correction_rate=0.0,
            denial_rate=0.0,
            input_rejection_rate=0.0,
            avg_iteration_count=0.0,
            avg_latency_ms=None,
        )

    latencies = [r.latency_ms for r in results if r.latency_ms is not None]

    return ExecutionEvaluationSummary(
        total=total,
        passed=sum(1 for r in results if r.passed),
        pass_rate=sum(1 for r in results if r.passed) / total,
        completion_rate=sum(1 for r in results if r.completed) / total,
        correction_rate=sum(1 for r in results if r.correction_count > 0) / total,
        denial_rate=sum(1 for r in results if r.denied_tools) / total,
        input_rejection_rate=sum(1 for r in results if r.input_rejected_count > 0) / total,
        avg_iteration_count=sum(r.iteration_count for r in results) / total,
        avg_latency_ms=(sum(latencies) / len(latencies)) if latencies else None,
    )


class _ScriptedLLM:
    """The minimal fake LLM every other test in this codebase already
    hand-rolls (see tests/test_production_tool_authorization.py's
    `FakeLLM`) — defined once here so `run_case` needs no dependency on
    any specific test file."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)

    def generate(self, messages: list[dict[str, str]], *, json_mode: bool = False) -> str:
        try:
            return next(self._responses)
        except StopIteration:
            raise AssertionError(
                "ExecutionCase ran out of scripted llm_responses — the case's script is shorter "
                "than the number of decide() calls the execution actually made."
            ) from None
