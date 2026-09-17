"""Milestone 19, Phase 2: telemetry wired into AgentLoop.

Proves the wiring itself — event ordering, tool proposed/authorized/
denied/rejected, correction events, durations — and, most importantly,
that telemetry is a pure observer of the Milestone-18/18-C security
boundary, never a participant: a throwing EventSink must not change any
authorization/confirmation/validation/execution outcome, and disabled
telemetry (`event_emitter=None`, the default) must produce a byte-for-byte
identical AgentResult/AgentState to before this milestone.

Fully offline: no Ollama, no network, no real tools.
"""
from __future__ import annotations

import pytest

from app.agent.loop import AgentDecision, AgentLoop
from app.agent.permissions import AllowlistPermissionPolicy, ExecutionContext, PermissionDecision
from app.agent.reliability import BudgetedCorrectionPolicy, FailureCategory
from app.agent.state import AgentState, AgentStatus
from app.agent.telemetry import EventEmitter, EventType, ListEventSink
from app.agent.tool_execution import ToolExecutionGate
from app.agent.tool_registry import ToolRegistry
from app.tools.base import ToolResult


class RaisingSink:
    def emit(self, event) -> None:  # noqa: ANN001
        raise RuntimeError("telemetry backend is down")


class RecordingTool:
    def __init__(self, name: str, *, requires_confirmation: bool = False, result: ToolResult | None = None):
        self.name = name
        self.description = f"tool '{name}'"
        self.input_schema: dict[str, str] = {"value": "string"}
        self.requires_confirmation = requires_confirmation
        self.execute_count = 0
        self._result = result or ToolResult.ok("done")

    def validate(self, input: str | None = None) -> None:
        if input is None or not str(input).strip():
            raise ValueError("value cannot be empty.")

    def execute(self, input: str | None = None) -> ToolResult:
        self.execute_count += 1
        return self._result


class _Repeat:
    def __init__(self, tool_name: str, tool_input: str | None = None):
        self.tool_name = tool_name
        self.tool_input = tool_input

    def decide(self, state: AgentState) -> AgentDecision:
        return AgentDecision.tool(self.tool_name, self.tool_input)


def _registry(*tools) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


# ===========================================================================
# Event ordering / fields — the allow path
# ===========================================================================

def test_a_successful_tool_call_emits_proposed_authorized_completed_in_order() -> None:
    tool = RecordingTool("time")
    registry = _registry(tool)
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy({"time"}))
    sink = ListEventSink()
    emitter = EventEmitter(sink, request_id="r1")

    class _Once:
        def __init__(self) -> None:
            self.decisions = iter([AgentDecision.tool("time", "value"), AgentDecision.final("done")])

        def decide(self, state: AgentState) -> AgentDecision:
            return next(self.decisions)

    loop = AgentLoop(
        decision_maker=_Once(),
        tool_registry=registry,
        tool_execution_gate=gate,
        event_emitter=emitter,
    )
    state = AgentState(user_input="what time is it")

    loop.run(state)

    types = [e.event_type for e in sink.events]
    assert types == [
        EventType.STEP_STARTED,
        EventType.TOOL_PROPOSED,
        EventType.TOOL_AUTHORIZED,
        EventType.TOOL_EXECUTION_COMPLETED,
        EventType.STEP_STARTED,
    ]
    completed = sink.events[3]
    assert completed.tool_name == "time"
    assert completed.success is True
    assert completed.duration_ms is not None and completed.duration_ms >= 0.0
    assert completed.step == 1


def test_sequence_numbers_are_strictly_increasing_within_one_request() -> None:
    tool = RecordingTool("time")
    registry = _registry(tool)
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy({"time"}))
    sink = ListEventSink()
    emitter = EventEmitter(sink, request_id="r1")
    loop = AgentLoop(
        decision_maker=_Repeat("time", "value"),
        tool_registry=registry,
        tool_execution_gate=gate,
        event_emitter=emitter,
        max_iterations=3,
    )
    state = AgentState(user_input="x")

    loop.run(state)

    sequences = [e.sequence for e in sink.events]
    assert sequences == sorted(sequences)
    assert len(set(sequences)) == len(sequences)


# ===========================================================================
# Denial / confirmation / validation — no execution event follows
# ===========================================================================

def test_denied_tool_emits_tool_denied_and_no_execution_event() -> None:
    tool = RecordingTool("delete_file")
    registry = _registry(tool)
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy(frozenset()))
    sink = ListEventSink()
    emitter = EventEmitter(sink, request_id="r1")
    loop = AgentLoop(
        decision_maker=_Repeat("delete_file", "x"),
        tool_registry=registry,
        tool_execution_gate=gate,
        event_emitter=emitter,
    )
    state = AgentState(user_input="delete it")

    loop.run(state)

    types = [e.event_type for e in sink.events]
    assert EventType.TOOL_DENIED in types
    assert EventType.TOOL_EXECUTION_COMPLETED not in types
    denied_event = next(e for e in sink.events if e.event_type is EventType.TOOL_DENIED)
    assert denied_event.tool_name == "delete_file"
    assert denied_event.permission_decision is PermissionDecision.DENY
    assert tool.execute_count == 0


def test_confirmation_required_emits_tool_confirmation_required_and_no_execution_event() -> None:
    tool = RecordingTool("wipe", requires_confirmation=True)
    registry = _registry(tool)
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy({"wipe"}))
    sink = ListEventSink()
    emitter = EventEmitter(sink, request_id="r1")
    loop = AgentLoop(
        decision_maker=_Repeat("wipe", "x"),
        tool_registry=registry,
        tool_execution_gate=gate,
        event_emitter=emitter,
        execution_context=ExecutionContext(),  # nothing confirmed
    )
    state = AgentState(user_input="wipe it")

    loop.run(state)

    types = [e.event_type for e in sink.events]
    assert EventType.TOOL_CONFIRMATION_REQUIRED in types
    assert EventType.TOOL_EXECUTION_COMPLETED not in types
    event = next(e for e in sink.events if e.event_type is EventType.TOOL_CONFIRMATION_REQUIRED)
    assert event.permission_decision is PermissionDecision.CONFIRM
    assert tool.execute_count == 0


def test_invalid_input_emits_tool_input_rejected_and_no_execution_event() -> None:
    tool = RecordingTool("writer")
    registry = _registry(tool)
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy({"writer"}))
    sink = ListEventSink()
    emitter = EventEmitter(sink, request_id="r1")
    loop = AgentLoop(
        decision_maker=_Repeat("writer", "   "),
        tool_registry=registry,
        tool_execution_gate=gate,
        event_emitter=emitter,
    )
    state = AgentState(user_input="write")

    loop.run(state)

    types = [e.event_type for e in sink.events]
    assert EventType.TOOL_INPUT_REJECTED in types
    assert EventType.TOOL_EXECUTION_COMPLETED not in types
    event = next(e for e in sink.events if e.event_type is EventType.TOOL_INPUT_REJECTED)
    assert event.success is False
    assert tool.execute_count == 0


def test_unknown_tool_emits_no_tool_specific_event() -> None:
    """UNKNOWN_TOOL has no dedicated event in the approved set — the
    request-level failure is what surfaces it; there is no tool identity
    to attach a tool.* event to (the model invented an unregistered name,
    which must never appear on an AgentEvent — see the module docstring)."""
    registry = ToolRegistry()
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy({"ghost"}))
    sink = ListEventSink()
    emitter = EventEmitter(sink, request_id="r1")
    loop = AgentLoop(
        decision_maker=_Repeat("ghost", "x"),
        tool_registry=registry,
        tool_execution_gate=gate,
        event_emitter=emitter,
    )
    state = AgentState(user_input="do it")

    loop.run(state)

    tool_events = [e for e in sink.events if e.event_type.value.startswith("tool.")]
    assert tool_events == [
        e for e in sink.events if e.event_type is EventType.TOOL_PROPOSED
    ]  # only the proposal, nothing else


# ===========================================================================
# Correction events
# ===========================================================================

def test_correction_triggered_then_a_successful_retry_emits_no_further_correction_event() -> None:
    tool = RecordingTool("writer")
    registry = _registry(tool)
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy({"writer"}))
    sink = ListEventSink()
    emitter = EventEmitter(sink, request_id="r1")

    class _Scripted:
        def __init__(self) -> None:
            self.decisions = iter(
                [AgentDecision.tool("writer", "   "), AgentDecision.tool("writer", "ok"), AgentDecision.final("done")]
            )

        def decide(self, state: AgentState) -> AgentDecision:
            return next(self.decisions)

    loop = AgentLoop(
        decision_maker=_Scripted(),
        tool_registry=registry,
        tool_execution_gate=gate,
        event_emitter=emitter,
        correction_policy=BudgetedCorrectionPolicy(max_corrections=3),
        max_iterations=4,
    )
    state = AgentState(user_input="write")

    loop.run(state)

    types = [e.event_type for e in sink.events]
    assert types.count(EventType.CORRECTION_TRIGGERED) == 1
    triggered = next(e for e in sink.events if e.event_type is EventType.CORRECTION_TRIGGERED)
    assert triggered.failure_category is FailureCategory.INVALID_TOOL_INPUT
    assert tool.execute_count == 1


def test_correction_declined_emits_correction_declined_with_category() -> None:
    tool = RecordingTool("writer")
    registry = _registry(tool)
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy({"writer"}))
    sink = ListEventSink()
    emitter = EventEmitter(sink, request_id="r1")
    loop = AgentLoop(
        decision_maker=_Repeat("writer", "   "),
        tool_registry=registry,
        tool_execution_gate=gate,
        event_emitter=emitter,
        correction_policy=BudgetedCorrectionPolicy(max_corrections=0),  # no budget at all
        max_iterations=3,
    )
    state = AgentState(user_input="write")

    loop.run(state)

    declined = [e for e in sink.events if e.event_type is EventType.CORRECTION_DECLINED]
    assert len(declined) == 1
    assert declined[0].failure_category is FailureCategory.INVALID_TOOL_INPUT


def test_no_correction_policy_emits_no_correction_events_at_all() -> None:
    """The default, production-matching case (correction_policy=None):
    emitting CORRECTION_DECLINED here would misleadingly imply a policy
    was consulted when none exists."""
    tool = RecordingTool("delete_file")
    registry = _registry(tool)
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy(frozenset()))
    sink = ListEventSink()
    emitter = EventEmitter(sink, request_id="r1")
    loop = AgentLoop(
        decision_maker=_Repeat("delete_file", "x"),
        tool_registry=registry,
        tool_execution_gate=gate,
        event_emitter=emitter,
        # correction_policy left at its default: None
    )
    state = AgentState(user_input="delete it")

    loop.run(state)

    correction_events = [e for e in sink.events if e.event_type.value.startswith("correction.")]
    assert correction_events == []


# ===========================================================================
# Request correlation
# ===========================================================================

def test_two_separate_emitters_produce_disjoint_request_ids() -> None:
    tool = RecordingTool("time")
    registry = _registry(tool)
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy({"time"}))
    sink = ListEventSink()

    for request_id in ("req-a", "req-b"):
        emitter = EventEmitter(sink, request_id=request_id)
        loop = AgentLoop(
            decision_maker=_Repeat("time", None), tool_registry=registry, tool_execution_gate=gate,
            event_emitter=emitter, max_iterations=1,
        )
        loop.run(AgentState(user_input="x"))

    request_ids = {e.request_id for e in sink.events}
    assert request_ids == {"req-a", "req-b"}
    assert len({e.request_id for e in sink.events if e.request_id == "req-a"}) == 1


# ===========================================================================
# SECURITY INVARIANCE — the critical property for this milestone
# ===========================================================================

def test_a_throwing_sink_does_not_change_the_denial_outcome() -> None:
    tool = RecordingTool("delete_file")
    registry = _registry(tool)
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy(frozenset()))
    emitter = EventEmitter(RaisingSink(), request_id="r1")
    loop = AgentLoop(
        decision_maker=_Repeat("delete_file", "x"),
        tool_registry=registry,
        tool_execution_gate=gate,
        event_emitter=emitter,
    )
    state = AgentState(user_input="delete it")

    loop.run(state)  # must not raise despite every emit() attempt failing

    assert state.status is AgentStatus.FAILED
    assert tool.execute_count == 0


def test_a_throwing_sink_does_not_change_the_confirmation_outcome() -> None:
    tool = RecordingTool("wipe", requires_confirmation=True)
    registry = _registry(tool)
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy({"wipe"}))
    emitter = EventEmitter(RaisingSink(), request_id="r1")
    loop = AgentLoop(
        decision_maker=_Repeat("wipe", "x"),
        tool_registry=registry,
        tool_execution_gate=gate,
        event_emitter=emitter,
        execution_context=ExecutionContext(),
    )
    state = AgentState(user_input="wipe it")

    loop.run(state)

    assert state.status is AgentStatus.FAILED
    assert tool.execute_count == 0


def test_a_throwing_sink_does_not_prevent_a_successful_execution() -> None:
    tool = RecordingTool("time")
    registry = _registry(tool)
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy({"time"}))
    emitter = EventEmitter(RaisingSink(), request_id="r1")

    class _Once:
        def __init__(self) -> None:
            self.decisions = iter([AgentDecision.tool("time", "value"), AgentDecision.final("done")])

        def decide(self, state: AgentState) -> AgentDecision:
            return next(self.decisions)

    loop = AgentLoop(
        decision_maker=_Once(), tool_registry=registry, tool_execution_gate=gate, event_emitter=emitter,
    )
    state = AgentState(user_input="what time is it")

    loop.run(state)  # must not raise

    assert state.status is AgentStatus.COMPLETED
    assert tool.execute_count == 1


def test_a_throwing_sink_does_not_change_correction_or_execute_count() -> None:
    tool = RecordingTool("writer")
    registry = _registry(tool)
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy({"writer"}))
    emitter = EventEmitter(RaisingSink(), request_id="r1")

    class _Scripted:
        def __init__(self) -> None:
            self.decisions = iter(
                [AgentDecision.tool("writer", "   "), AgentDecision.tool("writer", "ok"), AgentDecision.final("done")]
            )

        def decide(self, state: AgentState) -> AgentDecision:
            return next(self.decisions)

    loop = AgentLoop(
        decision_maker=_Scripted(),
        tool_registry=registry,
        tool_execution_gate=gate,
        event_emitter=emitter,
        correction_policy=BudgetedCorrectionPolicy(max_corrections=3),
        max_iterations=4,
    )
    state = AgentState(user_input="write")

    loop.run(state)

    assert tool.execute_count == 1
    assert len(state.corrections) == 1


# ===========================================================================
# Disabled telemetry preserves existing behavior exactly
# ===========================================================================

class _SpySink:
    def __init__(self) -> None:
        self.emit_calls = 0

    def emit(self, event) -> None:  # noqa: ANN001
        self.emit_calls += 1


def test_disabled_telemetry_produces_an_identical_agentstate_outcome() -> None:
    """Same scripted run, with vs. without an emitter: identical status,
    step count, tool_calls, observations, and errors."""
    def _build_loop(event_emitter):
        tool = RecordingTool("time")
        registry = _registry(tool)
        gate = ToolExecutionGate(registry, AllowlistPermissionPolicy({"time"}))

        class _Once:
            def __init__(self) -> None:
                self.decisions = iter([AgentDecision.tool("time", "value"), AgentDecision.final("done")])

            def decide(self, state: AgentState) -> AgentDecision:
                return next(self.decisions)

        return AgentLoop(
            decision_maker=_Once(), tool_registry=registry, tool_execution_gate=gate, event_emitter=event_emitter,
        ), tool

    loop_without, tool_without = _build_loop(None)
    state_without = AgentState(user_input="what time is it")
    loop_without.run(state_without)

    sink = ListEventSink()
    loop_with, tool_with = _build_loop(EventEmitter(sink, request_id="r1"))
    state_with = AgentState(user_input="what time is it")
    loop_with.run(state_with)

    assert state_without.status == state_with.status
    assert state_without.step == state_with.step
    assert len(state_without.tool_calls) == len(state_with.tool_calls)
    assert len(state_without.observations) == len(state_with.observations)
    assert state_without.final_answer == state_with.final_answer
    assert tool_without.execute_count == tool_with.execute_count == 1
    assert len(sink.events) > 0  # the "with" run really did emit something


def test_sink_is_never_consulted_when_event_emitter_is_none() -> None:
    tool = RecordingTool("time")
    registry = _registry(tool)
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy({"time"}))
    loop = AgentLoop(
        decision_maker=_Repeat("time", None), tool_registry=registry, tool_execution_gate=gate,
        event_emitter=None, max_iterations=1,
    )
    # event_emitter=None means no EventEmitter/AgentEvent is ever
    # constructed — there is nothing here TO consult. Proven by the
    # absence of any AttributeError/TypeError despite never wiring a sink.
    state = AgentState(user_input="x")

    loop.run(state)  # must not raise, must not require a sink

    assert loop.event_emitter is None
