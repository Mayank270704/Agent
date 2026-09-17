"""Milestone 23, item 3: a cooperative, global per-request deadline.

Individual Ollama calls already have their own client-side timeout
(app/models/llm.py, unchanged by this milestone — verified below). This
file proves the SEPARATE, new guarantee: `AgentLoop` stops starting new
iterations once an externally-supplied absolute deadline has passed,
without bypassing security (ToolExecutionGate) or correction-budget
enforcement, and without altering the public response schema.

Fully offline: no Ollama, no network. A few tests use short REAL sleeps
(tens of milliseconds) to deterministically advance wall-clock time past
a short deadline — this is a monotonic-clock comparison, not a retry
loop, so it is not flaky.
"""
from __future__ import annotations

import inspect
import json
import time

import pytest

import app.main as main_module
from app.agent.loop import ActionType, AgentDecision, AgentLoop
from app.agent.orchestrator import AgentOrchestrator
from app.agent.permissions import AllowlistPermissionPolicy, ExecutionContext
from app.agent.reliability import BudgetedCorrectionPolicy
from app.agent.state import AgentState, AgentStatus
from app.agent.tool_execution import ToolExecutionGate
from app.agent.tool_registry import ToolRegistry
from app.config import Settings
from app.tools.base import ToolResult


class FakeLLM:
    def __init__(self, responses: list[str]):
        self.responses = iter(responses)
        self.calls = 0

    def generate(self, messages, *, json_mode: bool = False) -> str:
        self.calls += 1
        try:
            return next(self.responses)
        except StopIteration:
            raise AssertionError("FakeLLM ran out of scripted responses") from None


class RecordingTool:
    def __init__(self, name: str, *, requires_confirmation: bool = False):
        self.name = name
        self.description = f"tool '{name}'"
        self.input_schema: dict[str, str] = {}
        self.requires_confirmation = requires_confirmation
        self.execute_count = 0

    def execute(self, input: str | None = None) -> ToolResult:
        self.execute_count += 1
        return ToolResult.ok("done")


class AlwaysToolDecisionMaker:
    """Never returns FINAL — the ONLY way this loop can terminate is via
    max_iterations or the deadline. `delay_seconds` lets a test control
    real wall-clock pacing precisely."""

    def __init__(self, tool_name: str, *, delay_seconds: float = 0.0):
        self.tool_name = tool_name
        self.delay_seconds = delay_seconds
        self.calls = 0

    def decide(self, state: AgentState) -> AgentDecision:
        self.calls += 1
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        return AgentDecision.tool(self.tool_name, None)


def _registry(*tools) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


# ===========================================================================
# Deadline already passed -> zero further work, deterministic
# ===========================================================================

def test_a_deadline_already_in_the_past_stops_the_loop_before_any_decision() -> None:
    decision_maker = AlwaysToolDecisionMaker("time")
    loop = AgentLoop(
        decision_maker=decision_maker,
        tool_registry=ToolRegistry(),
        max_iterations=10,
        deadline=time.monotonic() - 1.0,  # already expired
    )
    state = AgentState(user_input="hello")

    loop.run(state)

    assert state.status is AgentStatus.FAILED
    assert decision_maker.calls == 0  # no further work was ever attempted
    assert "exceeded the maximum allowed time" in state.errors[-1].message


def test_deadline_exceeded_behavior_is_deterministic_across_repeated_runs() -> None:
    def run_once() -> AgentStatus:
        decision_maker = AlwaysToolDecisionMaker("time")
        loop = AgentLoop(
            decision_maker=decision_maker, tool_registry=ToolRegistry(), deadline=time.monotonic() - 1.0
        )
        state = AgentState(user_input="hello")
        loop.run(state)
        return state.status

    assert run_once() is AgentStatus.FAILED
    assert run_once() is AgentStatus.FAILED


# ===========================================================================
# Deadline expiring MID-run -> no further iteration starts after it passes
# ===========================================================================

def test_no_additional_iteration_occurs_once_the_deadline_passes_mid_run() -> None:
    tool = RecordingTool("time")
    registry = _registry(tool)
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy({"time"}))
    decision_maker = AlwaysToolDecisionMaker("time", delay_seconds=0.05)
    loop = AgentLoop(
        decision_maker=decision_maker,
        tool_registry=registry,
        tool_execution_gate=gate,
        max_iterations=100,  # deliberately high -- the deadline must win, not this
        deadline=time.monotonic() + 0.12,  # expires after roughly 2-3 iterations
    )
    state = AgentState(user_input="hello")

    loop.run(state)

    assert state.status is AgentStatus.FAILED
    assert "exceeded the maximum allowed time" in state.errors[-1].message
    # It stopped well short of max_iterations=100 -- the deadline, not the
    # iteration cap, is what terminated this run.
    assert decision_maker.calls < 10


def test_max_iterations_still_wins_when_it_is_reached_before_the_deadline() -> None:
    """The two bounds are independent -- whichever is hit first governs,
    and max_iterations exhaustion must still report its OWN message."""
    decision_maker = AlwaysToolDecisionMaker("time")
    tool = RecordingTool("time")
    registry = _registry(tool)
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy({"time"}))
    loop = AgentLoop(
        decision_maker=decision_maker,
        tool_registry=registry,
        tool_execution_gate=gate,
        max_iterations=2,
        deadline=time.monotonic() + 60,  # generous; must not fire first
    )
    state = AgentState(user_input="hello")

    loop.run(state)

    assert state.status is AgentStatus.FAILED
    assert "Maximum iterations" in state.errors[-1].message


# ===========================================================================
# Normal requests are unaffected
# ===========================================================================

def test_a_normal_fast_request_completes_unaffected_by_a_generous_deadline() -> None:
    llm = FakeLLM([json.dumps({"action_type": "final", "final_answer": "hello there"})])
    decision_maker = None
    from app.agent.decision_maker import LLMDecisionMaker

    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=ToolRegistry())
    loop = AgentLoop(decision_maker=decision_maker, tool_registry=ToolRegistry(), deadline=time.monotonic() + 60)
    state = AgentState(user_input="hi")

    loop.run(state)

    assert state.status is AgentStatus.COMPLETED
    assert state.final_answer == "hello there"


def test_no_deadline_configured_is_byte_identical_to_before_this_milestone() -> None:
    llm = FakeLLM([json.dumps({"action_type": "final", "final_answer": "ok"})])
    from app.agent.decision_maker import LLMDecisionMaker

    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=ToolRegistry())
    loop = AgentLoop(decision_maker=decision_maker, tool_registry=ToolRegistry())  # deadline left at default: None

    state = AgentState(user_input="hi")
    loop.run(state)

    assert loop.deadline is None
    assert state.status is AgentStatus.COMPLETED


# ===========================================================================
# Security remains enforced with a deadline configured
# ===========================================================================

def test_permission_denial_remains_enforced_and_terminal_with_a_deadline_set() -> None:
    tool = RecordingTool("delete_file")
    registry = _registry(tool)
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy(frozenset()))  # never allowed
    decision_maker = AlwaysToolDecisionMaker("delete_file")
    loop = AgentLoop(
        decision_maker=decision_maker,
        tool_registry=registry,
        tool_execution_gate=gate,
        deadline=time.monotonic() + 60,
        correction_policy=BudgetedCorrectionPolicy(max_corrections=5),
    )
    state = AgentState(user_input="delete it")

    loop.run(state)

    assert state.status is AgentStatus.FAILED
    assert tool.execute_count == 0
    assert state.step == 1  # denial terminated it immediately, not the deadline


def test_confirmation_requirement_remains_enforced_with_a_deadline_set() -> None:
    tool = RecordingTool("wipe", requires_confirmation=True)
    registry = _registry(tool)
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy({"wipe"}))
    decision_maker = AlwaysToolDecisionMaker("wipe")
    loop = AgentLoop(
        decision_maker=decision_maker,
        tool_registry=registry,
        tool_execution_gate=gate,
        execution_context=ExecutionContext(),  # nothing confirmed
        deadline=time.monotonic() + 60,
    )
    state = AgentState(user_input="wipe it")

    loop.run(state)

    assert state.status is AgentStatus.FAILED
    assert tool.execute_count == 0


def test_deadline_exceeded_is_never_offered_to_the_correction_policy() -> None:
    class AlwaysCorrectPolicy:
        def __init__(self) -> None:
            self.calls = 0

        def evaluate(self, state, failure):
            self.calls += 1
            from app.agent.reliability import CorrectionAction, CorrectionVerdict

            return CorrectionVerdict(CorrectionAction.CORRECT, "keep going", signature="s")

    policy = AlwaysCorrectPolicy()
    decision_maker = AlwaysToolDecisionMaker("time")
    loop = AgentLoop(
        decision_maker=decision_maker,
        tool_registry=ToolRegistry(),
        correction_policy=policy,
        deadline=time.monotonic() - 1.0,
    )
    state = AgentState(user_input="hello")

    loop.run(state)

    assert state.status is AgentStatus.FAILED
    assert policy.calls == 0  # never consulted -- deadline is terminal, like max_iterations


# ===========================================================================
# Existing per-call LLM timeout is untouched by this milestone
# ===========================================================================

def test_llm_client_per_call_timeout_is_unchanged() -> None:
    import app.models.llm as llm_module

    source = inspect.getsource(llm_module.LLMClient._generate_with_ollama)
    assert "timeout=60" in source


# ===========================================================================
# Configuration
# ===========================================================================

def test_request_timeout_seconds_has_a_sensible_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REQUEST_TIMEOUT_SECONDS", raising=False)
    assert Settings().request_timeout_seconds == 120


def test_request_timeout_seconds_env_var_is_respected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REQUEST_TIMEOUT_SECONDS", "30")
    assert Settings().request_timeout_seconds == 30


# ===========================================================================
# Production wiring
# ===========================================================================

def test_production_agent_loop_always_receives_a_real_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.agent import loop as loop_module

    captured: list[object] = []
    original_init = loop_module.AgentLoop.__init__

    def spy_init(self, *args, **kwargs):
        captured.append(kwargs.get("deadline"))
        return original_init(self, *args, **kwargs)

    monkeypatch.setattr(loop_module.AgentLoop, "__init__", spy_init)

    llm = FakeLLM([json.dumps({"action_type": "final", "final_answer": "hi"})])
    monkeypatch.setattr(main_module.chat_service, "llm", llm)
    main_module.chat_service.ask("hello")

    assert len(captured) == 1
    assert captured[0] is not None
    assert isinstance(captured[0], float)


def test_a_request_that_exceeds_the_configured_timeout_fails_gracefully_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end through the real production ChatService/AgentOrchestrator,
    with an already-expired configured timeout, proving the public
    response contract (a graceful string answer, never an exception) is
    preserved. `Settings` is a frozen dataclass, so the module-level
    `settings` binding in app.services.chat is REPLACED with a fresh
    instance built from a patched env var — never mutated in place."""
    import app.services.chat as chat_module
    from app.services.chat import ChatService

    monkeypatch.setenv("REQUEST_TIMEOUT_SECONDS", "0")  # expired by the time it's read
    monkeypatch.setattr(chat_module, "settings", Settings())

    llm = FakeLLM([])  # must never even be called
    service = ChatService(
        llm_client=llm,
        tool_registry=main_module._tool_registry,
        tool_execution_gate=main_module._tool_execution_gate,
    )

    answer = service.ask("hello")

    assert isinstance(answer, str)
    assert "could not complete this request" in answer
    assert llm.calls == 0
