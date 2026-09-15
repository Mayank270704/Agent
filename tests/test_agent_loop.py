from __future__ import annotations

import pytest

from app.agent.loop import ActionType, AgentDecision, AgentLoop, DecisionMaker, DecisionMakerError
from app.agent.plan import Plan, PlanStatus, PlanStep
from app.agent.state import AgentState, AgentStatus
from app.agent.tool_registry import ToolRegistry
from app.tools.base import ToolResult


# ---------------------------------------------------------------------------
# Fakes — fully deterministic, no LLM/network involved anywhere in this file.
# ---------------------------------------------------------------------------

class ScriptedDecisionMaker:
    """Returns a fixed, pre-scripted sequence of decisions, one per call."""

    def __init__(self, decisions: list[AgentDecision]):
        self._decisions = iter(decisions)
        self.calls = 0

    def decide(self, state: AgentState) -> AgentDecision:
        self.calls += 1
        try:
            return next(self._decisions)
        except StopIteration:
            raise AssertionError("ScriptedDecisionMaker ran out of scripted decisions") from None


class AlwaysToolDecisionMaker:
    """Always requests the same tool action — used to prove the loop is bounded."""

    def __init__(self, tool_name: str, tool_input: str | None = None):
        self.tool_name = tool_name
        self.tool_input = tool_input
        self.calls = 0

    def decide(self, state: AgentState) -> AgentDecision:
        self.calls += 1
        return AgentDecision.tool(self.tool_name, self.tool_input)


class FailingDecisionMaker:
    """Raises a given exception every time decide() is called."""

    def __init__(self, exc: Exception):
        self._exc = exc
        self.calls = 0

    def decide(self, state: AgentState) -> AgentDecision:
        self.calls += 1
        raise self._exc


class FakeTool:
    name_default = "fake_tool"

    def __init__(
        self,
        name: str = "fake_tool",
        *,
        results: list[ToolResult] | None = None,
        raise_value_error: str | None = None,
    ):
        self.name = name
        self.description = f"Fake tool '{name}' for loop tests."
        self.calls: list[str | None] = []
        self._results = iter(results) if results is not None else None
        self._raise_value_error = raise_value_error

    def execute(self, input: str | None = None) -> ToolResult:
        self.calls.append(input)
        if self._raise_value_error is not None:
            raise ValueError(self._raise_value_error)
        if self._results is not None:
            return next(self._results)
        return ToolResult.ok(f"handled: {input}")


# ---------------------------------------------------------------------------
# 1. A final-answer decision completes the state.
# ---------------------------------------------------------------------------

def test_final_answer_decision_completes_the_state() -> None:
    registry = ToolRegistry()
    decision_maker = ScriptedDecisionMaker([AgentDecision.final("done")])
    loop = AgentLoop(decision_maker=decision_maker, tool_registry=registry, max_iterations=5)
    state = AgentState(user_input="hello")

    result_state = loop.run(state)

    assert result_state is state
    assert state.status is AgentStatus.COMPLETED
    assert state.final_answer == "done"
    assert state.step == 1


# ---------------------------------------------------------------------------
# 2/3/4. A single tool decision executes the correct tool; ToolCall and
# Observation are recorded.
# ---------------------------------------------------------------------------

def test_single_tool_decision_executes_the_correct_registered_tool_and_records_history() -> None:
    registry = ToolRegistry()
    tool = FakeTool("demo_tool")
    registry.register(tool)
    decision_maker = ScriptedDecisionMaker([
        AgentDecision.tool("demo_tool", "some input"),
        AgentDecision.final("done"),
    ])
    loop = AgentLoop(decision_maker=decision_maker, tool_registry=registry, max_iterations=5)
    state = AgentState(user_input="hello")

    loop.run(state)

    assert tool.calls == ["some input"]

    assert len(state.tool_calls) == 1
    assert state.tool_calls[0].tool_name == "demo_tool"
    assert state.tool_calls[0].tool_input == "some input"
    assert state.tool_calls[0].step == 1

    assert len(state.observations) == 1
    assert state.observations[0].tool_name == "demo_tool"
    assert state.observations[0].success is True
    assert state.observations[0].data == "handled: some input"

    assert state.status is AgentStatus.COMPLETED


# ---------------------------------------------------------------------------
# 5. A tool result failure causes the state to fail.
# ---------------------------------------------------------------------------

def test_tool_result_failure_fails_the_state() -> None:
    registry = ToolRegistry()
    tool = FakeTool("flaky", results=[ToolResult.fail("boom")])
    registry.register(tool)
    decision_maker = ScriptedDecisionMaker([AgentDecision.tool("flaky", "x")])
    loop = AgentLoop(decision_maker=decision_maker, tool_registry=registry, max_iterations=5)
    state = AgentState(user_input="hello")

    loop.run(state)

    assert state.status is AgentStatus.FAILED
    assert len(state.observations) == 1
    assert state.observations[0].success is False
    assert state.observations[0].error == "boom"
    assert any("boom" in error.message for error in state.errors)


def test_tool_invalid_input_value_error_fails_the_state_without_fabricating_an_observation() -> None:
    registry = ToolRegistry()
    tool = FakeTool("bad_input", raise_value_error="Missing required configuration.")
    registry.register(tool)
    decision_maker = ScriptedDecisionMaker([AgentDecision.tool("bad_input", "x")])
    loop = AgentLoop(decision_maker=decision_maker, tool_registry=registry, max_iterations=5)
    state = AgentState(user_input="hello")

    loop.run(state)

    assert state.status is AgentStatus.FAILED
    assert state.observations == []  # no ToolResult was ever produced
    assert any("Missing required configuration." in error.message for error in state.errors)


# ---------------------------------------------------------------------------
# 6. Unknown tool causes deterministic failure.
# ---------------------------------------------------------------------------

def test_unknown_tool_causes_deterministic_failure() -> None:
    registry = ToolRegistry()  # nothing registered
    decision_maker = ScriptedDecisionMaker([AgentDecision.tool("missing", "x")])
    loop = AgentLoop(decision_maker=decision_maker, tool_registry=registry, max_iterations=5)
    state = AgentState(user_input="hello")

    loop.run(state)

    assert state.status is AgentStatus.FAILED
    assert state.observations == []
    assert len(state.tool_calls) == 1  # the attempt itself is still recorded
    assert len(state.errors) == 1
    assert "missing" in state.errors[0].message


# ---------------------------------------------------------------------------
# 7/8. Multiple tool decisions execute across multiple iterations, and the
# loop stops as soon as a final-answer decision is returned.
# ---------------------------------------------------------------------------

def test_multiple_tool_decisions_execute_across_multiple_iterations_then_stop_at_final() -> None:
    registry = ToolRegistry()
    tool_a = FakeTool("tool_a")
    tool_b = FakeTool("tool_b")
    registry.register(tool_a)
    registry.register(tool_b)
    decision_maker = ScriptedDecisionMaker([
        AgentDecision.tool("tool_a", "first"),
        AgentDecision.tool("tool_b", "second"),
        AgentDecision.final("done"),
    ])
    loop = AgentLoop(decision_maker=decision_maker, tool_registry=registry, max_iterations=5)
    state = AgentState(user_input="hello")

    loop.run(state)

    assert tool_a.calls == ["first"]
    assert tool_b.calls == ["second"]
    assert len(state.tool_calls) == 2
    assert len(state.observations) == 2
    assert state.step == 3  # 2 tool iterations + 1 final iteration
    assert state.status is AgentStatus.COMPLETED
    assert decision_maker.calls == 3  # loop must not call decide() again after completion


# ---------------------------------------------------------------------------
# DecisionMakerError handling (Step 7): an unreliable decision maker fails
# the state gracefully instead of crashing the whole execution; any other
# exception type is NOT caught and propagates normally.
# ---------------------------------------------------------------------------

def test_decision_maker_error_fails_the_state_and_stops_the_loop() -> None:
    registry = ToolRegistry()
    decision_maker = FailingDecisionMaker(DecisionMakerError("model output was not valid JSON"))
    loop = AgentLoop(decision_maker=decision_maker, tool_registry=registry, max_iterations=5)
    state = AgentState(user_input="hello")

    result_state = loop.run(state)

    assert result_state.status is AgentStatus.FAILED
    assert len(state.errors) == 1
    assert "model output was not valid JSON" in state.errors[0].message
    assert decision_maker.calls == 1  # loop must not retry


def test_unrelated_exception_from_decision_maker_is_not_caught() -> None:
    registry = ToolRegistry()
    decision_maker = FailingDecisionMaker(RuntimeError("Ollama connection failed"))
    loop = AgentLoop(decision_maker=decision_maker, tool_registry=registry, max_iterations=5)
    state = AgentState(user_input="hello")

    with pytest.raises(RuntimeError, match="Ollama connection failed"):
        loop.run(state)

    assert state.status is AgentStatus.RUNNING  # loop never got the chance to fail the state cleanly


# ---------------------------------------------------------------------------
# 9. max_iterations prevents infinite execution.
# ---------------------------------------------------------------------------

def test_max_iterations_prevents_infinite_execution() -> None:
    registry = ToolRegistry()
    tool = FakeTool("loopy")
    registry.register(tool)
    decision_maker = AlwaysToolDecisionMaker("loopy")  # never returns FINAL
    loop = AgentLoop(decision_maker=decision_maker, tool_registry=registry, max_iterations=3)
    state = AgentState(user_input="hello")

    loop.run(state)

    assert state.status is AgentStatus.FAILED
    assert state.step == 3
    assert len(tool.calls) == 3  # never more than max_iterations
    assert decision_maker.calls == 3
    assert "3" in state.errors[-1].message
    assert "iteration" in state.errors[-1].message.lower()


# ---------------------------------------------------------------------------
# 10. Invalid max_iterations is rejected.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_value", [0, -1, -5])
def test_invalid_max_iterations_is_rejected(bad_value: int) -> None:
    registry = ToolRegistry()
    decision_maker = ScriptedDecisionMaker([])

    with pytest.raises(ValueError):
        AgentLoop(decision_maker=decision_maker, tool_registry=registry, max_iterations=bad_value)


# ---------------------------------------------------------------------------
# 11. Invalid AgentDecision combinations are rejected.
# ---------------------------------------------------------------------------

def test_final_decision_without_final_answer_is_rejected() -> None:
    with pytest.raises(ValueError):
        AgentDecision(action_type=ActionType.FINAL, final_answer=None)


def test_final_decision_with_blank_final_answer_is_rejected() -> None:
    with pytest.raises(ValueError):
        AgentDecision(action_type=ActionType.FINAL, final_answer="   ")


def test_tool_decision_without_tool_name_is_rejected() -> None:
    with pytest.raises(ValueError):
        AgentDecision(action_type=ActionType.TOOL, tool_name=None)


def test_tool_decision_with_blank_tool_name_is_rejected() -> None:
    with pytest.raises(ValueError):
        AgentDecision(action_type=ActionType.TOOL, tool_name="   ")


# ---------------------------------------------------------------------------
# 12. Fake dependencies make execution fully deterministic and repeatable.
# ---------------------------------------------------------------------------

def test_loop_execution_is_deterministic_with_fake_dependencies() -> None:
    def build_and_run() -> AgentState:
        registry = ToolRegistry()
        registry.register(FakeTool("demo_tool"))
        decision_maker = ScriptedDecisionMaker([
            AgentDecision.tool("demo_tool", "x"),
            AgentDecision.final("done"),
        ])
        loop = AgentLoop(decision_maker=decision_maker, tool_registry=registry, max_iterations=5)
        return loop.run(AgentState(user_input="hello"))

    state_a = build_and_run()
    state_b = build_and_run()

    assert state_a.status == state_b.status == AgentStatus.COMPLETED
    assert state_a.final_answer == state_b.final_answer == "done"
    assert len(state_a.tool_calls) == len(state_b.tool_calls) == 1


# ---------------------------------------------------------------------------
# Structural: fakes satisfy the DecisionMaker Protocol without inheritance.
# ---------------------------------------------------------------------------

def test_fake_decision_makers_conform_to_the_decision_maker_protocol() -> None:
    assert isinstance(ScriptedDecisionMaker([]), DecisionMaker)
    assert isinstance(AlwaysToolDecisionMaker("x"), DecisionMaker)


# ---------------------------------------------------------------------------
# Step 11: plan-aware execution. state.plan=None (the default, exercised by
# every test above) is completely unaffected — these tests exercise the
# behavior that only activates when a Plan is attached.
# ---------------------------------------------------------------------------

def _plan(n: int) -> Plan:
    return Plan(steps=[PlanStep(i, f"step {i}") for i in range(1, n + 1)])


def test_pending_plan_is_started_when_the_loop_runs() -> None:
    registry = ToolRegistry()
    decision_maker = ScriptedDecisionMaker([AgentDecision.final("done")])
    loop = AgentLoop(decision_maker=decision_maker, tool_registry=registry)
    state = AgentState(user_input="hello", plan=_plan(1))

    loop.run(state)

    # The plan reached RUNNING (and, since its one step completed via the
    # final-answer path below, ultimately COMPLETED) — either way it was
    # started, not left PENDING.
    assert state.plan.status is not PlanStatus.PENDING


def test_successful_tool_action_completes_the_current_plan_step_and_advances() -> None:
    registry = ToolRegistry()
    tool = FakeTool("demo_tool")
    registry.register(tool)
    decision_maker = ScriptedDecisionMaker([
        AgentDecision.tool("demo_tool", "input for step 1"),
        AgentDecision.tool("demo_tool", "input for step 2"),
        AgentDecision.final("done"),
    ])
    loop = AgentLoop(decision_maker=decision_maker, tool_registry=registry, max_iterations=5)
    plan = _plan(2)
    state = AgentState(user_input="do two things", plan=plan)

    loop.run(state)

    assert plan.get_step(1).status is PlanStatus.COMPLETED
    assert plan.get_step(2).status is PlanStatus.COMPLETED
    assert plan.status is PlanStatus.COMPLETED
    assert state.status is AgentStatus.COMPLETED
    assert tool.calls == ["input for step 1", "input for step 2"]


def test_one_step_plan_completes_after_its_single_tool_action() -> None:
    registry = ToolRegistry()
    tool = FakeTool("demo_tool")
    registry.register(tool)
    decision_maker = ScriptedDecisionMaker([
        AgentDecision.tool("demo_tool", "x"),
        AgentDecision.final("done"),
    ])
    loop = AgentLoop(decision_maker=decision_maker, tool_registry=registry)
    plan = _plan(1)
    state = AgentState(user_input="do one thing", plan=plan)

    loop.run(state)

    assert plan.status is PlanStatus.COMPLETED
    assert state.status is AgentStatus.COMPLETED


def test_one_step_plan_completes_via_final_directly_with_no_tool_call() -> None:
    """Part 3/8: a one-step plan that needs no tool at all (e.g. "Explain
    what Python is") must be able to complete via a direct FINAL answer —
    not be forced through an artificial tool call just to satisfy the plan."""
    registry = ToolRegistry()
    decision_maker = ScriptedDecisionMaker([AgentDecision.final("Python is a programming language.")])
    loop = AgentLoop(decision_maker=decision_maker, tool_registry=registry)
    plan = _plan(1)
    state = AgentState(user_input="What is Python?", plan=plan)

    loop.run(state)

    assert state.status is AgentStatus.COMPLETED
    assert state.final_answer == "Python is a programming language."
    assert plan.status is PlanStatus.COMPLETED
    assert plan.get_step(1).status is PlanStatus.COMPLETED


def test_final_decision_with_multiple_plan_steps_still_pending_fails_state_and_plan() -> None:
    """FINAL is only rejected as premature when it would skip work BEYOND
    the current step — i.e. real future steps that were never attempted at
    all. No new AgentDecision field exists to say "step done" (see the
    module docstring), so this is treated as skipped plan work rather than
    silently accepted. No retry."""
    registry = ToolRegistry()
    decision_maker = ScriptedDecisionMaker([AgentDecision.final("premature answer")])
    loop = AgentLoop(decision_maker=decision_maker, tool_registry=registry)
    plan = _plan(2)
    state = AgentState(user_input="do two things", plan=plan)

    loop.run(state)

    assert state.status is AgentStatus.FAILED
    assert plan.status is PlanStatus.FAILED
    assert plan.get_step(1).status is PlanStatus.FAILED  # never fabricated as COMPLETED
    assert plan.get_step(2).status is PlanStatus.PENDING  # never even attempted
    assert state.final_answer is None


def test_final_decision_completing_the_last_of_two_steps_succeeds() -> None:
    """The mirror image: once step 1 is done via a tool call, FINAL is
    allowed to finish off the single remaining step (step 2) directly."""
    registry = ToolRegistry()
    tool = FakeTool("demo_tool")
    registry.register(tool)
    decision_maker = ScriptedDecisionMaker([
        AgentDecision.tool("demo_tool", "x"),
        AgentDecision.final("done"),
    ])
    loop = AgentLoop(decision_maker=decision_maker, tool_registry=registry)
    plan = _plan(2)
    state = AgentState(user_input="do two things", plan=plan)

    loop.run(state)

    assert state.status is AgentStatus.COMPLETED
    assert plan.status is PlanStatus.COMPLETED
    assert plan.get_step(1).status is PlanStatus.COMPLETED
    assert plan.get_step(2).status is PlanStatus.COMPLETED


def test_failed_tool_result_fails_the_plan_and_does_not_complete_the_step() -> None:
    registry = ToolRegistry()
    tool = FakeTool("demo_tool", results=[ToolResult.fail("boom")])
    registry.register(tool)
    decision_maker = ScriptedDecisionMaker([AgentDecision.tool("demo_tool", "x")])
    loop = AgentLoop(decision_maker=decision_maker, tool_registry=registry)
    plan = _plan(2)
    state = AgentState(user_input="do two things", plan=plan)

    loop.run(state)

    assert state.status is AgentStatus.FAILED
    assert plan.status is PlanStatus.FAILED
    assert plan.get_step(1).status is PlanStatus.FAILED
    assert plan.get_step(2).status is PlanStatus.PENDING  # never started — no later step executes


def test_unknown_tool_fails_the_plan_too() -> None:
    registry = ToolRegistry()  # nothing registered
    decision_maker = ScriptedDecisionMaker([AgentDecision.tool("missing", "x")])
    loop = AgentLoop(decision_maker=decision_maker, tool_registry=registry)
    plan = _plan(1)
    state = AgentState(user_input="hello", plan=plan)

    loop.run(state)

    assert state.status is AgentStatus.FAILED
    assert plan.status is PlanStatus.FAILED


def test_max_iterations_reached_fails_the_plan_too() -> None:
    registry = ToolRegistry()
    tool = FakeTool("loopy")
    registry.register(tool)
    decision_maker = AlwaysToolDecisionMaker("loopy")  # never returns FINAL, never completes the step
    loop = AgentLoop(decision_maker=decision_maker, tool_registry=registry, max_iterations=2)
    plan = _plan(5)  # far more steps than max_iterations allows
    state = AgentState(user_input="hello", plan=plan)

    loop.run(state)

    assert state.status is AgentStatus.FAILED
    assert plan.status is PlanStatus.FAILED


def test_plan_step_id_and_agent_state_step_are_independent_counters() -> None:
    """Part 8: PlanStep.step_id (logical plan ordering) must never be
    confused with AgentState.step (decision/execution iteration count)."""
    registry = ToolRegistry()
    tool = FakeTool("demo_tool")
    registry.register(tool)
    decision_maker = ScriptedDecisionMaker([
        AgentDecision.tool("demo_tool", "a"),
        AgentDecision.tool("demo_tool", "b"),
        AgentDecision.final("done"),
    ])
    loop = AgentLoop(decision_maker=decision_maker, tool_registry=registry, max_iterations=5)
    plan = _plan(2)
    state = AgentState(user_input="do two things", plan=plan)

    loop.run(state)

    # 2 tool iterations + 1 final iteration = 3 AgentState.step increments,
    # completely independent of the plan only ever having step_ids 1 and 2.
    assert state.step == 3
    assert [step.step_id for step in plan.steps] == [1, 2]


def test_plan_none_behavior_is_unaffected_by_plan_aware_code_paths() -> None:
    """Explicit regression check for Part 12/13: a state with plan=None
    behaves exactly as it did before Step 11 — FINAL is accepted
    immediately, no plan-related failure path is ever consulted."""
    registry = ToolRegistry()
    decision_maker = ScriptedDecisionMaker([AgentDecision.final("done")])
    loop = AgentLoop(decision_maker=decision_maker, tool_registry=registry)
    state = AgentState(user_input="hello")  # plan defaults to None

    loop.run(state)

    assert state.status is AgentStatus.COMPLETED
    assert state.final_answer == "done"
    assert state.plan is None
