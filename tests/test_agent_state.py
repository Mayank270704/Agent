from __future__ import annotations

import pytest

from app.agent.plan import Plan, PlanStep
from app.agent.state import AgentState, AgentStatus, ExecutionError, Observation, ToolCall


# ---------------------------------------------------------------------------
# Initialization + invariants.
# ---------------------------------------------------------------------------

def test_state_initializes_correctly() -> None:
    state = AgentState(user_input="What time is it?")

    assert state.user_input == "What time is it?"
    assert state.messages == []
    assert state.step == 0
    assert state.tool_calls == []
    assert state.observations == []
    assert state.errors == []
    assert state.final_answer is None
    assert state.plan is None


@pytest.mark.parametrize("bad_input", ["", "   ", None])
def test_blank_user_input_is_rejected(bad_input: str | None) -> None:
    with pytest.raises(ValueError):
        AgentState(user_input=bad_input)  # type: ignore[arg-type]


def test_state_can_hold_an_optional_plan_when_provided() -> None:
    """Step 9: `plan` defaults to None (existing single-step behavior is
    unaffected), but AgentState can also simply hold a Plan value when one
    is given — this is storage only, AgentLoop does not read this field."""
    plan = Plan(steps=[PlanStep(1, "do something")])

    state = AgentState(user_input="hello", plan=plan)

    assert state.plan is plan


def test_initial_status_is_running() -> None:
    state = AgentState(user_input="hello")

    assert state.status is AgentStatus.RUNNING


def test_negative_iteration_count_is_rejected() -> None:
    with pytest.raises(ValueError):
        AgentState(user_input="hello", step=-1)


# ---------------------------------------------------------------------------
# Recording observations / tool calls / errors.
# ---------------------------------------------------------------------------

def test_observation_can_be_added() -> None:
    state = AgentState(user_input="What is the current gold price?")

    state.add_observation("web_search", success=True, data=[{"title": "A"}])

    assert len(state.observations) == 1
    observation = state.observations[0]
    assert isinstance(observation, Observation)
    assert observation.tool_name == "web_search"
    assert observation.success is True
    assert observation.data == [{"title": "A"}]
    assert observation.error is None
    assert observation.step == state.step


def test_tool_call_can_be_recorded() -> None:
    state = AgentState(user_input="What day was 27 July 2026?")

    state.record_tool_call("date", "27 July 2026")

    assert len(state.tool_calls) == 1
    call = state.tool_calls[0]
    assert isinstance(call, ToolCall)
    assert call.tool_name == "date"
    assert call.tool_input == "27 July 2026"
    assert call.step == state.step


def test_error_can_be_recorded() -> None:
    state = AgentState(user_input="hello")

    state.record_error("Tavily request failed due to network or timeout")

    assert len(state.errors) == 1
    error = state.errors[0]
    assert isinstance(error, ExecutionError)
    assert error.message == "Tavily request failed due to network or timeout"
    assert error.step == state.step
    assert state.status is AgentStatus.RUNNING  # recording an error alone doesn't change status


def test_record_error_rejects_blank_message() -> None:
    state = AgentState(user_input="hello")

    with pytest.raises(ValueError):
        state.record_error("   ")


# ---------------------------------------------------------------------------
# Terminal transitions: complete() / fail().
# ---------------------------------------------------------------------------

def test_complete_stores_final_answer_and_changes_status() -> None:
    state = AgentState(user_input="What is backpropagation?")

    state.complete("Backpropagation is an algorithm for training neural networks.")

    assert state.final_answer == "Backpropagation is an algorithm for training neural networks."
    assert state.status is AgentStatus.COMPLETED


def test_complete_rejects_blank_final_answer() -> None:
    state = AgentState(user_input="hello")

    with pytest.raises(ValueError):
        state.complete("   ")


def test_fail_stores_error_and_changes_status() -> None:
    state = AgentState(user_input="What is the current gold price?")

    state.fail("Tavily request failed due to network or timeout")

    assert state.status is AgentStatus.FAILED
    assert len(state.errors) == 1
    assert state.errors[0].message == "Tavily request failed due to network or timeout"


def test_complete_cannot_be_called_twice() -> None:
    state = AgentState(user_input="hello")
    state.complete("first answer")

    with pytest.raises(ValueError):
        state.complete("second answer")


def test_fail_cannot_be_called_after_complete() -> None:
    state = AgentState(user_input="hello")
    state.complete("first answer")

    with pytest.raises(ValueError):
        state.fail("late error")


# ---------------------------------------------------------------------------
# Independence between instances (no shared mutable defaults).
# ---------------------------------------------------------------------------

def test_state_instances_are_independent_and_do_not_share_mutable_lists() -> None:
    state_a = AgentState(user_input="first execution")
    state_b = AgentState(user_input="second execution")

    state_a.record_tool_call("web_search", "some query")
    state_a.add_observation("web_search", success=True, data=["result"])
    state_a.record_error("boom")

    assert state_a.tool_calls != state_b.tool_calls
    assert state_b.tool_calls == []
    assert state_b.observations == []
    assert state_b.errors == []
    assert state_a.tool_calls is not state_b.tool_calls
    assert state_a.observations is not state_b.observations
    assert state_a.errors is not state_b.errors
