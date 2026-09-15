from __future__ import annotations

import dataclasses

import pytest

from app.agent.plan import Plan, PlanStatus, PlanStep


def _steps(n: int) -> list[PlanStep]:
    return [PlanStep(i, f"step {i}") for i in range(1, n + 1)]


# ---------------------------------------------------------------------------
# PlanStep invariants.
# ---------------------------------------------------------------------------

def test_plan_step_defaults_to_pending() -> None:
    step = PlanStep(1, "do something")

    assert step.status is PlanStatus.PENDING


@pytest.mark.parametrize("blank_description", ["", "   ", None])
def test_blank_step_description_rejected(blank_description) -> None:
    with pytest.raises(ValueError):
        PlanStep(1, blank_description)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_step_id", [0, -1, -100])
def test_non_positive_step_id_rejected(bad_step_id: int) -> None:
    with pytest.raises(ValueError):
        PlanStep(bad_step_id, "do something")


def test_plan_step_is_frozen() -> None:
    step = PlanStep(1, "do something")

    with pytest.raises(dataclasses.FrozenInstanceError):
        step.status = PlanStatus.RUNNING  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Plan invariants (Part 4 / Part 9 items 1-4).
# ---------------------------------------------------------------------------

def test_empty_plan_rejected() -> None:
    with pytest.raises(ValueError):
        Plan(steps=[])


def test_duplicate_step_ids_rejected() -> None:
    with pytest.raises(ValueError):
        Plan(steps=[PlanStep(1, "first"), PlanStep(1, "duplicate")])


def test_plan_preserves_deterministic_step_ordering() -> None:
    steps = [PlanStep(3, "third"), PlanStep(1, "first"), PlanStep(2, "second")]

    plan = Plan(steps=steps)

    assert [step.step_id for step in plan.steps] == [3, 1, 2]
    assert [step.description for step in plan.steps] == ["third", "first", "second"]


def test_new_plan_starts_as_pending() -> None:
    plan = Plan(steps=_steps(2))

    assert plan.status is PlanStatus.PENDING


def test_current_step_is_the_first_non_completed_step() -> None:
    plan = Plan(steps=_steps(3))
    plan.start()

    assert plan.current_step().step_id == 1

    plan.start_step(1)
    assert plan.current_step().step_id == 1  # RUNNING still counts as current

    plan.complete_step(1)
    assert plan.current_step().step_id == 2

    plan.start_step(2)
    plan.complete_step(2)
    assert plan.current_step().step_id == 3


def test_current_step_is_none_once_every_step_is_completed() -> None:
    plan = Plan(steps=_steps(1))
    plan.start()
    plan.start_step(1)
    plan.complete_step(1)

    assert plan.current_step() is None


def test_current_step_on_a_pending_plan_is_the_first_step() -> None:
    plan = Plan(steps=_steps(2))

    assert plan.current_step().step_id == 1


def test_get_step_returns_the_matching_step() -> None:
    plan = Plan(steps=_steps(3))

    step = plan.get_step(2)

    assert step.step_id == 2
    assert step.description == "step 2"


def test_get_step_unknown_id_raises() -> None:
    plan = Plan(steps=_steps(2))

    with pytest.raises(ValueError):
        plan.get_step(999)


def test_two_plan_instances_do_not_share_mutable_state() -> None:
    plan_a = Plan(steps=_steps(1))
    plan_b = Plan(steps=_steps(1))

    plan_a.start()
    plan_a.start_step(1)

    assert plan_a.status is PlanStatus.RUNNING
    assert plan_b.status is PlanStatus.PENDING
    assert plan_b.get_step(1).status is PlanStatus.PENDING


# ---------------------------------------------------------------------------
# Plan-level transitions (Part 9 items 6, 14).
# ---------------------------------------------------------------------------

def test_start_transitions_pending_to_running() -> None:
    plan = Plan(steps=_steps(1))

    plan.start()

    assert plan.status is PlanStatus.RUNNING


def test_start_on_a_non_pending_plan_raises() -> None:
    plan = Plan(steps=_steps(1))
    plan.start()

    with pytest.raises(ValueError):
        plan.start()  # already RUNNING


def test_complete_on_a_pending_plan_raises() -> None:
    """Item 14: invalid terminal transitions are rejected — you cannot
    complete a plan that was never started."""
    plan = Plan(steps=_steps(1))

    with pytest.raises(ValueError):
        plan.complete()


def test_fail_on_a_pending_plan_raises() -> None:
    plan = Plan(steps=_steps(1))

    with pytest.raises(ValueError):
        plan.fail()


def test_start_after_plan_already_failed_raises() -> None:
    plan = Plan(steps=_steps(1))
    plan.start()
    plan.fail()

    with pytest.raises(ValueError):
        plan.start()


def test_complete_after_plan_already_completed_raises() -> None:
    plan = Plan(steps=_steps(1))
    plan.start()
    plan.start_step(1)
    plan.complete_step(1)
    plan.complete()

    with pytest.raises(ValueError):
        plan.complete()


# ---------------------------------------------------------------------------
# Step-level transitions (Part 9 items 7, 8, 9, 10, 11).
# ---------------------------------------------------------------------------

def test_start_step_changes_only_the_selected_step_to_running() -> None:
    plan = Plan(steps=_steps(2))
    plan.start()

    plan.start_step(1)

    assert plan.get_step(1).status is PlanStatus.RUNNING
    assert plan.get_step(2).status is PlanStatus.PENDING


def test_start_step_before_plan_start_raises() -> None:
    plan = Plan(steps=_steps(1))

    with pytest.raises(ValueError):
        plan.start_step(1)


def test_complete_step_changes_running_to_completed() -> None:
    plan = Plan(steps=_steps(1))
    plan.start()
    plan.start_step(1)

    plan.complete_step(1)

    assert plan.get_step(1).status is PlanStatus.COMPLETED


def test_fail_step_changes_running_to_failed() -> None:
    plan = Plan(steps=_steps(1))
    plan.start()
    plan.start_step(1)

    plan.fail_step(1)

    assert plan.get_step(1).status is PlanStatus.FAILED


def test_cannot_complete_a_step_that_is_not_running() -> None:
    """Item 10: a PENDING step cannot be completed directly."""
    plan = Plan(steps=_steps(1))
    plan.start()

    with pytest.raises(ValueError):
        plan.complete_step(1)


def test_cannot_fail_a_step_that_is_not_running() -> None:
    plan = Plan(steps=_steps(1))
    plan.start()

    with pytest.raises(ValueError):
        plan.fail_step(1)


def test_cannot_start_a_completed_step() -> None:
    """Item 11."""
    plan = Plan(steps=_steps(1))
    plan.start()
    plan.start_step(1)
    plan.complete_step(1)

    with pytest.raises(ValueError):
        plan.start_step(1)


def test_cannot_start_an_already_running_step() -> None:
    plan = Plan(steps=_steps(1))
    plan.start()
    plan.start_step(1)

    with pytest.raises(ValueError):
        plan.start_step(1)


def test_cannot_complete_an_already_completed_step() -> None:
    plan = Plan(steps=_steps(1))
    plan.start()
    plan.start_step(1)
    plan.complete_step(1)

    with pytest.raises(ValueError):
        plan.complete_step(1)


def test_cannot_fail_an_already_failed_step() -> None:
    plan = Plan(steps=_steps(1))
    plan.start()
    plan.start_step(1)
    plan.fail_step(1)

    with pytest.raises(ValueError):
        plan.fail_step(1)


# ---------------------------------------------------------------------------
# Whole-plan completion / failure behavior (Part 9 items 12, 13).
# ---------------------------------------------------------------------------

def test_completing_all_steps_allows_the_plan_to_become_completed() -> None:
    """Item 12."""
    plan = Plan(steps=_steps(2))
    plan.start()
    plan.start_step(1)
    plan.complete_step(1)
    plan.start_step(2)
    plan.complete_step(2)

    plan.complete()

    assert plan.status is PlanStatus.COMPLETED


def test_complete_rejected_while_any_step_is_not_yet_completed() -> None:
    plan = Plan(steps=_steps(2))
    plan.start()
    plan.start_step(1)
    plan.complete_step(1)
    # step 2 is still PENDING

    with pytest.raises(ValueError):
        plan.complete()

    assert plan.status is PlanStatus.RUNNING  # rejected attempt must not change status


def test_failing_a_step_causes_the_plan_itself_to_fail() -> None:
    """Item 13: failing a step causes appropriate plan failure behavior —
    in this minimal model (no retries/replanning), that means the whole
    plan is immediately marked FAILED too."""
    plan = Plan(steps=_steps(2))
    plan.start()
    plan.start_step(1)

    plan.fail_step(1)

    assert plan.status is PlanStatus.FAILED
    assert plan.get_step(1).status is PlanStatus.FAILED
    assert plan.get_step(2).status is PlanStatus.PENDING  # untouched step left as-is


def test_cannot_start_a_step_after_the_plan_has_failed() -> None:
    plan = Plan(steps=_steps(2))
    plan.start()
    plan.start_step(1)
    plan.fail_step(1)

    with pytest.raises(ValueError):
        plan.start_step(2)


def test_cannot_complete_a_plan_after_it_has_failed() -> None:
    plan = Plan(steps=_steps(1))
    plan.start()
    plan.start_step(1)
    plan.fail_step(1)

    with pytest.raises(ValueError):
        plan.complete()
