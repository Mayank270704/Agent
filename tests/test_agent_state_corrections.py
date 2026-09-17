"""Step 17, Phase 2: AgentState's `corrections` field and
`record_correction()`.

Mirrors the existing test_agent_state.py conventions for `record_error`/
`errors` exactly, since `corrections` is deliberately the same "recorded
fact, not a decision" shape.
"""
from __future__ import annotations

import pytest

from app.agent.reliability import FailureCategory
from app.agent.state import AgentState, AgentStatus, CorrectionNote


def _note(step: int = 1, category: FailureCategory = FailureCategory.DECISION_PARSE) -> CorrectionNote:
    return CorrectionNote(category=category, safe_message="fixed message", step=step, signature=category.value)


# ===========================================================================
# 1 — default state
# ===========================================================================

def test_corrections_defaults_to_an_empty_list() -> None:
    state = AgentState(user_input="hello")

    assert state.corrections == []


def test_each_state_gets_its_own_corrections_list() -> None:
    """Guards against the classic mutable-default-argument bug — every
    AgentState instance must have an independent list."""
    state_a = AgentState(user_input="a")
    state_b = AgentState(user_input="b")

    state_a.record_correction(_note())

    assert state_a.corrections != state_b.corrections
    assert state_b.corrections == []


# ===========================================================================
# 2 — record_correction()
# ===========================================================================

def test_record_correction_appends_the_note() -> None:
    state = AgentState(user_input="hello")
    note = _note(step=2, category=FailureCategory.UNKNOWN_TOOL)

    state.record_correction(note)

    assert state.corrections == [note]


def test_record_correction_does_not_change_status() -> None:
    """The entire point of a correction: the execution keeps RUNNING."""
    state = AgentState(user_input="hello")

    state.record_correction(_note())

    assert state.status is AgentStatus.RUNNING


def test_record_correction_preserves_order() -> None:
    state = AgentState(user_input="hello")
    first = _note(step=1, category=FailureCategory.DECISION_PARSE)
    second = _note(step=2, category=FailureCategory.INVALID_TOOL_INPUT)

    state.record_correction(first)
    state.record_correction(second)

    assert state.corrections == [first, second]


def test_multiple_corrections_can_be_recorded_across_categories() -> None:
    state = AgentState(user_input="hello")

    for category in FailureCategory:
        state.record_correction(_note(category=category))

    assert len(state.corrections) == len(list(FailureCategory))


def test_record_correction_works_alongside_record_error() -> None:
    """Corrections and terminal errors are independent lists — recording
    one must not disturb the other."""
    state = AgentState(user_input="hello")

    state.record_correction(_note())
    state.record_error("an unrelated terminal error")

    assert len(state.corrections) == 1
    assert len(state.errors) == 1


# ===========================================================================
# 3 — CorrectionNote shape
# ===========================================================================

def test_correction_note_is_frozen() -> None:
    note = _note()

    with pytest.raises((AttributeError, TypeError)):
        note.step = 99  # type: ignore[misc]


def test_correction_note_carries_category_message_step_and_signature() -> None:
    note = CorrectionNote(
        category=FailureCategory.TOOL_EXECUTION_FAILED,
        safe_message="try something else",
        step=3,
        signature="tool_execution_failed:web_search",
    )

    assert note.category is FailureCategory.TOOL_EXECUTION_FAILED
    assert note.safe_message == "try something else"
    assert note.step == 3
    assert note.signature == "tool_execution_failed:web_search"


# ===========================================================================
# 4 — attempt/remaining are DERIVED, never stored (Step 17 design §4)
# ===========================================================================

def test_attempt_number_is_derived_from_corrections_length_not_stored() -> None:
    """The design explicitly rejects storing attempt_number/remaining_
    attempts as separate fields — this proves the derivation works and
    that CorrectionNote itself carries no such field."""
    state = AgentState(user_input="hello")
    state.record_correction(_note(step=1))
    state.record_correction(_note(step=2))

    attempt_number = len(state.corrections)

    assert attempt_number == 2
    assert not hasattr(state.corrections[-1], "attempt_number")
    assert not hasattr(state.corrections[-1], "remaining_attempts")


def test_last_failure_is_derived_via_indexing_not_a_stored_field() -> None:
    state = AgentState(user_input="hello")
    first = _note(step=1, category=FailureCategory.DECISION_PARSE)
    second = _note(step=2, category=FailureCategory.PLAN_SKIPPED)
    state.record_correction(first)
    state.record_correction(second)

    assert state.corrections[-1] is second
    assert not hasattr(state, "last_failure")
