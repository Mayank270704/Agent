"""Step 17, Phase 1: the failure taxonomy and CorrectionPolicy in isolation.

Nothing here touches AgentLoop, AgentState, or an LLM — this file proves
`BudgetedCorrectionPolicy` is a PURE function of (state, failure): same
inputs, same verdict, no mutation, no I/O. A minimal stand-in state object
is used instead of the real AgentState (Phase 2 adds `corrections` there);
this file only depends on the one attribute the policy actually reads.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from app.agent.reliability import (
    DEFAULT_MAX_CORRECTIONS,
    BudgetedCorrectionPolicy,
    CorrectionAction,
    CorrectionPolicy,
    CorrectionVerdict,
    Failure,
    FailureCategory,
)


@dataclass
class _Note:
    """Stands in for state.py's CorrectionNote (not yet built in Phase 1) —
    only `signature` is read by the policy."""

    signature: str


@dataclass
class _FakeState:
    """Stands in for AgentState — only `corrections` is read."""

    corrections: list[_Note] = field(default_factory=list)


def _push(state: _FakeState, signature: str) -> None:
    state.corrections.append(_Note(signature=signature))


# ===========================================================================
# 1 — FailureCategory: the closed vocabulary
# ===========================================================================

def test_failure_category_has_exactly_the_seven_documented_members() -> None:
    """Milestone 18 added exactly two new members (PERMISSION_DENIED,
    CONFIRMATION_REQUIRED) to this closed set — every pre-existing member
    is preserved unchanged, matching Milestone 18's explicit constraint
    not to replace this enum."""
    assert {member.value for member in FailureCategory} == {
        "decision_parse",
        "unknown_tool",
        "invalid_tool_input",
        "tool_execution_failed",
        "plan_skipped",
        "permission_denied",
        "confirmation_required",
    }


def test_max_iterations_and_programming_errors_have_no_category() -> None:
    """Structural proof that these two are simply not representable here —
    there is no FailureCategory member for them, so no policy can ever be
    asked to classify one."""
    names = {member.name for member in FailureCategory}
    assert "MAX_ITERATIONS" not in names
    assert "UNEXPECTED_ERROR" not in names
    assert "SECURITY" not in names
    assert "SESSION_ISOLATION" not in names


# ===========================================================================
# 2 — Failure: safety of tool_name/detail
# ===========================================================================

def test_failure_defaults_to_no_tool_name_and_no_detail() -> None:
    failure = Failure(category=FailureCategory.DECISION_PARSE)

    assert failure.tool_name is None
    assert failure.detail is None


def test_failure_is_frozen() -> None:
    failure = Failure(category=FailureCategory.UNKNOWN_TOOL)

    with pytest.raises((AttributeError, TypeError)):
        failure.tool_name = "x"  # type: ignore[misc]


# ===========================================================================
# 3 — CorrectionPolicy Protocol conformance
# ===========================================================================

def test_budgeted_policy_conforms_to_the_protocol() -> None:
    assert isinstance(BudgetedCorrectionPolicy(), CorrectionPolicy)


# ===========================================================================
# 4 — BudgetedCorrectionPolicy: construction validation
# ===========================================================================

def test_default_max_corrections_is_two() -> None:
    assert DEFAULT_MAX_CORRECTIONS == 2
    assert BudgetedCorrectionPolicy().max_corrections == 2


def test_include_tool_error_text_defaults_to_false() -> None:
    assert BudgetedCorrectionPolicy().include_tool_error_text is False


@pytest.mark.parametrize("bad", [-1, 1.5, "2", True])
def test_invalid_max_corrections_is_rejected(bad: object) -> None:
    with pytest.raises(ValueError, match="max_corrections"):
        BudgetedCorrectionPolicy(max_corrections=bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [0, 1, "true", None])
def test_invalid_include_tool_error_text_is_rejected(bad: object) -> None:
    with pytest.raises(ValueError, match="include_tool_error_text"):
        BudgetedCorrectionPolicy(include_tool_error_text=bad)  # type: ignore[arg-type]


def test_zero_max_corrections_is_allowed_and_always_terminates() -> None:
    """A degenerate but legal configuration: classify failures but never
    actually grant a correction."""
    policy = BudgetedCorrectionPolicy(max_corrections=0)
    state = _FakeState()

    verdict = policy.evaluate(state, Failure(category=FailureCategory.DECISION_PARSE))

    assert verdict.action is CorrectionAction.TERMINATE


# ===========================================================================
# 5 — evaluate() is pure: no mutation, deterministic
# ===========================================================================

def test_evaluate_does_not_mutate_state() -> None:
    policy = BudgetedCorrectionPolicy()
    state = _FakeState()

    policy.evaluate(state, Failure(category=FailureCategory.DECISION_PARSE))

    assert state.corrections == []  # policy never appends; that is the caller's job


def test_evaluate_is_deterministic_for_the_same_inputs() -> None:
    policy = BudgetedCorrectionPolicy()
    state = _FakeState()

    v1 = policy.evaluate(state, Failure(category=FailureCategory.UNKNOWN_TOOL))
    v2 = policy.evaluate(state, Failure(category=FailureCategory.UNKNOWN_TOOL))

    assert v1 == v2


# ===========================================================================
# 6 — budget enforcement
# ===========================================================================

def test_correction_granted_while_under_budget() -> None:
    policy = BudgetedCorrectionPolicy(max_corrections=2)
    state = _FakeState()  # 0 corrections so far

    verdict = policy.evaluate(state, Failure(category=FailureCategory.DECISION_PARSE))

    assert verdict.action is CorrectionAction.CORRECT


def test_correction_denied_once_budget_is_exhausted() -> None:
    policy = BudgetedCorrectionPolicy(max_corrections=2)
    state = _FakeState()
    _push(state, "decision_parse")
    _push(state, "unknown_tool")  # 2 corrections already spent, budget = 2

    verdict = policy.evaluate(state, Failure(category=FailureCategory.INVALID_TOOL_INPUT, tool_name="date"))

    assert verdict.action is CorrectionAction.TERMINATE


def test_budget_is_shared_across_categories_not_per_category() -> None:
    """A separate-per-category budget would let the model burn through
    max_corrections FAILURES OF EACH KIND — this proves the ceiling is
    flat across the whole request."""
    policy = BudgetedCorrectionPolicy(max_corrections=1)
    state = _FakeState()
    _push(state, "decision_parse")  # budget spent on category A

    verdict = policy.evaluate(state, Failure(category=FailureCategory.UNKNOWN_TOOL))  # different category

    assert verdict.action is CorrectionAction.TERMINATE


def test_budget_is_never_reset_by_an_intervening_success() -> None:
    """state.corrections only grows; a policy re-consulted after a
    successful iteration must still see the earlier spend."""
    policy = BudgetedCorrectionPolicy(max_corrections=1)
    state = _FakeState()
    _push(state, "decision_parse")
    # (a hypothetical successful iteration happened here — nothing removes
    # the entry from state.corrections)

    verdict = policy.evaluate(state, Failure(category=FailureCategory.UNKNOWN_TOOL))

    assert verdict.action is CorrectionAction.TERMINATE


# ===========================================================================
# 7 — repetition detection
# ===========================================================================

def test_identical_consecutive_failure_terminates_even_with_budget_left() -> None:
    policy = BudgetedCorrectionPolicy(max_corrections=5)
    state = _FakeState()
    _push(state, "invalid_tool_input:date")

    verdict = policy.evaluate(
        state, Failure(category=FailureCategory.INVALID_TOOL_INPUT, tool_name="date")
    )

    assert verdict.action is CorrectionAction.TERMINATE


def test_different_tool_with_the_same_category_is_not_a_repetition() -> None:
    policy = BudgetedCorrectionPolicy(max_corrections=5)
    state = _FakeState()
    _push(state, "invalid_tool_input:date")

    verdict = policy.evaluate(
        state, Failure(category=FailureCategory.INVALID_TOOL_INPUT, tool_name="web_search")
    )

    assert verdict.action is CorrectionAction.CORRECT


def test_non_consecutive_repetition_is_not_terminated() -> None:
    """A fails, then B, then A again: still exploring, not stuck — only
    CONSECUTIVE identical failures trigger early termination."""
    policy = BudgetedCorrectionPolicy(max_corrections=5)
    state = _FakeState()
    _push(state, "decision_parse")
    _push(state, "unknown_tool")

    verdict = policy.evaluate(state, Failure(category=FailureCategory.DECISION_PARSE))

    assert verdict.action is CorrectionAction.CORRECT


def test_unknown_tool_repetition_is_detected_by_category_alone() -> None:
    """UNKNOWN_TOOL never fingerprints on the model's invented tool name
    (untrusted text) — two consecutive UNKNOWN_TOOL failures are always a
    repetition regardless of what name was invented each time."""
    policy = BudgetedCorrectionPolicy(max_corrections=5)
    state = _FakeState()
    _push(state, "unknown_tool")

    verdict = policy.evaluate(state, Failure(category=FailureCategory.UNKNOWN_TOOL))

    assert verdict.action is CorrectionAction.TERMINATE


# ===========================================================================
# 8 — safe_message: fixed vocabulary, never echoes untrusted content
# ===========================================================================

@pytest.mark.parametrize("category", list(FailureCategory))
def test_every_category_has_a_nonempty_fixed_message(category: FailureCategory) -> None:
    policy = BudgetedCorrectionPolicy()
    state = _FakeState()

    verdict = policy.evaluate(state, Failure(category=category))

    assert verdict.safe_message.strip() != ""


def test_default_safe_message_never_includes_tool_detail() -> None:
    policy = BudgetedCorrectionPolicy(include_tool_error_text=False)
    state = _FakeState()
    secret = "Missing TAVILY_API_KEY configuration. Set it in the .env file."

    verdict = policy.evaluate(
        state,
        Failure(category=FailureCategory.TOOL_EXECUTION_FAILED, tool_name="web_search", detail=secret),
    )

    assert secret not in verdict.safe_message


def test_safe_message_never_echoes_an_invented_tool_name() -> None:
    """Even if a caller mistakenly populated tool_name on an UNKNOWN_TOOL
    Failure, the fixed message template contains no interpolation slot for
    it -- the vocabulary itself is the safety boundary, not caller
    discipline alone."""
    policy = BudgetedCorrectionPolicy()
    state = _FakeState()

    verdict = policy.evaluate(
        state, Failure(category=FailureCategory.UNKNOWN_TOOL, tool_name="'; DROP TABLE users; --")
    )

    assert "DROP TABLE" not in verdict.safe_message


def test_include_tool_error_text_true_appends_a_bounded_detail() -> None:
    policy = BudgetedCorrectionPolicy(include_tool_error_text=True)
    state = _FakeState()

    verdict = policy.evaluate(
        state,
        Failure(category=FailureCategory.INVALID_TOOL_INPUT, tool_name="date", detail="bad date format"),
    )

    assert "bad date format" in verdict.safe_message


def test_include_tool_error_text_true_still_truncates_long_detail() -> None:
    policy = BudgetedCorrectionPolicy(include_tool_error_text=True)
    state = _FakeState()
    long_detail = "x" * 5000

    verdict = policy.evaluate(
        state, Failure(category=FailureCategory.TOOL_EXECUTION_FAILED, tool_name="web_search", detail=long_detail)
    )

    assert len(verdict.safe_message) < 5000
    assert "...[truncated]" in verdict.safe_message


def test_no_detail_means_no_appended_text_even_when_enabled() -> None:
    policy = BudgetedCorrectionPolicy(include_tool_error_text=True)
    state = _FakeState()

    verdict = policy.evaluate(
        state, Failure(category=FailureCategory.TOOL_EXECUTION_FAILED, tool_name="web_search", detail=None)
    )

    assert verdict.safe_message == BudgetedCorrectionPolicy()._safe_message(
        Failure(category=FailureCategory.TOOL_EXECUTION_FAILED)
    )


# ===========================================================================
# 9 — CorrectionVerdict shape
# ===========================================================================

def test_correct_verdict_carries_a_nonempty_signature() -> None:
    policy = BudgetedCorrectionPolicy()
    state = _FakeState()

    verdict = policy.evaluate(state, Failure(category=FailureCategory.DECISION_PARSE))

    assert verdict.action is CorrectionAction.CORRECT
    assert verdict.signature == "decision_parse"


def test_tool_scoped_signature_includes_the_tool_name() -> None:
    policy = BudgetedCorrectionPolicy()
    state = _FakeState()

    verdict = policy.evaluate(
        state, Failure(category=FailureCategory.INVALID_TOOL_INPUT, tool_name="date")
    )

    assert verdict.signature == "invalid_tool_input:date"


def test_verdict_is_frozen() -> None:
    verdict = CorrectionVerdict(CorrectionAction.CORRECT, "msg", "sig")

    with pytest.raises((AttributeError, TypeError)):
        verdict.action = CorrectionAction.TERMINATE  # type: ignore[misc]
