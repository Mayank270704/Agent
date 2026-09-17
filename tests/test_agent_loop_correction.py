"""Step 17, Phase 3: Tier-1 self-correction wired into AgentLoop.

Every test here is fully deterministic and offline (no LLM, no network),
matching test_agent_loop.py's own conventions exactly. This file proves:

1. correction_policy=None (the default) leaves the five call sites
   byte-for-byte unchanged — the pinned regressions in test_agent_loop.py
   already prove this at the whole-suite level; this file adds direct
   proof for each individual site.
2. With a real BudgetedCorrectionPolicy injected, each of the five
   Tier-1 categories (DECISION_PARSE, UNKNOWN_TOOL x2 call sites,
   INVALID_TOOL_INPUT, PLAN_SKIPPED) can be corrected: the state stays
   RUNNING, a CorrectionNote is recorded, and the SAME decide -> validate
   -> execute pipeline runs again on the next iteration.
3. Budget exhaustion still terminates.
4. Termination is STRUCTURAL: an always-permissive fake policy cannot
   defeat max_iterations.
5. A correction cannot bypass tool-registry validation — the identical
   check runs again on the corrected attempt.
"""
from __future__ import annotations

import pytest

from app.agent.loop import ActionType, AgentDecision, AgentLoop, DecisionMakerError
from app.agent.plan import Plan, PlanStep
from app.agent.reliability import (
    BudgetedCorrectionPolicy,
    CorrectionAction,
    CorrectionVerdict,
    FailureCategory,
)
from app.agent.state import AgentState, AgentStatus
from app.agent.tool_registry import ToolRegistry
from app.tools.base import ToolResult


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class MixedDecisionMaker:
    """Returns a fixed sequence where each entry is EITHER an AgentDecision
    (returned) or an Exception (raised) — unlike test_agent_loop.py's
    ScriptedDecisionMaker/FailingDecisionMaker, which each only do one of
    the two for their whole lifetime. Needed here because a correction
    test must fail on one call and succeed on the next."""

    def __init__(self, script: list[AgentDecision | Exception]):
        self._script = iter(script)
        self.calls = 0

    def decide(self, state: AgentState) -> AgentDecision:
        self.calls += 1
        try:
            item = next(self._script)
        except StopIteration:
            raise AssertionError("MixedDecisionMaker ran out of scripted items") from None
        if isinstance(item, Exception):
            raise item
        return item


class AlwaysUnknownToolDecisionMaker:
    """Always requests the same NEVER-registered tool — used to prove
    max_iterations bounds correction regardless of policy permissiveness."""

    def __init__(self, tool_name: str = "never_registered"):
        self.tool_name = tool_name
        self.calls = 0

    def decide(self, state: AgentState) -> AgentDecision:
        self.calls += 1
        return AgentDecision.tool(self.tool_name, None)


class FlakyTool:
    """Returns a fixed sequence of outcomes, one per execute() call, where
    each outcome is either a ToolResult or a ValueError to raise."""

    def __init__(self, name: str, outcomes: list[ToolResult | ValueError]):
        self.name = name
        self.description = f"Flaky tool '{name}' for correction tests."
        self.input_schema: dict[str, str] = {}
        self._outcomes = iter(outcomes)
        self.calls: list[str | None] = []

    def execute(self, input: str | None = None) -> ToolResult:
        self.calls.append(input)
        outcome = next(self._outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class AlwaysCorrectPolicy:
    """A deliberately permissive fake CorrectionPolicy: ALWAYS grants a
    correction, with no budget and no repetition detection whatsoever.
    Used to prove AgentLoop's termination is a STRUCTURAL property
    (max_iterations), not something that depends on any policy behaving
    reasonably."""

    def __init__(self):
        self.calls = 0

    def evaluate(self, state, failure):
        self.calls += 1
        return CorrectionVerdict(CorrectionAction.CORRECT, "keep going", signature=f"sig-{self.calls}")


class RecordingPolicy:
    """Wraps a real policy and records every Failure it was asked to
    evaluate, so a test can assert exactly what AgentLoop constructed."""

    def __init__(self, inner):
        self._inner = inner
        self.failures = []

    def evaluate(self, state, failure):
        self.failures.append(failure)
        return self._inner.evaluate(state, failure)


# ===========================================================================
# 1 — correction_policy=None: every site behaves exactly as before Step 17
# ===========================================================================

def test_no_policy_decision_maker_error_still_fails_immediately() -> None:
    registry = ToolRegistry()
    decision_maker = MixedDecisionMaker(
        [DecisionMakerError("bad output", category=FailureCategory.DECISION_PARSE)]
    )
    loop = AgentLoop(decision_maker=decision_maker, tool_registry=registry, max_iterations=5)
    state = AgentState(user_input="hello")

    loop.run(state)

    assert state.status is AgentStatus.FAILED
    assert decision_maker.calls == 1
    assert state.corrections == []


def test_no_policy_unknown_tool_still_fails_immediately() -> None:
    registry = ToolRegistry()  # nothing registered
    decision_maker = MixedDecisionMaker([AgentDecision.tool("missing", None)])
    loop = AgentLoop(decision_maker=decision_maker, tool_registry=registry, max_iterations=5)
    state = AgentState(user_input="hello")

    loop.run(state)

    assert state.status is AgentStatus.FAILED
    assert state.corrections == []


def test_no_policy_invalid_tool_input_still_fails_immediately() -> None:
    registry = ToolRegistry()
    tool = FlakyTool("bad_input", [ValueError("invalid")])
    registry.register(tool)
    decision_maker = MixedDecisionMaker([AgentDecision.tool("bad_input", "x")])
    loop = AgentLoop(decision_maker=decision_maker, tool_registry=registry, max_iterations=5)
    state = AgentState(user_input="hello")

    loop.run(state)

    assert state.status is AgentStatus.FAILED
    assert state.corrections == []


def test_no_policy_tool_execution_failure_still_fails_immediately() -> None:
    registry = ToolRegistry()
    tool = FlakyTool("flaky", [ToolResult.fail("boom")])
    registry.register(tool)
    decision_maker = MixedDecisionMaker([AgentDecision.tool("flaky", "x")])
    loop = AgentLoop(decision_maker=decision_maker, tool_registry=registry, max_iterations=5)
    state = AgentState(user_input="hello")

    loop.run(state)

    assert state.status is AgentStatus.FAILED
    assert state.corrections == []


def test_no_policy_plan_skipped_still_fails_immediately() -> None:
    registry = ToolRegistry()
    plan = Plan(steps=[PlanStep(step_id=1, description="s1"), PlanStep(step_id=2, description="s2")])
    decision_maker = MixedDecisionMaker([AgentDecision.final("done early")])
    loop = AgentLoop(decision_maker=decision_maker, tool_registry=registry, max_iterations=5)
    state = AgentState(user_input="hello", plan=plan)

    loop.run(state)

    assert state.status is AgentStatus.FAILED
    assert state.corrections == []


def test_no_policy_matches_the_no_correction_policy_argument_at_all() -> None:
    """AgentLoop constructed without even mentioning correction_policy must
    behave identically to one explicitly given None."""
    registry = ToolRegistry()
    decision_maker = MixedDecisionMaker([AgentDecision.final("done")])
    loop_default = AgentLoop(decision_maker=decision_maker, tool_registry=registry, max_iterations=5)

    state = AgentState(user_input="hello")
    loop_default.run(state)

    assert state.status is AgentStatus.COMPLETED
    assert state.corrections == []


# ===========================================================================
# 2 — Tier-1 corrections actually work, one category at a time
# ===========================================================================

def test_decision_parse_failure_is_corrected() -> None:
    registry = ToolRegistry()
    decision_maker = MixedDecisionMaker(
        [
            DecisionMakerError("bad json", category=FailureCategory.DECISION_PARSE),
            AgentDecision.final("recovered answer"),
        ]
    )
    loop = AgentLoop(
        decision_maker=decision_maker,
        tool_registry=registry,
        max_iterations=5,
        correction_policy=BudgetedCorrectionPolicy(),
    )
    state = AgentState(user_input="hello")

    loop.run(state)

    assert state.status is AgentStatus.COMPLETED
    assert state.final_answer == "recovered answer"
    assert decision_maker.calls == 2
    assert len(state.corrections) == 1
    assert state.corrections[0].category is FailureCategory.DECISION_PARSE


def test_unknown_tool_via_decision_maker_error_is_corrected() -> None:
    """The common production path: LLMDecisionMaker validates the tool
    name itself and raises a categorized DecisionParseError BEFORE an
    AgentDecision.tool(...) is ever constructed."""
    registry = ToolRegistry()
    decision_maker = MixedDecisionMaker(
        [
            DecisionMakerError("unknown tool 'ghost'", category=FailureCategory.UNKNOWN_TOOL),
            AgentDecision.final("recovered answer"),
        ]
    )
    loop = AgentLoop(
        decision_maker=decision_maker,
        tool_registry=registry,
        max_iterations=5,
        correction_policy=BudgetedCorrectionPolicy(),
    )
    state = AgentState(user_input="hello")

    loop.run(state)

    assert state.status is AgentStatus.COMPLETED
    assert len(state.corrections) == 1
    assert state.corrections[0].category is FailureCategory.UNKNOWN_TOOL


def test_unknown_tool_via_loop_registry_lookup_is_corrected() -> None:
    """The second path: a DecisionMaker that does NOT pre-validate against
    the registry, so AgentLoop's own ToolRegistry.get() raises
    ToolNotFoundError."""
    registry = ToolRegistry()  # nothing registered
    decision_maker = MixedDecisionMaker(
        [AgentDecision.tool("missing", None), AgentDecision.final("recovered answer")]
    )
    loop = AgentLoop(
        decision_maker=decision_maker,
        tool_registry=registry,
        max_iterations=5,
        correction_policy=BudgetedCorrectionPolicy(),
    )
    state = AgentState(user_input="hello")

    loop.run(state)

    assert state.status is AgentStatus.COMPLETED
    assert len(state.corrections) == 1
    assert state.corrections[0].category is FailureCategory.UNKNOWN_TOOL


def test_invalid_tool_input_is_corrected() -> None:
    registry = ToolRegistry()
    tool = FlakyTool("date_tool", [ValueError("bad date"), ToolResult.ok("2026-01-01")])
    registry.register(tool)
    decision_maker = MixedDecisionMaker(
        [
            AgentDecision.tool("date_tool", "not-a-date"),
            AgentDecision.final("recovered answer"),
        ]
    )
    loop = AgentLoop(
        decision_maker=decision_maker,
        tool_registry=registry,
        max_iterations=5,
        correction_policy=BudgetedCorrectionPolicy(),
    )
    state = AgentState(user_input="hello")

    loop.run(state)

    assert state.status is AgentStatus.COMPLETED
    assert len(state.corrections) == 1
    assert state.corrections[0].category is FailureCategory.INVALID_TOOL_INPUT
    assert tool.calls == ["not-a-date"]  # the failed call is still recorded


def test_tool_execution_failure_is_corrected() -> None:
    registry = ToolRegistry()
    tool = FlakyTool("web_search", [ToolResult.fail("network error"), ToolResult.ok(["result"])])
    registry.register(tool)
    decision_maker = MixedDecisionMaker(
        [AgentDecision.tool("web_search", "q"), AgentDecision.final("recovered answer")]
    )
    loop = AgentLoop(
        decision_maker=decision_maker,
        tool_registry=registry,
        max_iterations=5,
        correction_policy=BudgetedCorrectionPolicy(),
    )
    state = AgentState(user_input="hello")

    loop.run(state)

    assert state.status is AgentStatus.COMPLETED
    assert len(state.corrections) == 1
    assert state.corrections[0].category is FailureCategory.TOOL_EXECUTION_FAILED


def test_plan_skipped_is_corrected_and_the_plan_still_completes() -> None:
    """A FINAL returned before the plan is done gets corrected; the model
    then finishes the plan's steps normally and a later FINAL succeeds."""
    registry = ToolRegistry()
    tool = FlakyTool("step_tool", [ToolResult.ok("step 1 done")])
    registry.register(tool)
    plan = Plan(steps=[PlanStep(step_id=1, description="s1"), PlanStep(step_id=2, description="s2")])
    decision_maker = MixedDecisionMaker(
        [
            AgentDecision.final("too early"),  # blocked: 2 steps still pending
            AgentDecision.tool("step_tool", None),  # completes step 1
            AgentDecision.final("plan finished"),  # completes step 2 + plan
        ]
    )
    loop = AgentLoop(
        decision_maker=decision_maker,
        tool_registry=registry,
        max_iterations=5,
        correction_policy=BudgetedCorrectionPolicy(),
    )
    state = AgentState(user_input="hello", plan=plan)

    loop.run(state)

    assert state.status is AgentStatus.COMPLETED
    assert state.final_answer == "plan finished"
    assert len(state.corrections) == 1
    assert state.corrections[0].category is FailureCategory.PLAN_SKIPPED
    assert plan.status.value == "completed"


# ===========================================================================
# 3 — a correction still consumes a normal iteration (no free retries)
# ===========================================================================

def test_a_correction_increments_step_like_any_other_iteration() -> None:
    registry = ToolRegistry()
    decision_maker = MixedDecisionMaker(
        [
            DecisionMakerError("bad json", category=FailureCategory.DECISION_PARSE),
            AgentDecision.final("done"),
        ]
    )
    loop = AgentLoop(
        decision_maker=decision_maker,
        tool_registry=registry,
        max_iterations=5,
        correction_policy=BudgetedCorrectionPolicy(),
    )
    state = AgentState(user_input="hello")

    loop.run(state)

    assert state.step == 2  # one for the failed attempt, one for the recovery


# ===========================================================================
# 4 — budget exhaustion still terminates
# ===========================================================================

def test_budget_exhaustion_terminates_with_two_different_categories() -> None:
    """Two DIFFERENT categories (not consecutive-identical, so repetition
    detection does not fire) exhaust the default budget of 2; a third
    failure of any kind must then terminate."""
    registry = ToolRegistry()
    decision_maker = MixedDecisionMaker(
        [
            DecisionMakerError("bad json", category=FailureCategory.DECISION_PARSE),
            DecisionMakerError("unknown tool", category=FailureCategory.UNKNOWN_TOOL),
            DecisionMakerError("bad json again", category=FailureCategory.DECISION_PARSE),
        ]
    )
    loop = AgentLoop(
        decision_maker=decision_maker,
        tool_registry=registry,
        max_iterations=10,
        correction_policy=BudgetedCorrectionPolicy(max_corrections=2),
    )
    state = AgentState(user_input="hello")

    loop.run(state)

    assert state.status is AgentStatus.FAILED
    assert len(state.corrections) == 2  # exactly the budget, not more
    assert decision_maker.calls == 3


# ===========================================================================
# 5 — termination is STRUCTURAL, independent of policy behavior
# ===========================================================================

def test_max_iterations_bounds_correction_even_with_an_always_correct_policy() -> None:
    """THE key safety test: a policy that always says CORRECT, forever,
    with no budget and no repetition detection of its own, must still be
    bounded by max_iterations. This is what makes runaway self-correction
    structurally impossible rather than merely policy-discouraged."""
    registry = ToolRegistry()  # the tool is never registered
    decision_maker = AlwaysUnknownToolDecisionMaker()
    policy = AlwaysCorrectPolicy()
    loop = AgentLoop(
        decision_maker=decision_maker,
        tool_registry=registry,
        max_iterations=4,
        correction_policy=policy,
    )
    state = AgentState(user_input="hello")

    loop.run(state)

    assert state.status is AgentStatus.FAILED
    assert state.step == 4
    assert decision_maker.calls == 4
    assert policy.calls == 4
    assert "iteration" in state.errors[-1].message.lower()


class CyclingToolDecisionMaker:
    """Requests a DIFFERENT registered-but-failing tool each call, cycling
    through the given names — used so consecutive failures have DIFFERENT
    signatures (category:tool_name) and never trip repetition detection,
    isolating the max_iterations-vs-budget interaction from it."""

    def __init__(self, tool_names: list[str]):
        self._names = tool_names
        self.calls = 0

    def decide(self, state: AgentState) -> AgentDecision:
        name = self._names[self.calls % len(self._names)]
        self.calls += 1
        return AgentDecision.tool(name, "x")


def test_max_iterations_bound_holds_regardless_of_max_corrections_value() -> None:
    """Even a policy configured with a very large correction budget cannot
    exceed max_iterations, because every correction consumes exactly one
    iteration. Three DIFFERENT failing tools are used so each failure has a
    distinct signature and repetition detection never fires — isolating
    this from test_identical_consecutive_failure_terminates_even_with_
    budget_left's concern."""
    registry = ToolRegistry()
    for name in ("tool_a", "tool_b", "tool_c"):
        registry.register(FlakyTool(name, [ValueError("always invalid")] * 10))
    decision_maker = CyclingToolDecisionMaker(["tool_a", "tool_b", "tool_c"])
    loop = AgentLoop(
        decision_maker=decision_maker,
        tool_registry=registry,
        max_iterations=3,
        correction_policy=BudgetedCorrectionPolicy(max_corrections=1000),
    )
    state = AgentState(user_input="hello")

    loop.run(state)

    assert state.status is AgentStatus.FAILED
    assert state.step == 3
    assert len(state.corrections) == 3  # all three were granted; budget never bound this
    assert "iteration" in state.errors[-1].message.lower()


# ===========================================================================
# 6 — correction cannot bypass tool-registry validation
# ===========================================================================

def test_repeated_request_for_the_same_unregistered_tool_is_rejected_every_time() -> None:
    """After a correction, the NEXT attempt at the SAME unregistered tool
    must go through the identical ToolRegistry check and be rejected again
    — there is no "trusted" bypass for a corrected attempt."""
    registry = ToolRegistry()  # never registers "ghost"
    policy = AlwaysCorrectPolicy()  # so we can observe multiple attempts
    decision_maker = AlwaysUnknownToolDecisionMaker("ghost")
    loop = AgentLoop(
        decision_maker=decision_maker,
        tool_registry=registry,
        max_iterations=3,
        correction_policy=policy,
    )
    state = AgentState(user_input="hello")

    loop.run(state)

    # Every single attempt hit the SAME registry check and failed it —
    # nothing was ever silently accepted.
    assert len(state.tool_calls) == 3
    assert all(call.tool_name == "ghost" for call in state.tool_calls)
    assert state.observations == []  # never actually executed


def test_correction_never_calls_tool_execute_for_an_unregistered_tool() -> None:
    """Structural proof, not just an outcome check: registering a tool
    named 'ghost' would make this pass for the wrong reason, so this test
    deliberately leaves the registry empty and asserts execute() is never
    reachable regardless of how many corrections occur."""
    registry = ToolRegistry()
    executed = []

    class SpyTool:
        name = "ghost"
        description = "spy"
        input_schema: dict[str, str] = {}

        def execute(self, input=None):
            executed.append(input)
            return ToolResult.ok("should never happen")

    # Deliberately NOT registered.
    decision_maker = AlwaysUnknownToolDecisionMaker("ghost")
    loop = AgentLoop(
        decision_maker=decision_maker,
        tool_registry=registry,
        max_iterations=3,
        correction_policy=AlwaysCorrectPolicy(),
    )
    state = AgentState(user_input="hello")

    loop.run(state)

    assert executed == []


# ===========================================================================
# 7 — a policy is never consulted when nothing fails
# ===========================================================================

def test_policy_is_never_consulted_on_a_clean_run() -> None:
    registry = ToolRegistry()
    decision_maker = MixedDecisionMaker([AgentDecision.final("done")])
    policy = RecordingPolicy(BudgetedCorrectionPolicy())
    loop = AgentLoop(
        decision_maker=decision_maker, tool_registry=registry, max_iterations=5, correction_policy=policy
    )
    state = AgentState(user_input="hello")

    loop.run(state)

    assert state.status is AgentStatus.COMPLETED
    assert policy.failures == []
    assert state.corrections == []


# ===========================================================================
# 8 — Failure objects never carry an invented/unregistered tool name
# ===========================================================================

def test_unknown_tool_failure_never_carries_the_invented_tool_name() -> None:
    """Safety proof at the wiring level: whatever AgentLoop builds for
    UNKNOWN_TOOL must never include the model's own invented name."""
    registry = ToolRegistry()
    decision_maker = MixedDecisionMaker(
        [AgentDecision.tool("totally made up name", None), AgentDecision.final("done")]
    )
    policy = RecordingPolicy(BudgetedCorrectionPolicy())
    loop = AgentLoop(
        decision_maker=decision_maker, tool_registry=registry, max_iterations=5, correction_policy=policy
    )
    state = AgentState(user_input="hello")

    loop.run(state)

    assert len(policy.failures) == 1
    assert policy.failures[0].category is FailureCategory.UNKNOWN_TOOL
    assert policy.failures[0].tool_name is None


def test_invalid_tool_input_failure_carries_the_validated_tool_name() -> None:
    """By contrast, a REGISTERED tool's name is safe to carry — the
    registry already validated it before this failure could occur."""
    registry = ToolRegistry()
    tool = FlakyTool("date_tool", [ValueError("bad date"), ToolResult.ok("ok")])
    registry.register(tool)
    decision_maker = MixedDecisionMaker(
        [AgentDecision.tool("date_tool", "bad"), AgentDecision.final("done")]
    )
    policy = RecordingPolicy(BudgetedCorrectionPolicy())
    loop = AgentLoop(
        decision_maker=decision_maker, tool_registry=registry, max_iterations=5, correction_policy=policy
    )
    state = AgentState(user_input="hello")

    loop.run(state)

    assert policy.failures[0].tool_name == "date_tool"


# ===========================================================================
# 9 — DecisionMakerError with no category is never eligible for correction
# ===========================================================================

def test_uncategorized_decision_maker_error_is_always_terminal() -> None:
    """A safety default: a hand-rolled DecisionMakerError with no category
    (exactly what every existing test fake constructs) must stay terminal
    even with a permissive policy injected."""
    registry = ToolRegistry()
    decision_maker = MixedDecisionMaker([DecisionMakerError("uncategorized failure")])
    loop = AgentLoop(
        decision_maker=decision_maker,
        tool_registry=registry,
        max_iterations=5,
        correction_policy=AlwaysCorrectPolicy(),
    )
    state = AgentState(user_input="hello")

    loop.run(state)

    assert state.status is AgentStatus.FAILED
    assert state.corrections == []


def test_decision_maker_error_category_defaults_to_none() -> None:
    exc = DecisionMakerError("plain failure")

    assert exc.category is None
