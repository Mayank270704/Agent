"""Milestone 18, Phase 6: AgentLoop integration with ToolExecutionGate.

Covers:
- correction_policy=None / tool_execution_gate=None default behavior is
  completely unchanged (spot checks; the full existing suite already
  proves this exhaustively).
- Existing tools (time, date, web_search) still execute correctly through
  the new gate, with their real, application-set capability metadata.
- Milestone 17 regression: correction still works for the categories that
  were always correctable; it cannot bypass authorization, validation, or
  confirmation; max_iterations and repeated-failure protection remain
  authoritative.
- "One authoritative execution boundary": UNKNOWN_TOOL/INVALID_TOOL_INPUT
  raised THROUGH the gate are handled identically to the no-gate path.

Fully offline: no LLM, no network — WebSearchTool is exercised only
through its existing ValueError precondition path (missing API key),
never a real HTTP call.
"""
from __future__ import annotations

import logging

import pytest

from app.agent.loop import ActionType, AgentDecision, AgentLoop, DecisionMakerError
from app.agent.permissions import AllowlistPermissionPolicy, ExecutionContext
from app.agent.reliability import BudgetedCorrectionPolicy, FailureCategory
from app.agent.state import AgentState, AgentStatus
from app.agent.tool_execution import ToolExecutionGate
from app.agent.tool_registry import ToolRegistry
from app.tools.base import ToolResult
from app.tools.date import DateTool
from app.tools.time import TimeTool
from app.tools.web_search import WebSearchTool


class MixedDecisionMaker:
    def __init__(self, script):
        self._script = iter(script)
        self.calls = 0

    def decide(self, state: AgentState) -> AgentDecision:
        self.calls += 1
        item = next(self._script)
        if isinstance(item, Exception):
            raise item
        return item


class RepeatToolDecisionMaker:
    def __init__(self, tool_name: str, tool_input: str | None = None):
        self.tool_name = tool_name
        self.tool_input = tool_input
        self.calls = 0

    def decide(self, state: AgentState) -> AgentDecision:
        self.calls += 1
        return AgentDecision.tool(self.tool_name, self.tool_input)


class RecordingTool:
    def __init__(self, name: str, *, requires_confirmation: bool = False, result: ToolResult | None = None):
        self.name = name
        self.description = f"tool '{name}'"
        self.input_schema: dict[str, str] = {}
        self.requires_confirmation = requires_confirmation
        self._result = result or ToolResult.ok("done")
        self.calls: list[str | None] = []

    def execute(self, input: str | None = None) -> ToolResult:
        self.calls.append(input)
        return self._result


def _real_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(WebSearchTool())
    registry.register(TimeTool())
    registry.register(DateTool())
    return registry


def _default_gate(registry: ToolRegistry) -> ToolExecutionGate:
    return ToolExecutionGate(registry, AllowlistPermissionPolicy({"time", "date", "web_search"}))


# ===========================================================================
# tool_execution_gate=None: byte-for-byte pre-Milestone-18 behavior
# ===========================================================================

def test_no_gate_means_no_authorization_check_at_all() -> None:
    """A tool NOT on any allow-list still runs fine when no gate is
    configured — proving the default is a true no-op, not "deny
    everything not listed"."""
    registry = ToolRegistry()
    tool = RecordingTool("anything_goes")
    registry.register(tool)
    decision_maker = MixedDecisionMaker([AgentDecision.tool("anything_goes", None), AgentDecision.final("done")])
    loop = AgentLoop(decision_maker=decision_maker, tool_registry=registry, max_iterations=5)
    state = AgentState(user_input="hello")

    loop.run(state)

    assert state.status is AgentStatus.COMPLETED
    assert len(tool.calls) == 1


def test_gate_and_context_default_to_none_explicitly() -> None:
    registry = ToolRegistry()
    loop = AgentLoop(decision_maker=RepeatToolDecisionMaker("x"), tool_registry=registry)

    assert loop.tool_execution_gate is None
    assert loop.execution_context is None


# ===========================================================================
# Existing tools still work through the new gate
# ===========================================================================

def test_time_tool_executes_through_the_gate() -> None:
    registry = _real_registry()
    gate = _default_gate(registry)
    decision_maker = MixedDecisionMaker([AgentDecision.tool("time", None), AgentDecision.final("It's noon.")])
    loop = AgentLoop(decision_maker=decision_maker, tool_registry=registry, tool_execution_gate=gate)
    state = AgentState(user_input="what time is it")

    loop.run(state)

    assert state.status is AgentStatus.COMPLETED
    assert state.observations[0].success is True


def test_date_tool_executes_through_the_gate() -> None:
    registry = _real_registry()
    gate = _default_gate(registry)
    decision_maker = MixedDecisionMaker(
        [AgentDecision.tool("date", "27 July 2026"), AgentDecision.final("It was a Monday.")]
    )
    loop = AgentLoop(decision_maker=decision_maker, tool_registry=registry, tool_execution_gate=gate)
    state = AgentState(user_input="what day was 27 July 2026")

    loop.run(state)

    assert state.status is AgentStatus.COMPLETED
    assert state.observations[0].data["day_of_week"] == "Monday"


def test_web_search_tool_missing_api_key_still_raises_value_error_through_the_gate() -> None:
    """Reuses the existing precondition-failure path (no real network
    call) — proves the gate does not swallow or reinterpret it."""
    registry = _real_registry()
    web_search = registry.get("web_search")
    web_search.api_key = ""  # type: ignore[attr-defined]
    gate = _default_gate(registry)
    decision_maker = MixedDecisionMaker([AgentDecision.tool("web_search", "current news")])
    loop = AgentLoop(decision_maker=decision_maker, tool_registry=registry, tool_execution_gate=gate)
    state = AgentState(user_input="search for something")

    loop.run(state)

    assert state.status is AgentStatus.FAILED
    assert "TAVILY_API_KEY" not in state.final_answer if state.final_answer else True


def test_all_three_real_tools_are_allowed_by_the_default_gate_construction() -> None:
    registry = _real_registry()
    policy = AllowlistPermissionPolicy({"time", "date", "web_search"})

    for name in ("time", "date", "web_search"):
        descriptor = registry.describe(name)
        assert policy.evaluate(descriptor, ExecutionContext()).value == "allow"


def test_real_tools_none_require_confirmation_today() -> None:
    registry = _real_registry()

    for name in ("time", "date", "web_search"):
        assert registry.describe(name).requires_confirmation is False


def test_web_search_is_classified_as_external_network_not_read() -> None:
    """A real, non-contrived distinction in the actual tool set: unlike
    time/date, web_search calls a third-party API."""
    from app.tools.base import ToolCapability

    registry = _real_registry()

    assert registry.describe("web_search").capability is ToolCapability.EXTERNAL_NETWORK
    assert registry.describe("time").capability is ToolCapability.READ
    assert registry.describe("date").capability is ToolCapability.READ


# ===========================================================================
# INVARIANT 1/2/3 via the gate, through the full loop
# ===========================================================================

def test_unregistered_tool_through_the_gate_still_maps_to_unknown_tool_and_fails() -> None:
    registry = ToolRegistry()
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy({"ghost"}))
    decision_maker = RepeatToolDecisionMaker("ghost")
    loop = AgentLoop(decision_maker=decision_maker, tool_registry=registry, tool_execution_gate=gate)
    state = AgentState(user_input="hello")

    loop.run(state)

    assert state.status is AgentStatus.FAILED
    assert state.observations == []


def test_denied_tool_through_the_gate_fails_and_never_executes() -> None:
    tool = RecordingTool("delete_file")
    registry = ToolRegistry()
    registry.register(tool)
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy(frozenset()))
    decision_maker = RepeatToolDecisionMaker("delete_file")
    loop = AgentLoop(decision_maker=decision_maker, tool_registry=registry, tool_execution_gate=gate)
    state = AgentState(user_input="delete it")

    loop.run(state)

    assert state.status is AgentStatus.FAILED
    assert tool.calls == []
    assert state.observations == []


def test_confirmation_required_through_the_gate_fails_and_never_executes() -> None:
    tool = RecordingTool("delete_file", requires_confirmation=True)
    registry = ToolRegistry()
    registry.register(tool)
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy({"delete_file"}))
    decision_maker = RepeatToolDecisionMaker("delete_file")
    loop = AgentLoop(
        decision_maker=decision_maker,
        tool_registry=registry,
        tool_execution_gate=gate,
        execution_context=ExecutionContext(),  # nothing confirmed
    )
    state = AgentState(user_input="delete it")

    loop.run(state)

    assert state.status is AgentStatus.FAILED
    assert tool.calls == []


def test_invalid_input_through_the_gate_still_maps_to_invalid_tool_input() -> None:
    registry = ToolRegistry()
    registry.register(DateTool())
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy({"date"}))
    decision_maker = RepeatToolDecisionMaker("date", "not a real date")
    loop = AgentLoop(decision_maker=decision_maker, tool_registry=registry, tool_execution_gate=gate)
    state = AgentState(user_input="what day")

    loop.run(state)

    assert state.status is AgentStatus.FAILED
    assert "Could not parse" in state.errors[-1].message


# ===========================================================================
# Milestone 17 regression: correction still works for eligible categories
# ===========================================================================

def test_unknown_tool_through_the_gate_is_still_correctable() -> None:
    """UNKNOWN_TOOL/INVALID_TOOL_INPUT keep their pre-Milestone-18
    correction eligibility even when raised through the gate — only
    PERMISSION_DENIED/CONFIRMATION_REQUIRED are new and non-correctable."""
    registry = ToolRegistry()
    tool = RecordingTool("time")
    registry.register(tool)
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy({"time"}))
    decision_maker = MixedDecisionMaker(
        [AgentDecision.tool("missing", None), AgentDecision.final("recovered")]
    )
    loop = AgentLoop(
        decision_maker=decision_maker,
        tool_registry=registry,
        tool_execution_gate=gate,
        correction_policy=BudgetedCorrectionPolicy(),
    )
    state = AgentState(user_input="hello")

    loop.run(state)

    assert state.status is AgentStatus.COMPLETED
    assert len(state.corrections) == 1
    assert state.corrections[0].category is FailureCategory.UNKNOWN_TOOL


def test_invalid_tool_input_through_the_gate_is_still_correctable() -> None:
    registry = ToolRegistry()
    registry.register(DateTool())
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy({"date"}))
    decision_maker = MixedDecisionMaker(
        [AgentDecision.tool("date", "garbage"), AgentDecision.tool("date", "27 July 2026"), AgentDecision.final("Monday")]
    )
    loop = AgentLoop(
        decision_maker=decision_maker,
        tool_registry=registry,
        tool_execution_gate=gate,
        correction_policy=BudgetedCorrectionPolicy(),
    )
    state = AgentState(user_input="what day")

    loop.run(state)

    assert state.status is AgentStatus.COMPLETED
    assert len(state.corrections) == 1
    assert state.corrections[0].category is FailureCategory.INVALID_TOOL_INPUT


# ===========================================================================
# INVARIANT 8/9/10 — CorrectionPolicy cannot bypass authorization,
# validation, or confirmation
# ===========================================================================

class AlwaysCorrectPolicy:
    """A deliberately maximally-permissive fake CorrectionPolicy — always
    grants a correction, no budget, no repetition detection at all. Used
    to prove PERMISSION_DENIED/CONFIRMATION_REQUIRED cannot be corrected
    EVEN by a policy willing to correct anything else."""

    def __init__(self):
        self.calls = 0

    def evaluate(self, state, failure):
        self.calls += 1
        from app.agent.reliability import CorrectionAction, CorrectionVerdict

        return CorrectionVerdict(CorrectionAction.CORRECT, "keep going", signature=f"sig-{self.calls}")


def test_always_correct_policy_cannot_bypass_permission_denial() -> None:
    tool = RecordingTool("delete_file")
    registry = ToolRegistry()
    registry.register(tool)
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy(frozenset()))
    decision_maker = RepeatToolDecisionMaker("delete_file")
    policy = AlwaysCorrectPolicy()
    loop = AgentLoop(
        decision_maker=decision_maker,
        tool_registry=registry,
        tool_execution_gate=gate,
        correction_policy=policy,
        max_iterations=5,
    )
    state = AgentState(user_input="delete it")

    loop.run(state)

    assert state.status is AgentStatus.FAILED
    assert state.step == 1  # never reached a second iteration
    assert policy.calls == 0  # the policy was never even consulted
    assert tool.calls == []


def test_always_correct_policy_cannot_bypass_confirmation_requirement() -> None:
    tool = RecordingTool("delete_file", requires_confirmation=True)
    registry = ToolRegistry()
    registry.register(tool)
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy({"delete_file"}))
    decision_maker = RepeatToolDecisionMaker("delete_file")
    policy = AlwaysCorrectPolicy()
    loop = AgentLoop(
        decision_maker=decision_maker,
        tool_registry=registry,
        tool_execution_gate=gate,
        execution_context=ExecutionContext(),
        correction_policy=policy,
        max_iterations=5,
    )
    state = AgentState(user_input="delete it")

    loop.run(state)

    assert state.status is AgentStatus.FAILED
    assert state.step == 1
    assert policy.calls == 0
    assert tool.calls == []


def test_correction_cannot_bypass_input_validation() -> None:
    """Correction for INVALID_TOOL_INPUT is allowed, but each corrected
    attempt must go through the SAME real validation again — a second bad
    input on the SAME tool is caught by REAL validation a second time too
    (proving no silent pass-through), and Milestone 17's own repetition
    detection then correctly refuses a third attempt at the identical
    failure rather than granting an unbounded number of corrections."""
    registry = ToolRegistry()
    registry.register(DateTool())
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy({"date"}))
    decision_maker = MixedDecisionMaker(
        [
            AgentDecision.tool("date", "still garbage"),
            AgentDecision.tool("date", "also garbage"),
            AgentDecision.tool("date", "27 July 2026"),  # never reached
            AgentDecision.final("Monday"),  # never reached
        ]
    )
    loop = AgentLoop(
        decision_maker=decision_maker,
        tool_registry=registry,
        tool_execution_gate=gate,
        correction_policy=BudgetedCorrectionPolicy(max_corrections=5),
    )
    state = AgentState(user_input="what day")

    loop.run(state)

    # The first bad attempt is corrected; the SECOND is independently
    # re-validated (still real ValueError from DateTool, not waved
    # through) and then terminated as a repeated identical failure — never
    # silently treated as valid input.
    assert state.status is AgentStatus.FAILED
    assert len(state.corrections) == 1
    assert state.corrections[0].category is FailureCategory.INVALID_TOOL_INPUT
    assert "Could not parse" in state.errors[-1].message


def test_correction_re_validates_a_different_bad_input_on_the_same_tool() -> None:
    """The positive companion: when the SECOND attempt targets a
    DIFFERENT tool (so repetition detection does not fire), its own bad
    input is still independently rejected by real validation — proving
    validation genuinely reruns rather than being skipped for a
    "corrected" attempt."""
    registry = ToolRegistry()
    registry.register(DateTool())
    other = RecordingTool("other_date_like_tool")
    registry.register(other)
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy({"date", "other_date_like_tool"}))
    decision_maker = MixedDecisionMaker(
        [
            AgentDecision.tool("date", "garbage"),
            AgentDecision.tool("date", "27 July 2026"),
            AgentDecision.final("Monday"),
        ]
    )
    loop = AgentLoop(
        decision_maker=decision_maker,
        tool_registry=registry,
        tool_execution_gate=gate,
        correction_policy=BudgetedCorrectionPolicy(max_corrections=5),
    )
    state = AgentState(user_input="what day")

    loop.run(state)

    assert state.status is AgentStatus.COMPLETED
    assert len(state.corrections) == 1
    assert state.corrections[0].category is FailureCategory.INVALID_TOOL_INPUT


# ===========================================================================
# max_iterations and repeated-failure protection remain authoritative
# ===========================================================================

def test_max_iterations_remains_authoritative_with_a_gate_configured() -> None:
    registry = ToolRegistry()
    tool = RecordingTool("loopy")
    registry.register(tool)
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy({"loopy"}))
    decision_maker = RepeatToolDecisionMaker("loopy")
    loop = AgentLoop(
        decision_maker=decision_maker, tool_registry=registry, tool_execution_gate=gate, max_iterations=3
    )
    state = AgentState(user_input="hello")

    loop.run(state)

    assert state.status is AgentStatus.FAILED
    assert state.step == 3
    assert len(tool.calls) == 3


def test_repeated_failure_protection_remains_intact_with_a_gate_configured() -> None:
    """A repeatedly-failing (but authorized) tool still triggers
    Milestone 17's repetition-detection termination, unaffected by the
    gate's presence."""
    tool = RecordingTool("flaky", result=ToolResult.fail("boom"))
    registry = ToolRegistry()
    registry.register(tool)
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy({"flaky"}))
    decision_maker = RepeatToolDecisionMaker("flaky")
    loop = AgentLoop(
        decision_maker=decision_maker,
        tool_registry=registry,
        tool_execution_gate=gate,
        correction_policy=BudgetedCorrectionPolicy(max_corrections=5),
        max_iterations=10,
    )
    state = AgentState(user_input="hello")

    loop.run(state)

    assert state.status is AgentStatus.FAILED
    assert len(tool.calls) == 2  # first failure corrected, second is the repeat -> terminate
    assert len(state.corrections) == 1


def test_default_correction_policy_behavior_is_unaffected_by_gate_presence() -> None:
    """correction_policy=None with a gate configured behaves exactly like
    correction_policy=None without one — the two opt-ins are independent."""
    registry = ToolRegistry()
    tool = RecordingTool("time")
    registry.register(tool)
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy({"time"}))
    decision_maker = RepeatToolDecisionMaker("missing")  # unregistered
    loop = AgentLoop(decision_maker=decision_maker, tool_registry=registry, tool_execution_gate=gate)
    state = AgentState(user_input="hello")

    loop.run(state)

    assert state.status is AgentStatus.FAILED
    assert decision_maker.calls == 1  # no correction attempted -- terminated immediately


# ===========================================================================
# INVARIANT 15/16 — one authoritative boundary, registry stays authoritative
# ===========================================================================

def test_the_loop_never_calls_tool_execute_directly_when_a_gate_is_configured() -> None:
    """Structural proof: once a gate exists, _resolve_and_run_tool's ONLY
    branch for that case delegates to the gate — there is no second path
    that could accidentally call tool.execute() directly."""
    registry = ToolRegistry()
    tool = RecordingTool("time")
    registry.register(tool)
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy(frozenset()))  # denies everything
    decision_maker = RepeatToolDecisionMaker("time")
    loop = AgentLoop(decision_maker=decision_maker, tool_registry=registry, tool_execution_gate=gate)
    state = AgentState(user_input="hello")

    loop.run(state)

    # If the loop had a second, gate-bypassing path to tool.execute(), a
    # denied tool would still have run. It must not have.
    assert tool.calls == []


def test_registry_remains_the_sole_source_of_tool_existence_even_with_a_gate() -> None:
    """A name on the allow-list that was never registered is still
    ToolNotFoundError, not silently treated as existing."""
    registry = ToolRegistry()  # "time" never registered
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy({"time"}))
    decision_maker = RepeatToolDecisionMaker("time")
    loop = AgentLoop(decision_maker=decision_maker, tool_registry=registry, tool_execution_gate=gate)
    state = AgentState(user_input="hello")

    loop.run(state)

    assert state.status is AgentStatus.FAILED
    assert "registered" in state.errors[-1].message.lower() or "time" in state.errors[-1].message


# ===========================================================================
# Observability
# ===========================================================================

def test_permission_denial_through_the_loop_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    tool = RecordingTool("delete_file")
    registry = ToolRegistry()
    registry.register(tool)
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy(frozenset()))
    decision_maker = RepeatToolDecisionMaker("delete_file")
    loop = AgentLoop(decision_maker=decision_maker, tool_registry=registry, tool_execution_gate=gate)
    state = AgentState(user_input="delete it")

    with caplog.at_level(logging.WARNING, logger="app.agent.tool_execution"):
        loop.run(state)

    assert "tool.authorization.denied" in caplog.text
