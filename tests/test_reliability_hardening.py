"""Step 17, Phase 6: F7/F8 hardening + observability.

F7 — `_format_history` must bound `observation.error` and `error.message`,
     matching the existing bound already applied to `observation.data`.
F8 — `AgentOrchestrator._failure_answer` must never echo internal detail
     (an invented tool name, a tool's own error text, iteration-limit
     detail) into the user-facing answer; the detail remains available via
     `AgentResult.errors` for internal diagnostics/logs only.
Observability — structured, safe log lines for the correction lifecycle:
     correction.triggered / correction.declined / correction.succeeded /
     execution.terminal — with a fixed vocabulary that never contains raw
     model output, tool output, or secrets.

Fully offline; no LLM, no network.
"""
from __future__ import annotations

import logging

import pytest

from app.agent.decision_maker import LLMDecisionMaker, _MAX_OBSERVATION_DATA_CHARS
from app.agent.loop import ActionType, AgentDecision, AgentLoop, DecisionMakerError
from app.agent.orchestrator import AgentOrchestrator
from app.agent.reliability import BudgetedCorrectionPolicy, FailureCategory
from app.agent.state import AgentState, AgentStatus
from app.agent.tool_registry import ToolRegistry
from app.tools.base import ToolResult


class FakeLLM:
    def __init__(self, responses: list[str] | None = None):
        self.responses = iter(responses or [])

    def generate(self, messages, *, json_mode: bool = False) -> str:
        return next(self.responses)


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


class FlakyTool:
    def __init__(self, name: str, outcomes):
        self.name = name
        self.description = f"Flaky tool '{name}'."
        self.input_schema: dict[str, str] = {}
        self._outcomes = iter(outcomes)
        self.calls: list[str | None] = []

    def execute(self, input: str | None = None) -> ToolResult:
        self.calls.append(input)
        outcome = next(self._outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class RepeatToolDecisionMaker:
    def __init__(self, tool_name: str):
        self.tool_name = tool_name
        self.calls = 0

    def decide(self, state: AgentState) -> AgentDecision:
        self.calls += 1
        return AgentDecision.tool(self.tool_name, "x")


class AlwaysFailingTool:
    def __init__(self, name: str, error: str = "always fails"):
        self.name = name
        self.description = f"Tool '{name}'."
        self.input_schema: dict[str, str] = {}
        self._error = error
        self.calls: list[str | None] = []

    def execute(self, input: str | None = None) -> ToolResult:
        self.calls.append(input)
        return ToolResult.fail(self._error)


def _build_orchestrator(decision_maker, *, tools=None, max_iterations: int = 5) -> AgentOrchestrator:
    registry = ToolRegistry()
    for tool in tools or []:
        registry.register(tool)
    return AgentOrchestrator(
        llm_client=FakeLLM(),
        tool_registry=registry,
        decision_maker=decision_maker,
        max_iterations=max_iterations,
    )


# ===========================================================================
# F7 — bounding observation.error and error.message in the prompt
# ===========================================================================

def test_observation_error_is_truncated_like_observation_data() -> None:
    tool = FlakyTool("flaky", [ToolResult.fail("x" * 5000)])
    registry = ToolRegistry()
    registry.register(tool)
    decision_maker = LLMDecisionMaker(llm_client=FakeLLM([]), tool_registry=registry)
    state = AgentState(user_input="hello")
    state.record_tool_call("flaky", "x")
    state.add_observation("flaky", success=False, error="x" * 5000)

    history = decision_maker._format_history(state)

    assert len(history) < 5000
    assert "...[truncated]" in history


def test_error_message_is_truncated_in_history() -> None:
    decision_maker = LLMDecisionMaker(llm_client=FakeLLM([]), tool_registry=ToolRegistry())
    state = AgentState(user_input="hello")
    # record_error requires a non-empty message but places no upper bound —
    # exactly the gap F7 closes.
    state.record_error("y" * 5000)

    history = decision_maker._format_history(state)

    assert len(history) < 5000
    assert "...[truncated]" in history


def test_short_observation_error_and_error_message_are_unaffected() -> None:
    """The bound must not alter ordinary, already-short content."""
    decision_maker = LLMDecisionMaker(llm_client=FakeLLM([]), tool_registry=ToolRegistry())
    state = AgentState(user_input="hello")
    state.record_tool_call("flaky", "x")
    state.add_observation("flaky", success=False, error="network timeout")
    state.record_error("a short error")

    history = decision_maker._format_history(state)

    assert "network timeout" in history
    assert "a short error" in history


def test_observation_error_bound_matches_the_existing_data_bound() -> None:
    """F7 reuses the SAME constant as observation.data — one bound, not
    two independently-tunable ones that could silently drift apart."""
    decision_maker = LLMDecisionMaker(llm_client=FakeLLM([]), tool_registry=ToolRegistry())
    state = AgentState(user_input="hello")
    exact = "z" * _MAX_OBSERVATION_DATA_CHARS
    state.record_tool_call("flaky", "x")
    state.add_observation("flaky", success=False, error=exact)

    history = decision_maker._format_history(state)

    assert "...[truncated]" not in history  # exactly at the bound, not over it


# ===========================================================================
# F8 — generic failure answer, detail preserved in AgentResult.errors
# ===========================================================================

def test_failure_answer_never_echoes_an_unregistered_tool_name() -> None:
    from app.agent.loop import AgentDecision as _AD

    class ScriptedDecisionMaker:
        def decide(self, state):
            return _AD.tool("some_invented_tool_name", "x")

    orchestrator = _build_orchestrator(ScriptedDecisionMaker())

    result = orchestrator.process("do something")

    assert result.status is AgentStatus.FAILED
    assert "some_invented_tool_name" not in result.answer
    assert "some_invented_tool_name" in result.errors[-1].message


def test_failure_answer_never_echoes_tool_error_text() -> None:
    from app.agent.loop import AgentDecision as _AD

    class ScriptedDecisionMaker:
        def decide(self, state):
            return _AD.tool("web_search", "x")

    tool = FlakyTool("web_search", [ToolResult.fail("Missing TAVILY_API_KEY configuration.")])
    orchestrator = _build_orchestrator(ScriptedDecisionMaker(), tools=[tool])

    result = orchestrator.process("search something")

    assert result.status is AgentStatus.FAILED
    assert "TAVILY_API_KEY" not in result.answer
    assert "TAVILY_API_KEY" in result.errors[-1].message


def test_failure_answer_is_a_fixed_generic_sentence() -> None:
    """Every terminal failure produces the SAME answer text, regardless of
    cause — proving no per-cause detail leaks through by varying it."""
    from app.agent.loop import AgentDecision as _AD

    class ScriptedDecisionMaker:
        def decide(self, state):
            return _AD.tool("missing", "x")

    result_a = _build_orchestrator(ScriptedDecisionMaker()).process("a")
    result_b = _build_orchestrator(ScriptedDecisionMaker()).process("b")

    assert result_a.answer == result_b.answer


def test_failure_answer_logs_the_detail_server_side(caplog: pytest.LogCaptureFixture) -> None:
    """The detail is not LOST — it moves to a server-side log, which an
    operator (not the end user, and not the model) can see."""
    from app.agent.loop import AgentDecision as _AD

    class ScriptedDecisionMaker:
        def decide(self, state):
            return _AD.tool("ghost_tool", "x")

    orchestrator = _build_orchestrator(ScriptedDecisionMaker())

    with caplog.at_level(logging.WARNING, logger="app.agent.orchestrator"):
        orchestrator.process("do something")

    assert "ghost_tool" in caplog.text


def test_failure_answer_still_contains_the_pinned_substring() -> None:
    """Regression: the exact substring existing tests assert on
    (test_chat_service.py, test_memory_integration.py, test_orchestrator.py)
    must still be present."""
    from app.agent.loop import AgentDecision as _AD

    class ScriptedDecisionMaker:
        def decide(self, state):
            return _AD.tool("missing", "x")

    result = _build_orchestrator(ScriptedDecisionMaker()).process("do something")

    assert "could not complete this request" in result.answer


# ===========================================================================
# Observability — correction.triggered / declined / succeeded / terminal
# ===========================================================================

def test_correction_triggered_is_logged_with_category_step_and_attempt(
    caplog: pytest.LogCaptureFixture,
) -> None:
    registry = ToolRegistry()
    decision_maker = MixedDecisionMaker(
        [
            DecisionMakerError("bad json", category=FailureCategory.DECISION_PARSE),
            AgentDecision.final("recovered"),
        ]
    )
    loop = AgentLoop(
        decision_maker=decision_maker,
        tool_registry=registry,
        max_iterations=5,
        correction_policy=BudgetedCorrectionPolicy(),
    )
    state = AgentState(user_input="hello")

    with caplog.at_level(logging.INFO, logger="app.agent.loop"):
        loop.run(state)

    assert "correction.triggered" in caplog.text
    assert "category=decision_parse" in caplog.text
    assert "attempt=1" in caplog.text


def test_correction_declined_logs_repeated_reason(caplog: pytest.LogCaptureFixture) -> None:
    registry = ToolRegistry()
    tool = AlwaysFailingTool("web_search")
    registry.register(tool)
    decision_maker = RepeatToolDecisionMaker("web_search")
    loop = AgentLoop(
        decision_maker=decision_maker,
        tool_registry=registry,
        max_iterations=5,
        correction_policy=BudgetedCorrectionPolicy(max_corrections=5),
    )
    state = AgentState(user_input="hello")

    with caplog.at_level(logging.WARNING, logger="app.agent.loop"):
        loop.run(state)

    assert "correction.declined" in caplog.text
    assert "reason=repeated" in caplog.text


def test_correction_declined_logs_budget_exhausted_reason(caplog: pytest.LogCaptureFixture) -> None:
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

    with caplog.at_level(logging.WARNING, logger="app.agent.loop"):
        loop.run(state)

    assert "correction.declined" in caplog.text
    assert "reason=budget_exhausted" in caplog.text


def test_correction_succeeded_is_logged_after_recovery(caplog: pytest.LogCaptureFixture) -> None:
    registry = ToolRegistry()
    decision_maker = MixedDecisionMaker(
        [
            DecisionMakerError("bad json", category=FailureCategory.DECISION_PARSE),
            AgentDecision.final("recovered"),
        ]
    )
    loop = AgentLoop(
        decision_maker=decision_maker,
        tool_registry=registry,
        max_iterations=5,
        correction_policy=BudgetedCorrectionPolicy(),
    )
    state = AgentState(user_input="hello")

    with caplog.at_level(logging.INFO, logger="app.agent.loop"):
        loop.run(state)

    assert "correction.succeeded" in caplog.text
    assert "category=decision_parse" in caplog.text


def test_correction_succeeded_is_not_logged_when_no_correction_happened(
    caplog: pytest.LogCaptureFixture,
) -> None:
    registry = ToolRegistry()
    decision_maker = MixedDecisionMaker([AgentDecision.final("done")])
    loop = AgentLoop(
        decision_maker=decision_maker,
        tool_registry=registry,
        max_iterations=5,
        correction_policy=BudgetedCorrectionPolicy(),
    )
    state = AgentState(user_input="hello")

    with caplog.at_level(logging.INFO, logger="app.agent.loop"):
        loop.run(state)

    assert "correction.succeeded" not in caplog.text


def test_execution_terminal_is_logged_on_max_iterations(caplog: pytest.LogCaptureFixture) -> None:
    """The tool must SUCCEED every time, or the state fails on the tool
    failure itself (step 1) before max_iterations is ever reached."""
    registry = ToolRegistry()
    tool = FlakyTool("loopy", [ToolResult.ok("keep going")] * 10)
    registry.register(tool)

    class AlwaysToolDecisionMaker:
        def decide(self, state):
            return AgentDecision.tool("loopy", None)

    loop = AgentLoop(decision_maker=AlwaysToolDecisionMaker(), tool_registry=registry, max_iterations=3)
    state = AgentState(user_input="hello")

    with caplog.at_level(logging.ERROR, logger="app.agent.loop"):
        loop.run(state)

    assert "execution.terminal" in caplog.text
    assert "category=max_iterations" in caplog.text


# ===========================================================================
# Never logged: raw model output, tool output, secrets
# ===========================================================================

def test_correction_logs_never_contain_the_safe_message_text(caplog: pytest.LogCaptureFixture) -> None:
    """Log lines carry category/step/attempt/reason only -- never the
    rendered message text itself (which, while safe/fixed, is a prompt
    concern, not a log concern; keeping logs to pure metadata avoids ever
    needing to reason about message content growing unsafe later)."""
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

    with caplog.at_level(logging.INFO, logger="app.agent.loop"):
        loop.run(state)

    assert "Respond with STRICT JSON ONLY" not in caplog.text


def test_correction_logs_never_contain_tool_detail_text(caplog: pytest.LogCaptureFixture) -> None:
    registry = ToolRegistry()
    tool = FlakyTool(
        "web_search", [ValueError("Missing TAVILY_API_KEY configuration."), ToolResult.ok("ok")]
    )
    registry.register(tool)
    decision_maker = MixedDecisionMaker(
        [AgentDecision.tool("web_search", "q"), AgentDecision.final("done")]
    )
    loop = AgentLoop(
        decision_maker=decision_maker,
        tool_registry=registry,
        max_iterations=5,
        correction_policy=BudgetedCorrectionPolicy(include_tool_error_text=True),
    )
    state = AgentState(user_input="hello")

    with caplog.at_level(logging.DEBUG, logger="app.agent.loop"):
        loop.run(state)

    assert "TAVILY_API_KEY" not in caplog.text


def test_correction_logs_never_contain_the_user_input(caplog: pytest.LogCaptureFixture) -> None:
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
    secret_input = "my secret question about hunter2-correct-horse"
    state = AgentState(user_input=secret_input)

    with caplog.at_level(logging.DEBUG, logger="app.agent.loop"):
        loop.run(state)

    assert secret_input not in caplog.text
    assert "hunter2" not in caplog.text
