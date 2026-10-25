"""Duplicate-successful-web-search guard (audit follow-up to Milestone 44.1).

Context: a read-only audit found that the ONLY defense against the model
repeating an identical, already-successful `web_search` query on the
general LLM decision path was a soft prompt instruction ("Do not call the
same tool again if an existing observation already answers the need") —
nothing enforced it in code. `LLMDecisionMaker._is_duplicate_successful_web_search`
and `DuplicateWebSearchError` (app/agent/decision_maker.py) close that gap:
a TOOL decision requesting `web_search` with input that normalizes
identically to an already-SUCCESSFUL `web_search` call earlier in the same
execution is rejected with a categorized `DecisionMakerError`
(`FailureCategory.DUPLICATE_TOOL_CALL`), which flows through AgentLoop's
existing, unmodified correction machinery exactly like DECISION_PARSE or
UNKNOWN_TOOL already do.

Deliberately narrow: only `web_search`, only an exact (normalized) query
match, and only when the prior attempt SUCCEEDED -- a different query, a
retry after a prior failure, or any other tool are all untouched.

This file matches tests/test_decision_maker.py's own fixtures/conventions
exactly (same FakeLLM/FakeTool/_registry_with shapes) and is fully
offline: no network, no Ollama, no Tavily.
"""
from __future__ import annotations

import json

import pytest

from app.agent.decision_maker import DuplicateWebSearchError, LLMDecisionMaker
from app.agent.loop import ActionType
from app.agent.reliability import FailureCategory
from app.agent.state import AgentState
from app.agent.tool_registry import ToolRegistry
from app.tools.base import ToolResult


class FakeLLM:
    """Replays a fixed sequence of responses; records every call. Matches
    tests/test_decision_maker.py's own FakeLLM exactly."""

    def __init__(self, responses: list[str]):
        self.responses = iter(responses)
        self.calls: list[dict[str, object]] = []

    def generate(self, messages: list[dict[str, str]], *, json_mode: bool = False) -> str:
        self.calls.append({"messages": messages, "json_mode": json_mode})
        return next(self.responses)


class FakeTool:
    """A registry-only tool double. execute() must never be called by the
    decision maker -- it raises loudly if it ever is."""

    def __init__(self, name: str, description: str, input_schema: dict[str, str] | None = None):
        self.name = name
        self.description = description
        self.input_schema = input_schema if input_schema is not None else {}
        self.output_description = ""

    def execute(self, input: str | None = None) -> ToolResult:
        raise AssertionError("LLMDecisionMaker must never execute a tool")


def _registry_with(*tools: FakeTool) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def _decision_maker(llm: FakeLLM, registry: ToolRegistry | None = None) -> LLMDecisionMaker:
    return LLMDecisionMaker(llm_client=llm, tool_registry=registry or ToolRegistry())


# ===========================================================================
# 1. An identical (normalized) query after a SUCCESSFUL prior call is
#    rejected with the new categorized error.
# ===========================================================================

def test_identical_normalized_query_after_success_raises_categorized_error() -> None:
    registry = _registry_with(FakeTool("web_search", "Search the web."))
    llm = FakeLLM([json.dumps({"action_type": "tool", "tool_name": "web_search", "tool_input": "Latest AI News"})])
    decision_maker = _decision_maker(llm, registry)
    state = AgentState(user_input="What's the latest AI news?")
    state.record_tool_call("web_search", "latest ai news")
    state.add_observation("web_search", success=True, data=[{"title": "x", "url": "https://x", "content": "y"}])

    with pytest.raises(DuplicateWebSearchError) as exc_info:
        decision_maker.decide(state)

    assert exc_info.value.category is FailureCategory.DUPLICATE_TOOL_CALL


def test_whitespace_and_case_variants_count_as_the_same_query() -> None:
    """Normalization (strip/lower/collapse-whitespace) means trivial
    formatting differences still count as a duplicate -- only a genuinely
    different query should pass."""
    registry = _registry_with(FakeTool("web_search", "Search the web."))
    llm = FakeLLM(
        [json.dumps({"action_type": "tool", "tool_name": "web_search", "tool_input": "  latest   AI News  "})]
    )
    decision_maker = _decision_maker(llm, registry)
    state = AgentState(user_input="What's the latest AI news?")
    state.record_tool_call("web_search", "latest ai news")
    state.add_observation("web_search", success=True, data=[{"content": "evidence"}])

    with pytest.raises(DuplicateWebSearchError):
        decision_maker.decide(state)


# ===========================================================================
# 2. A genuinely different query after the same prior success is allowed.
# ===========================================================================

def test_different_query_after_success_is_not_blocked() -> None:
    registry = _registry_with(FakeTool("web_search", "Search the web."))
    llm = FakeLLM(
        [json.dumps({"action_type": "tool", "tool_name": "web_search", "tool_input": "ai funding rounds 2026"})]
    )
    decision_maker = _decision_maker(llm, registry)
    state = AgentState(user_input="What's the latest AI news, and AI funding rounds?")
    state.record_tool_call("web_search", "latest ai news")
    state.add_observation("web_search", success=True, data=[{"content": "evidence"}])

    decision = decision_maker.decide(state)

    assert decision.action_type is ActionType.TOOL
    assert decision.tool_name == "web_search"
    assert decision.tool_input == "ai funding rounds 2026"


# ===========================================================================
# 3. An identical query after a FAILED prior attempt remains retryable --
#    this guard must never block a legitimate retry.
# ===========================================================================

def test_identical_query_after_a_failed_search_is_not_blocked() -> None:
    registry = _registry_with(FakeTool("web_search", "Search the web."))
    llm = FakeLLM([json.dumps({"action_type": "tool", "tool_name": "web_search", "tool_input": "latest ai news"})])
    decision_maker = _decision_maker(llm, registry)
    state = AgentState(user_input="What's the latest AI news?")
    state.record_tool_call("web_search", "latest ai news")
    state.add_observation("web_search", success=False, error="Tavily request failed due to network or timeout.")

    decision = decision_maker.decide(state)

    assert decision.action_type is ActionType.TOOL
    assert decision.tool_name == "web_search"
    assert decision.tool_input == "latest ai news"


def test_tool_call_with_no_matching_observation_is_not_blocked() -> None:
    """A recorded ToolCall whose execution never reached add_observation
    (e.g. it was denied/rejected before a ToolResult existed) must not be
    mistaken for a successful prior attempt."""
    registry = _registry_with(FakeTool("web_search", "Search the web."))
    llm = FakeLLM([json.dumps({"action_type": "tool", "tool_name": "web_search", "tool_input": "latest ai news"})])
    decision_maker = _decision_maker(llm, registry)
    state = AgentState(user_input="What's the latest AI news?")
    state.record_tool_call("web_search", "latest ai news")  # no corresponding add_observation at all

    decision = decision_maker.decide(state)

    assert decision.action_type is ActionType.TOOL
    assert decision.tool_input == "latest ai news"


# ===========================================================================
# 4. Non-web-search tools are entirely unaffected by this guard.
# ===========================================================================

def test_duplicate_check_does_not_apply_to_non_web_search_tools() -> None:
    registry = _registry_with(FakeTool("date", "Look up a date."))
    llm = FakeLLM([json.dumps({"action_type": "tool", "tool_name": "date", "tool_input": "25 December 2026"})])
    decision_maker = _decision_maker(llm, registry)
    state = AgentState(user_input="What day is 25 December 2026?")
    state.record_tool_call("date", "25 December 2026")
    state.add_observation("date", success=True, data={"day_of_week": "Friday"})

    decision = decision_maker.decide(state)

    assert decision.action_type is ActionType.TOOL
    assert decision.tool_name == "date"
    assert decision.tool_input == "25 December 2026"


def test_is_duplicate_helper_returns_false_for_non_web_search_tool_name() -> None:
    llm = FakeLLM([])
    decision_maker = _decision_maker(llm)
    state = AgentState(user_input="hello")
    state.record_tool_call("date", "25 December 2026")
    state.add_observation("date", success=True, data={"day_of_week": "Friday"})

    assert decision_maker._is_duplicate_successful_web_search(state, "date", "25 December 2026") is False


def test_is_duplicate_helper_returns_false_for_blank_or_missing_input() -> None:
    llm = FakeLLM([])
    decision_maker = _decision_maker(llm)
    state = AgentState(user_input="hello")

    assert decision_maker._is_duplicate_successful_web_search(state, "web_search", None) is False
    assert decision_maker._is_duplicate_successful_web_search(state, "web_search", "   ") is False


# ===========================================================================
# 5. Integration with the existing correction mechanism: the exact same
#    machinery that already handles DECISION_PARSE/UNKNOWN_TOOL.
# ===========================================================================

def test_duplicate_error_carries_the_category_agentloop_expects() -> None:
    """AgentLoop's existing, unmodified `except DecisionMakerError` handler
    (app/agent/loop.py) only offers the failure to a CorrectionPolicy when
    `exc.category is not None` -- confirm that invariant holds for this
    new error without needing to touch loop.py at all."""
    error = DuplicateWebSearchError("duplicate query")

    assert error.category is FailureCategory.DUPLICATE_TOOL_CALL
    assert str(error) == "duplicate query"
