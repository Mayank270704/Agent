"""Step 17, Phase 5: Tier-2 (tool-outcome) correction + repetition
detection, end to end through AgentLoop + BudgetedCorrectionPolicy +
LLMDecisionMaker together.

Categories D (INVALID_TOOL_INPUT) and E (TOOL_EXECUTION_FAILED) are
"Tier 2" because, unlike Tier 1 (a malformed decision that never touched
anything external), a real tool call already happened. This file's
distinct concern beyond test_agent_loop_correction.py's per-category
wiring proof is:

1. Repetition detection's EXACT semantics for tool failures — including
   the deliberate decision that an intervening SUCCESS on a DIFFERENT
   action does not "reset" a repeated failure of the SAME tool (see the
   Milestone 17 report for the reasoning).
2. include_tool_error_text's full data flow, end to end: AgentLoop
   captures a real tool's ValueError/ToolResult.error text as
   `Failure.detail` -> BudgetedCorrectionPolicy decides whether to render
   it -> LLMDecisionMaker's prompt either contains it or provably does
   not, depending on the flag — proven with a REAL sensitive-looking
   message (a credential-shaped string), not a synthetic one.

Fully offline, no network, no LLM.
"""
from __future__ import annotations

import json
import logging

from app.agent.decision_maker import LLMDecisionMaker
from app.agent.loop import ActionType, AgentDecision, AgentLoop
from app.agent.reliability import BudgetedCorrectionPolicy, FailureCategory
from app.agent.state import AgentState, AgentStatus
from app.agent.tool_registry import ToolRegistry
from app.tools.base import ToolResult


class FakeLLM:
    def __init__(self, responses: list[str]):
        self.responses = iter(responses)
        self.calls: list[dict[str, object]] = []

    def generate(self, messages, *, json_mode: bool = False) -> str:
        self.calls.append({"messages": messages})
        return next(self.responses)


class MixedDecisionMaker:
    def __init__(self, script: list[AgentDecision | Exception]):
        self._script = iter(script)
        self.calls = 0

    def decide(self, state: AgentState) -> AgentDecision:
        self.calls += 1
        item = next(self._script)
        if isinstance(item, Exception):
            raise item
        return item


class FlakyTool:
    def __init__(self, name: str, outcomes: list[ToolResult | ValueError]):
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


class AlwaysFailingTool:
    """Never succeeds — used for repetition-termination tests where the
    loop must stop before exhausting a scripted decision list."""

    def __init__(self, name: str, error: str = "always fails"):
        self.name = name
        self.description = f"Tool '{name}' that always fails."
        self.input_schema: dict[str, str] = {}
        self._error = error
        self.calls: list[str | None] = []

    def execute(self, input: str | None = None) -> ToolResult:
        self.calls.append(input)
        return ToolResult.fail(self._error)


class RepeatToolDecisionMaker:
    """Always requests the same tool with the same input — pairs with
    AlwaysFailingTool to drive repeated identical TOOL_EXECUTION_FAILED
    failures."""

    def __init__(self, tool_name: str):
        self.tool_name = tool_name
        self.calls = 0

    def decide(self, state: AgentState) -> AgentDecision:
        self.calls += 1
        return AgentDecision.tool(self.tool_name, "x")


def _tool_json(tool_name: str, tool_input: str) -> str:
    return json.dumps({"action_type": "tool", "tool_name": tool_name, "tool_input": tool_input})


def _final_json(answer: str) -> str:
    return json.dumps({"action_type": "final", "final_answer": answer})


# ===========================================================================
# 1 — repeated identical tool failure terminates immediately
# ===========================================================================

def test_same_tool_failing_twice_in_a_row_terminates_on_the_second_failure() -> None:
    """Repetition fires immediately — the loop must NOT spend the full
    correction budget on a tool that keeps failing identically."""
    registry = ToolRegistry()
    tool = AlwaysFailingTool("web_search")
    registry.register(tool)
    decision_maker = RepeatToolDecisionMaker("web_search")
    loop = AgentLoop(
        decision_maker=decision_maker,
        tool_registry=registry,
        max_iterations=10,
        correction_policy=BudgetedCorrectionPolicy(max_corrections=5),  # plenty of budget left
    )
    state = AgentState(user_input="hello")

    loop.run(state)

    assert state.status is AgentStatus.FAILED
    assert len(tool.calls) == 2  # first failure corrected, second is the repeat -> terminate
    assert len(state.corrections) == 1  # only the FIRST failure was granted a correction


def test_same_invalid_tool_input_repeated_terminates_on_the_second_failure() -> None:
    """The identical repetition guarantee for category D, not just E."""
    registry = ToolRegistry()
    tool = FlakyTool("date_tool", [ValueError("bad date")] * 5)
    registry.register(tool)
    decision_maker = RepeatToolDecisionMaker("date_tool")
    loop = AgentLoop(
        decision_maker=decision_maker,
        tool_registry=registry,
        max_iterations=10,
        correction_policy=BudgetedCorrectionPolicy(max_corrections=5),
    )
    state = AgentState(user_input="hello")

    loop.run(state)

    assert state.status is AgentStatus.FAILED
    assert len(tool.calls) == 2
    assert len(state.corrections) == 1


# ===========================================================================
# 2 — different tool, or different category, is NOT a repetition
# ===========================================================================

def test_two_different_tools_failing_are_each_corrected_independently() -> None:
    registry = ToolRegistry()
    tool_a = FlakyTool("tool_a", [ToolResult.fail("boom a"), ToolResult.ok("ok a")])
    tool_b = FlakyTool("tool_b", [ToolResult.fail("boom b"), ToolResult.ok("ok b")])
    registry.register(tool_a)
    registry.register(tool_b)
    decision_maker = MixedDecisionMaker(
        [
            AgentDecision.tool("tool_a", "x"),
            AgentDecision.tool("tool_b", "x"),
            AgentDecision.tool("tool_a", "x"),
            AgentDecision.tool("tool_b", "x"),
            AgentDecision.final("done"),
        ]
    )
    loop = AgentLoop(
        decision_maker=decision_maker,
        tool_registry=registry,
        max_iterations=10,
        correction_policy=BudgetedCorrectionPolicy(max_corrections=5),
    )
    state = AgentState(user_input="hello")

    loop.run(state)

    assert state.status is AgentStatus.COMPLETED
    assert len(state.corrections) == 2
    assert {n.category for n in state.corrections} == {FailureCategory.TOOL_EXECUTION_FAILED}


# ===========================================================================
# 3 — an intervening SUCCESS does not reset repetition for the same tool
# ===========================================================================

def test_an_intervening_success_on_a_different_tool_does_not_reset_repetition() -> None:
    """Deliberate design decision, locked in by this test: repetition
    compares against the LAST recorded correction, not the last N real
    iterations. A tool that keeps failing every time it is actually tried
    is still "not converging" even if the model did something else (and
    succeeded) in between attempts."""
    registry = ToolRegistry()
    web_search = AlwaysFailingTool("web_search")
    time_tool = FlakyTool("time_tool", [ToolResult.ok("12:00")])
    registry.register(web_search)
    registry.register(time_tool)
    decision_maker = MixedDecisionMaker(
        [
            AgentDecision.tool("web_search", "q"),  # fails -> corrected
            AgentDecision.tool("time_tool", None),  # succeeds, unrelated
            AgentDecision.tool("web_search", "q"),  # fails again -> REPEATED, terminate
        ]
    )
    loop = AgentLoop(
        decision_maker=decision_maker,
        tool_registry=registry,
        max_iterations=10,
        correction_policy=BudgetedCorrectionPolicy(max_corrections=5),
    )
    state = AgentState(user_input="hello")

    loop.run(state)

    assert state.status is AgentStatus.FAILED
    assert len(web_search.calls) == 2
    assert len(time_tool.calls) == 1
    assert len(state.corrections) == 1  # only the FIRST web_search failure was corrected


def test_a_different_failure_in_between_does_reset_the_repetition_check() -> None:
    """By contrast: if the INTERVENING event is itself a different
    correctable FAILURE (not a success), the later same-tool failure is
    compared against THAT intervening failure's signature, not the
    earlier one -- so it is not seen as a repeat of the original."""
    registry = ToolRegistry()
    web_search = FlakyTool("web_search", [ToolResult.fail("boom"), ToolResult.fail("boom again")])
    other_tool = FlakyTool("other_tool", [ToolResult.fail("boom other")])
    registry.register(web_search)
    registry.register(other_tool)
    decision_maker = MixedDecisionMaker(
        [
            AgentDecision.tool("web_search", "q"),  # fails -> corrected (sig: ...web_search)
            AgentDecision.tool("other_tool", "q"),  # fails -> corrected (sig: ...other_tool, different)
            AgentDecision.tool("web_search", "q"),  # fails again -> compared to other_tool's sig -> NOT a repeat
            AgentDecision.final("done"),
        ]
    )
    loop = AgentLoop(
        decision_maker=decision_maker,
        tool_registry=registry,
        max_iterations=10,
        correction_policy=BudgetedCorrectionPolicy(max_corrections=5),
    )
    state = AgentState(user_input="hello")

    loop.run(state)

    assert state.status is AgentStatus.COMPLETED
    assert len(state.corrections) == 3


# ===========================================================================
# 4 — include_tool_error_text: full data flow, both settings
# ===========================================================================

def test_tool_error_text_reaches_the_prompt_only_when_explicitly_enabled() -> None:
    """The strict default (False): a tool's own error text, even though
    AgentLoop captures it as Failure.detail, must NEVER reach the model's
    prompt."""
    registry = ToolRegistry()
    tool = FlakyTool(
        "web_search",
        [ValueError("Missing TAVILY_API_KEY configuration. Set it in the .env file."), ToolResult.ok("ok")],
    )
    registry.register(tool)
    decision_maker_llm = FakeLLM([_tool_json("web_search", "q"), _final_json("done")])
    decision_maker = LLMDecisionMaker(llm_client=decision_maker_llm, tool_registry=registry)
    loop = AgentLoop(
        decision_maker=decision_maker,
        tool_registry=registry,
        max_iterations=5,
        correction_policy=BudgetedCorrectionPolicy(include_tool_error_text=False),
    )
    state = AgentState(user_input="search something")

    loop.run(state)

    assert state.status is AgentStatus.COMPLETED
    second_prompt = decision_maker_llm.calls[1]["messages"][-1]["content"]
    assert "TAVILY_API_KEY" not in second_prompt
    assert "Missing TAVILY_API_KEY configuration" not in second_prompt


def test_tool_error_text_reaches_the_prompt_when_explicitly_enabled() -> None:
    """When a caller has explicitly opted in, the (bounded) tool detail
    DOES reach the prompt -- proving the flag actually does something end
    to end, not just at the policy-unit level."""
    registry = ToolRegistry()
    tool = FlakyTool("date_tool", [ValueError("Could not parse a supported date."), ToolResult.ok("ok")])
    registry.register(tool)
    decision_maker_llm = FakeLLM([_tool_json("date_tool", "bad date"), _final_json("done")])
    decision_maker = LLMDecisionMaker(llm_client=decision_maker_llm, tool_registry=registry)
    loop = AgentLoop(
        decision_maker=decision_maker,
        tool_registry=registry,
        max_iterations=5,
        correction_policy=BudgetedCorrectionPolicy(include_tool_error_text=True),
    )
    state = AgentState(user_input="what date")

    loop.run(state)

    assert state.status is AgentStatus.COMPLETED
    second_prompt = decision_maker_llm.calls[1]["messages"][-1]["content"]
    assert "Could not parse a supported date." in second_prompt


# ===========================================================================
# 5 — repeated failures are logged, not just terminated silently
# ===========================================================================

def test_repeated_failure_termination_is_logged(caplog) -> None:
    registry = ToolRegistry()
    tool = AlwaysFailingTool("web_search")
    registry.register(tool)
    decision_maker = RepeatToolDecisionMaker("web_search")
    loop = AgentLoop(
        decision_maker=decision_maker,
        tool_registry=registry,
        max_iterations=10,
        correction_policy=BudgetedCorrectionPolicy(max_corrections=5),
    )
    state = AgentState(user_input="hello")

    with caplog.at_level(logging.WARNING, logger="app.agent.loop"):
        loop.run(state)

    assert "correction.declined" in caplog.text
    assert "tool_execution_failed" in caplog.text


# ===========================================================================
# 6 — Milestone 21: dict/list tool_input normalization, end to end
# ===========================================================================
#
# The Milestone 20 diagnostic found 3/20 cases where the model emitted a
# JSON object/array for tool_input, which LLMDecisionMaker rejected as a
# DECISION_PARSE failure BEFORE any AgentDecision existed. That failure
# was already correctable via correction_policy (one retry). Milestone 21
# makes the parser normalize dict/list into a deterministic JSON string
# instead, so these cases succeed on the FIRST attempt -- no correction
# needed at all. These tests prove that end to end through the real
# AgentLoop + ToolExecutionGate-free pipeline (this file's existing
# pattern uses no gate), and prove the change does not weaken the
# UNRELATED INVALID_TOOL_INPUT correction path that already existed.

class RecordingValidatingTool:
    """A real tool double with its OWN validate() hook, exactly like the
    ones in tests/test_tool_execution_gate.py — used here to prove a
    NORMALIZED (dict->string) tool_input is still just ordinary input the
    tool is free to accept or reject; normalization never vouches for
    usability."""

    def __init__(self, name: str, *, rejects: bool = False):
        self.name = name
        self.description = f"tool '{name}'"
        self.input_schema: dict[str, str] = {"value": "string"}
        self.rejects = rejects
        self.calls: list[str | None] = []

    def execute(self, input: str | None = None) -> ToolResult:
        if self.rejects:
            raise ValueError("this tool rejects its input")
        self.calls.append(input)
        return ToolResult.ok({"received": input})


def test_dict_tool_input_no_longer_needs_correction_at_all() -> None:
    """The headline fix: a dict tool_input now parses cleanly on the FIRST
    attempt -- zero corrections, in contrast to the pre-Milestone-21
    behavior where this shape was a DECISION_PARSE failure."""
    tool = RecordingValidatingTool("web_search")
    registry = ToolRegistry()
    registry.register(tool)
    llm = FakeLLM([
        _tool_json_object("web_search", {"query": "latest AI news"}),
        _final_json("Recent developments include..."),
    ])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=registry)
    loop = AgentLoop(
        decision_maker=decision_maker,
        tool_registry=registry,
        correction_policy=BudgetedCorrectionPolicy(max_corrections=3),
    )
    state = AgentState(user_input="what's new in AI")

    loop.run(state)

    assert state.status is AgentStatus.COMPLETED
    assert state.corrections == []  # no correction was needed
    assert tool.calls == ['{"query": "latest AI news"}']


def test_normalized_tool_input_can_still_be_rejected_by_the_tool_and_still_corrects() -> None:
    """The UNRELATED INVALID_TOOL_INPUT correction path is unaffected:
    parsing succeeds (this method's own job, and the thing Milestone 21
    changed), but the tool itself still rejects the resulting string and
    that failure still triggers a normal correction cycle -- normalization
    never vouches for usability, it only stops a dict/list SHAPE from
    being rejected before the tool ever sees it."""
    registry = ToolRegistry()
    registry.register(RecordingValidatingTool("web_search", rejects=True))
    llm = FakeLLM([
        _tool_json_object("web_search", {"query": "x"}),  # parses fine now; tool rejects it
        _final_json("giving up on the tool, answering directly"),
    ])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=registry)
    loop = AgentLoop(
        decision_maker=decision_maker,
        tool_registry=registry,
        correction_policy=BudgetedCorrectionPolicy(max_corrections=3),
    )
    state = AgentState(user_input="search for x")

    loop.run(state)

    assert state.status is AgentStatus.COMPLETED
    assert len(state.corrections) == 1
    assert state.corrections[0].category == FailureCategory.INVALID_TOOL_INPUT


def _tool_json_object(tool_name: str, tool_input_obj: dict) -> str:
    import json as _json

    return _json.dumps({"action_type": "tool", "tool_name": tool_name, "tool_input": tool_input_obj})
