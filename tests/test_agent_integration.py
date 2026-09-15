"""End-to-end tests for the real agent execution path introduced in Step 7:

    AgentState -> LLMDecisionMaker -> AgentDecision -> AgentLoop ->
    ToolRegistry -> Tool -> ToolResult -> Observation -> AgentState -> ...

Every test in this file except the one marked `@pytest.mark.integration` uses
a ScriptedLLM (a fake LLMClient) and fake tools — no real Ollama or Tavily
call happens in the default suite. The single integration-marked test talks
to a real local Ollama instance and is skipped automatically if it isn't
reachable; it deliberately registers zero tools, so it can never trigger a
Tavily request.
"""
from __future__ import annotations

import json

import pytest

from app.agent.decision_maker import DecisionParseError, LLMDecisionMaker
from app.agent.loop import AgentLoop
from app.agent.state import AgentState, AgentStatus
from app.agent.tool_registry import ToolRegistry
from app.config import settings
from app.models.llm import LLMClient
from app.tools.base import ToolResult


class ScriptedLLM:
    """A fake LLMClient: returns a fixed sequence of raw text responses, one
    per generate() call. Records every call, including the exact messages
    sent and whether json_mode was requested. No network involved."""

    def __init__(self, responses: list[str]):
        self.responses = iter(responses)
        self.calls: list[dict[str, object]] = []

    def generate(self, messages: list[dict[str, str]], *, json_mode: bool = False) -> str:
        self.calls.append({"messages": messages, "json_mode": json_mode})
        try:
            return next(self.responses)
        except StopIteration:
            raise AssertionError("ScriptedLLM ran out of scripted responses") from None


class FakeTool:
    def __init__(
        self,
        name: str,
        description: str = "A fake tool for tests.",
        *,
        input_schema: dict[str, str] | None = None,
        results: list[ToolResult] | None = None,
    ):
        self.name = name
        self.description = description
        self.input_schema = input_schema if input_schema is not None else {}
        self.calls: list[str | None] = []
        self._results = iter(results) if results is not None else None

    def execute(self, input: str | None = None) -> ToolResult:
        self.calls.append(input)
        if self._results is not None:
            return next(self._results)
        return ToolResult.ok(f"handled: {input}")


def _final_json(answer: str) -> str:
    return json.dumps({"action_type": "final", "final_answer": answer})


def _tool_json(name: str, tool_input: str | None) -> str:
    return json.dumps({"action_type": "tool", "tool_name": name, "tool_input": tool_input})


def _prompt(llm: ScriptedLLM, call_index: int) -> str:
    return llm.calls[call_index]["messages"][0]["content"]


# ---------------------------------------------------------------------------
# 1. Full pipeline wiring: real AgentLoop + real LLMDecisionMaker + real
# ToolRegistry, only the LLM client and the tools are fakes.
# ---------------------------------------------------------------------------

def test_full_pipeline_state_decision_action_observation_decision_final() -> None:
    registry = ToolRegistry()
    fake_tool = FakeTool("fake_tool", results=[ToolResult.ok("tool output")])
    registry.register(fake_tool)

    llm = ScriptedLLM([
        _tool_json("fake_tool", "example"),
        _final_json("Task completed."),
    ])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=registry)
    loop = AgentLoop(decision_maker=decision_maker, tool_registry=registry, max_iterations=5)

    state = AgentState(user_input="Do the thing")
    result_state = loop.run(state)

    assert result_state is state  # the same AgentState instance flows through the whole loop
    assert state.status is AgentStatus.COMPLETED
    assert state.final_answer == "Task completed."

    assert fake_tool.calls == ["example"]
    assert len(state.tool_calls) == 1
    assert state.tool_calls[0].tool_name == "fake_tool"
    assert state.tool_calls[0].tool_input == "example"
    assert len(state.observations) == 1
    assert state.observations[0].success is True
    assert state.observations[0].data == "tool output"

    # Every LLM call uses the message shape LLMClient.generate() already
    # expects, and requests structured JSON output.
    for call in llm.calls:
        assert call["json_mode"] is True
        messages = call["messages"]
        assert isinstance(messages, list) and len(messages) == 1
        assert messages[0]["role"] == "user"
        assert isinstance(messages[0]["content"], str)


# ---------------------------------------------------------------------------
# A. Simple question: LLM -> FINAL, no tool call at all.
# ---------------------------------------------------------------------------

def test_scenario_a_simple_question_goes_straight_to_final() -> None:
    registry = ToolRegistry()
    registry.register(FakeTool("web_search"))
    llm = ScriptedLLM([_final_json("Backpropagation trains neural networks via gradient descent.")])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=registry)
    loop = AgentLoop(decision_maker=decision_maker, tool_registry=registry)

    state = AgentState(user_input="What is backpropagation?")
    loop.run(state)

    assert state.status is AgentStatus.COMPLETED
    assert state.final_answer == "Backpropagation trains neural networks via gradient descent."
    assert state.tool_calls == []
    assert llm.calls[0]["json_mode"] is True


# ---------------------------------------------------------------------------
# B. Tool-required task: TOOL -> Observation -> FINAL.
# ---------------------------------------------------------------------------

def test_scenario_b_tool_required_task_completes_after_one_observation() -> None:
    registry = ToolRegistry()
    web_search = FakeTool("web_search", results=[ToolResult.ok([{"title": "Gold price today", "content": "$2400/oz"}])])
    registry.register(web_search)
    llm = ScriptedLLM([
        _tool_json("web_search", "current gold price"),
        _final_json("The current gold price is about $2400/oz."),
    ])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=registry)
    loop = AgentLoop(decision_maker=decision_maker, tool_registry=registry)

    state = AgentState(user_input="What is the current gold price?")
    loop.run(state)

    assert web_search.calls == ["current gold price"]
    assert state.status is AgentStatus.COMPLETED
    assert state.final_answer == "The current gold price is about $2400/oz."
    assert len(state.observations) == 1


# ---------------------------------------------------------------------------
# C. Multi-step task: TOOL 1 -> Observation -> TOOL 2 -> Observation -> FINAL.
# ---------------------------------------------------------------------------

def test_scenario_c_multi_step_task_chains_two_tools_before_final() -> None:
    registry = ToolRegistry()
    tool_one = FakeTool("tool_one", results=[ToolResult.ok("result one")])
    tool_two = FakeTool("tool_two", results=[ToolResult.ok("result two")])
    registry.register(tool_one)
    registry.register(tool_two)
    llm = ScriptedLLM([
        _tool_json("tool_one", "step one input"),
        _tool_json("tool_two", "step two input"),
        _final_json("Combined result: result one + result two."),
    ])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=registry)
    loop = AgentLoop(decision_maker=decision_maker, tool_registry=registry, max_iterations=5)

    state = AgentState(user_input="Do a two-step task")
    loop.run(state)

    assert tool_one.calls == ["step one input"]
    assert tool_two.calls == ["step two input"]
    assert state.status is AgentStatus.COMPLETED
    assert state.step == 3
    assert len(state.tool_calls) == 2
    assert len(state.observations) == 2


# ---------------------------------------------------------------------------
# D. Unknown tool chosen by the model -> validation failure -> deterministic
# failure (never executed).
# ---------------------------------------------------------------------------

def test_scenario_d_unknown_tool_chosen_by_llm_fails_deterministically() -> None:
    registry = ToolRegistry()  # nothing registered
    llm = ScriptedLLM([_tool_json("made_up_tool", "x")])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=registry)
    loop = AgentLoop(decision_maker=decision_maker, tool_registry=registry)

    state = AgentState(user_input="Do something")
    loop.run(state)

    assert state.status is AgentStatus.FAILED
    assert state.observations == []  # never executed — rejected before it could run
    assert any("made_up_tool" in error.message for error in state.errors)


# ---------------------------------------------------------------------------
# E. Tool failure: Tool executes -> ToolResult(success=False) -> Observation
# -> FAILED.
# ---------------------------------------------------------------------------

def test_scenario_e_tool_result_failure_leads_to_failed_state() -> None:
    registry = ToolRegistry()
    web_search = FakeTool("web_search", results=[ToolResult.fail("Tavily request failed due to network or timeout")])
    registry.register(web_search)
    llm = ScriptedLLM([_tool_json("web_search", "current gold price")])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=registry)
    loop = AgentLoop(decision_maker=decision_maker, tool_registry=registry)

    state = AgentState(user_input="What is the current gold price?")
    loop.run(state)

    assert state.status is AgentStatus.FAILED
    assert len(state.observations) == 1
    assert state.observations[0].success is False
    assert any("Tavily request failed" in error.message for error in state.errors)


# ---------------------------------------------------------------------------
# Step 8 scenarios A-J: decision reliability + advisory hint behavior for
# the specific cases observed to be unreliable with llama3.2:3b (see the
# Step 8 report for the real "calculator" hallucination this targets).
# ---------------------------------------------------------------------------

def test_step8_a_simple_arithmetic_goes_straight_to_final() -> None:
    registry = ToolRegistry()
    registry.register(FakeTool("web_search"))
    registry.register(FakeTool("time"))
    registry.register(FakeTool("date"))
    llm = ScriptedLLM([_final_json("4")])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=registry)
    loop = AgentLoop(decision_maker=decision_maker, tool_registry=registry)

    state = AgentState(user_input="What is 2 + 2?")
    loop.run(state)

    assert state.status is AgentStatus.COMPLETED
    assert state.tool_calls == []
    assert "ROUTING HINT" not in _prompt(llm, 0)  # GENERAL bucket: no hint noise


def test_step8_b_conceptual_explanation_goes_straight_to_final() -> None:
    registry = ToolRegistry()
    registry.register(FakeTool("web_search"))
    llm = ScriptedLLM([_final_json("Machine learning is learning patterns from data.")])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=registry)
    loop = AgentLoop(decision_maker=decision_maker, tool_registry=registry)

    state = AgentState(user_input="Explain what machine learning is.")
    loop.run(state)

    assert state.status is AgentStatus.COMPLETED
    assert state.tool_calls == []


def test_step8_c_todays_date_question_uses_a_time_tool() -> None:
    registry = ToolRegistry()
    time_tool = FakeTool("time", results=[ToolResult.ok({"date": "2026-09-15"})])
    registry.register(time_tool)
    llm = ScriptedLLM([
        _tool_json("time", None),
        _final_json("Today's date is 2026-09-15."),
    ])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=registry)
    loop = AgentLoop(decision_maker=decision_maker, tool_registry=registry)

    state = AgentState(user_input="What is today's date?")
    loop.run(state)

    assert time_tool.calls == [None]
    assert state.status is AgentStatus.COMPLETED
    assert "ROUTING HINT" in _prompt(llm, 0)
    assert "deterministic date/time operation" in _prompt(llm, 0)


def test_step8_d_specific_date_question_uses_the_date_tool() -> None:
    registry = ToolRegistry()
    date_tool = FakeTool("date", results=[ToolResult.ok({"day_of_week": "Friday"})])
    registry.register(date_tool)
    llm = ScriptedLLM([
        _tool_json("date", "25 December 2026"),
        _final_json("25 December 2026 is a Friday."),
    ])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=registry)
    loop = AgentLoop(decision_maker=decision_maker, tool_registry=registry)

    state = AgentState(user_input="What day is 25 December 2026?")
    loop.run(state)

    assert date_tool.calls == ["25 December 2026"]
    assert state.status is AgentStatus.COMPLETED
    assert "ROUTING HINT" in _prompt(llm, 0)


def test_step8_e_latest_news_request_uses_web_search() -> None:
    registry = ToolRegistry()
    web_search = FakeTool("web_search", results=[ToolResult.ok([{"title": "AI News", "content": "GPT-5 released"}])])
    registry.register(web_search)
    llm = ScriptedLLM([
        _tool_json("web_search", "latest AI news"),
        _final_json("The latest AI news is about GPT-5's release."),
    ])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=registry)
    loop = AgentLoop(decision_maker=decision_maker, tool_registry=registry)

    state = AgentState(user_input="What is the latest AI news?")
    loop.run(state)

    assert web_search.calls == ["latest AI news"]
    assert state.status is AgentStatus.COMPLETED
    # As of Step 8C, "latest" is one of the small web-intent keywords, so this
    # now gets an explicit TOOL_LIKELY routing hint (previously GENERAL).
    assert "ROUTING HINT" in _prompt(llm, 0)
    assert "likely requires a tool" in _prompt(llm, 0)


def test_step8_f_search_for_current_topic_uses_web_search() -> None:
    registry = ToolRegistry()
    web_search = FakeTool(
        "web_search", results=[ToolResult.ok([{"title": "NVIDIA", "content": "NVIDIA announced a new GPU."}])]
    )
    registry.register(web_search)
    llm = ScriptedLLM([
        _tool_json("web_search", "current NVIDIA news"),
        _final_json("NVIDIA recently announced a new GPU."),
    ])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=registry)
    loop = AgentLoop(decision_maker=decision_maker, tool_registry=registry)

    state = AgentState(user_input="Search for current NVIDIA news.")
    loop.run(state)

    assert web_search.calls == ["current NVIDIA news"]
    assert state.status is AgentStatus.COMPLETED


def test_step8_g_unknown_tool_like_the_observed_calculator_bug_raises_decision_parse_error() -> None:
    """Mirrors the real failure this step was written to fix: llama3.2:3b
    inventing a "calculator" tool for "What is 2 + 2?" when none is
    registered. The parser must still reject it deterministically."""
    registry = ToolRegistry()  # no calculator tool registered
    llm = ScriptedLLM([_tool_json("calculator", "2 + 2")])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=registry)

    with pytest.raises(DecisionParseError, match="calculator"):
        decision_maker.decide(AgentState(user_input="What is 2 + 2?"))


def test_step8_h_final_answer_synthesizes_the_tool_observation() -> None:
    registry = ToolRegistry()
    web_search = FakeTool(
        "web_search", results=[ToolResult.ok([{"title": "NVIDIA", "content": "NVIDIA announced a new GPU architecture."}])]
    )
    registry.register(web_search)
    llm = ScriptedLLM([
        _tool_json("web_search", "current NVIDIA news"),
        _final_json("NVIDIA recently announced a new GPU architecture."),
    ])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=registry)
    loop = AgentLoop(decision_maker=decision_maker, tool_registry=registry)

    state = AgentState(user_input="Search for current NVIDIA news.")
    loop.run(state)

    assert state.status is AgentStatus.COMPLETED
    assert "GPU architecture" in state.final_answer
    # The second prompt (which produced the final answer) must have carried
    # the tool's observation content forward — this is what "synthesis" means
    # in this architecture: the model sees the real evidence before FINAL.
    assert "NVIDIA announced a new GPU architecture." in _prompt(llm, 1)


def test_step8_i_multi_step_state_continuity_across_three_iterations() -> None:
    registry = ToolRegistry()
    tool_a = FakeTool("tool_a", results=[ToolResult.ok("result from tool a")])
    tool_b = FakeTool("tool_b", results=[ToolResult.ok("result from tool b")])
    registry.register(tool_a)
    registry.register(tool_b)
    llm = ScriptedLLM([
        _tool_json("tool_a", "input a"),
        _tool_json("tool_b", "input b"),
        _final_json("combined answer"),
    ])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=registry)
    loop = AgentLoop(decision_maker=decision_maker, tool_registry=registry, max_iterations=5)

    state = AgentState(user_input="multi-step task")
    loop.run(state)

    third_prompt = _prompt(llm, 2)
    assert "input a" in third_prompt
    assert "result from tool a" in third_prompt
    assert "input b" in third_prompt
    assert "result from tool b" in third_prompt
    assert state.status is AgentStatus.COMPLETED


def test_step8_j_no_unnecessary_web_search_for_a_static_question() -> None:
    registry = ToolRegistry()
    web_search = FakeTool("web_search")
    registry.register(web_search)
    llm = ScriptedLLM([_final_json("Water boils at 100C at sea level.")])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=registry)
    loop = AgentLoop(decision_maker=decision_maker, tool_registry=registry)

    state = AgentState(user_input="At what temperature does water boil?")
    loop.run(state)

    assert web_search.calls == []
    assert state.status is AgentStatus.COMPLETED


# ---------------------------------------------------------------------------
# State continuity: the SAME AgentState instance flows through the loop, and
# the second decision's prompt must reflect what happened in the first.
# ---------------------------------------------------------------------------

def test_state_continuity_second_decision_sees_first_iterations_history() -> None:
    registry = ToolRegistry()
    web_search = FakeTool("web_search", results=[ToolResult.ok([{"title": "AI News", "content": "GPT-5 released"}])])
    registry.register(web_search)
    llm = ScriptedLLM([
        _tool_json("web_search", "latest AI news"),
        _final_json("The latest AI news is about GPT-5's release."),
    ])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=registry)
    loop = AgentLoop(decision_maker=decision_maker, tool_registry=registry)

    state = AgentState(user_input="What's the latest AI news?")

    # Sanity-check the pre-loop state.
    assert state.step == 0
    assert state.observations == []

    loop.run(state)

    # No second, hidden state object was created — same instance throughout.
    assert state.user_input == "What's the latest AI news?"

    first_prompt = _prompt(llm, 0)
    second_prompt = _prompt(llm, 1)

    # First decision is made with no prior history.
    assert '"tool_calls": []' in first_prompt
    assert '"observations": []' in first_prompt

    # Second decision's prompt must contain the first iteration's ToolCall
    # and Observation — proving the SAME state (not a fresh one) was passed
    # back into the decision maker.
    assert "latest AI news" in second_prompt
    assert "GPT-5 released" in second_prompt
    assert '"step": 1' in second_prompt  # the recorded step of the first tool call/observation

    assert state.step == 2
    assert len(state.tool_calls) == 1
    assert len(state.observations) == 1


# ---------------------------------------------------------------------------
# Optional live smoke test against the real local Ollama model. Skips
# automatically if Ollama isn't reachable. NOT part of the default suite
# (`python -m pytest -q -m "not integration"` never runs this).
# ---------------------------------------------------------------------------

def _ollama_is_available() -> bool:
    import urllib.error
    import urllib.request

    try:
        urllib.request.urlopen(f"{settings.ollama_base_url}/api/tags", timeout=2)
        return True
    except (urllib.error.URLError, OSError):
        return False


@pytest.mark.integration
def test_real_ollama_smoke_decision_without_tools() -> None:
    """Uses the existing LLMClient against the existing local Ollama model —
    no paid API, no OpenAI/Anthropic/Gemini. An EMPTY tool registry is used
    on purpose: with no tools to choose from, this test can never trigger a
    Tavily request, regardless of what the model decides."""
    if not _ollama_is_available():
        pytest.skip(f"Ollama is not reachable at {settings.ollama_base_url}; skipping live smoke test.")

    llm_client = LLMClient(
        provider="ollama",
        model_name=settings.model_name,
        base_url=settings.ollama_base_url,
    )
    registry = ToolRegistry()
    decision_maker = LLMDecisionMaker(llm_client=llm_client, tool_registry=registry)
    loop = AgentLoop(decision_maker=decision_maker, tool_registry=registry, max_iterations=3)

    state = AgentState(user_input="What is 2 + 2? Answer with just the number.")
    loop.run(state)

    # A real 3B model's structured-output reliability can vary, so this only
    # asserts the loop always terminates cleanly (COMPLETED or a deterministic
    # FAILED) rather than requiring a specific answer.
    assert state.status in (AgentStatus.COMPLETED, AgentStatus.FAILED)
    if state.status is AgentStatus.COMPLETED:
        assert state.final_answer
