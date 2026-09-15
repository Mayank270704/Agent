from __future__ import annotations

import json

import pytest

from app.agent.decision_maker import DecisionParseError, LLMDecisionMaker
from app.agent.loop import ActionType
from app.agent.plan import Plan, PlanStep
from app.agent.router import RoutingHint
from app.agent.state import AgentState
from app.agent.tool_registry import ToolRegistry
from app.tools.base import ToolResult


class FakeLLM:
    """Replays a fixed sequence of responses; records every call, including
    whether json_mode was requested. No network/Ollama/OpenAI involved."""

    def __init__(self, responses: list[str]):
        self.responses = iter(responses)
        self.calls: list[dict[str, object]] = []

    def generate(self, messages: list[dict[str, str]], *, json_mode: bool = False) -> str:
        self.calls.append({"messages": messages, "json_mode": json_mode})
        return next(self.responses)


class FakeTool:
    """A registry-only tool double. execute() must never be called by the
    decision maker — it raises loudly if it ever is."""

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: dict[str, str] | None = None,
        output_description: str = "",
    ):
        self.name = name
        self.description = description
        self.input_schema = input_schema if input_schema is not None else {}
        self.output_description = output_description

    def execute(self, input: str | None = None) -> ToolResult:
        raise AssertionError("LLMDecisionMaker must never execute a tool")


def _registry_with(*tools: FakeTool) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def _prompt_of(llm: FakeLLM, call_index: int = 0) -> str:
    return llm.calls[call_index]["messages"][0]["content"]


# ---------------------------------------------------------------------------
# 1/2. Valid decisions.
# ---------------------------------------------------------------------------

def test_valid_final_json_becomes_agent_decision_final() -> None:
    llm = FakeLLM([json.dumps({"action_type": "final", "final_answer": "42"})])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=ToolRegistry())

    decision = decision_maker.decide(AgentState(user_input="What is 6 * 7?"))

    assert decision.action_type is ActionType.FINAL
    assert decision.final_answer == "42"


def test_valid_tool_json_becomes_agent_decision_tool() -> None:
    registry = _registry_with(FakeTool("web_search", "Search the web."))
    llm = FakeLLM([json.dumps({
        "action_type": "tool", "tool_name": "web_search", "tool_input": "current AI news",
    })])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=registry)

    decision = decision_maker.decide(AgentState(user_input="What's the latest AI news?"))

    assert decision.action_type is ActionType.TOOL
    assert decision.tool_name == "web_search"
    assert decision.tool_input == "current AI news"


# ---------------------------------------------------------------------------
# Step 8A / Part 5, items 2-6: specific decision-reliability scenarios with
# the exact inputs called out in the spec.
# ---------------------------------------------------------------------------

def _standard_registry() -> ToolRegistry:
    return _registry_with(
        FakeTool("web_search", "Search the web for current information.", input_schema={"query": "string"}),
        FakeTool("time", "Provides the current local/system date and time."),
        FakeTool("date", "Determines the weekday for a specified date.", input_schema={"date": "string"}),
    )


def test_ordinary_static_question_returns_final() -> None:
    llm = FakeLLM([json.dumps({
        "action_type": "final", "final_answer": "Machine learning is learning patterns from data.",
    })])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=_standard_registry())

    decision = decision_maker.decide(AgentState(user_input="What is machine learning?"))

    assert decision.action_type is ActionType.FINAL


def test_current_information_question_returns_web_search_tool_decision() -> None:
    llm = FakeLLM([json.dumps({
        "action_type": "tool", "tool_name": "web_search", "tool_input": "latest AI news",
    })])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=_standard_registry())

    decision = decision_maker.decide(AgentState(user_input="What is the latest AI news?"))

    assert decision.action_type is ActionType.TOOL
    assert decision.tool_name == "web_search"


def test_specific_date_question_returns_date_tool_decision() -> None:
    llm = FakeLLM([json.dumps({
        "action_type": "tool", "tool_name": "date", "tool_input": "25 December 2026",
    })])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=_standard_registry())

    decision = decision_maker.decide(AgentState(user_input="What day is 25 December 2026?"))

    assert decision.action_type is ActionType.TOOL
    assert decision.tool_name == "date"


def test_current_time_question_returns_time_tool_decision() -> None:
    llm = FakeLLM([json.dumps({"action_type": "tool", "tool_name": "time", "tool_input": None})])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=_standard_registry())

    decision = decision_maker.decide(AgentState(user_input="What time is it?"))

    assert decision.action_type is ActionType.TOOL
    assert decision.tool_name == "time"


def test_unknown_calculator_tool_raises_decision_parse_error() -> None:
    """The exact scenario this whole reliability effort was written around:
    the model inventing a "calculator" tool that was never registered."""
    llm = FakeLLM([json.dumps({
        "action_type": "tool", "tool_name": "calculator", "tool_input": "2 + 2",
    })])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=_standard_registry())

    with pytest.raises(DecisionParseError, match="calculator"):
        decision_maker.decide(AgentState(user_input="What is 2 + 2?"))


# ---------------------------------------------------------------------------
# 3-8. Malformed / invalid model output must never silently become a decision.
# ---------------------------------------------------------------------------

def test_malformed_json_raises_decision_parse_error() -> None:
    llm = FakeLLM(["not valid json {"])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=ToolRegistry())

    with pytest.raises(DecisionParseError):
        decision_maker.decide(AgentState(user_input="hello"))


def test_missing_action_type_raises_decision_parse_error() -> None:
    llm = FakeLLM([json.dumps({"final_answer": "42"})])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=ToolRegistry())

    with pytest.raises(DecisionParseError):
        decision_maker.decide(AgentState(user_input="hello"))


def test_unknown_action_type_raises_decision_parse_error() -> None:
    llm = FakeLLM([json.dumps({"action_type": "plan", "final_answer": "42"})])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=ToolRegistry())

    with pytest.raises(DecisionParseError):
        decision_maker.decide(AgentState(user_input="hello"))


def test_tool_without_tool_name_raises_decision_parse_error() -> None:
    llm = FakeLLM([json.dumps({"action_type": "tool", "tool_input": "x"})])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=ToolRegistry())

    with pytest.raises(DecisionParseError):
        decision_maker.decide(AgentState(user_input="hello"))


def test_final_without_final_answer_raises_decision_parse_error() -> None:
    llm = FakeLLM([json.dumps({"action_type": "final"})])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=ToolRegistry())

    with pytest.raises(DecisionParseError):
        decision_maker.decide(AgentState(user_input="hello"))


def test_tool_referencing_unregistered_tool_raises_decision_parse_error() -> None:
    registry = _registry_with(FakeTool("time", "Current local time."))
    llm = FakeLLM([json.dumps({"action_type": "tool", "tool_name": "nonexistent_tool"})])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=registry)

    with pytest.raises(DecisionParseError):
        decision_maker.decide(AgentState(user_input="hello"))


# -- Additional edge cases ---------------------------------------------------

def test_action_type_case_is_normalized() -> None:
    """Observed in practice with llama3.2:3b: it sometimes emits "FINAL"
    instead of the requested lowercase "final". action_type is a small
    closed set where case carries no meaning, so this must still parse."""
    llm = FakeLLM([json.dumps({"action_type": "FINAL", "final_answer": "42"})])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=ToolRegistry())

    decision = decision_maker.decide(AgentState(user_input="hello"))

    assert decision.action_type is ActionType.FINAL
    assert decision.final_answer == "42"


def test_non_dict_json_raises_decision_parse_error() -> None:
    llm = FakeLLM([json.dumps(["action_type", "final"])])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=ToolRegistry())

    with pytest.raises(DecisionParseError):
        decision_maker.decide(AgentState(user_input="hello"))


def test_blank_tool_name_raises_decision_parse_error() -> None:
    llm = FakeLLM([json.dumps({"action_type": "tool", "tool_name": "   "})])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=ToolRegistry())

    with pytest.raises(DecisionParseError):
        decision_maker.decide(AgentState(user_input="hello"))


def test_blank_final_answer_raises_decision_parse_error() -> None:
    llm = FakeLLM([json.dumps({"action_type": "final", "final_answer": "   "})])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=ToolRegistry())

    with pytest.raises(DecisionParseError):
        decision_maker.decide(AgentState(user_input="hello"))


def test_non_string_tool_input_raises_decision_parse_error() -> None:
    registry = _registry_with(FakeTool("web_search", "Search the web."))
    llm = FakeLLM([json.dumps({"action_type": "tool", "tool_name": "web_search", "tool_input": 123})])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=registry)

    with pytest.raises(DecisionParseError):
        decision_maker.decide(AgentState(user_input="hello"))


def test_empty_llm_response_raises_decision_parse_error() -> None:
    llm = FakeLLM([""])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=ToolRegistry())

    with pytest.raises(DecisionParseError):
        decision_maker.decide(AgentState(user_input="hello"))


def test_tool_input_literal_null_string_is_treated_as_none() -> None:
    """Observed in practice with llama3.2:3b: it sometimes returns the
    literal string "null" (not JSON null) for a no-input tool. This must be
    normalized to None rather than passed through as literal text input."""
    registry = _registry_with(FakeTool("time", "Get the current local time."))
    llm = FakeLLM([json.dumps({"action_type": "tool", "tool_name": "time", "tool_input": "null"})])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=registry)

    decision = decision_maker.decide(AgentState(user_input="hello"))

    assert decision.tool_name == "time"
    assert decision.tool_input is None


def test_tool_decision_with_null_tool_input_is_allowed_for_no_input_tools() -> None:
    registry = _registry_with(FakeTool("time", "Get the current local time."))
    llm = FakeLLM([json.dumps({"action_type": "tool", "tool_name": "time", "tool_input": None})])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=registry)

    decision = decision_maker.decide(AgentState(user_input="hello"))

    assert decision.tool_name == "time"
    assert decision.tool_input is None


# ---------------------------------------------------------------------------
# 9. Tool metadata is dynamically included from ToolRegistry.
# ---------------------------------------------------------------------------

def test_tool_metadata_is_dynamically_included_in_the_prompt() -> None:
    """Part 5 test 1: name, description, input_schema, and output_description
    must all reach the prompt, built entirely from ToolRegistry.describe_all()."""
    registry = _registry_with(
        FakeTool(
            "web_search",
            "Search the web for current information.",
            input_schema={"query": "string"},
            output_description="A list of web search results.",
        ),
        FakeTool("time", "Get the current local time."),  # no input_schema/output_description
    )
    llm = FakeLLM([json.dumps({"action_type": "final", "final_answer": "ok"})])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=registry)

    decision_maker.decide(AgentState(user_input="hello"))

    prompt = _prompt_of(llm)
    assert "web_search" in prompt
    assert "Search the web for current information." in prompt
    assert '{"query": "string"}' in prompt  # input_schema
    assert "A list of web search results." in prompt  # output_description
    assert "time" in prompt
    assert "Get the current local time." in prompt


def test_tool_with_no_input_schema_or_output_description_is_handled_gracefully() -> None:
    """A tool with input_schema={} and no output_description must not crash
    or produce a malformed prompt — both are genuinely optional."""
    registry = _registry_with(FakeTool("time", "Get the current local time."))
    llm = FakeLLM([json.dumps({"action_type": "final", "final_answer": "ok"})])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=registry)

    decision_maker.decide(AgentState(user_input="hello"))

    prompt = _prompt_of(llm)
    assert "- time: Get the current local time." in prompt
    assert "input: (none)" in prompt
    assert "returns:" not in prompt  # no output_description on this tool -> no "returns:" line at all


def test_empty_registry_produces_a_prompt_with_no_tools_listed() -> None:
    llm = FakeLLM([json.dumps({"action_type": "final", "final_answer": "ok"})])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=ToolRegistry())

    decision_maker.decide(AgentState(user_input="hello"))

    assert "no tools are currently registered" in _prompt_of(llm)


# ---------------------------------------------------------------------------
# 10-13. AgentState context is represented in the prompt.
# ---------------------------------------------------------------------------

def test_agent_state_information_is_included_in_the_prompt() -> None:
    llm = FakeLLM([json.dumps({"action_type": "final", "final_answer": "ok"})])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=ToolRegistry())
    state = AgentState(user_input="What is the current gold price?")
    state.step = 2

    decision_maker.decide(state)

    prompt = _prompt_of(llm)
    assert "What is the current gold price?" in prompt
    assert '"step": 2' in prompt


def test_conversation_history_in_state_messages_reaches_the_prompt() -> None:
    """Step 12: AgentState.messages is where ConversationMemory's previous
    turns land (see app/agent/orchestrator.py) — no decision_maker.py
    changes were needed for this, since _format_history already serializes
    state.messages; this test proves that wiring actually works once the
    field is populated (previously nothing ever populated it)."""
    llm = FakeLLM([json.dumps({"action_type": "final", "final_answer": "Your name is Alice."})])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=ToolRegistry())
    state = AgentState(
        user_input="what is my name?",
        messages=[
            {"role": "user", "content": "my name is Alice"},
            {"role": "assistant", "content": "Nice to meet you, Alice."},
        ],
    )

    decision_maker.decide(state)

    prompt = _prompt_of(llm)
    assert "my name is Alice" in prompt
    assert "Nice to meet you, Alice." in prompt


def test_tool_observations_are_represented_in_the_prompt() -> None:
    registry = _registry_with(FakeTool("web_search", "Search the web."))
    llm = FakeLLM([json.dumps({"action_type": "final", "final_answer": "ok"})])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=registry)
    state = AgentState(user_input="hello")
    state.add_observation("web_search", success=True, data=["result one"])

    decision_maker.decide(state)

    prompt = _prompt_of(llm)
    assert "result one" in prompt
    assert '"tool_name": "web_search"' in prompt


def test_previous_tool_calls_are_represented_in_the_prompt() -> None:
    llm = FakeLLM([json.dumps({"action_type": "final", "final_answer": "ok"})])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=ToolRegistry())
    state = AgentState(user_input="hello")
    state.record_tool_call("web_search", "current gold price")

    decision_maker.decide(state)

    assert "current gold price" in _prompt_of(llm)


def test_observation_grounding_previous_tool_call_and_observation_reach_the_prompt() -> None:
    """Part 5 test 7: a prior web_search ToolCall + Observation, together,
    must both be visible in the prompt sent to the model for the NEXT
    decision — this is what lets the model ground FINAL in real evidence."""
    registry = _registry_with(FakeTool("web_search", "Search the web.", input_schema={"query": "string"}))
    llm = FakeLLM([json.dumps({"action_type": "final", "final_answer": "GPT-5 was recently released."})])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=registry)

    state = AgentState(user_input="What is the latest AI news?")
    state.record_tool_call("web_search", "latest AI news")
    state.add_observation(
        "web_search", success=True, data=[{"title": "AI News", "content": "GPT-5 released"}]
    )

    decision_maker.decide(state)

    prompt = _prompt_of(llm)
    assert "latest AI news" in prompt  # the prior tool_input
    assert "GPT-5 released" in prompt  # the prior observation's evidence
    assert '"tool_name": "web_search"' in prompt
    assert "real tool-returned evidence" in prompt  # explicit grounding framing (Part 4)


def test_final_decision_after_observation_is_returned_correctly() -> None:
    """Part 5 test 8: once a FakeLLM returns FINAL referencing prior
    evidence, the parsed AgentDecision must simply be FINAL with that answer
    — no separate answer-generation step is introduced."""
    registry = _registry_with(FakeTool("web_search", "Search the web."))
    llm = FakeLLM([json.dumps({
        "action_type": "final", "final_answer": "Based on the search results, GPT-5 was recently released.",
    })])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=registry)

    state = AgentState(user_input="What is the latest AI news?")
    state.record_tool_call("web_search", "latest AI news")
    state.add_observation("web_search", success=True, data=[{"content": "GPT-5 released"}])

    decision = decision_maker.decide(state)

    assert decision.action_type is ActionType.FINAL
    assert decision.final_answer == "Based on the search results, GPT-5 was recently released."


def test_prompt_explicitly_warns_against_unnecessary_web_search() -> None:
    """Part 5 test 9: the prompt must explicitly tell the model not to use
    web_search for ordinary static questions."""
    registry = _registry_with(FakeTool("web_search", "Search the web for current information."))
    llm = FakeLLM([json.dumps({"action_type": "final", "final_answer": "ok"})])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=registry)

    decision_maker.decide(AgentState(user_input="What is 2 + 2?"))

    prompt = _prompt_of(llm)
    assert "web_search" in prompt
    assert "ordinary static questions" in prompt
    normalized = " ".join(prompt.lower().split())  # collapse line-wrap whitespace before matching
    assert "do not call a tool just because one exists" in normalized


def test_dynamically_registered_tool_appears_in_prompt_without_decision_maker_changes() -> None:
    """Part 5 test 10: register an additional tool never referenced by
    LLMDecisionMaker's own code, and confirm its metadata shows up purely
    because it's in the registry — proving the tool listing is fully dynamic."""
    registry = _standard_registry()
    registry.register(FakeTool(
        "calculator_stub",
        "A brand-new tool this test invents on the fly.",
        input_schema={"expression": "string"},
        output_description="The computed numeric result.",
    ))
    llm = FakeLLM([json.dumps({"action_type": "final", "final_answer": "ok"})])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=registry)

    decision_maker.decide(AgentState(user_input="hello"))

    prompt = _prompt_of(llm)
    assert "calculator_stub" in prompt
    assert "A brand-new tool this test invents on the fly." in prompt
    assert '{"expression": "string"}' in prompt
    assert "The computed numeric result." in prompt


def test_recorded_errors_are_represented_in_the_prompt() -> None:
    llm = FakeLLM([json.dumps({"action_type": "final", "final_answer": "ok"})])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=ToolRegistry())
    state = AgentState(user_input="hello")
    state.record_error("Tavily request failed due to network or timeout")

    decision_maker.decide(state)

    assert "Tavily request failed due to network or timeout" in _prompt_of(llm)


def test_large_observation_data_is_truncated_in_the_prompt() -> None:
    llm = FakeLLM([json.dumps({"action_type": "final", "final_answer": "ok"})])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=ToolRegistry())
    state = AgentState(user_input="hello")
    state.add_observation("web_search", success=True, data="x" * 5000)

    decision_maker.decide(state)

    prompt = _prompt_of(llm)
    assert "...[truncated]" in prompt
    assert "x" * 5000 not in prompt


# ---------------------------------------------------------------------------
# json_mode / trust boundary.
# ---------------------------------------------------------------------------

def test_decision_maker_requests_json_mode_from_the_llm_client() -> None:
    llm = FakeLLM([json.dumps({"action_type": "final", "final_answer": "ok"})])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=ToolRegistry())

    decision_maker.decide(AgentState(user_input="hello"))

    assert llm.calls[0]["json_mode"] is True


def test_decision_maker_never_executes_the_selected_tool() -> None:
    tool = FakeTool("web_search", "Search the web.")
    registry = _registry_with(tool)
    llm = FakeLLM([json.dumps({"action_type": "tool", "tool_name": "web_search", "tool_input": "x"})])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=registry)

    # FakeTool.execute() raises AssertionError if the decision maker ever calls it.
    decision = decision_maker.decide(AgentState(user_input="hello"))

    assert decision.tool_name == "web_search"


# ---------------------------------------------------------------------------
# 14/15. Fake LLM makes tests deterministic; no real external API is called.
# ---------------------------------------------------------------------------

def test_decision_maker_is_deterministic_with_fake_llm() -> None:
    def build_and_decide():
        llm = FakeLLM([json.dumps({"action_type": "final", "final_answer": "42"})])
        decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=ToolRegistry())
        return decision_maker.decide(AgentState(user_input="hello"))

    first = build_and_decide()
    second = build_and_decide()

    assert first.final_answer == second.final_answer == "42"


# ---------------------------------------------------------------------------
# Step 8: Router-derived advisory hint in the prompt. The hint is a single
# extra line, never a bypass — the LLM's decision is still what's parsed and
# validated exactly as before.
# ---------------------------------------------------------------------------

class FakeRouter:
    """A duck-typed router double — proves DI works without depending on
    Router's actual regex logic."""

    def __init__(self, hint: RoutingHint):
        self._hint = hint
        self.calls: list[str] = []

    def classify_hint(self, user_message: str) -> RoutingHint:
        self.calls.append(user_message)
        return self._hint


def test_deterministic_tool_hint_is_included_in_the_prompt() -> None:
    llm = FakeLLM([json.dumps({"action_type": "final", "final_answer": "ok"})])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=ToolRegistry())

    decision_maker.decide(AgentState(user_input="What time is it?"))

    prompt = _prompt_of(llm)
    assert "ROUTING HINT" in prompt
    assert "deterministic date/time operation" in prompt


def test_tool_likely_hint_is_included_in_the_prompt() -> None:
    llm = FakeLLM([json.dumps({"action_type": "final", "final_answer": "ok"})])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=ToolRegistry())

    decision_maker.decide(AgentState(user_input="What happened on 12 September 2026?"))

    prompt = _prompt_of(llm)
    assert "ROUTING HINT" in prompt
    assert "likely requires a tool" in prompt


def test_general_hint_adds_no_routing_hint_text() -> None:
    llm = FakeLLM([json.dumps({"action_type": "final", "final_answer": "ok"})])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=ToolRegistry())

    decision_maker.decide(AgentState(user_input="What is 2 + 2?"))

    assert "ROUTING HINT" not in _prompt_of(llm)


def test_a_custom_router_can_be_injected_and_is_used_for_the_hint() -> None:
    llm = FakeLLM([json.dumps({"action_type": "final", "final_answer": "ok"})])
    fake_router = FakeRouter(RoutingHint.TOOL_LIKELY)
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=ToolRegistry(), router=fake_router)

    decision_maker.decide(AgentState(user_input="anything at all"))

    assert fake_router.calls == ["anything at all"]
    assert "likely requires a tool" in _prompt_of(llm)


def test_hint_is_advisory_only_llm_can_still_choose_final_despite_web_hint() -> None:
    """The hint nudges but never decides — a FakeRouter claiming this needs
    web search must not stop a FINAL decision from being parsed normally."""
    llm = FakeLLM([json.dumps({"action_type": "final", "final_answer": "42"})])
    fake_router = FakeRouter(RoutingHint.TOOL_LIKELY)
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=ToolRegistry(), router=fake_router)

    decision = decision_maker.decide(AgentState(user_input="What is 6 * 7?"))

    assert decision.action_type is ActionType.FINAL
    assert decision.final_answer == "42"


# ---------------------------------------------------------------------------
# Part 9: hint presence must not disturb anything else about decision-making.
# ---------------------------------------------------------------------------

def test_hint_presence_does_not_interfere_with_validation_metadata_or_observations() -> None:
    """Items 4-7 together: a DETERMINISTIC_TOOL hint is present in the
    prompt, but validation, tool metadata, and observation history all still
    work exactly as they did before the hint existed."""
    registry = _registry_with(
        FakeTool("date", "Determines the weekday for a specified date.", input_schema={"date": "string"}),
    )
    llm = FakeLLM([json.dumps({"action_type": "tool", "tool_name": "date", "tool_input": "25 December 2026"})])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=registry)

    state = AgentState(user_input="What day is 25 December 2026?")
    state.record_tool_call("date", "1 January 2025")
    state.add_observation("date", success=True, data={"day_of_week": "Wednesday"})

    decision = decision_maker.decide(state)

    prompt = _prompt_of(llm)
    assert "ROUTING HINT" in prompt  # the hint itself is present
    assert '{"date": "string"}' in prompt  # item 6: tool metadata still present
    assert "Wednesday" in prompt  # item 7: observation history still present
    # item 4: the model's actual JSON decision is still what gets validated/returned
    assert decision.action_type is ActionType.TOOL
    assert decision.tool_name == "date"
    assert decision.tool_input == "25 December 2026"


def test_hint_does_not_prevent_decision_parse_error_for_unregistered_tool() -> None:
    """Item 5: even with a TOOL_LIKELY hint present, an unregistered tool
    name is still rejected exactly as before."""
    llm = FakeLLM([json.dumps({"action_type": "tool", "tool_name": "calculator", "tool_input": "2 + 2"})])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=ToolRegistry())

    with pytest.raises(DecisionParseError):
        decision_maker.decide(AgentState(user_input="What is the latest calculator result?"))


def test_hint_remains_consistent_across_multiple_decide_calls_on_the_same_state() -> None:
    """Item 8 (decision-maker-level slice): user_input never changes across
    loop iterations, so the same hint must be produced on every decide()
    call for a given state — multi-step continuity shouldn't destabilize it."""
    registry = _registry_with(FakeTool("web_search", "Search the web."))
    llm = FakeLLM([
        json.dumps({"action_type": "tool", "tool_name": "web_search", "tool_input": "latest AI news"}),
        json.dumps({"action_type": "final", "final_answer": "done"}),
    ])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=registry)
    state = AgentState(user_input="What is the latest AI news?")

    state.step = 1
    decision_maker.decide(state)
    first_prompt = _prompt_of(llm, 0)

    state.step = 2
    state.record_tool_call("web_search", "latest AI news")
    state.add_observation("web_search", success=True, data=["result"])
    decision_maker.decide(state)
    second_prompt = _prompt_of(llm, 1)

    assert "ROUTING HINT" in first_prompt
    assert "ROUTING HINT" in second_prompt


# ---------------------------------------------------------------------------
# Step 11: plan context in the prompt. state.plan=None (the default,
# exercised by every test above) produces a byte-identical prompt to before
# Step 11 — verified explicitly below.
# ---------------------------------------------------------------------------

def _plan(*descriptions: str) -> Plan:
    return Plan(steps=[PlanStep(i, desc) for i, desc in enumerate(descriptions, start=1)])


def test_plan_none_omits_the_plan_section_entirely() -> None:
    llm = FakeLLM([json.dumps({"action_type": "final", "final_answer": "ok"})])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=ToolRegistry())

    decision_maker.decide(AgentState(user_input="hello"))  # plan defaults to None

    prompt = _prompt_of(llm)
    # The static rules text always mentions "a CURRENT PLAN STEP" generically
    # (see the "if shown above" wording) — what must be absent is the actual
    # dynamic section, which only appears when a plan exists.
    assert "CURRENT PLAN:" not in prompt
    assert "CURRENT PLAN STEP (your immediate task):" not in prompt


def test_current_plan_and_current_plan_step_are_visible_in_the_prompt() -> None:
    plan = _plan("Search for the latest AI news", "Summarize the findings")
    llm = FakeLLM([json.dumps({"action_type": "final", "final_answer": "ok"})])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=ToolRegistry())
    state = AgentState(user_input="Find and summarize the latest AI news", plan=plan)

    decision_maker.decide(state)

    prompt = _prompt_of(llm)
    assert "CURRENT PLAN:" in prompt
    assert "1. Search for the latest AI news" in prompt
    assert "2. Summarize the findings" in prompt
    assert "CURRENT PLAN STEP" in prompt
    # step 1 hasn't been started by the loop in this unit test, so it's
    # still the first (PENDING) step — this only asserts it is SHOWN,
    # not who is responsible for advancing it (that's AgentLoop's job).
    assert "1. Search for the latest AI news" in prompt.split("CURRENT PLAN STEP")[1]


def test_plan_context_does_not_remove_existing_tool_metadata() -> None:
    """Part 18 item 17: existing tool metadata remains available alongside
    plan context."""
    registry = _registry_with(
        FakeTool("web_search", "Search the web.", input_schema={"query": "string"}, output_description="Results."),
    )
    plan = _plan("Search for something")
    llm = FakeLLM([json.dumps({"action_type": "final", "final_answer": "ok"})])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=registry)
    state = AgentState(user_input="Search for something", plan=plan)

    decision_maker.decide(state)

    prompt = _prompt_of(llm)
    assert "CURRENT PLAN:" in prompt
    assert "web_search" in prompt
    assert '{"query": "string"}' in prompt
    assert "Results." in prompt


def test_plan_context_coexists_with_routing_hint() -> None:
    """Part 18 item 18: existing routing hint remains available if
    configured, alongside plan context."""
    plan = _plan("Find out what time it is")
    llm = FakeLLM([json.dumps({"action_type": "final", "final_answer": "ok"})])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=ToolRegistry())
    state = AgentState(user_input="What time is it?", plan=plan)  # deterministic hint input

    decision_maker.decide(state)

    prompt = _prompt_of(llm)
    assert "CURRENT PLAN:" in prompt
    assert "ROUTING HINT" in prompt


def test_plan_context_reflects_observations_and_previous_tool_calls_too() -> None:
    """Part 18 item 16: previous observations remain visible between plan
    steps, alongside the plan context itself."""
    plan = _plan("Search for AI news", "Summarize it")
    plan.start()
    plan.start_step(1)
    plan.complete_step(1)  # step 1 done, step 2 is now current

    llm = FakeLLM([json.dumps({"action_type": "final", "final_answer": "ok"})])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=ToolRegistry())
    state = AgentState(user_input="Find and summarize AI news", plan=plan)
    state.record_tool_call("web_search", "AI news")
    state.add_observation("web_search", success=True, data=[{"content": "GPT-5 released"}])

    decision_maker.decide(state)

    prompt = _prompt_of(llm)
    assert "GPT-5 released" in prompt  # prior observation still visible
    assert "CURRENT PLAN STEP" in prompt
    assert "2. Summarize it" in prompt.split("CURRENT PLAN STEP")[1]  # step 2 is now current


def test_plan_with_all_steps_completed_shows_no_current_step() -> None:
    plan = _plan("Only step")
    plan.start()
    plan.start_step(1)
    plan.complete_step(1)
    plan.complete()

    llm = FakeLLM([json.dumps({"action_type": "final", "final_answer": "ok"})])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=ToolRegistry())
    state = AgentState(user_input="Do the only thing", plan=plan)

    decision_maker.decide(state)

    prompt = _prompt_of(llm)
    assert "every plan step is already complete" in prompt


def test_plan_context_does_not_change_json_decision_validation() -> None:
    """Part 18 item 4: LLMDecisionMaker still validates the model's actual
    JSON decision even when plan context is present — an unregistered tool
    is still rejected."""
    plan = _plan("Do something")
    llm = FakeLLM([json.dumps({"action_type": "tool", "tool_name": "calculator", "tool_input": "2 + 2"})])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=ToolRegistry())
    state = AgentState(user_input="What is 2 + 2?", plan=plan)

    with pytest.raises(DecisionParseError):
        decision_maker.decide(state)
