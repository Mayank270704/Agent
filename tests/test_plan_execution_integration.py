"""End-to-end tests for the Step 11 plan/execution integration:

    User request -> AgentOrchestrator -> PlanGenerator -> AgentState.plan
        -> AgentLoop -> LLMDecisionMaker -> AgentDecision -> Tool
        -> Observation -> next plan step -> ... -> Final answer

Every test except the one marked `@pytest.mark.integration` uses a
ScriptedLLM (a fake LLMClient) and fake tools — real AgentLoop, real
LLMDecisionMaker, real Plan/PlanStep, but no network, no Ollama, no Tavily.
The integration test talks to real local Ollama, skips automatically if
unreachable, and never executes web_search (so it can never call Tavily).
"""
from __future__ import annotations

import json

import pytest

from app.agent.decision_maker import LLMDecisionMaker
from app.agent.loop import ActionType, AgentDecision, AgentLoop
from app.agent.plan import Plan, PlanStatus, PlanStep
from app.agent.plan_generator import LLMPlanGenerator
from app.agent.state import AgentState, AgentStatus
from app.agent.tool_registry import ToolRegistry
from app.config import settings
from app.models.llm import LLMClient
from app.tools.base import ToolResult


class ScriptedLLM:
    """Returns a fixed sequence of raw text responses, one per generate()
    call. Records every call, including the exact messages and json_mode
    flag. No network involved."""

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
        self, name: str, description: str = "A fake tool for tests.", *, results: list[ToolResult] | None = None
    ):
        self.name = name
        self.description = description
        self.input_schema: dict[str, str] = {}
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


def _two_step_plan() -> Plan:
    return Plan(steps=[
        PlanStep(1, "Search for information"),
        PlanStep(2, "Summarize the information"),
    ])


# ---------------------------------------------------------------------------
# Part 19: multi-step state continuity. The second decision must see the
# original user request, the plan, the current plan step, the first tool
# call, and the first observation — and the SAME AgentState instance flows
# through the whole run (no fresh state is created between plan steps).
# ---------------------------------------------------------------------------

def test_multi_step_state_continuity_second_decision_sees_full_plan_context() -> None:
    registry = ToolRegistry()
    tool = FakeTool("web_search", results=[ToolResult.ok([{"title": "Result", "content": "found information"}])])
    registry.register(tool)
    llm = ScriptedLLM([
        _tool_json("web_search", "information query"),
        _final_json("Here is a summary of the information."),
    ])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=registry)
    loop = AgentLoop(decision_maker=decision_maker, tool_registry=registry, max_iterations=5)

    plan = _two_step_plan()
    state = AgentState(user_input="Search for information and summarize it.", plan=plan)

    result_state = loop.run(state)

    assert result_state is state  # no fresh AgentState was ever created

    second_prompt = _prompt(llm, 1)
    assert "Search for information and summarize it." in second_prompt  # original user request
    assert "1. Search for information" in second_prompt  # plan is visible
    assert "2. Summarize the information" in second_prompt
    assert "information query" in second_prompt  # first tool call
    assert "found information" in second_prompt  # first observation
    # step 1 is done, step 2 is now the current plan step for the 2nd decision
    assert "2. Summarize the information" in second_prompt.split("CURRENT PLAN STEP")[1]

    assert state.status is AgentStatus.COMPLETED
    assert plan.status is PlanStatus.COMPLETED


# ---------------------------------------------------------------------------
# Part 20: tool failure. Observation recorded per existing semantics, plan
# step not completed, plan becomes FAILED, AgentState becomes FAILED, no
# later plan step executes.
# ---------------------------------------------------------------------------

def test_tool_failure_fails_the_plan_and_never_reaches_later_steps() -> None:
    registry = ToolRegistry()
    tool = FakeTool("web_search", results=[ToolResult.fail("Tavily request failed due to network or timeout")])
    registry.register(tool)
    llm = ScriptedLLM([_tool_json("web_search", "information query")])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=registry)
    loop = AgentLoop(decision_maker=decision_maker, tool_registry=registry, max_iterations=5)

    plan = _two_step_plan()
    state = AgentState(user_input="Search for information and summarize it.", plan=plan)

    loop.run(state)

    assert len(state.observations) == 1
    assert state.observations[0].success is False
    assert state.observations[0].error == "Tavily request failed due to network or timeout"

    assert plan.get_step(1).status is PlanStatus.FAILED  # not completed
    assert plan.get_step(2).status is PlanStatus.PENDING  # never even attempted
    assert plan.status is PlanStatus.FAILED
    assert state.status is AgentStatus.FAILED
    assert tool.calls == ["information query"]  # only ever called once — step 2 never ran


# ---------------------------------------------------------------------------
# Part 21: simple one-step request. No artificial second execution
# architecture — the same AgentLoop/LLMDecisionMaker handles it.
# ---------------------------------------------------------------------------

def test_simple_one_step_request_executes_normally_and_returns_a_final_answer() -> None:
    registry = ToolRegistry()
    registry.register(FakeTool("web_search"))  # registered but never needed/called
    llm = ScriptedLLM([_final_json("Python is a high-level programming language.")])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=registry)
    loop = AgentLoop(decision_maker=decision_maker, tool_registry=registry, max_iterations=5)

    plan = Plan(steps=[PlanStep(1, "Explain what Python is")])
    state = AgentState(user_input="What is Python?", plan=plan)

    loop.run(state)

    assert state.status is AgentStatus.COMPLETED
    assert state.final_answer == "Python is a high-level programming language."
    assert plan.status is PlanStatus.COMPLETED
    assert plan.get_step(1).status is PlanStatus.COMPLETED
    # No tool call happened at all — the plan step was satisfied directly.
    assert registry.get("web_search").calls == []


# ---------------------------------------------------------------------------
# Part 22: multi-step request. Step 1 executes before step 2; step 2 sees
# step 1's observation; no step is skipped.
# ---------------------------------------------------------------------------

def test_multi_step_request_executes_steps_in_order_without_skipping() -> None:
    registry = ToolRegistry()
    tool = FakeTool(
        "web_search", results=[ToolResult.ok([{"title": "AI News", "content": "GPT-5 released"}])]
    )
    registry.register(tool)
    llm = ScriptedLLM([
        _tool_json("web_search", "latest AI news"),
        _final_json("The latest AI news is about GPT-5's release."),
    ])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=registry)
    loop = AgentLoop(decision_maker=decision_maker, tool_registry=registry, max_iterations=5)

    plan = Plan(steps=[
        PlanStep(1, "Search for the requested information"),
        PlanStep(2, "Summarize the returned information"),
    ])
    state = AgentState(user_input="Find the latest AI news and summarize it.", plan=plan)

    loop.run(state)

    assert tool.calls == ["latest AI news"]  # step 1's tool call, exactly once
    assert plan.get_step(1).status is PlanStatus.COMPLETED
    assert plan.get_step(2).status is PlanStatus.COMPLETED  # completed via FINAL, no tool needed
    assert plan.status is PlanStatus.COMPLETED
    assert state.status is AgentStatus.COMPLETED
    assert state.final_answer == "The latest AI news is about GPT-5's release."


# ---------------------------------------------------------------------------
# Part 8 (cross-check at the integration level): PlanStep.step_id and
# AgentState.step never get confused with each other.
# ---------------------------------------------------------------------------

def test_plan_step_ids_and_agent_state_step_stay_independent_end_to_end() -> None:
    """Uses deliberately non-sequential, non-matching step_id values (10,
    20) so any accidental conflation with AgentState.step's own 1, 2, 3...
    sequence would be immediately obvious."""
    registry = ToolRegistry()
    tool = FakeTool("web_search", results=[ToolResult.ok(["result"])])
    registry.register(tool)
    llm = ScriptedLLM([
        _tool_json("web_search", "query"),
        _final_json("done"),
    ])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=registry)
    loop = AgentLoop(decision_maker=decision_maker, tool_registry=registry, max_iterations=5)

    plan = Plan(steps=[PlanStep(10, "Search for information"), PlanStep(20, "Summarize the information")])
    state = AgentState(user_input="do two things", plan=plan)

    loop.run(state)

    assert [step.step_id for step in plan.steps] == [10, 20]  # untouched by AgentLoop's own counting
    assert state.step == 2  # 1 tool iteration + 1 final iteration — a completely separate count
    assert plan.status is PlanStatus.COMPLETED
    # The recorded ToolCall's own `step` field is AgentState.step (1), not
    # any PlanStep.step_id (10) — confirming the loop never substitutes one
    # for the other anywhere it records history.
    assert state.tool_calls[0].step == 1


# ---------------------------------------------------------------------------
# Part 24: optional real Ollama smoke test. Verifies only that the planner
# generates a valid plan and the decision maker can see plan context without
# crashing — never executes web_search, never calls Tavily, never asserts
# exact wording or claims multi-step reliability from one run.
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
def test_real_ollama_plan_and_decision_context_smoke() -> None:
    if not _ollama_is_available():
        pytest.skip(f"Ollama is not reachable at {settings.ollama_base_url}; skipping live smoke test.")

    llm_client = LLMClient(provider="ollama", model_name=settings.model_name, base_url=settings.ollama_base_url)

    plan_generator = LLMPlanGenerator(llm_client=llm_client)
    plan = plan_generator.generate("What is the latest AI news?")
    assert isinstance(plan, Plan)
    assert len(plan.steps) >= 1
    assert plan.status is PlanStatus.PENDING

    registry = ToolRegistry()  # no tools registered: cannot call Tavily even if it wanted to
    decision_maker = LLMDecisionMaker(llm_client=llm_client, tool_registry=registry)
    loop = AgentLoop(decision_maker=decision_maker, tool_registry=registry, max_iterations=3)

    state = AgentState(user_input="What is the latest AI news?", plan=plan)
    loop.run(state)  # must not crash regardless of what the model decides

    assert state.status in (AgentStatus.COMPLETED, AgentStatus.FAILED)
