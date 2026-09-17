from __future__ import annotations

import pytest

from app.agent.episodic_memory import InMemoryEpisodicMemory
from app.agent.loop import AgentDecision
from app.agent.memory import InMemoryConversationMemory, InMemorySessionMemoryStore
from app.agent.orchestrator import AgentOrchestrator
from app.agent.plan import Plan, PlanStatus, PlanStep
from app.agent.plan_generator import PlanGenerationError
from app.agent.state import AgentState, AgentStatus
from app.agent.tool_registry import ToolRegistry
from app.tools.base import ToolResult


class FakeLLM:
    """Never actually used by these tests (a FakeDecisionMaker is injected
    instead), but AgentOrchestrator always constructs/stores an LLMClient-
    shaped object, so a harmless stand-in is supplied everywhere."""

    def generate(self, messages: list[dict[str, str]], *, json_mode: bool = False) -> str:
        raise AssertionError("The real LLM client must never be called in these tests")


class ScriptedDecisionMaker:
    """Returns a fixed, pre-scripted sequence of decisions, one per call."""

    def __init__(self, decisions: list[AgentDecision]):
        self._decisions = iter(decisions)
        self.calls = 0

    def decide(self, state: AgentState) -> AgentDecision:
        self.calls += 1
        try:
            return next(self._decisions)
        except StopIteration:
            raise AssertionError("ScriptedDecisionMaker ran out of scripted decisions") from None


class FakeTool:
    def __init__(self, name: str, description: str = "A fake tool for tests.", *, results: list[ToolResult] | None = None):
        self.name = name
        self.description = description
        self.calls: list[str | None] = []
        self._results = iter(results) if results is not None else None

    def execute(self, input: str | None = None) -> ToolResult:
        self.calls.append(input)
        if self._results is not None:
            return next(self._results)
        return ToolResult.ok(f"handled: {input}")


def _build_orchestrator(decision_maker, *, tools: list[FakeTool] | None = None, max_iterations: int = 5):
    registry = ToolRegistry()
    for tool in tools or []:
        registry.register(tool)
    orchestrator = AgentOrchestrator(
        llm_client=FakeLLM(),
        tool_registry=registry,
        decision_maker=decision_maker,
        max_iterations=max_iterations,
    )
    return orchestrator


# ---------------------------------------------------------------------------
# Simple question: straight to FINAL, no tool involved.
# ---------------------------------------------------------------------------

def test_final_decision_produces_a_completed_result_with_the_answer() -> None:
    decision_maker = ScriptedDecisionMaker([AgentDecision.final("Backpropagation trains neural nets.")])
    orchestrator = _build_orchestrator(decision_maker)

    result = orchestrator.process("What is backpropagation?")

    assert result.status is AgentStatus.COMPLETED
    assert result.answer == "Backpropagation trains neural nets."
    assert result.steps == 1
    assert result.tool_calls == []
    assert result.observations == []
    assert result.errors == []


# ---------------------------------------------------------------------------
# Tool-required task: TOOL -> Observation -> FINAL.
# ---------------------------------------------------------------------------

def test_tool_then_final_executes_the_correct_tool_and_completes() -> None:
    tool = FakeTool("web_search")
    decision_maker = ScriptedDecisionMaker([
        AgentDecision.tool("web_search", "current gold price"),
        AgentDecision.final("The current gold price is approximately $X."),
    ])
    orchestrator = _build_orchestrator(decision_maker, tools=[tool])

    result = orchestrator.process("What is the current gold price?")

    assert result.status is AgentStatus.COMPLETED
    assert result.answer == "The current gold price is approximately $X."
    assert tool.calls == ["current gold price"]
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].tool_name == "web_search"
    assert len(result.observations) == 1
    assert result.observations[0].success is True


# ---------------------------------------------------------------------------
# Multi-step task: TOOL 1 -> Observation -> TOOL 2 -> Observation -> FINAL.
# ---------------------------------------------------------------------------

def test_multiple_tools_execute_in_sequence_before_final() -> None:
    tool_a = FakeTool("tool_a")
    tool_b = FakeTool("tool_b")
    decision_maker = ScriptedDecisionMaker([
        AgentDecision.tool("tool_a", "first"),
        AgentDecision.tool("tool_b", "second"),
        AgentDecision.final("done"),
    ])
    orchestrator = _build_orchestrator(decision_maker, tools=[tool_a, tool_b])

    result = orchestrator.process("multi-step task")

    assert result.status is AgentStatus.COMPLETED
    assert tool_a.calls == ["first"]
    assert tool_b.calls == ["second"]
    assert result.steps == 3
    assert len(result.tool_calls) == 2
    assert len(result.observations) == 2


# ---------------------------------------------------------------------------
# Unknown tool -> deterministic failure, graceful answer, no exception.
# ---------------------------------------------------------------------------

def test_unregistered_tool_produces_a_failed_result_with_a_graceful_answer() -> None:
    """Step 17 hardening (F8): the user-facing answer is now a fixed,
    generic sentence — it must NOT echo internal detail like the invented
    tool name. That detail is still fully captured in `result.errors`,
    which is application-internal data, not part of the HTTP response
    body (see app/main.py)."""
    decision_maker = ScriptedDecisionMaker([AgentDecision.tool("nonexistent_tool", "x")])
    orchestrator = _build_orchestrator(decision_maker, tools=[])

    result = orchestrator.process("do something")

    assert result.status is AgentStatus.FAILED
    assert "could not complete this request" in result.answer
    assert "nonexistent_tool" not in result.answer
    assert "nonexistent_tool" in result.errors[-1].message
    assert result.observations == []
    assert len(result.errors) == 1


# ---------------------------------------------------------------------------
# Tool failure -> ToolResult(success=False) -> Observation -> FAILED.
# ---------------------------------------------------------------------------

def test_tool_result_failure_produces_a_failed_result_with_a_graceful_answer() -> None:
    """Step 17 hardening (F8): a tool's own error text must not reach the
    user-facing answer either — it is exactly the kind of tool-authored
    detail (here, network/operational detail; elsewhere a credential
    config name) this hardening is meant to keep out of an HTTP response.
    It remains available via `result.errors` for internal diagnostics."""
    tool = FakeTool("web_search", results=[ToolResult.fail("Tavily request failed due to network or timeout")])
    decision_maker = ScriptedDecisionMaker([AgentDecision.tool("web_search", "x")])
    orchestrator = _build_orchestrator(decision_maker, tools=[tool])

    result = orchestrator.process("What is the current gold price?")

    assert result.status is AgentStatus.FAILED
    assert "could not complete this request" in result.answer
    assert "Tavily request failed due to network or timeout" not in result.answer
    assert "Tavily request failed due to network or timeout" in result.errors[-1].message
    assert len(result.observations) == 1
    assert result.observations[0].success is False


# ---------------------------------------------------------------------------
# max_iterations is enforced end-to-end through the orchestrator.
# ---------------------------------------------------------------------------

def test_max_iterations_is_enforced_and_produces_a_failed_result() -> None:
    tool = FakeTool("loopy")

    class AlwaysToolDecisionMaker:
        def decide(self, state: AgentState) -> AgentDecision:
            return AgentDecision.tool("loopy", "x")

    orchestrator = _build_orchestrator(AlwaysToolDecisionMaker(), tools=[tool], max_iterations=3)

    result = orchestrator.process("never-ending task")

    assert result.status is AgentStatus.FAILED
    assert result.steps == 3
    assert len(tool.calls) == 3
    # Step 17 hardening (F8): the generic user-facing answer no longer
    # embeds this detail; it is still fully captured in result.errors.
    assert "iteration" not in result.answer.lower()
    assert "iteration" in result.errors[-1].message.lower()


# ---------------------------------------------------------------------------
# Construction / DI / basic validation.
# ---------------------------------------------------------------------------

def test_orchestrator_registers_default_tools_when_none_are_injected() -> None:
    orchestrator = AgentOrchestrator(llm_client=FakeLLM())

    registered = {tool.name for tool in orchestrator.tools.list_tools()}
    assert registered == {"web_search", "time", "date"}


def test_orchestrator_accepts_a_prebuilt_tool_registry() -> None:
    registry = ToolRegistry()
    tool = FakeTool("only_tool")
    registry.register(tool)
    decision_maker = ScriptedDecisionMaker([AgentDecision.final("ok")])

    orchestrator = AgentOrchestrator(llm_client=FakeLLM(), tool_registry=registry, decision_maker=decision_maker)

    assert orchestrator.tools is registry


def test_empty_message_raises_value_error() -> None:
    orchestrator = _build_orchestrator(ScriptedDecisionMaker([AgentDecision.final("ok")]))

    with pytest.raises(ValueError):
        orchestrator.process("   ")


# ---------------------------------------------------------------------------
# Step 11: plan generation is opt-in. Every test above constructs an
# orchestrator with no plan_generator and is completely unaffected — this
# is the explicit regression check plus the new opt-in behavior.
# ---------------------------------------------------------------------------

class CapturingDecisionMaker:
    """Records every AgentState it was handed, so tests can inspect
    state.plan after the fact without AgentResult needing to expose it."""

    def __init__(self, decisions: list[AgentDecision]):
        self._decisions = iter(decisions)
        self.captured_states: list[AgentState] = []

    def decide(self, state: AgentState) -> AgentDecision:
        self.captured_states.append(state)
        return next(self._decisions)


class FakePlanGenerator:
    def __init__(self, plan: Plan | None = None, *, raise_error: bool = False):
        self._plan = plan
        self._raise_error = raise_error
        self.calls: list[str] = []

    def generate(self, user_input: str) -> Plan:
        self.calls.append(user_input)
        if self._raise_error:
            raise PlanGenerationError("simulated planner failure")
        return self._plan


def test_orchestrator_does_not_generate_a_plan_by_default() -> None:
    """Regression: no plan_generator injected -> plan generation never
    happens, exactly like before Step 11."""
    decision_maker = CapturingDecisionMaker([AgentDecision.final("ok")])
    orchestrator = _build_orchestrator(decision_maker)

    orchestrator.process("What is Python?")

    assert decision_maker.captured_states[0].plan is None


def test_orchestrator_generates_a_plan_when_plan_generator_is_injected() -> None:
    plan = Plan(steps=[PlanStep(1, "Explain what Python is")])
    plan_generator = FakePlanGenerator(plan)
    decision_maker = CapturingDecisionMaker([AgentDecision.final("Python is a programming language.")])
    registry = ToolRegistry()
    orchestrator = AgentOrchestrator(
        llm_client=FakeLLM(),
        tool_registry=registry,
        decision_maker=decision_maker,
        plan_generator=plan_generator,
    )

    orchestrator.process("What is Python?")

    assert plan_generator.calls == ["What is Python?"]
    assert decision_maker.captured_states[0].plan is plan  # same object, attached before AgentLoop runs


def test_generated_plan_is_attached_to_the_agent_state_passed_to_the_loop() -> None:
    """Part 18 items 1-3: orchestrator generates a plan, attaches it to
    AgentState, and AgentLoop (via the decision maker) sees that same state."""
    plan = Plan(steps=[PlanStep(1, "Search for AI news"), PlanStep(2, "Summarize it")])
    plan_generator = FakePlanGenerator(plan)
    tool = FakeTool("web_search")
    decision_maker = CapturingDecisionMaker([
        AgentDecision.tool("web_search", "AI news"),
        AgentDecision.final("done"),
    ])
    orchestrator = _build_orchestrator(decision_maker, tools=[tool])
    orchestrator.plan_generator = plan_generator  # opt in after construction, for this test's convenience

    result = orchestrator.process("Find and summarize AI news")

    assert result.status is AgentStatus.COMPLETED
    seen_plan = decision_maker.captured_states[0].plan
    assert seen_plan is plan
    assert seen_plan.status is PlanStatus.COMPLETED


def test_plan_generation_failure_falls_back_to_no_plan_gracefully() -> None:
    """A PlanGenerationError must not crash the request — it degrades to
    the already-supported plan=None case rather than failing the whole
    request over a planning-layer hiccup."""
    plan_generator = FakePlanGenerator(raise_error=True)
    decision_maker = CapturingDecisionMaker([AgentDecision.final("Python is a programming language.")])
    orchestrator = _build_orchestrator(decision_maker)
    orchestrator.plan_generator = plan_generator

    result = orchestrator.process("What is Python?")

    assert result.status is AgentStatus.COMPLETED
    assert result.answer == "Python is a programming language."
    assert decision_maker.captured_states[0].plan is None


def test_one_step_plan_executes_through_the_same_orchestrator_and_loop() -> None:
    """Part 21: a one-step plan for a simple request works through the same
    AgentLoop/DecisionMaker architecture — no separate execution path."""
    plan = Plan(steps=[PlanStep(1, "Explain what Python is")])
    plan_generator = FakePlanGenerator(plan)
    decision_maker = ScriptedDecisionMaker([AgentDecision.final("Python is a programming language.")])
    orchestrator = _build_orchestrator(decision_maker)
    orchestrator.plan_generator = plan_generator

    result = orchestrator.process("What is Python?")

    assert result.status is AgentStatus.COMPLETED
    assert result.answer == "Python is a programming language."
    assert isinstance(orchestrator, AgentOrchestrator)  # no SimpleRequestHandler or similar was introduced


def test_multi_step_plan_executes_steps_in_order_through_the_orchestrator() -> None:
    """Part 22: multi-step request — step 1 executes before step 2, and
    step 2 sees step 1's observation."""
    plan = Plan(steps=[PlanStep(1, "Search for AI news"), PlanStep(2, "Summarize it")])
    plan_generator = FakePlanGenerator(plan)
    tool = FakeTool("web_search", results=[ToolResult.ok([{"title": "AI News", "content": "GPT-5 released"}])])
    decision_maker = ScriptedDecisionMaker([
        AgentDecision.tool("web_search", "latest AI news"),
        AgentDecision.final("The latest AI news is about GPT-5's release."),
    ])
    orchestrator = _build_orchestrator(decision_maker, tools=[tool])
    orchestrator.plan_generator = plan_generator

    result = orchestrator.process("Find the latest AI news and summarize it")

    assert result.status is AgentStatus.COMPLETED
    assert tool.calls == ["latest AI news"]  # step 1 executed, step 2 needed no tool
    assert len(result.observations) == 1
    assert plan.status is PlanStatus.COMPLETED
    assert plan.get_step(1).status is PlanStatus.COMPLETED
    assert plan.get_step(2).status is PlanStatus.COMPLETED


# ---------------------------------------------------------------------------
# Step 12: conversation memory is opt-in. Every test above constructs an
# orchestrator with no memory and is completely unaffected — this is the
# explicit regression check (item 16) plus the new opt-in behavior.
# ---------------------------------------------------------------------------

def test_orchestrator_without_memory_remains_backward_compatible() -> None:
    """Item 16."""
    decision_maker = CapturingDecisionMaker([AgentDecision.final("Backpropagation trains neural nets.")])
    orchestrator = _build_orchestrator(decision_maker)

    result = orchestrator.process("What is backpropagation?")

    assert result.status is AgentStatus.COMPLETED
    assert decision_maker.captured_states[0].messages == []


def test_orchestrator_with_memory_receives_previous_messages() -> None:
    """Item 17."""
    memory = InMemoryConversationMemory()
    memory.add_user_message("my name is Alice")
    memory.add_assistant_message("Nice to meet you, Alice.")
    decision_maker = CapturingDecisionMaker([AgentDecision.final("Your name is Alice.")])
    registry = ToolRegistry()
    orchestrator = AgentOrchestrator(
        llm_client=FakeLLM(), tool_registry=registry, decision_maker=decision_maker, memory=memory
    )

    orchestrator.process("what is my name?")

    seen_messages = decision_maker.captured_states[0].messages
    assert seen_messages == [
        {"role": "user", "content": "my name is Alice"},
        {"role": "assistant", "content": "Nice to meet you, Alice."},
    ]


def test_successful_execution_commits_user_and_assistant_turn() -> None:
    """Item 18."""
    memory = InMemoryConversationMemory()
    decision_maker = ScriptedDecisionMaker([AgentDecision.final("Python is a programming language.")])
    registry = ToolRegistry()
    orchestrator = AgentOrchestrator(
        llm_client=FakeLLM(), tool_registry=registry, decision_maker=decision_maker, memory=memory
    )

    orchestrator.process("What is Python?")

    assert memory.get_messages() == [
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a programming language."},
    ]


def test_failed_execution_does_not_commit_a_completed_turn() -> None:
    """Item 19."""
    memory = InMemoryConversationMemory()
    decision_maker = ScriptedDecisionMaker([AgentDecision.tool("nonexistent_tool", "x")])
    registry = ToolRegistry()  # nothing registered -> deterministic failure
    orchestrator = AgentOrchestrator(
        llm_client=FakeLLM(), tool_registry=registry, decision_maker=decision_maker, memory=memory
    )

    result = orchestrator.process("do something")

    assert result.status is AgentStatus.FAILED
    assert memory.get_messages() == []  # nothing committed — neither the user turn nor the failure "answer"


def test_memory_persists_across_two_orchestrator_process_calls() -> None:
    """Items 20-21: the second request sees the first conversation turn."""
    memory = InMemoryConversationMemory()
    decision_maker = CapturingDecisionMaker([
        AgentDecision.final("Nice to meet you, Alice."),
        AgentDecision.final("Your name is Alice."),
    ])
    registry = ToolRegistry()
    orchestrator = AgentOrchestrator(
        llm_client=FakeLLM(), tool_registry=registry, decision_maker=decision_maker, memory=memory
    )

    first_result = orchestrator.process("my name is Alice")
    second_result = orchestrator.process("what is my name?")

    assert first_result.status is AgentStatus.COMPLETED
    assert second_result.status is AgentStatus.COMPLETED
    # The SECOND call's AgentState.messages must contain the FIRST turn.
    assert decision_maker.captured_states[1].messages == [
        {"role": "user", "content": "my name is Alice"},
        {"role": "assistant", "content": "Nice to meet you, Alice."},
    ]
    # And by now memory holds both full turns.
    assert memory.get_messages() == [
        {"role": "user", "content": "my name is Alice"},
        {"role": "assistant", "content": "Nice to meet you, Alice."},
        {"role": "user", "content": "what is my name?"},
        {"role": "assistant", "content": "Your name is Alice."},
    ]


def test_tool_observations_are_not_stored_as_conversation_messages() -> None:
    """Item 22."""
    memory = InMemoryConversationMemory()
    tool = FakeTool("web_search", results=[ToolResult.ok([{"title": "Result", "content": "some search content"}])])
    decision_maker = ScriptedDecisionMaker([
        AgentDecision.tool("web_search", "some query"),
        AgentDecision.final("Here is what I found."),
    ])
    orchestrator = _build_orchestrator(decision_maker, tools=[tool])
    orchestrator.memory = memory

    orchestrator.process("search for something")

    assert memory.get_messages() == [
        {"role": "user", "content": "search for something"},
        {"role": "assistant", "content": "Here is what I found."},
    ]
    # Only the two conversational turns — no tool_name, tool_input, or the
    # raw search content ever appears as its own memory entry.
    for message in memory.get_messages():
        assert "web_search" not in message["content"]
        assert "some search content" not in message["content"]


def test_planner_data_is_not_stored_as_conversation_messages() -> None:
    """Item 23."""
    memory = InMemoryConversationMemory()
    plan = Plan(steps=[PlanStep(1, "Explain what Python is")])
    plan_generator = FakePlanGenerator(plan)
    decision_maker = ScriptedDecisionMaker([AgentDecision.final("Python is a programming language.")])
    registry = ToolRegistry()
    orchestrator = AgentOrchestrator(
        llm_client=FakeLLM(),
        tool_registry=registry,
        decision_maker=decision_maker,
        plan_generator=plan_generator,
        memory=memory,
    )

    orchestrator.process("What is Python?")

    assert memory.get_messages() == [
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a programming language."},
    ]
    for message in memory.get_messages():
        assert "Explain what Python is" not in message["content"]  # the plan step description itself


# ---------------------------------------------------------------------------
# Part 11: deterministic cross-request test — the second execution must
# actually receive the first turn through ConversationMemory, proven via a
# CapturingDecisionMaker that records the exact AgentState it was handed.
# ---------------------------------------------------------------------------

def test_cross_request_second_execution_receives_first_turn_via_memory() -> None:
    memory = InMemoryConversationMemory()
    decision_maker = CapturingDecisionMaker([
        AgentDecision.final("Nice to meet you, Alice."),
        AgentDecision.final("Your name is Alice."),
    ])
    registry = ToolRegistry()
    orchestrator = AgentOrchestrator(
        llm_client=FakeLLM(), tool_registry=registry, decision_maker=decision_maker, memory=memory
    )

    first = orchestrator.process("my name is Alice")
    assert first.answer == "Nice to meet you, Alice."

    # Proof the SECOND decision actually SAW the first turn, not just that
    # memory happens to hold it afterward: inspect the exact AgentState the
    # decision maker was called with.
    assert decision_maker.captured_states[0].messages == []  # first call: no history yet

    second = orchestrator.process("what is my name?")
    assert second.answer == "Your name is Alice."
    assert decision_maker.captured_states[1].messages == [
        {"role": "user", "content": "my name is Alice"},
        {"role": "assistant", "content": "Nice to meet you, Alice."},
    ]


# ---------------------------------------------------------------------------
# Part 12: isolation — two independent memories, and two independent
# orchestrators using separate memories, never share state.
# ---------------------------------------------------------------------------

def test_two_orchestrators_with_separate_memories_are_isolated() -> None:
    memory_a = InMemoryConversationMemory()
    memory_b = InMemoryConversationMemory()

    decision_maker_a = ScriptedDecisionMaker([AgentDecision.final("Nice to meet you, Alice.")])
    decision_maker_b = ScriptedDecisionMaker([AgentDecision.final("Nice to meet you, Bob.")])

    orchestrator_a = AgentOrchestrator(
        llm_client=FakeLLM(), tool_registry=ToolRegistry(), decision_maker=decision_maker_a, memory=memory_a
    )
    orchestrator_b = AgentOrchestrator(
        llm_client=FakeLLM(), tool_registry=ToolRegistry(), decision_maker=decision_maker_b, memory=memory_b
    )

    orchestrator_a.process("my name is Alice")
    orchestrator_b.process("my name is Bob")

    assert memory_a.get_messages() == [
        {"role": "user", "content": "my name is Alice"},
        {"role": "assistant", "content": "Nice to meet you, Alice."},
    ]
    assert memory_b.get_messages() == [
        {"role": "user", "content": "my name is Bob"},
        {"role": "assistant", "content": "Nice to meet you, Bob."},
    ]
    assert orchestrator_a.memory is not orchestrator_b.memory


# ---------------------------------------------------------------------------
# Step 13, Part 14: deterministic cross-session test via
# InMemorySessionMemoryStore. Session A (Alice) and session B (Bob), each
# with two requests, verified via the EXACT AgentState.messages the second
# decision in each session actually saw — not just the final answers.
# ---------------------------------------------------------------------------

def test_two_sessions_via_session_store_never_cross_contaminate() -> None:
    session_store = InMemorySessionMemoryStore()
    decision_maker = CapturingDecisionMaker([
        AgentDecision.final("Nice to meet you, Alice."),  # session A, request 1
        AgentDecision.final("Nice to meet you, Bob."),  # session B, request 1
        AgentDecision.final("Your name is Alice."),  # session A, request 2
        AgentDecision.final("Your name is Bob."),  # session B, request 2
    ])
    registry = ToolRegistry()

    def _process(message: str, session_id: str):
        memory = session_store.get_memory(session_id)
        orchestrator = AgentOrchestrator(
            llm_client=FakeLLM(), tool_registry=registry, decision_maker=decision_maker, memory=memory
        )
        return orchestrator.process(message)

    _process("My name is Alice", "A")
    _process("My name is Bob", "B")
    result_a2 = _process("What is my name?", "A")
    result_b2 = _process("What is my name?", "B")

    assert result_a2.answer == "Your name is Alice."
    assert result_b2.answer == "Your name is Bob."

    # captured_states[2] is session A's 2nd call, [3] is session B's 2nd call
    session_a_second_history = decision_maker.captured_states[2].messages
    session_b_second_history = decision_maker.captured_states[3].messages

    assert session_a_second_history == [
        {"role": "user", "content": "My name is Alice"},
        {"role": "assistant", "content": "Nice to meet you, Alice."},
    ]
    assert session_b_second_history == [
        {"role": "user", "content": "My name is Bob"},
        {"role": "assistant", "content": "Nice to meet you, Bob."},
    ]
    # Explicit no-contamination check: neither session's history mentions
    # the other session's name anywhere.
    for message in session_a_second_history:
        assert "Bob" not in message["content"]
    for message in session_b_second_history:
        assert "Alice" not in message["content"]


# ---------------------------------------------------------------------------
# Step 14: episodic memory is opt-in. Every test above constructs an
# orchestrator with no episodic_memory and is completely unaffected — this
# is the explicit regression check (Part 19 item 1) plus the new behavior.
# ---------------------------------------------------------------------------

def test_orchestrator_without_episodic_memory_remains_backward_compatible() -> None:
    """Part 19 item 1."""
    decision_maker = ScriptedDecisionMaker([AgentDecision.final("Backpropagation trains neural nets.")])
    orchestrator = _build_orchestrator(decision_maker)

    result = orchestrator.process("What is backpropagation?")

    assert result.status is AgentStatus.COMPLETED
    assert orchestrator.episodic_memory is None


def test_episodic_memory_without_session_id_is_rejected_at_construction() -> None:
    """session_id is required whenever episodic_memory is supplied — see
    the orchestrator module docstring for why (Part 12/26 forbid adding
    session_id to AgentState, so the constructor is the only place left)."""
    episodic_memory = InMemoryEpisodicMemory()

    with pytest.raises(ValueError):
        AgentOrchestrator(
            llm_client=FakeLLM(),
            tool_registry=ToolRegistry(),
            decision_maker=ScriptedDecisionMaker([AgentDecision.final("ok")]),
            episodic_memory=episodic_memory,
        )


@pytest.mark.parametrize("bad_session_id", ["", "   "])
def test_episodic_memory_with_blank_session_id_is_rejected(bad_session_id: str) -> None:
    episodic_memory = InMemoryEpisodicMemory()

    with pytest.raises(ValueError):
        AgentOrchestrator(
            llm_client=FakeLLM(),
            tool_registry=ToolRegistry(),
            decision_maker=ScriptedDecisionMaker([AgentDecision.final("ok")]),
            episodic_memory=episodic_memory,
            session_id=bad_session_id,
        )


def test_successful_execution_creates_exactly_one_episodic_record() -> None:
    """Part 19 item 2."""
    episodic_memory = InMemoryEpisodicMemory()
    decision_maker = ScriptedDecisionMaker([AgentDecision.final("Python is a programming language.")])
    orchestrator = AgentOrchestrator(
        llm_client=FakeLLM(),
        tool_registry=ToolRegistry(),
        decision_maker=decision_maker,
        episodic_memory=episodic_memory,
        session_id="A",
    )

    orchestrator.process("What is Python?")

    assert len(episodic_memory.get_recent("A")) == 1


def test_failed_execution_creates_zero_episodic_records() -> None:
    """Part 19 item 3 / Part 14 / Part 21."""
    episodic_memory = InMemoryEpisodicMemory()
    decision_maker = ScriptedDecisionMaker([AgentDecision.tool("nonexistent_tool", "x")])
    orchestrator = AgentOrchestrator(
        llm_client=FakeLLM(),
        tool_registry=ToolRegistry(),  # nothing registered -> deterministic failure
        decision_maker=decision_maker,
        episodic_memory=episodic_memory,
        session_id="A",
    )

    result = orchestrator.process("do something")

    assert result.status is AgentStatus.FAILED
    assert episodic_memory.get_recent("A") == []


def test_failed_execution_leaves_conversation_memory_unchanged_and_records_no_episode() -> None:
    """Part 21: a fake DecisionMaker causing failure must not disturb
    ConversationMemory (Step 12 behavior) and must not create an episode."""
    memory = InMemoryConversationMemory()
    episodic_memory = InMemoryEpisodicMemory()
    decision_maker = ScriptedDecisionMaker([AgentDecision.tool("nonexistent_tool", "x")])
    orchestrator = AgentOrchestrator(
        llm_client=FakeLLM(),
        tool_registry=ToolRegistry(),
        decision_maker=decision_maker,
        memory=memory,
        episodic_memory=episodic_memory,
        session_id="A",
    )

    result = orchestrator.process("do something")

    assert result.status is AgentStatus.FAILED
    assert memory.get_messages() == []
    assert episodic_memory.get_recent("A") == []


def test_episodic_record_contains_correct_session_id() -> None:
    """Part 19 item 4."""
    episodic_memory = InMemoryEpisodicMemory()
    decision_maker = ScriptedDecisionMaker([AgentDecision.final("Nice to meet you, Alice.")])
    orchestrator = AgentOrchestrator(
        llm_client=FakeLLM(),
        tool_registry=ToolRegistry(),
        decision_maker=decision_maker,
        episodic_memory=episodic_memory,
        session_id="session-A",
    )

    orchestrator.process("My name is Alice")

    record = episodic_memory.get_recent("session-A")[0]
    assert record.session_id == "session-A"


def test_episodic_record_has_a_meaningful_event_type() -> None:
    """Part 19 item 5."""
    episodic_memory = InMemoryEpisodicMemory()
    decision_maker = ScriptedDecisionMaker([AgentDecision.final("ok")])
    orchestrator = AgentOrchestrator(
        llm_client=FakeLLM(),
        tool_registry=ToolRegistry(),
        decision_maker=decision_maker,
        episodic_memory=episodic_memory,
        session_id="A",
    )

    orchestrator.process("hello")

    record = episodic_memory.get_recent("A")[0]
    assert record.event_type == "conversation_completed"


def test_episodic_record_summary_is_deterministic_and_bounded() -> None:
    """Part 19 item 6 / Part 9: no LLM call, bounded length, derived only
    from the user's message and the final answer."""
    episodic_memory = InMemoryEpisodicMemory()
    long_message = "x" * 500
    long_answer = "y" * 500
    decision_maker = ScriptedDecisionMaker([AgentDecision.final(long_answer)])
    orchestrator = AgentOrchestrator(
        llm_client=FakeLLM(),
        tool_registry=ToolRegistry(),
        decision_maker=decision_maker,
        episodic_memory=episodic_memory,
        session_id="A",
    )

    orchestrator.process(long_message)

    record = episodic_memory.get_recent("A")[0]
    assert len(record.summary) < 250  # well below the raw 500+500 input
    assert "x" in record.summary and "y" in record.summary


def test_episodic_record_has_a_timezone_aware_timestamp() -> None:
    """Part 19 item 7."""
    episodic_memory = InMemoryEpisodicMemory()
    decision_maker = ScriptedDecisionMaker([AgentDecision.final("ok")])
    orchestrator = AgentOrchestrator(
        llm_client=FakeLLM(),
        tool_registry=ToolRegistry(),
        decision_maker=decision_maker,
        episodic_memory=episodic_memory,
        session_id="A",
    )

    orchestrator.process("hello")

    record = episodic_memory.get_recent("A")[0]
    assert record.timestamp.tzinfo is not None


def test_episodic_record_metadata_contains_no_hidden_internal_data() -> None:
    """Part 19 item 8 / Part 17: metadata stays small/structured — no
    prompts, tool payloads, or raw model output leak into it."""
    tool = FakeTool("web_search", results=[ToolResult.ok([{"title": "Result", "content": "secret payload text"}])])
    episodic_memory = InMemoryEpisodicMemory()
    decision_maker = ScriptedDecisionMaker([
        AgentDecision.tool("web_search", "some query"),
        AgentDecision.final("Here is what I found."),
    ])
    orchestrator = _build_orchestrator(decision_maker, tools=[tool])
    orchestrator.episodic_memory = episodic_memory
    orchestrator.session_id = "A"

    orchestrator.process("search for something")

    record = episodic_memory.get_recent("A")[0]
    assert set(record.metadata.keys()) == {"steps", "tool_count"}
    assert record.metadata["tool_count"] == 1
    for value in record.metadata.values():
        assert "secret payload text" not in str(value)
    assert "secret payload text" not in record.summary


def test_multiple_successful_requests_create_multiple_episodic_records() -> None:
    """Part 19 item 9."""
    episodic_memory = InMemoryEpisodicMemory()
    decision_maker = ScriptedDecisionMaker([
        AgentDecision.final("first answer"),
        AgentDecision.final("second answer"),
    ])
    orchestrator = AgentOrchestrator(
        llm_client=FakeLLM(),
        tool_registry=ToolRegistry(),
        decision_maker=decision_maker,
        episodic_memory=episodic_memory,
        session_id="A",
    )

    orchestrator.process("first request")
    orchestrator.process("second request")

    assert len(episodic_memory.get_recent("A", limit=10)) == 2


def test_separate_sessions_remain_isolated_in_episodic_memory() -> None:
    """Part 19 item 10."""
    episodic_memory = InMemoryEpisodicMemory()
    decision_maker_a = ScriptedDecisionMaker([AgentDecision.final("Nice to meet you, Alice.")])
    decision_maker_b = ScriptedDecisionMaker([AgentDecision.final("Nice to meet you, Bob.")])

    orchestrator_a = AgentOrchestrator(
        llm_client=FakeLLM(),
        tool_registry=ToolRegistry(),
        decision_maker=decision_maker_a,
        episodic_memory=episodic_memory,
        session_id="A",
    )
    orchestrator_b = AgentOrchestrator(
        llm_client=FakeLLM(),
        tool_registry=ToolRegistry(),
        decision_maker=decision_maker_b,
        episodic_memory=episodic_memory,
        session_id="B",
    )

    orchestrator_a.process("My name is Alice")
    orchestrator_b.process("My name is Bob")

    records_a = episodic_memory.get_recent("A")
    records_b = episodic_memory.get_recent("B")
    assert len(records_a) == 1 and len(records_b) == 1
    assert "Alice" in records_a[0].summary
    assert "Bob" in records_b[0].summary


# ---------------------------------------------------------------------------
# Part 20: cross-layer deterministic test — session_id-scoped episodic
# memory shared across two orchestrators (mirroring the Step 13 pattern of
# sharing one SessionMemoryStore across per-session orchestrators), then
# clear("A") leaves B untouched.
# ---------------------------------------------------------------------------

def test_cross_layer_episodic_isolation_and_clear_session() -> None:
    episodic_memory = InMemoryEpisodicMemory()

    def _process(message: str, session_id: str, answer: str):
        decision_maker = ScriptedDecisionMaker([AgentDecision.final(answer)])
        orchestrator = AgentOrchestrator(
            llm_client=FakeLLM(),
            tool_registry=ToolRegistry(),
            decision_maker=decision_maker,
            episodic_memory=episodic_memory,
            session_id=session_id,
        )
        return orchestrator.process(message)

    _process("My name is Alice.", "A", "Nice to meet you, Alice.")
    _process("My name is Bob.", "B", "Nice to meet you, Bob.")

    records_a = episodic_memory.get_recent("A")
    records_b = episodic_memory.get_recent("B")
    assert len(records_a) == 1
    assert len(records_b) == 1
    assert "Alice" in records_a[0].summary
    assert "Bob" not in records_a[0].summary
    assert "Bob" in records_b[0].summary
    assert "Alice" not in records_b[0].summary

    episodic_memory.clear("A")

    assert episodic_memory.get_recent("A") == []
    assert len(episodic_memory.get_recent("B")) == 1  # B untouched
