"""Authoritative deterministic time/date routing.

Capability Assessment finding #2: llama3.2:3b missed the `time` tool for
"What time is it?" (76.5% missed-tool-call rate even with the advisory
routing hint active; 0% unnecessary calls). Router's LLM-free matcher
already recognizes that narrow class with certainty, so the application —
not the model — now selects the tool for it.

These tests prove the authority boundary in both directions: the model
cannot redirect a deterministic temporal request away from time/date, AND
the deterministic path cannot bypass any part of the Milestone
18/18-A/18-B/18-C security pipeline, skip validation, skip confirmation,
or acquire a privileged retry that Milestone 17 correction does not
already grant.

Fully offline: FakeLLM replays scripted JSON; no Ollama, no network.
"""
from __future__ import annotations

import json

import pytest

import app.main as main_module
from app.agent.loop import ActionType, AgentDecision, AgentLoop
from app.agent.decision_maker import LLMDecisionMaker
from app.agent.orchestrator import AgentOrchestrator
from app.agent.permissions import AllowlistPermissionPolicy, ExecutionContext
from app.agent.reliability import BudgetedCorrectionPolicy
from app.agent.router import Router
from app.agent.state import AgentState, AgentStatus
from app.agent.telemetry import EventEmitter, EventType, ListEventSink
from app.agent.tool_execution import ConfirmationRequiredError, PermissionDeniedError, ToolExecutionGate
from app.agent.tool_registry import ToolRegistry
from app.tools.base import ToolResult
from app.tools.date import DateTool
from app.tools.time import TimeTool


class FakeLLM:
    """Raises if consulted — lets a test prove the LLM was never called."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = iter(responses or [])
        self.calls: list[str] = []

    def generate(self, messages: list[dict[str, str]], *, json_mode: bool = False) -> str:
        self.calls.append(messages[-1]["content"])
        try:
            return next(self.responses)
        except StopIteration:
            raise AssertionError("FakeLLM was consulted more times than the test scripted") from None


class RecordingTool:
    def __init__(self, name: str, *, requires_confirmation: bool = False, result: ToolResult | None = None):
        self.name = name
        self.description = f"tool '{name}'"
        self.input_schema: dict[str, str] = {}
        self.requires_confirmation = requires_confirmation
        self.execute_count = 0
        self._result = result or ToolResult.ok({"ok": True})
        self.inputs: list[str | None] = []

    def execute(self, input: str | None = None) -> ToolResult:
        self.execute_count += 1
        self.inputs.append(input)
        return self._result


def _final_json(answer: str) -> str:
    return json.dumps({"action_type": "final", "final_answer": answer})


def _tool_json(tool_name: str, tool_input: str | None) -> str:
    return json.dumps({"action_type": "tool", "tool_name": tool_name, "tool_input": tool_input})


def _registry(*tools) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def _decision_maker(llm, registry, **kwargs) -> LLMDecisionMaker:
    return LLMDecisionMaker(
        llm_client=llm, tool_registry=registry, deterministic_temporal_routing=True, **kwargs
    )


# ===========================================================================
# Router accessor — adds no new matching, only exposes the existing verdict
# ===========================================================================

@pytest.mark.parametrize(
    "message,expected",
    [
        ("What time is it?", ("time", None)),
        ("What is today's date?", ("time", None)),
        ("What day is it?", ("time", None)),
        ("What day is 25 December 2026?", ("date", "25 december 2026")),
        ("What weekday is 1 January 2027?", ("date", "1 january 2027")),
    ],
)
def test_router_exposes_the_existing_deterministic_verdict(message: str, expected: tuple) -> None:
    assert Router.__new__(Router).deterministic_tool_route(message) == expected


@pytest.mark.parametrize(
    "message",
    [
        "What happened on 12 September 2026?",  # route="web" — NOT a local tool case
        "Explain today's date conceptually.",
        "What is the current price of Bitcoin?",
        "Tell me about current events.",
        "What are the latest developments in AI?",
        "What is a transformer in deep learning?",
        "How does Google Search work?",
        "",
        "   ",
    ],
)
def test_router_declines_everything_outside_the_deterministic_local_tool_class(message: str) -> None:
    assert Router.__new__(Router).deterministic_tool_route(message) is None


# ===========================================================================
# 1/2 — deterministic requests select the right tool, with no LLM call
# ===========================================================================

def test_deterministic_time_request_selects_the_time_tool_without_consulting_the_llm() -> None:
    llm = FakeLLM()  # any call raises
    decision_maker = _decision_maker(llm, _registry(TimeTool(), DateTool()))

    decision = decision_maker.decide(AgentState(user_input="What time is it?"))

    assert decision.action_type is ActionType.TOOL
    assert decision.tool_name == "time"
    assert decision.tool_input is None
    assert llm.calls == []  # the model was never consulted


def test_deterministic_date_request_selects_the_date_tool_with_the_matched_date() -> None:
    llm = FakeLLM()
    decision_maker = _decision_maker(llm, _registry(TimeTool(), DateTool()))

    decision = decision_maker.decide(AgentState(user_input="What day is 25 December 2026?"))

    assert decision.action_type is ActionType.TOOL
    assert decision.tool_name == "date"
    assert decision.tool_input == "25 december 2026"
    assert llm.calls == []


def test_the_routed_date_input_is_actually_parseable_by_the_real_date_tool() -> None:
    """The router lowercases its matched text; prove the real DateTool
    accepts exactly what deterministic routing hands it."""
    llm = FakeLLM()
    decision_maker = _decision_maker(llm, _registry(DateTool()))

    decision = decision_maker.decide(AgentState(user_input="What weekday is 1 January 2027?"))
    result = DateTool().execute(decision.tool_input)

    assert result.success is True
    assert result.data["day_of_week"] == "Friday"


# ===========================================================================
# 3 — deterministic routing reaches the gate (no direct tool execution)
# ===========================================================================

def test_deterministic_routing_reaches_the_tool_execution_gate() -> None:
    tool = RecordingTool("time")
    registry = _registry(tool)
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy({"time"}))
    llm = FakeLLM([_final_json("It is noon.")])  # only the FINAL turn is scripted
    sink = ListEventSink()
    loop = AgentLoop(
        decision_maker=_decision_maker(llm, registry),
        tool_registry=registry,
        tool_execution_gate=gate,
        event_emitter=EventEmitter(sink, request_id="r1"),
    )
    state = AgentState(user_input="What time is it?")

    loop.run(state)

    assert state.status is AgentStatus.COMPLETED
    assert tool.execute_count == 1
    # The full gate pipeline ran: authorization was evaluated and recorded.
    types = [e.event_type for e in sink.events]
    assert EventType.TOOL_PROPOSED in types
    assert EventType.TOOL_AUTHORIZED in types
    assert EventType.TOOL_EXECUTION_COMPLETED in types


def test_no_direct_tool_execution_bypass_exists_in_the_deterministic_path() -> None:
    """Structural proof: the deterministic path returns an ordinary
    AgentDecision and never touches a tool object. A registry whose tools
    would explode if executed directly is still safe, because only the
    gate ever calls execute()."""
    import ast
    import inspect
    import textwrap

    from app.agent import decision_maker as module

    tree = ast.parse(textwrap.dedent(inspect.getsource(module.LLMDecisionMaker._deterministic_temporal_decision)))
    function = tree.body[0]
    # Drop the docstring so this asserts on real code, never on prose.
    body = function.body[1:] if ast.get_docstring(function) else function.body
    code = "\n".join(ast.unparse(node) for node in body)

    assert ".execute(" not in code
    assert "self.tools.get(" not in code  # only has() — a membership check, never resolution
    assert "AgentDecision.tool(" in code


# ===========================================================================
# 4/5/6 — permission, validation and confirmation still apply
# ===========================================================================

def test_permission_policy_still_applies_to_a_deterministically_routed_tool() -> None:
    tool = RecordingTool("time")
    registry = _registry(tool)
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy(frozenset()))  # time NOT allowed
    llm = FakeLLM()
    loop = AgentLoop(
        decision_maker=_decision_maker(llm, registry), tool_registry=registry, tool_execution_gate=gate
    )
    state = AgentState(user_input="What time is it?")

    loop.run(state)

    assert state.status is AgentStatus.FAILED
    assert tool.execute_count == 0


def test_permission_denial_of_a_deterministic_tool_is_raised_by_the_gate_itself() -> None:
    tool = RecordingTool("time")
    gate = ToolExecutionGate(_registry(tool), AllowlistPermissionPolicy(frozenset()))

    with pytest.raises(PermissionDeniedError):
        gate.execute("time", None, ExecutionContext())

    assert tool.execute_count == 0


def test_confirmation_still_applies_to_a_deterministically_routed_tool() -> None:
    tool = RecordingTool("time", requires_confirmation=True)
    registry = _registry(tool)
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy({"time"}))
    llm = FakeLLM()
    loop = AgentLoop(
        decision_maker=_decision_maker(llm, registry),
        tool_registry=registry,
        tool_execution_gate=gate,
        execution_context=ExecutionContext(),  # nothing confirmed
    )
    state = AgentState(user_input="What time is it?")

    loop.run(state)

    assert state.status is AgentStatus.FAILED
    assert tool.execute_count == 0


def test_trusted_confirmation_still_permits_a_deterministically_routed_tool() -> None:
    tool = RecordingTool("time", requires_confirmation=True)
    registry = _registry(tool)
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy({"time"}))
    llm = FakeLLM([_final_json("It is noon.")])
    loop = AgentLoop(
        decision_maker=_decision_maker(llm, registry),
        tool_registry=registry,
        tool_execution_gate=gate,
        execution_context=ExecutionContext(confirmed_tools=frozenset({"time"})),
    )
    state = AgentState(user_input="What time is it?")

    loop.run(state)

    assert state.status is AgentStatus.COMPLETED
    assert tool.execute_count == 1


def test_validation_still_applies_to_a_deterministically_routed_tool() -> None:
    """A date tool whose validate() hook rejects the routed input must
    still block execution — deterministic routing chooses the tool, it
    does not vouch for the input."""

    class StrictDateTool(RecordingTool):
        def validate(self, input: str | None = None) -> None:
            raise ValueError("validation always rejects in this test")

    tool = StrictDateTool("date")
    registry = _registry(tool)
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy({"date"}))
    llm = FakeLLM()
    loop = AgentLoop(
        decision_maker=_decision_maker(llm, registry), tool_registry=registry, tool_execution_gate=gate
    )
    state = AgentState(user_input="What day is 25 December 2026?")

    loop.run(state)

    assert state.status is AgentStatus.FAILED
    assert tool.execute_count == 0


# ===========================================================================
# 7 — the model cannot override deterministic routing
# ===========================================================================

def test_model_cannot_redirect_a_deterministic_time_request_to_web_search() -> None:
    """The exact failure this change exists to fix: the model wants
    web_search, the application knows it is a local `time` request."""
    time_tool = RecordingTool("time")
    web_tool = RecordingTool("web_search")
    registry = _registry(time_tool, web_tool)
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy({"time", "web_search"}))
    # The model would pick web_search if it were ever asked. It isn't.
    llm = FakeLLM([_tool_json("web_search", "what time is it"), _final_json("done")])
    loop = AgentLoop(
        decision_maker=_decision_maker(llm, registry), tool_registry=registry, tool_execution_gate=gate,
        max_iterations=2,
    )
    state = AgentState(user_input="What time is it?")

    loop.run(state)

    assert time_tool.execute_count == 1  # the application's choice won
    assert state.tool_calls[0].tool_name == "time"


def test_model_cannot_redirect_a_deterministic_date_request_to_web_search() -> None:
    date_tool = RecordingTool("date")
    web_tool = RecordingTool("web_search")
    registry = _registry(date_tool, web_tool)
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy({"date", "web_search"}))
    llm = FakeLLM([_tool_json("web_search", "25 december 2026"), _final_json("done")])
    loop = AgentLoop(
        decision_maker=_decision_maker(llm, registry), tool_registry=registry, tool_execution_gate=gate,
        max_iterations=2,
    )
    state = AgentState(user_input="What day is 25 December 2026?")

    loop.run(state)

    assert date_tool.execute_count == 1
    assert state.tool_calls[0].tool_name == "date"


def test_deterministic_authority_is_limited_to_the_first_attempt() -> None:
    """Once the tool has run, the LLM regains full control — otherwise
    `decide()` would return the same tool forever and the loop could
    never reach a FINAL answer."""
    tool = RecordingTool("time")
    registry = _registry(tool)
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy({"time"}))
    llm = FakeLLM([_final_json("It is noon.")])
    loop = AgentLoop(
        decision_maker=_decision_maker(llm, registry), tool_registry=registry, tool_execution_gate=gate
    )
    state = AgentState(user_input="What time is it?")

    loop.run(state)

    assert state.status is AgentStatus.COMPLETED
    assert state.final_answer == "It is noon."
    assert tool.execute_count == 1  # exactly once, not once per iteration
    assert len(llm.calls) == 1  # the LLM was consulted only for the FINAL synthesis


# ===========================================================================
# 8/9 — ambiguous and non-temporal requests stay on the normal LLM path
# ===========================================================================

@pytest.mark.parametrize(
    "message",
    [
        "Explain today's date conceptually.",
        "What is the current price of Bitcoin?",
        "What happened on 12 September 2026?",
        "Tell me about current events.",
    ],
)
def test_ambiguous_temporal_requests_remain_on_the_normal_decision_path(message: str) -> None:
    llm = FakeLLM([_final_json("an ordinary model answer")])
    decision_maker = _decision_maker(llm, _registry(TimeTool(), DateTool()))

    decision = decision_maker.decide(AgentState(user_input=message))

    assert decision.action_type is ActionType.FINAL
    assert len(llm.calls) == 1  # the model WAS consulted


@pytest.mark.parametrize(
    "message",
    ["What is a transformer in deep learning?", "What is 25 + 17?", "How does Google Search work?"],
)
def test_non_temporal_questions_are_unchanged(message: str) -> None:
    llm = FakeLLM([_final_json("an ordinary model answer")])
    decision_maker = _decision_maker(llm, _registry(TimeTool(), DateTool()))

    decision = decision_maker.decide(AgentState(user_input=message))

    assert decision.action_type is ActionType.FINAL
    assert len(llm.calls) == 1


def test_an_unregistered_time_tool_falls_through_to_the_llm() -> None:
    """Deterministic routing must never name a tool this deployment has
    not registered — that would manufacture an UNKNOWN_TOOL failure."""
    llm = FakeLLM([_final_json("model answered instead")])
    decision_maker = _decision_maker(llm, ToolRegistry())  # empty registry

    decision = decision_maker.decide(AgentState(user_input="What time is it?"))

    assert decision.action_type is ActionType.FINAL
    assert len(llm.calls) == 1


def test_disabled_flag_restores_the_pre_existing_llm_only_behavior() -> None:
    llm = FakeLLM([_tool_json("web_search", "what time is it")])
    decision_maker = LLMDecisionMaker(
        llm_client=llm, tool_registry=_registry(TimeTool(), DateTool(), RecordingTool("web_search"))
    )  # deterministic_temporal_routing left at its default: False

    decision = decision_maker.decide(AgentState(user_input="What time is it?"))

    assert decision.tool_name == "web_search"  # exactly what the model asked for
    assert len(llm.calls) == 1


# ===========================================================================
# 10 — correction still works after a deterministic tool failure
# ===========================================================================

def test_correction_recovers_after_a_deterministically_routed_tool_fails() -> None:
    """A deterministic tool that reports an operational failure must
    re-enter the NORMAL path: the next iteration goes to the LLM carrying
    correction feedback, with no privileged deterministic retry."""
    failing = RecordingTool("time", result=ToolResult.fail("clock unavailable"))
    registry = _registry(failing)
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy({"time"}))
    llm = FakeLLM([_final_json("I could not read the clock, but here is what I can say.")])
    loop = AgentLoop(
        decision_maker=_decision_maker(llm, registry),
        tool_registry=registry,
        tool_execution_gate=gate,
        correction_policy=BudgetedCorrectionPolicy(max_corrections=3),
        max_iterations=4,
    )
    state = AgentState(user_input="What time is it?")

    loop.run(state)

    assert state.status is AgentStatus.COMPLETED
    assert failing.execute_count == 1  # NOT retried deterministically
    assert len(state.corrections) == 1
    assert len(llm.calls) == 1  # recovery came from the normal LLM path


def test_a_denied_deterministic_tool_does_not_loop_and_stays_terminal() -> None:
    tool = RecordingTool("time")
    registry = _registry(tool)
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy(frozenset()))
    llm = FakeLLM()
    loop = AgentLoop(
        decision_maker=_decision_maker(llm, registry),
        tool_registry=registry,
        tool_execution_gate=gate,
        correction_policy=BudgetedCorrectionPolicy(max_corrections=5),
        max_iterations=5,
    )
    state = AgentState(user_input="What time is it?")

    loop.run(state)

    assert state.status is AgentStatus.FAILED
    assert state.step == 1  # terminated on the first denial
    assert state.corrections == []
    assert tool.execute_count == 0


# ===========================================================================
# Production composition
# ===========================================================================

def test_production_chat_service_has_deterministic_temporal_routing_enabled() -> None:
    assert main_module.chat_service.deterministic_temporal_routing is True


def test_production_time_request_routes_deterministically_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Through the REAL production chat_service: only the FINAL turn is
    scripted, so if the deterministic path did not fire the FakeLLM would
    be asked for a decision it does not have."""
    llm = FakeLLM([_final_json("It is noon.")])
    monkeypatch.setattr(main_module.chat_service, "llm", llm)

    answer = main_module.chat_service.ask("What time is it?")

    assert answer == "It is noon."
    assert len(llm.calls) == 1


def test_production_date_request_routes_deterministically_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = FakeLLM([_final_json("It was a Friday.")])
    monkeypatch.setattr(main_module.chat_service, "llm", llm)

    answer = main_module.chat_service.ask("What day is 25 December 2026?")

    assert answer == "It was a Friday."
    assert len(llm.calls) == 1


def test_production_non_temporal_request_still_consults_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = FakeLLM([_final_json("A transformer is a neural network architecture.")])
    monkeypatch.setattr(main_module.chat_service, "llm", llm)

    answer = main_module.chat_service.ask("What is a transformer in deep learning?")

    assert "transformer" in answer.lower()
    assert len(llm.calls) == 1
