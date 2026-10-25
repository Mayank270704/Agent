"""Authoritative deterministic web-search-COMMAND routing (Step 26).

The `web_search` counterpart to `tests/test_deterministic_temporal_routing.py`
(Step 20): llama3.2:3b was found to under-call `web_search` even for an
EXPLICIT instruction to search, despite the existing advisory ROUTING HINT
(Router.classify_hint / RoutingHint.TOOL_LIKELY, Step 8C) already present in
every prompt. `Router.deterministic_web_search_route` recognizes a narrow
subset of that same territory with certainty — an imperative search command
that ALSO names "web"/"internet"/"online" — and `LLMDecisionMaker.
_deterministic_web_decision` makes the resulting tool choice authoritative,
mirroring the temporal fix's exact discipline.

These tests prove the same authority boundary in both directions the
temporal suite does: the model cannot redirect an explicit search command
away from `web_search`, AND the deterministic path cannot bypass any part of
the Milestone 18/18-A/18-B/18-C security pipeline, skip validation, skip
confirmation, or acquire a privileged retry Milestone 17 correction does not
already grant. A dedicated block also proves the narrowing this milestone's
own adversarial testing required: a bare, unqualified "search for X"/"look
up X" (no web/online/internet reference) is deliberately NOT deterministic,
because testing found it false-positives on ordinary non-web requests.

Fully offline: FakeLLM replays scripted JSON; no Ollama, no network, no
Tavily — the real `web_search` tool is used only via a `RecordingTool` stand-
in, so nothing here ever performs a real HTTP call.
"""
from __future__ import annotations

import json

import pytest

import app.main as main_module
from app.agent.loop import ActionType, AgentLoop
from app.agent.decision_maker import LLMDecisionMaker
from app.agent.permissions import AllowlistPermissionPolicy, ExecutionContext
from app.agent.reliability import BudgetedCorrectionPolicy
from app.agent.router import Router
from app.agent.state import AgentState, AgentStatus
from app.agent.telemetry import EventEmitter, EventType, ListEventSink
from app.agent.tool_execution import PermissionDeniedError, ToolExecutionGate
from app.agent.tool_registry import ToolRegistry
from app.tools.base import ToolResult
from app.tools.date import DateTool
from app.tools.time import TimeTool


class FakeLLM:
    """Raises if consulted more times than scripted — lets a test prove the
    LLM was never called, exactly like the temporal suite's fake."""

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
        self.input_schema: dict[str, str] = {"query": "string"} if name == "web_search" else {}
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
        llm_client=llm, tool_registry=registry, deterministic_web_search_routing=True, **kwargs
    )


# ===========================================================================
# Router accessor — positive cases: an explicit command naming web/online/
# internet is recognized, and the query text is extracted correctly.
# ===========================================================================

@pytest.mark.parametrize(
    "message,expected",
    [
        ("Search the web for information about RAG.", ("web_search", "information about rag")),
        ("search the internet for python tutorials", ("web_search", "python tutorials")),
        ("Please search online for the latest AI news", ("web_search", "the latest ai news")),
        ("Look up the current price of Bitcoin online.", ("web_search", "the current price of bitcoin")),
        ("look up the definition of osmosis on the web", ("web_search", "the definition of osmosis")),
        ("Find online the best pizza in Rome", ("web_search", "the best pizza in rome")),
        ("find the best pizza in Rome online", ("web_search", "the best pizza in rome")),
        # Step 28 — closes the "find X on the internet/web" coverage gap
        # found during Step 27's evaluation (symmetric to the already-
        # supported "look up X on the internet/web").
        ("find X on the internet", ("web_search", "x")),
        ("find information about X on the internet", ("web_search", "information about x")),
        (
            "find the latest information about X on the internet",
            ("web_search", "the latest information about x"),
        ),
        ("find the best pizza in Rome on the web", ("web_search", "the best pizza in rome")),
    ],
)
def test_router_recognizes_explicit_web_search_commands(message: str, expected: tuple) -> None:
    assert Router.__new__(Router).deterministic_web_search_route(message) == expected


# ===========================================================================
# Router accessor — negative cases required by the Step 26 design brief:
# static/historical questions (Category D), bare current-info QUESTIONS
# (Category B, no command), and existing time/date deterministic coverage
# (Category E) must never be captured by this new route.
# ===========================================================================

@pytest.mark.parametrize(
    "message",
    [
        # Category D — historical/static, no command at all
        "What year was Python created?",
        "When was the Eiffel Tower built?",
        "What is the capital of France?",
        # Category B — current-information QUESTIONS, not commands
        "What is the latest AI news?",
        "What is the current price of Bitcoin?",
        "Tell me about current events.",
        "What are the latest developments in AI?",
        # Category E / existing temporal coverage — untouched by this route
        "What time is it?",
        "What day is 25 December 2026?",
        # Ordinary static questions
        "What is a transformer in deep learning?",
        "How does Google Search work?",
        "",
        "   ",
        # Step 28 — the new "find X on the web/internet" pattern must not
        # widen coverage beyond its literal qualifier: bare "find X" and
        # "find X in <local content>" stay unmatched, exactly like the
        # pre-existing "find X online" pattern already required.
        "find X",
        "find the bug in this code",
        "find the value in this array",
        "find X in the document",
        "find X in the text above",
    ],
)
def test_router_declines_everything_outside_explicit_web_commands(message: str) -> None:
    assert Router.__new__(Router).deterministic_web_search_route(message) is None


# ===========================================================================
# Router accessor — adversarial cases found during THIS milestone's own
# testing: a bare, unqualified search/look-up verb with no web/online/
# internet reference must stay advisory-only (RoutingHint.TOOL_LIKELY), not
# become authoritative, because it reads naturally as something other than
# a web search in ordinary chat text.
# ===========================================================================

@pytest.mark.parametrize(
    "message",
    [
        "search for a job",
        "search for the mistake in my code",
        "search for a bug in this function",
        "search for meaning in life",
        "I will search for you at the park",
        "look up to my role model",  # a different phrasal verb entirely
        "help me look up my order status",
        "Look for my keys.",
        "I am looking for a new job.",
        "search for",  # trigger word with nothing after it
        "look up",
    ],
)
def test_router_declines_bare_unqualified_search_verbs(message: str) -> None:
    """The Step 26 design brief's Category A list includes bare "search for
    ...", but adversarial testing during this milestone found real false
    positives from it (see the module docstring and Router's own
    `_WEB_SEARCH_COMMAND_PATTERNS` comment) — this is the deliberate,
    evidence-driven narrowing away from that literal list."""
    assert Router.__new__(Router).deterministic_web_search_route(message) is None


# ===========================================================================
# Router accessor — negated/hypothetical phrasing never routes
# deterministically, regardless of which command phrase it wraps.
# ===========================================================================

@pytest.mark.parametrize(
    "message",
    [
        "Don't search the web for anything.",
        "Do not search the web for my address.",
        "No need to look up the weather online.",
        "You shouldn't search the web for that.",
        "Never search the internet for personal information.",
    ],
)
def test_router_declines_negated_web_search_commands(message: str) -> None:
    assert Router.__new__(Router).deterministic_web_search_route(message) is None


# ===========================================================================
# 1/2 — deterministic requests select web_search, with no LLM call
# ===========================================================================

def test_deterministic_web_search_command_selects_the_tool_without_consulting_the_llm() -> None:
    llm = FakeLLM()  # any call raises
    decision_maker = _decision_maker(llm, _registry(RecordingTool("web_search")))

    decision = decision_maker.decide(AgentState(user_input="Search the web for the latest AI news."))

    assert decision.action_type is ActionType.TOOL
    assert decision.tool_name == "web_search"
    assert decision.tool_input == "the latest ai news"
    assert llm.calls == []  # the model was never consulted


# ===========================================================================
# 3 — deterministic routing reaches the tool execution gate
# ===========================================================================

def test_deterministic_web_search_routing_reaches_the_tool_execution_gate() -> None:
    tool = RecordingTool("web_search")
    registry = _registry(tool)
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy({"web_search"}))
    llm = FakeLLM([_final_json("Here is what I found.")])  # only the FINAL turn is scripted
    sink = ListEventSink()
    loop = AgentLoop(
        decision_maker=_decision_maker(llm, registry),
        tool_registry=registry,
        tool_execution_gate=gate,
        event_emitter=EventEmitter(sink, request_id="r1"),
    )
    state = AgentState(user_input="Search the web for the latest AI news.")

    loop.run(state)

    assert state.status is AgentStatus.COMPLETED
    assert tool.execute_count == 1
    types = [e.event_type for e in sink.events]
    assert EventType.TOOL_PROPOSED in types
    assert EventType.TOOL_AUTHORIZED in types
    assert EventType.TOOL_EXECUTION_COMPLETED in types


def test_no_direct_tool_execution_bypass_exists_in_the_deterministic_web_path() -> None:
    """Structural proof, mirroring the temporal suite's equivalent test: the
    deterministic path returns an ordinary AgentDecision and never touches a
    tool object directly."""
    import ast
    import inspect
    import textwrap

    from app.agent import decision_maker as module

    tree = ast.parse(textwrap.dedent(inspect.getsource(module.LLMDecisionMaker._deterministic_web_decision)))
    function = tree.body[0]
    body = function.body[1:] if ast.get_docstring(function) else function.body
    code = "\n".join(ast.unparse(node) for node in body)

    assert ".execute(" not in code
    assert "self.tools.get(" not in code  # only has() — a membership check, never resolution
    assert "AgentDecision.tool(" in code


# ===========================================================================
# 4/5/6 — permission, validation and confirmation still apply
# ===========================================================================

def test_permission_policy_still_applies_to_a_deterministically_routed_web_search() -> None:
    tool = RecordingTool("web_search")
    registry = _registry(tool)
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy(frozenset()))  # web_search NOT allowed
    llm = FakeLLM()
    loop = AgentLoop(
        decision_maker=_decision_maker(llm, registry), tool_registry=registry, tool_execution_gate=gate
    )
    state = AgentState(user_input="Search the web for the latest AI news.")

    loop.run(state)

    assert state.status is AgentStatus.FAILED
    assert tool.execute_count == 0


def test_permission_denial_of_a_deterministic_web_search_is_raised_by_the_gate_itself() -> None:
    tool = RecordingTool("web_search")
    gate = ToolExecutionGate(_registry(tool), AllowlistPermissionPolicy(frozenset()))

    with pytest.raises(PermissionDeniedError):
        gate.execute("web_search", "the latest ai news", ExecutionContext())

    assert tool.execute_count == 0


def test_confirmation_still_applies_to_a_deterministically_routed_web_search() -> None:
    tool = RecordingTool("web_search", requires_confirmation=True)
    registry = _registry(tool)
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy({"web_search"}))
    llm = FakeLLM()
    loop = AgentLoop(
        decision_maker=_decision_maker(llm, registry),
        tool_registry=registry,
        tool_execution_gate=gate,
        execution_context=ExecutionContext(),  # nothing confirmed
    )
    state = AgentState(user_input="Search the web for the latest AI news.")

    loop.run(state)

    assert state.status is AgentStatus.FAILED
    assert tool.execute_count == 0


def test_trusted_confirmation_still_permits_a_deterministically_routed_web_search() -> None:
    tool = RecordingTool("web_search", requires_confirmation=True)
    registry = _registry(tool)
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy({"web_search"}))
    llm = FakeLLM([_final_json("Here is what I found.")])
    loop = AgentLoop(
        decision_maker=_decision_maker(llm, registry),
        tool_registry=registry,
        tool_execution_gate=gate,
        execution_context=ExecutionContext(confirmed_tools=frozenset({"web_search"})),
    )
    state = AgentState(user_input="Search the web for the latest AI news.")

    loop.run(state)

    assert state.status is AgentStatus.COMPLETED
    assert tool.execute_count == 1


def test_validation_still_applies_to_a_deterministically_routed_web_search() -> None:
    """A web_search stand-in whose validate() hook rejects the routed input
    must still block execution — deterministic routing chooses the tool, it
    does not vouch for the input."""

    class StrictWebSearchTool(RecordingTool):
        def validate(self, input: str | None = None) -> None:
            raise ValueError("validation always rejects in this test")

    tool = StrictWebSearchTool("web_search")
    registry = _registry(tool)
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy({"web_search"}))
    llm = FakeLLM()
    loop = AgentLoop(
        decision_maker=_decision_maker(llm, registry), tool_registry=registry, tool_execution_gate=gate
    )
    state = AgentState(user_input="Search the web for the latest AI news.")

    loop.run(state)

    assert state.status is AgentStatus.FAILED
    assert tool.execute_count == 0


# ===========================================================================
# 7 — the model cannot override deterministic web-search routing
# ===========================================================================

def test_model_cannot_redirect_an_explicit_web_search_command_to_final() -> None:
    """The exact failure this change exists to fix: the model would answer
    directly (FINAL) instead of honoring an explicit search instruction."""
    web_tool = RecordingTool("web_search")
    registry = _registry(web_tool)
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy({"web_search"}))
    # The model would answer directly if it were ever asked. It isn't.
    llm = FakeLLM([_final_json("I already know the answer.")])
    loop = AgentLoop(
        decision_maker=_decision_maker(llm, registry), tool_registry=registry, tool_execution_gate=gate,
        max_iterations=2,
    )
    state = AgentState(user_input="Search the web for the latest AI news.")

    loop.run(state)

    assert web_tool.execute_count == 1  # the application's choice won
    assert state.tool_calls[0].tool_name == "web_search"
    # The model IS consulted once, but only for FINAL synthesis AFTER the
    # deterministically-selected tool already ran — never for the decision
    # of WHICH action to take on the first iteration.
    assert len(llm.calls) == 1


def test_deterministic_web_authority_is_limited_to_the_first_attempt() -> None:
    """Once the tool has run, the LLM regains full control for FINAL
    synthesis — otherwise decide() would return the same tool forever."""
    tool = RecordingTool("web_search")
    registry = _registry(tool)
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy({"web_search"}))
    llm = FakeLLM([_final_json("Here is what I found.")])
    loop = AgentLoop(
        decision_maker=_decision_maker(llm, registry), tool_registry=registry, tool_execution_gate=gate
    )
    state = AgentState(user_input="Search the web for the latest AI news.")

    loop.run(state)

    assert state.status is AgentStatus.COMPLETED
    assert state.final_answer == "Here is what I found."
    assert tool.execute_count == 1  # exactly once, not once per iteration
    assert len(llm.calls) == 1  # the LLM was consulted only for the FINAL synthesis


# ===========================================================================
# 8/9 — non-command and ambiguous requests stay on the normal LLM path
# ===========================================================================

@pytest.mark.parametrize(
    "message",
    [
        "What is the latest AI news?",
        "What is the current price of Bitcoin?",
        "Tell me about current events.",
        "search for the mistake in my code",
        "look up to my role model",
    ],
)
def test_non_command_and_adversarial_requests_remain_on_the_normal_decision_path(message: str) -> None:
    llm = FakeLLM([_final_json("an ordinary model answer")])
    decision_maker = _decision_maker(llm, _registry(RecordingTool("web_search")))

    decision = decision_maker.decide(AgentState(user_input=message))

    assert decision.action_type is ActionType.FINAL
    assert len(llm.calls) == 1  # the model WAS consulted


def test_an_unregistered_web_search_tool_falls_through_to_the_llm() -> None:
    """Deterministic routing must never name a tool this deployment has not
    registered — that would manufacture an UNKNOWN_TOOL failure."""
    llm = FakeLLM([_final_json("model answered instead")])
    decision_maker = _decision_maker(llm, ToolRegistry())  # empty registry

    decision = decision_maker.decide(AgentState(user_input="Search the web for the latest AI news."))

    assert decision.action_type is ActionType.FINAL
    assert len(llm.calls) == 1


def test_disabled_flag_restores_the_pre_existing_llm_only_behavior() -> None:
    llm = FakeLLM([_final_json("I already know the answer.")])
    decision_maker = LLMDecisionMaker(
        llm_client=llm, tool_registry=_registry(RecordingTool("web_search"))
    )  # deterministic_web_search_routing left at its default: False

    decision = decision_maker.decide(AgentState(user_input="Search the web for the latest AI news."))

    assert decision.action_type is ActionType.FINAL  # exactly what the model chose
    assert len(llm.calls) == 1


# ===========================================================================
# 10 — precedence: existing temporal routing is completely unaffected, even
# for a message that could plausibly match both
# ===========================================================================

def test_existing_temporal_routing_still_takes_precedence_over_web_search_routing() -> None:
    """A message matching BOTH the (longer-tested) temporal route and the
    new web-search-command route resolves exactly as it already did before
    this milestone: `decide()` checks temporal first (see
    app/agent/decision_maker.py) and this method never runs at all."""
    time_tool = RecordingTool("time")
    web_tool = RecordingTool("web_search")
    registry = _registry(time_tool, web_tool)
    llm = FakeLLM()  # any call raises
    decision_maker = LLMDecisionMaker(
        llm_client=llm,
        tool_registry=registry,
        deterministic_temporal_routing=True,
        deterministic_web_search_routing=True,
    )

    decision = decision_maker.decide(AgentState(user_input="search the web for what time is it"))

    assert decision.tool_name == "time"
    assert llm.calls == []


def test_deterministic_time_request_is_unaffected_by_web_search_routing_being_enabled() -> None:
    llm = FakeLLM()
    decision_maker = LLMDecisionMaker(
        llm_client=llm,
        tool_registry=_registry(TimeTool(), DateTool(), RecordingTool("web_search")),
        deterministic_temporal_routing=True,
        deterministic_web_search_routing=True,
    )

    decision = decision_maker.decide(AgentState(user_input="What time is it?"))

    assert decision.tool_name == "time"
    assert llm.calls == []


# ===========================================================================
# 11 — correction still works after a deterministic web-search failure
# ===========================================================================

def test_correction_recovers_after_a_deterministically_routed_web_search_fails() -> None:
    """A deterministic web_search that reports an operational failure must
    re-enter the NORMAL path: the next iteration goes to the LLM carrying
    correction feedback, with no privileged deterministic retry."""
    failing = RecordingTool("web_search", result=ToolResult.fail("Tavily request timed out"))
    registry = _registry(failing)
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy({"web_search"}))
    llm = FakeLLM([_final_json("I could not search the web, but here is what I can say.")])
    loop = AgentLoop(
        decision_maker=_decision_maker(llm, registry),
        tool_registry=registry,
        tool_execution_gate=gate,
        correction_policy=BudgetedCorrectionPolicy(max_corrections=3),
        max_iterations=4,
    )
    state = AgentState(user_input="Search the web for the latest AI news.")

    loop.run(state)

    assert state.status is AgentStatus.COMPLETED
    assert failing.execute_count == 1  # NOT retried deterministically
    assert len(state.corrections) == 1
    assert len(llm.calls) == 1  # recovery came from the normal LLM path


def test_a_denied_deterministic_web_search_does_not_loop_and_stays_terminal() -> None:
    tool = RecordingTool("web_search")
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
    state = AgentState(user_input="Search the web for the latest AI news.")

    loop.run(state)

    assert state.status is AgentStatus.FAILED
    assert state.step == 1  # terminated on the first denial
    assert state.corrections == []
    assert tool.execute_count == 0


# ===========================================================================
# Production composition
# ===========================================================================

def test_production_chat_service_has_deterministic_web_search_routing_enabled() -> None:
    assert main_module.chat_service.deterministic_web_search_routing is True


def test_production_web_search_command_routes_deterministically_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full-stack smoke test through the REAL production chat_service:
    real ToolRegistry, real ToolExecutionGate/AllowlistPermissionPolicy,
    real WebSearchTool instance — only its `api_key` INSTANCE attribute is
    disabled here (not `settings`, which the already-constructed tool never
    re-reads), so `execute()` raises its existing configuration `ValueError`
    with NO network call ever attempted (see WebSearchTool.execute's own
    precondition check, which runs before any HTTP request). The resulting
    INVALID_TOOL_INPUT failure goes through the real, unconditionally-wired
    production CorrectionPolicy exactly as it would for any other tool
    failure — this test is a wiring/composition smoke test (paired with
    `test_production_chat_service_has_deterministic_web_search_routing_
    enabled` above, which is the actual flag proof), not a second proof of
    the dispatch logic itself — that is already proven, airtight, at the
    LLMDecisionMaker level above (a FakeLLM with zero scripted responses
    that raises on any call)."""
    web_search_tool = main_module.chat_service.tool_registry.get("web_search")
    monkeypatch.setattr(web_search_tool, "api_key", "", raising=False)

    llm = FakeLLM([_final_json("I could not search the web right now.")])
    monkeypatch.setattr(main_module.chat_service, "llm", llm)

    answer = main_module.chat_service.ask("Search the web for the latest AI news.")

    assert answer == "I could not search the web right now."
    assert len(llm.calls) == 1  # consulted once, for FINAL synthesis after the tool's own config error


# ===========================================================================
# Step 28 — "find X on the internet/web" closes a coverage gap discovered
# during Step 27's evaluation. Router-level positive/negative coverage
# lives in the parametrized tests above (extended, not duplicated); this
# block closes the Phase 4 checklist items that need the full decision-
# maker/loop/gate wiring: no extra LLM call, temporal precedence, the
# disabled-flag fallback, and ToolExecutionGate authority — all proven
# generically above for the pattern family as a whole, reconfirmed here
# specifically for this one new pattern so the gap-closing change has its
# own direct regression coverage.
# ===========================================================================

def test_find_x_on_the_internet_selects_web_search_without_consulting_the_llm() -> None:
    llm = FakeLLM()  # any call raises
    decision_maker = _decision_maker(llm, _registry(RecordingTool("web_search")))

    decision = decision_maker.decide(AgentState(user_input="Find the latest information about RAG on the internet."))

    assert decision.action_type is ActionType.TOOL
    assert decision.tool_name == "web_search"
    assert decision.tool_input == "the latest information about rag"
    assert llm.calls == []  # no LLM call for the deterministic decision itself


def test_find_x_on_the_internet_reaches_the_tool_execution_gate() -> None:
    """ToolExecutionGate remains authoritative for this pattern exactly
    like every other deterministic web-search command (see the shared
    gate/permission/confirmation/validation tests above, which already
    cover the pattern family generically)."""
    tool = RecordingTool("web_search")
    registry = _registry(tool)
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy({"web_search"}))
    llm = FakeLLM([_final_json("Here is what I found.")])
    loop = AgentLoop(
        decision_maker=_decision_maker(llm, registry), tool_registry=registry, tool_execution_gate=gate
    )
    state = AgentState(user_input="Find X on the internet.")

    loop.run(state)

    assert state.status is AgentStatus.COMPLETED
    assert tool.execute_count == 1
    assert tool.inputs == ["x"]


def test_find_x_on_the_internet_denied_by_permission_policy_never_executes() -> None:
    tool = RecordingTool("web_search")
    gate = ToolExecutionGate(_registry(tool), AllowlistPermissionPolicy(frozenset()))

    with pytest.raises(PermissionDeniedError):
        gate.execute("web_search", "x", ExecutionContext())

    assert tool.execute_count == 0


def test_find_x_on_the_internet_does_not_override_temporal_precedence() -> None:
    """Existing temporal routing (Step 20) still wins for an adversarial
    message matching both — unchanged by Step 28, same precedence rule
    Step 26/27 already established for the pattern family."""
    time_tool = RecordingTool("time")
    web_tool = RecordingTool("web_search")
    registry = _registry(time_tool, web_tool)
    llm = FakeLLM()
    decision_maker = LLMDecisionMaker(
        llm_client=llm,
        tool_registry=registry,
        deterministic_temporal_routing=True,
        deterministic_web_search_routing=True,
    )

    decision = decision_maker.decide(AgentState(user_input="find what time is it on the internet"))

    assert decision.tool_name == "time"
    assert llm.calls == []


def test_find_x_on_the_internet_falls_back_to_the_llm_when_the_flag_is_disabled() -> None:
    llm = FakeLLM([_final_json("I already know the answer.")])
    decision_maker = LLMDecisionMaker(
        llm_client=llm, tool_registry=_registry(RecordingTool("web_search"))
    )  # deterministic_web_search_routing left at its default: False

    decision = decision_maker.decide(AgentState(user_input="Find X on the internet."))

    assert decision.action_type is ActionType.FINAL  # exactly what the model chose
    assert len(llm.calls) == 1
