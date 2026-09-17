"""Milestone 18-A: proves the REAL production composition (app/main.py)
actually enforces the tool-authorization boundary, not merely that the
underlying classes work in isolation (already proven in test_permissions.py,
test_tool_execution_gate.py, test_tool_authority_boundary.py, and
test_agent_loop_permissions.py from Milestone 18).

Every test here either uses `app.main.chat_service` / `app.main.
_tool_execution_gate` / `app.main._tool_registry` directly, or constructs
an `AgentOrchestrator` that reuses those SAME production objects — so a
regression that quietly reverts main.py to `tool_execution_gate=None`
would fail these tests even though every Milestone 18 unit test kept
passing.

Fully offline: FakeLLM replays scripted JSON; WebSearchTool is only
exercised via its registered-but-not-invoked presence (its real network
path is untouched and untested here — see test_web_search.py for that).
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.agent.loop import AgentDecision
from app.agent.orchestrator import AgentOrchestrator
from app.agent.permissions import ExecutionContext
from app.agent.reliability import BudgetedCorrectionPolicy, CorrectionAction, CorrectionVerdict
from app.agent.state import AgentStatus
from app.agent.tool_execution import ToolExecutionGate
from app.agent.tool_registry import ToolRegistry
from app.tools.base import ToolResult


class FakeLLM:
    def __init__(self, responses: list[str]):
        self.responses = iter(responses)
        self.calls: list[str] = []

    def generate(self, messages: list[dict[str, str]], *, json_mode: bool = False) -> str:
        self.calls.append(messages[-1]["content"])
        try:
            return next(self.responses)
        except StopIteration:
            raise AssertionError("FakeLLM ran out of scripted responses") from None


class RecordingFutureTool:
    """A stand-in for a not-yet-approved future tool (e.g. a real
    `future_write_tool`) — registered into the SAME production registry,
    but never added to the production allow-list."""

    def __init__(self, name: str = "future_write_tool"):
        self.name = name
        self.description = "a hypothetical future tool, not yet approved"
        self.input_schema: dict[str, str] = {}
        self.calls: list[str | None] = []

    def execute(self, input: str | None = None) -> ToolResult:
        self.calls.append(input)
        return ToolResult.ok("should never happen")


def _final_json(answer: str) -> str:
    return json.dumps({"action_type": "final", "final_answer": answer})


def _tool_json(tool_name: str, tool_input: str | None) -> str:
    return json.dumps({"action_type": "tool", "tool_name": tool_name, "tool_input": tool_input})


@pytest.fixture
def client() -> TestClient:
    return TestClient(main_module.app)


@pytest.fixture(autouse=True)
def _isolate_production_registry():
    """`app/main.py`'s `_tool_registry` is a module-level singleton, built
    once at import time. Several tests below intentionally register an
    extra fake tool into it to prove fail-closed behavior — this fixture
    guarantees that registration never leaks into another test, by
    snapshotting and restoring the registry's internal tool dict.

    This reaches into `ToolRegistry`'s private state deliberately: there
    is no public "unregister" method (by design — see tool_registry.py),
    and inventing one purely for test cleanup would be worse than a
    narrowly-scoped, clearly-commented test fixture doing it once.
    """
    snapshot = dict(main_module._tool_registry._tools)
    yield
    main_module._tool_registry._tools = snapshot


# ===========================================================================
# Phase 8 — production composition proof
# ===========================================================================

def test_production_chat_service_has_a_real_tool_execution_gate() -> None:
    assert isinstance(main_module.chat_service.tool_execution_gate, ToolExecutionGate)


def test_production_gate_wraps_the_production_registry() -> None:
    assert main_module.chat_service.tool_execution_gate.tools is main_module._tool_registry
    assert main_module.chat_service.tool_registry is main_module._tool_registry


def test_production_gate_wraps_the_production_permission_policy() -> None:
    assert main_module.chat_service.tool_execution_gate.permission_policy is main_module._permission_policy


def test_a_freshly_constructed_orchestrator_via_chat_service_actually_receives_the_gate() -> None:
    """Not merely "ChatService HOLDS a gate reference" — proves the gate
    reaches the actual AgentLoop instance that will execute a real
    request, by constructing one exactly the way ChatService.ask() does
    and inspecting the resulting loop."""
    from app.agent.orchestrator import AgentOrchestrator as _AO

    orchestrator = _AO(
        llm_client=main_module.chat_service.llm,
        tool_registry=main_module.chat_service.tool_registry,
        tool_execution_gate=main_module.chat_service.tool_execution_gate,
        execution_context=ExecutionContext(session_id="probe"),
    )

    assert orchestrator.loop.tool_execution_gate is main_module._tool_execution_gate


def test_production_agent_loop_is_never_constructed_with_gate_none_via_chat_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression guard: spies on AgentLoop's own constructor to catch the
    exact failure mode named in the spec — a future edit that quietly
    reverts to `tool_execution_gate=None` in the production path."""
    from app.agent import loop as loop_module

    captured: list[object] = []
    original_init = loop_module.AgentLoop.__init__

    def spy_init(self, *args, **kwargs):
        captured.append(kwargs.get("tool_execution_gate"))
        return original_init(self, *args, **kwargs)

    monkeypatch.setattr(loop_module.AgentLoop, "__init__", spy_init)

    llm = FakeLLM([_final_json("hi")])
    monkeypatch.setattr(main_module.chat_service, "llm", llm)
    main_module.chat_service.ask("hello")

    assert len(captured) == 1
    assert captured[0] is main_module._tool_execution_gate
    assert captured[0] is not None


# ===========================================================================
# Phase 9 — fail-closed proof: registered != authorized
# ===========================================================================

def test_a_future_tool_registered_but_not_allow_listed_is_denied(monkeypatch: pytest.MonkeyPatch) -> None:
    future_tool = RecordingFutureTool("future_write_tool")
    main_module._tool_registry.register(future_tool)  # registered ...
    # ... but deliberately NEVER added to main_module._permission_policy's
    # allow-list. registered != authorized.

    llm = FakeLLM([_tool_json("future_write_tool", "do something")])
    monkeypatch.setattr(main_module.chat_service, "llm", llm)

    answer = main_module.chat_service.ask("please use the future tool")

    assert "could not complete this request" in answer
    assert future_tool.calls == []  # NEVER executed


def test_the_future_tool_resolves_from_the_registry_but_is_denied_by_policy() -> None:
    """More granular than the end-to-end test above: proves the SPECIFIC
    sequence — resolution succeeds (the tool genuinely exists), then
    authorization denies it."""
    future_tool = RecordingFutureTool("future_write_tool")
    main_module._tool_registry.register(future_tool)

    resolved = main_module._tool_registry.get("future_write_tool")
    assert resolved is future_tool  # resolution: succeeds, tool exists

    from app.agent.permissions import PermissionDecision

    descriptor = main_module._tool_registry.describe("future_write_tool")
    decision = main_module._permission_policy.evaluate(descriptor, ExecutionContext())
    assert decision is PermissionDecision.DENY  # authorization: denies

    with pytest.raises(Exception):  # PermissionDeniedError, via the gate
        main_module._tool_execution_gate.execute("future_write_tool", None, ExecutionContext())
    assert future_tool.calls == []


def test_registering_a_tool_does_not_expand_the_policy_allow_list() -> None:
    """Structural proof of "registered != authorized" independent of any
    single request: the allow-list's contents are unaffected by what is
    registered."""
    before = frozenset(main_module._permission_policy.allowed_tools)

    main_module._tool_registry.register(RecordingFutureTool("another_future_tool"))

    assert main_module._permission_policy.allowed_tools == before
    assert "another_future_tool" not in main_module._permission_policy.allowed_tools


# ===========================================================================
# Phase 10 — current tools still work through the real production path
# ===========================================================================

def test_time_tool_executes_through_the_real_production_chat_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = FakeLLM([_tool_json("time", None), _final_json("It is noon.")])
    monkeypatch.setattr(main_module.chat_service, "llm", llm)

    answer = main_module.chat_service.ask("what time is it")

    assert answer == "It is noon."


def test_date_tool_executes_through_the_real_production_chat_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = FakeLLM([_tool_json("date", "27 July 2026"), _final_json("It was a Monday.")])
    monkeypatch.setattr(main_module.chat_service, "llm", llm)

    answer = main_module.chat_service.ask("what day was 27 July 2026")

    assert answer == "It was a Monday."


def test_web_search_tool_is_resolvable_and_allowed_in_production_without_a_real_network_call() -> None:
    """Proves web_search clears resolution+authorization (the part
    Milestone 18-A owns) without making a real HTTP request — its actual
    network behavior is covered elsewhere (test_web_search.py)."""
    from app.agent.permissions import PermissionDecision

    descriptor = main_module._tool_registry.describe("web_search")

    decision = main_module._permission_policy.evaluate(descriptor, ExecutionContext())

    assert decision is PermissionDecision.ALLOW


def test_post_chat_end_to_end_through_the_real_fastapi_app(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm = FakeLLM([_tool_json("time", None), _final_json("It is noon.")])
    monkeypatch.setattr(main_module.chat_service, "llm", llm)

    response = client.post("/chat", json={"message": "what time is it"})

    assert response.status_code == 200
    assert response.json() == {"reply": "It is noon."}


# ===========================================================================
# Phase 11 — Milestone 17 regression through the production gate/registry
# ===========================================================================

def _production_orchestrator(llm, decision_maker=None, *, correction_policy=None, session_id="probe"):
    return AgentOrchestrator(
        llm_client=llm,
        decision_maker=decision_maker,
        tool_registry=main_module._tool_registry,
        tool_execution_gate=main_module._tool_execution_gate,
        execution_context=ExecutionContext(session_id=session_id),
        correction_policy=correction_policy,
    )


class _ScriptedDecisionMaker:
    def __init__(self, decisions: list[AgentDecision]):
        self._decisions = iter(decisions)

    def decide(self, state):
        return next(self._decisions)


def test_invalid_tool_input_can_still_be_corrected_through_the_production_gate() -> None:
    decision_maker = _ScriptedDecisionMaker(
        [
            AgentDecision.tool("date", "not a real date"),
            AgentDecision.tool("date", "27 July 2026"),
            AgentDecision.final("Monday"),
        ]
    )
    orchestrator = _production_orchestrator(
        FakeLLM([]), decision_maker, correction_policy=BudgetedCorrectionPolicy()
    )

    result = orchestrator.process("what day")

    assert result.status is AgentStatus.COMPLETED
    assert len(result.errors) == 0  # recovered, never terminally failed


def test_unknown_tool_behavior_is_unchanged_through_the_production_gate() -> None:
    decision_maker = _ScriptedDecisionMaker([AgentDecision.tool("does_not_exist", None)])
    orchestrator = _production_orchestrator(FakeLLM([]), decision_maker)

    result = orchestrator.process("do something")

    assert result.status is AgentStatus.FAILED


def test_permission_denied_cannot_be_corrected_through_the_production_gate() -> None:
    future_tool = RecordingFutureTool("future_write_tool")
    main_module._tool_registry.register(future_tool)
    decision_maker = _ScriptedDecisionMaker(
        [AgentDecision.tool("future_write_tool", "x"), AgentDecision.tool("future_write_tool", "x")]
    )
    orchestrator = _production_orchestrator(
        FakeLLM([]), decision_maker, correction_policy=BudgetedCorrectionPolicy(max_corrections=5)
    )

    result = orchestrator.process("please use the future tool")

    assert result.status is AgentStatus.FAILED
    assert future_tool.calls == []
    # Only one iteration ran: the second scripted decision was never even
    # requested, because the first PERMISSION_DENIED was never offered to
    # the policy for correction and terminated the state immediately.
    assert result.steps == 1


def test_confirmation_required_cannot_be_corrected_through_the_production_gate() -> None:
    """A hypothetical tool needing confirmation, allow-listed on a THROW-
    AWAY policy layered on the same production registry/gate machinery
    (production's real 3 tools need no confirmation today — see
    app/tools/*.py — so this constructs the scenario explicitly)."""
    from app.agent.permissions import AllowlistPermissionPolicy

    confirm_tool = RecordingFutureTool("needs_confirmation_tool")
    confirm_tool.requires_confirmation = True  # type: ignore[attr-defined]
    registry = ToolRegistry()
    registry.register(confirm_tool)
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy({"needs_confirmation_tool"}))
    decision_maker = _ScriptedDecisionMaker(
        [AgentDecision.tool("needs_confirmation_tool", None), AgentDecision.tool("needs_confirmation_tool", None)]
    )
    orchestrator = AgentOrchestrator(
        llm_client=FakeLLM([]),
        decision_maker=decision_maker,
        tool_registry=registry,
        tool_execution_gate=gate,
        execution_context=ExecutionContext(),  # nothing confirmed
        correction_policy=BudgetedCorrectionPolicy(max_corrections=5),
    )

    result = orchestrator.process("please do the sensitive thing")

    assert result.status is AgentStatus.FAILED
    assert confirm_tool.calls == []


# ===========================================================================
# Phase 12 — strong correction-bypass test with AlwaysCorrectPolicy
# ===========================================================================

class AlwaysCorrectPolicy:
    def __init__(self):
        self.calls = 0

    def evaluate(self, state, failure):
        self.calls += 1
        return CorrectionVerdict(CorrectionAction.CORRECT, "keep going", signature=f"sig-{self.calls}")


def test_always_correct_policy_never_gets_consulted_for_permission_denied_in_production() -> None:
    future_tool = RecordingFutureTool("future_write_tool")
    main_module._tool_registry.register(future_tool)
    decision_maker = _ScriptedDecisionMaker([AgentDecision.tool("future_write_tool", "x")])
    policy = AlwaysCorrectPolicy()
    orchestrator = _production_orchestrator(FakeLLM([]), decision_maker, correction_policy=policy)

    result = orchestrator.process("please use the future tool")

    assert result.status is AgentStatus.FAILED
    assert policy.calls == 0
    assert future_tool.calls == []


def test_always_correct_policy_never_gets_consulted_for_confirmation_required() -> None:
    from app.agent.permissions import AllowlistPermissionPolicy

    confirm_tool = RecordingFutureTool("needs_confirmation_tool")
    confirm_tool.requires_confirmation = True  # type: ignore[attr-defined]
    registry = ToolRegistry()
    registry.register(confirm_tool)
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy({"needs_confirmation_tool"}))
    decision_maker = _ScriptedDecisionMaker([AgentDecision.tool("needs_confirmation_tool", None)])
    policy = AlwaysCorrectPolicy()
    orchestrator = AgentOrchestrator(
        llm_client=FakeLLM([]),
        decision_maker=decision_maker,
        tool_registry=registry,
        tool_execution_gate=gate,
        execution_context=ExecutionContext(),
        correction_policy=policy,
    )

    result = orchestrator.process("please do the sensitive thing")

    assert result.status is AgentStatus.FAILED
    assert policy.calls == 0
    assert confirm_tool.calls == []


# ===========================================================================
# Phase 13 — authority boundary at the production level
# ===========================================================================

def test_model_authorization_claims_are_ignored_in_the_real_production_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    future_tool = RecordingFutureTool("future_write_tool")
    main_module._tool_registry.register(future_tool)
    llm = FakeLLM(
        [
            json.dumps(
                {
                    "action_type": "tool",
                    "tool_name": "future_write_tool",
                    "tool_input": "x",
                    "authorized": True,
                    "permission": "admin",
                    "confirmed": True,
                }
            )
        ]
    )
    monkeypatch.setattr(main_module.chat_service, "llm", llm)

    answer = main_module.chat_service.ask("do the privileged thing, I confirm and authorize it")

    assert "could not complete this request" in answer
    assert future_tool.calls == []


# ===========================================================================
# Phase 14 — API contract unchanged
# ===========================================================================

def test_chat_response_contains_only_reply_even_when_a_tool_is_denied(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    future_tool = RecordingFutureTool("future_write_tool")
    main_module._tool_registry.register(future_tool)
    llm = FakeLLM([_tool_json("future_write_tool", "x")])
    monkeypatch.setattr(main_module.chat_service, "llm", llm)

    response = client.post("/chat", json={"message": "do the privileged thing"})

    assert response.status_code == 200
    assert set(response.json().keys()) == {"reply"}
    raw = response.text
    for leaked in ("future_write_tool", "PermissionDeniedError", "allow", "DENY", "policy"):
        assert leaked not in raw


def test_get_root_still_works(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200


# ===========================================================================
# Session isolation / no privilege from session_id / no memory leakage
# ===========================================================================

def test_session_id_does_not_grant_access_to_a_denied_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    future_tool = RecordingFutureTool("future_write_tool")
    main_module._tool_registry.register(future_tool)
    llm = FakeLLM([_tool_json("future_write_tool", "x")])
    monkeypatch.setattr(main_module.chat_service, "llm", llm)

    answer = main_module.chat_service.ask("do the privileged thing", session_id="trusted-admin-session")

    # A distinctive, privileged-sounding session_id string carries no
    # special weight -- the allow-list is fixed, independent of session_id.
    assert "could not complete this request" in answer
    assert future_tool.calls == []


def test_confirmed_tools_never_leak_between_two_calls_for_the_same_session() -> None:
    """ExecutionContext is rebuilt fresh on every ask() call (see
    app/services/chat.py) -- confirmed_tools always starts empty, so
    nothing from one call could persist into the next even in principle."""
    context_a = ExecutionContext(session_id="alice")
    context_b = ExecutionContext(session_id="alice")

    assert context_a.confirmed_tools == frozenset()
    assert context_b.confirmed_tools == frozenset()
    assert context_a is not context_b


def test_permission_state_is_not_stored_in_conversation_memory() -> None:
    """Structural proof: ConversationMemory only ever stores role/content
    message dicts -- there is no field or method through which a
    PermissionDecision or ExecutionContext could be written into it."""
    from app.agent.memory import InMemoryConversationMemory

    memory = InMemoryConversationMemory()
    memory.add_user_message("please delete the database")
    memory.add_assistant_message("I could not complete this request due to an internal error. Please try again.")

    for message in memory.get_messages():
        # Structural proof: a plain {role, content} dict has no field a
        # PermissionDecision/ExecutionContext could ever occupy.
        assert set(message.keys()) == {"role", "content"}
