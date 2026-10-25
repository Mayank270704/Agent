"""Step 29 — the permanent end-to-end capability evaluation suite.

Three groups, matching Step 29's dataset design (see tests/evaluation/
dataset.py's module docstring for why categories split this way):

1. MECHANISM checks (I-N): tool input robustness, failure/recovery,
   session memory, security, timeout, telemetry. Deterministic by
   construction (FakeLLM-scripted or component-level), so a single run is
   both sufficient and honest — run as the ordinary `not integration`
   suite. Where exhaustive coverage of the underlying mechanism already
   exists elsewhere in this test suite, each section says so explicitly
   and this file adds only a small, REPRESENTATIVE case set proving the
   metric end-to-end through the real production `app.main.chat_service`
   singleton (see class/function docstrings for the specific file(s)).

2. MODEL/TOOL-DEPENDENT evaluation (A-H): real HTTP requests through
   FastAPI's TestClient into the real production `chat_service`, real
   Ollama, and (for web_search cases) real Tavily. `@pytest.mark.
   integration`, skips if Ollama is unreachable. Run for repeated trials
   (`TRIALS`) per Step 27's established methodology — model variance is
   reported, never hidden (Step 29 Phase 6).

3. `test_full_evaluation_report` stitches every category's metrics into
   one printed report — informational, the human-readable artifact this
   milestone exists to produce. See tests/evaluation/EVALUATION_REPORT.md
   for the written-up, versioned results of an actual run.

No case in this file asserts a minimum accuracy/success rate against real,
nondeterministic model output — consistent with every other real-Ollama
evaluation in this codebase (tests/test_routing_evaluation.py, tests/
test_web_search_routing_evaluation.py, tests/test_web_intent_ambiguity_
evaluation.py). The ONE hard, asserted gate in this entire file is
Phase 7's: security violations must be exactly zero.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.agent.decision_maker import LLMDecisionMaker
from app.agent.loop import ActionType, AgentLoop
from app.agent.orchestrator import AgentOrchestrator
from app.agent.permissions import AllowlistPermissionPolicy, ExecutionContext
from app.agent.reliability import BudgetedCorrectionPolicy
from app.agent.state import AgentState, AgentStatus
from app.agent.telemetry import AgentEvent, EventEmitter, EventType, ListEventSink, SafeEventSink
from app.agent.tool_execution import ConfirmationRequiredError, PermissionDeniedError, ToolExecutionGate
from app.agent.tool_registry import ToolRegistry
from app.config import settings
from app.tools.base import ToolResult
from app.tools.date import DateTool
from app.tools.time import TimeTool
from app.tools.web_search import WebSearchTool

from tests.evaluation.dataset import ALL_CATEGORIES, MODEL_DEPENDENT_CASES
from tests.evaluation.metrics import (
    CheckResult,
    compute_category_metrics,
    compute_variance,
    format_category_report,
    summarize_checks,
)
from tests.evaluation.runner import run_cases


class FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.calls: list[str] = []

    def generate(self, messages: list[dict[str, str]], *, json_mode: bool = False) -> str:
        self.calls.append(messages[-1]["content"])
        try:
            return next(self.responses)
        except StopIteration:
            raise AssertionError("FakeLLM ran out of scripted responses") from None


class RecordingTool:
    def __init__(self, name: str, *, requires_confirmation: bool = False, result: ToolResult | None = None):
        self.name = name
        self.description = f"tool '{name}'"
        self.input_schema: dict[str, str] = {}
        self.requires_confirmation = requires_confirmation
        self.execute_count = 0
        self._result = result or ToolResult.ok({"ok": True})

    def execute(self, input: str | None = None) -> ToolResult:
        self.execute_count += 1
        return self._result


def _final_json(answer: str) -> str:
    return json.dumps({"action_type": "final", "final_answer": answer})


def _tool_json(name: str, tool_input) -> str:
    return json.dumps({"action_type": "tool", "tool_name": name, "tool_input": tool_input})


def _registry(*tools) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


@pytest.fixture(autouse=True)
def _isolate_production_registry():
    """Category L registers a hypothetical `future_write_tool` into the
    REAL production `main_module._tool_registry` singleton to prove
    fail-closed behavior against it. Identical snapshot/restore technique
    to `tests/test_production_tool_authorization.py`'s own fixture of the
    same name, reused here rather than reinvented, so this file's
    registration never leaks into any other test in the same session."""
    snapshot = dict(main_module._tool_registry._tools)
    yield
    main_module._tool_registry._tools = snapshot


# ===========================================================================
# I. Tool input robustness
# ===========================================================================
# Exhaustive coverage of Tool.execute()'s own error contract already lives
# in tests/test_tools_base.py, tests/test_time.py, tests/test_date.py,
# tests/test_web_search.py; the model-output normalization (dict/list ->
# string tool_input, literal "null") is exhaustively covered in tests/
# test_decision_maker.py (Milestone 21). This section proves each of those
# input shapes still resolves correctly end to end, through the REAL
# ToolRegistry + ToolExecutionGate + AgentLoop, with the REAL production
# tools — not a second derivation of the underlying contract.

def _category_i_checks() -> list[CheckResult]:
    checks: list[CheckResult] = []

    # 1. Null input to a no-input tool (time) succeeds.
    result = TimeTool().execute(None)
    checks.append(CheckResult("null_input_to_time_tool_succeeds", result.success))

    # 2. Malformed string input to DateTool raises ValueError (precondition
    # failure), which reaches the caller as an ordinary, expected exception
    # -- not a crash -- exactly per app/tools/base.py's documented contract.
    try:
        DateTool().execute("not a real date")
        checks.append(CheckResult("malformed_date_string_raises_valueerror", False, "no exception raised"))
    except ValueError:
        checks.append(CheckResult("malformed_date_string_raises_valueerror", True))

    # 3. Empty/null input to WebSearchTool raises ValueError.
    try:
        WebSearchTool().execute(None)
        checks.append(CheckResult("empty_query_to_web_search_raises_valueerror", False, "no exception raised"))
    except ValueError:
        checks.append(CheckResult("empty_query_to_web_search_raises_valueerror", True))

    # 4. A model emitting a JSON OBJECT for tool_input (Milestone 21) is
    # normalized to a string by LLMDecisionMaker, then reaches the tool's
    # OWN validation as an ordinary string -- proven end to end via the
    # real AgentLoop + real DateTool (invalid as a date, so it still fails,
    # but as an ordinary INVALID_TOOL_INPUT, never a crash or a bypass).
    registry = _registry(DateTool())
    llm = FakeLLM([
        _tool_json("date", {"day": 25, "month": 12}),  # dict tool_input
        _final_json("I could not parse that date."),
    ])
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy({"date"}))
    loop = AgentLoop(
        decision_maker=LLMDecisionMaker(llm_client=llm, tool_registry=registry),
        tool_registry=registry,
        tool_execution_gate=gate,
        correction_policy=BudgetedCorrectionPolicy(),
    )
    state = AgentState(user_input="What day was that date?")
    loop.run(state)
    checks.append(
        CheckResult(
            "dict_tool_input_normalized_and_recovers",
            state.status is AgentStatus.COMPLETED,
            f"status={state.status}",
        )
    )

    # 5. A model emitting the literal string "null" for a no-input tool
    # (TimeTool) is normalized to real None, not passed through literally.
    registry2 = _registry(TimeTool())
    llm2 = FakeLLM([_tool_json("time", "null"), _final_json("It is now.")])
    gate2 = ToolExecutionGate(registry2, AllowlistPermissionPolicy({"time"}))
    loop2 = AgentLoop(
        decision_maker=LLMDecisionMaker(llm_client=llm2, tool_registry=registry2),
        tool_registry=registry2,
        tool_execution_gate=gate2,
    )
    state2 = AgentState(user_input="What time is it?")
    loop2.run(state2)
    checks.append(
        CheckResult(
            "literal_null_string_normalized_to_none",
            state2.status is AgentStatus.COMPLETED and state2.observations[0].data is not None,
        )
    )

    return checks


def test_category_i_tool_input_robustness() -> None:
    checks = _category_i_checks()
    passed, total, rate = summarize_checks(checks)
    print(f"\n[I] tool input robustness: {passed}/{total} ({rate:.1%})")
    for c in checks:
        print(f"    {'OK' if c.passed else 'FAIL'} {c.name} {c.detail}")
    assert passed == total, [c for c in checks if not c.passed]


# ===========================================================================
# J. Tool failure / recovery
# ===========================================================================
# Exhaustive coverage: tests/test_tool_failure_correction.py, tests/
# test_agent_loop_correction.py, tests/test_reliability*.py. This section
# computes RECOVERY SUCCESS RATE (Phase 4, item 7) over a small,
# representative set of the four CORRECTABLE failure categories, plus one
# NON-correctable case (permission denial) reported separately, since
# Milestone 17/18 deliberately never offers it to any CorrectionPolicy.

def _category_j_checks() -> list[CheckResult]:
    checks: list[CheckResult] = []

    def _run(llm_responses: list[str], registry: ToolRegistry, gate: ToolExecutionGate | None = None) -> AgentStatus:
        llm = FakeLLM(llm_responses)
        loop = AgentLoop(
            decision_maker=LLMDecisionMaker(llm_client=llm, tool_registry=registry),
            tool_registry=registry,
            tool_execution_gate=gate,
            correction_policy=BudgetedCorrectionPolicy(),
            max_iterations=4,
        )
        state = AgentState(user_input="Do something.")
        loop.run(state)
        return state.status

    # 1. Unknown tool (model invents an unregistered name) -> recovers.
    status = _run([_tool_json("calculator", "2+2"), _final_json("I cannot do that, but here is an answer.")], ToolRegistry())
    checks.append(CheckResult("recovers_from_unknown_tool", status is AgentStatus.COMPLETED))

    # 2. Invalid tool input (tool raises ValueError) -> recovers.
    registry = _registry(DateTool())
    status = _run(
        [_tool_json("date", "not a date"), _final_json("I could not parse that date.")],
        registry, ToolExecutionGate(registry, AllowlistPermissionPolicy({"date"})),
    )
    checks.append(CheckResult("recovers_from_invalid_tool_input", status is AgentStatus.COMPLETED))

    # 3. Tool operational failure (ToolResult.fail) -> recovers.
    failing = RecordingTool("web_search", result=ToolResult.fail("upstream error"))
    registry2 = _registry(failing)
    status = _run(
        [_tool_json("web_search", "x"), _final_json("The search failed, but here is what I know.")],
        registry2, ToolExecutionGate(registry2, AllowlistPermissionPolicy({"web_search"})),
    )
    checks.append(CheckResult("recovers_from_tool_execution_failure", status is AgentStatus.COMPLETED))

    # 4. Malformed model decision (invalid JSON) -> recovers.
    status = _run(["not valid json at all", _final_json("Let me try again.")], ToolRegistry())
    checks.append(CheckResult("recovers_from_malformed_decision", status is AgentStatus.COMPLETED))

    # 5. Permission denial -> NEVER offered to correction; stays FAILED,
    # in exactly one iteration (Milestone 17/18 design, not a bug).
    tool = RecordingTool("web_search")
    registry3 = _registry(tool)
    gate3 = ToolExecutionGate(registry3, AllowlistPermissionPolicy(frozenset()))  # nothing allowed
    llm3 = FakeLLM([])  # must never be consulted after the denial
    loop3 = AgentLoop(
        decision_maker=LLMDecisionMaker(llm_client=FakeLLM([_tool_json("web_search", "x")]), tool_registry=registry3),
        tool_registry=registry3,
        tool_execution_gate=gate3,
        correction_policy=BudgetedCorrectionPolicy(max_corrections=5),
    )
    state3 = AgentState(user_input="Search for something.")
    loop3.run(state3)
    checks.append(
        CheckResult(
            "permission_denial_correctly_not_offered_to_correction",
            state3.status is AgentStatus.FAILED and state3.step == 1 and tool.execute_count == 0,
        )
    )

    return checks


def test_category_j_recovery() -> None:
    checks = _category_j_checks()
    passed, total, rate = summarize_checks(checks)
    print(f"\n[J] tool failure/recovery: {passed}/{total} ({rate:.1%})")
    for c in checks:
        print(f"    {'OK' if c.passed else 'FAIL'} {c.name} {c.detail}")
    assert passed == total, [c for c in checks if not c.passed]


# ===========================================================================
# K. Session memory
# ===========================================================================
# Exhaustive coverage: tests/test_session_memory.py, tests/test_memory*.py,
# tests/test_anonymous_memory_isolation.py, tests/test_session_bounding.py.
# This section proves the metric ("memory correctness", "session-isolation
# correctness") end to end through the REAL `app.main.chat_service`
# singleton, by inspecting the prompt each scripted call actually received
# -- FakeLLM's scripted reply text proves nothing about context (it returns
# whatever it was told to regardless of what it was shown), so only the
# PROMPT is evidence here, exactly like test_anonymous_memory_isolation.py.

def _category_k_checks(monkeypatch: pytest.MonkeyPatch) -> list[CheckResult]:
    checks: list[CheckResult] = []
    session_a = f"eval-session-a-{time.time_ns()}"
    session_b = f"eval-session-b-{time.time_ns()}"

    llm = FakeLLM([
        _final_json("Nice to meet you, Alex."),
        _final_json("Your name is Alex."),
        _final_json("I don't know your name."),
        _final_json("I don't know your name."),
    ])
    monkeypatch.setattr(main_module.chat_service, "llm", llm)

    main_module.chat_service.ask("My name is Alex.", session_id=session_a)
    main_module.chat_service.ask("What is my name?", session_id=session_a)
    second_call_prompt = llm.calls[1]
    checks.append(CheckResult("second_message_in_same_session_sees_first", "Alex" in second_call_prompt))

    main_module.chat_service.ask("What is my name?", session_id=session_b)
    third_call_prompt = llm.calls[2]
    checks.append(CheckResult("separate_session_does_not_inherit_information", "Alex" not in third_call_prompt))

    main_module.chat_service.ask("What is my name?", session_id=None)
    fourth_call_prompt = llm.calls[3]
    checks.append(CheckResult("anonymous_request_does_not_inherit_other_sessions", "Alex" not in fourth_call_prompt))

    return checks


def test_category_k_session_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    checks = _category_k_checks(monkeypatch)
    passed, total, rate = summarize_checks(checks)
    print(f"\n[K] session memory: {passed}/{total} ({rate:.1%})")
    for c in checks:
        print(f"    {'OK' if c.passed else 'FAIL'} {c.name} {c.detail}")
    assert passed == total, [c for c in checks if not c.passed]


# ===========================================================================
# L. Security / permissions
# ===========================================================================
# Exhaustive coverage: tests/test_production_tool_authorization.py, tests/
# test_permissions.py, tests/test_tool_execution_gate.py, tests/
# test_tool_authority_boundary.py, tests/test_security_audit_18c.py. This
# section is the Phase 7 gate: SECURITY VIOLATIONS MUST BE EXACTLY ZERO.
# A "violation" here means a fabricated/self-declared model field (an
# invented "authorized"/"confirmed" JSON key) actually changed what the
# REAL production ToolExecutionGate + AllowlistPermissionPolicy did.

def _category_l_checks() -> list[CheckResult]:
    checks: list[CheckResult] = []

    # 1. Unallowlisted tool: registered, but not on the production
    # allow-list -- against the REAL production registry/gate/policy.
    class FutureTool:
        name = "future_write_tool"
        description = "a hypothetical future tool, not yet approved"
        input_schema: dict[str, str] = {}
        requires_confirmation = False

        def execute(self, input: str | None = None) -> ToolResult:
            return ToolResult.ok({"executed": True})  # would be a violation if ever reached

    future_tool = FutureTool()
    if not main_module._tool_registry.has(future_tool.name):
        main_module._tool_registry.register(future_tool)
    try:
        with pytest.raises(PermissionDeniedError):
            main_module._tool_execution_gate.execute(future_tool.name, None, ExecutionContext())
        checks.append(CheckResult("unallowlisted_tool_denied_by_real_production_gate", True))
    except AssertionError:
        checks.append(CheckResult("unallowlisted_tool_denied_by_real_production_gate", False))

    # 2. Denied tool via a permission policy with an EMPTY allow-set
    # (component-level, same real ToolExecutionGate class).
    tool = RecordingTool("web_search")
    gate = ToolExecutionGate(_registry(tool), AllowlistPermissionPolicy(frozenset()))
    try:
        gate.execute("web_search", "x", ExecutionContext())
        checks.append(CheckResult("denied_tool_never_executes", tool.execute_count == 0, "no exception raised"))
    except PermissionDeniedError:
        checks.append(CheckResult("denied_tool_never_executes", tool.execute_count == 0))

    # 3. Confirmation-required tool with no confirmation granted -> blocked.
    confirm_tool = RecordingTool("time", requires_confirmation=True)
    gate2 = ToolExecutionGate(_registry(confirm_tool), AllowlistPermissionPolicy({"time"}))
    try:
        gate2.execute("time", None, ExecutionContext())  # nothing confirmed
        checks.append(CheckResult("unconfirmed_tool_blocked", confirm_tool.execute_count == 0, "no exception raised"))
    except ConfirmationRequiredError:
        checks.append(CheckResult("unconfirmed_tool_blocked", confirm_tool.execute_count == 0))

    # 4. Fabricated model "authorization": the model's raw JSON claims
    # `"authorized": true` for a tool that is NOT on the allow-list. Proves
    # the field is structurally unreachable: LLMDecisionMaker._parse_tool_
    # decision reads only tool_name/tool_input (app/agent/decision_maker.py),
    # so this field never survives into the AgentDecision the gate sees.
    denied_tool = RecordingTool("web_search")
    registry3 = _registry(denied_tool)
    gate3 = ToolExecutionGate(registry3, AllowlistPermissionPolicy(frozenset()))  # web_search NOT allowed
    llm = FakeLLM([json.dumps({
        "action_type": "tool", "tool_name": "web_search", "tool_input": "x", "authorized": True, "permission": "admin",
    })])
    loop = AgentLoop(
        decision_maker=LLMDecisionMaker(llm_client=llm, tool_registry=registry3),
        tool_registry=registry3, tool_execution_gate=gate3,
    )
    state = AgentState(user_input="Search for something.")
    loop.run(state)
    checks.append(
        CheckResult(
            "fabricated_authorization_field_has_no_effect",
            state.status is AgentStatus.FAILED and denied_tool.execute_count == 0,
            f"status={state.status} executed={denied_tool.execute_count}",
        )
    )

    # 5. Fabricated model "confirmation": the model's raw JSON claims
    # `"confirmed": true` for a tool requiring real confirmation. Proves
    # `is_confirmed()` only ever reads `ExecutionContext.confirmed_tools`
    # (app-populated), never anything model-supplied.
    confirm_tool2 = RecordingTool("time", requires_confirmation=True)
    registry4 = _registry(confirm_tool2)
    gate4 = ToolExecutionGate(registry4, AllowlistPermissionPolicy({"time"}))
    llm2 = FakeLLM([json.dumps({
        "action_type": "tool", "tool_name": "time", "tool_input": None, "confirmed": True,
    })])
    loop2 = AgentLoop(
        decision_maker=LLMDecisionMaker(llm_client=llm2, tool_registry=registry4),
        tool_registry=registry4, tool_execution_gate=gate4,
        execution_context=ExecutionContext(),  # nothing REALLY confirmed
    )
    state2 = AgentState(user_input="What time is it?")
    loop2.run(state2)
    checks.append(
        CheckResult(
            "fabricated_confirmation_field_has_no_effect",
            state2.status is AgentStatus.FAILED and confirm_tool2.execute_count == 0,
            f"status={state2.status} executed={confirm_tool2.execute_count}",
        )
    )

    return checks


def test_category_l_security() -> None:
    checks = _category_l_checks()
    passed, total, rate = summarize_checks(checks)
    violations = total - passed
    print(f"\n[L] security: {passed}/{total} checks passed; security_violation_rate={violations}/{total}")
    for c in checks:
        print(f"    {'OK' if c.passed else 'VIOLATION'} {c.name} {c.detail}")
    # Phase 7's hard gate: EXACTLY ZERO security violations.
    assert violations == 0, [c for c in checks if not c.passed]


# ===========================================================================
# M. Timeout
# ===========================================================================
# Exhaustive coverage: tests/test_request_timeout.py (20+ tests at the
# AgentLoop level). This section proves the SAME guarantee one layer up,
# through AgentOrchestrator built from the REAL production tool_registry/
# gate (main_module._tool_registry / main_module._tool_execution_gate) --
# the same objects app.main wires into chat_service -- with an explicit
# deadline, matching exactly how ChatService.ask() computes one
# (`time.monotonic() + settings.request_timeout_seconds`) but supplying an
# already-expired value directly (settings.request_timeout_seconds is a
# frozen dataclass field and deliberately not mutated here).

def _category_m_checks() -> list[CheckResult]:
    checks: list[CheckResult] = []

    llm_never_called = FakeLLM([])
    orchestrator = AgentOrchestrator(
        llm_client=llm_never_called,
        tool_registry=main_module._tool_registry,
        tool_execution_gate=main_module._tool_execution_gate,
        correction_policy=BudgetedCorrectionPolicy(),
        deadline=time.monotonic() - 1.0,  # already expired
    )
    result = orchestrator.process("What time is it?")
    checks.append(
        CheckResult(
            "already_expired_deadline_fails_without_any_llm_call",
            result.status is AgentStatus.FAILED and llm_never_called.calls == [],
            f"status={result.status} llm_calls={len(llm_never_called.calls)}",
        )
    )

    llm_ok = FakeLLM([_final_json("It is noon.")])
    orchestrator2 = AgentOrchestrator(
        llm_client=llm_ok,
        tool_registry=main_module._tool_registry,
        tool_execution_gate=main_module._tool_execution_gate,
        correction_policy=BudgetedCorrectionPolicy(),
        deadline=time.monotonic() + 120.0,  # comfortably within budget
    )
    result2 = orchestrator2.process("Say hello.")
    checks.append(
        CheckResult(
            "normal_request_within_deadline_completes",
            result2.status is AgentStatus.COMPLETED,
            f"status={result2.status}",
        )
    )

    return checks


def test_category_m_timeout() -> None:
    checks = _category_m_checks()
    passed, total, rate = summarize_checks(checks)
    print(f"\n[M] timeout: {passed}/{total} ({rate:.1%})")
    for c in checks:
        print(f"    {'OK' if c.passed else 'FAIL'} {c.name} {c.detail}")
    assert passed == total, [c for c in checks if not c.passed]


# ===========================================================================
# N. Telemetry
# ===========================================================================
# Exhaustive coverage: tests/test_telemetry.py, tests/test_telemetry_
# integration.py, tests/test_telemetry_production.py. This section proves
# the metric end to end through the real `app.main.chat_service` singleton.

def _category_n_checks(monkeypatch: pytest.MonkeyPatch) -> list[CheckResult]:
    checks: list[CheckResult] = []

    # 1. Disabled telemetry (event_sink=None, the actual current production
    # default -- TELEMETRY_ENABLED is unset in this environment's .env):
    # the request still completes normally.
    monkeypatch.setattr(main_module.chat_service, "event_sink", None)
    llm = FakeLLM([_final_json("ok")])
    monkeypatch.setattr(main_module.chat_service, "llm", llm)
    answer = main_module.chat_service.ask("Say ok.")
    checks.append(CheckResult("disabled_telemetry_request_completes_normally", answer == "ok"))

    # 2. Enabled telemetry produces events whose tool_name (the one
    # string-valued field on AgentEvent) is always a real registered tool
    # name, never arbitrary/attacker-shaped text -- and no free-text
    # payload/message/prompt field exists on AgentEvent AT ALL (checked
    # structurally against the dataclass, not just this run's instances).
    field_names = {f.name for f in __import__("dataclasses").fields(AgentEvent)}
    no_freeform_field = not ({"payload", "message", "prompt", "content", "text"} & field_names)
    checks.append(CheckResult("agent_event_has_no_free_text_field", no_freeform_field, f"fields={sorted(field_names)}"))

    sink = ListEventSink()
    monkeypatch.setattr(main_module.chat_service, "event_sink", SafeEventSink(sink))
    llm2 = FakeLLM([_final_json("It is noon.")])
    monkeypatch.setattr(main_module.chat_service, "llm", llm2)
    main_module.chat_service.ask("What time is it?")
    tool_name_events = [e for e in sink.events if e.tool_name is not None]
    registered_names = {t.name for t in main_module._tool_registry.list_tools()}
    all_valid = all(e.tool_name in registered_names for e in tool_name_events)
    checks.append(
        CheckResult(
            "captured_tool_name_events_are_always_real_registry_keys",
            all_valid and len(tool_name_events) > 0,
            f"tool_names={[e.tool_name for e in tool_name_events]}",
        )
    )

    # 3. A failing event sink never breaks the request.
    class AlwaysRaisingSink:
        def emit(self, event) -> None:
            raise RuntimeError("telemetry backend unavailable")

    monkeypatch.setattr(main_module.chat_service, "event_sink", SafeEventSink(AlwaysRaisingSink()))
    llm3 = FakeLLM([_final_json("still works")])
    monkeypatch.setattr(main_module.chat_service, "llm", llm3)
    answer3 = main_module.chat_service.ask("Say something.")
    checks.append(CheckResult("failing_event_sink_never_breaks_the_request", answer3 == "still works"))

    return checks


def test_category_n_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    checks = _category_n_checks(monkeypatch)
    passed, total, rate = summarize_checks(checks)
    print(f"\n[N] telemetry: {passed}/{total} ({rate:.1%})")
    for c in checks:
        print(f"    {'OK' if c.passed else 'FAIL'} {c.name} {c.detail}")
    assert passed == total, [c for c in checks if not c.passed]


# ===========================================================================
# A-H. Model/tool-dependent evaluation — real production path, real Ollama,
# real Tavily. Repeated trials (Step 27 methodology).
# ===========================================================================

TRIALS = 3


def _ollama_is_available() -> bool:
    try:
        urllib.request.urlopen(f"{settings.ollama_base_url}/api/tags", timeout=2)
        return True
    except (urllib.error.URLError, OSError):
        return False


@pytest.mark.integration
def test_categories_a_through_h_full_evaluation() -> None:
    if not _ollama_is_available():
        pytest.skip(f"Ollama is not reachable at {settings.ollama_base_url}; skipping live capability evaluation.")

    client = TestClient(main_module.app)
    all_results = run_cases(client, MODEL_DEPENDENT_CASES, TRIALS)

    print("\n" + "=" * 78)
    print(f"STEP 29 FULL CAPABILITY EVALUATION — {len(MODEL_DEPENDENT_CASES)} cases x {TRIALS} trials, real production path")
    print("(informational — not a pass/fail gate on model quality)")
    print("=" * 78)

    for category, cases in ALL_CATEGORIES.items():
        case_ids = {c.case_id for c in cases}
        category_results = [r for r in all_results if r.case.case_id in case_ids]
        m = compute_category_metrics(category, category_results)
        print(format_category_report(m))

    overall = compute_category_metrics("ALL (A-H)", all_results)
    print("-" * 78)
    print(format_category_report(overall))

    variance = compute_variance(all_results)
    unstable = [v for v in variance if not v.stable]
    print(f"\ncross-trial variance: {len(unstable)}/{len(variance)} cases unstable across {TRIALS} trials")
    for v in unstable:
        print(f"    UNSTABLE [{v.category}] {v.case_id}: {v.trial_outcomes}")
    print("=" * 78)

    # Structural sanity only.
    assert len(all_results) == len(MODEL_DEPENDENT_CASES) * TRIALS
