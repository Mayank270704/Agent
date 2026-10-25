"""Step 29 — executes EvalCases through the REAL production path.

    FastAPI TestClient -> POST /chat
        -> app.main.chat_service.ask()          (the real, module-level singleton)
        -> AgentOrchestrator -> AgentLoop -> LLMDecisionMaker -> (real Ollama)
        -> ToolExecutionGate -> Tool -> (real Tavily for web_search)
        -> Observation -> ... -> final reply

Per-request instrumentation is done the SAME way Milestone 19 telemetry
already supports observing a request without touching its outcome: the
real `chat_service.event_sink` instance attribute is swapped for a
`SafeEventSink(ListEventSink())` for the DURATION OF THIS MODULE'S CALLS
ONLY (via `capture_events`, a context manager that restores the original
attribute afterward) — the identical, already-established technique
`tests/test_deterministic_web_search_routing.py` uses to swap `chat_service.
llm` for a fake. `EventEmitter`/`AgentEvent` are documented (app/agent/
telemetry.py's module docstring) as purely observational and unable to
affect execution, so this does not change agent behavior — it only makes
an already-real request's internal decisions (which tool was proposed,
how many LLM calls were made, whether execution succeeded) inspectable
without reading chat_service internals directly or duplicating its logic.
`TELEMETRY_ENABLED` in `.env`/`app/config.py` is NOT touched — this swap
is local to the evaluation run and reverted afterward, in a `finally`.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

from fastapi.testclient import TestClient

import app.main as main_module
from app.agent.telemetry import EventType, ListEventSink, SafeEventSink
from tests.evaluation.dataset import EvalCase


@contextmanager
def capture_events():
    """Swap the REAL production `chat_service.event_sink` for a fresh
    `ListEventSink` (wrapped in `SafeEventSink`, matching main.py's own
    wrapping exactly) for the duration of the `with` block, yielding the
    sink so callers can read `.events` afterward. Restores the ORIGINAL
    attribute on exit, always — even on an exception mid-evaluation — so
    a Step 29 test run never permanently alters the module-level
    singleton other tests in the same process depend on.
    """
    original_sink = main_module.chat_service.event_sink
    sink = ListEventSink()
    main_module.chat_service.event_sink = SafeEventSink(sink)
    try:
        yield sink
    finally:
        main_module.chat_service.event_sink = original_sink


@dataclass
class CaseExecutionResult:
    """One (case, trial) execution's full observable outcome."""

    case: EvalCase
    trial_index: int
    status_code: int
    reply: str | None
    latency_seconds: float
    proposed_tools: list[str] = field(default_factory=list)
    executed_tools: list[tuple[str, bool]] = field(default_factory=list)  # (tool_name, success)
    llm_call_count: int = 0
    llm_seconds: float = 0.0
    request_succeeded: bool | None = None  # from REQUEST_COMPLETED/REQUEST_FAILED, None if no event captured
    corrections_triggered: int = 0
    error: str | None = None

    @property
    def final_answer_ok(self) -> bool | None:
        """None when the case has no deterministic checker (Phase 3:
        genuinely open-ended cases are not force-graded)."""
        if self.case.final_answer_check is None or self.reply is None:
            return None
        try:
            return bool(self.case.final_answer_check(self.reply))
        except Exception:
            return False

    @property
    def tool_selection_ok(self) -> bool | None:
        """None for cases that don't require a tool at all (nothing to
        grade) — see `unnecessary_tool_call` for the companion check."""
        if not self.case.requires_tool:
            return None
        return self.case.expected_tool in self.proposed_tools

    @property
    def unnecessary_tool_call(self) -> bool:
        """True if a case expecting FINAL nonetheless proposed a tool."""
        return not self.case.requires_tool and len(self.proposed_tools) > 0

    @property
    def missed_tool_call(self) -> bool:
        """True if a case expecting a specific tool never proposed it."""
        return self.case.requires_tool and self.case.expected_tool not in self.proposed_tools


def _events_by_type(events: list, event_type: EventType) -> list:
    return [e for e in events if e.event_type is event_type]


def run_case_once(client: TestClient, case: EvalCase, trial_index: int) -> CaseExecutionResult:
    """Execute exactly one (case, trial) through the real HTTP endpoint,
    capturing this call's own telemetry slice only (events emitted by
    OTHER concurrent activity are not possible here — TestClient calls are
    synchronous/sequential within one process — but slicing by index is
    still done defensively rather than assumed)."""
    with capture_events() as sink:
        start_index = len(sink.events)
        start = time.monotonic()
        try:
            response = client.post(
                "/chat", json={"message": case.message, "session_id": case.session_id}
            )
            latency = time.monotonic() - start
            reply = response.json().get("reply") if response.status_code == 200 else None
            error = None if response.status_code == 200 else f"HTTP {response.status_code}: {response.text[:200]}"
        except Exception as exc:  # a genuine infrastructure failure (e.g. Ollama down mid-run)
            latency = time.monotonic() - start
            response = None
            reply = None
            error = f"{type(exc).__name__}: {exc}"

        this_call_events = sink.events[start_index:]

    proposed = [e.tool_name for e in _events_by_type(this_call_events, EventType.TOOL_PROPOSED) if e.tool_name]
    executed = [
        (e.tool_name, bool(e.success))
        for e in _events_by_type(this_call_events, EventType.TOOL_EXECUTION_COMPLETED)
        if e.tool_name
    ]
    llm_started = _events_by_type(this_call_events, EventType.LLM_CALL_STARTED)
    llm_completed = _events_by_type(this_call_events, EventType.LLM_CALL_COMPLETED)
    llm_seconds = sum((e.duration_ms or 0.0) for e in llm_completed) / 1000.0
    completed = _events_by_type(this_call_events, EventType.REQUEST_COMPLETED)
    failed = _events_by_type(this_call_events, EventType.REQUEST_FAILED)
    request_succeeded = True if completed else (False if failed else None)
    corrections = len(_events_by_type(this_call_events, EventType.CORRECTION_TRIGGERED))

    return CaseExecutionResult(
        case=case,
        trial_index=trial_index,
        status_code=response.status_code if response is not None else 0,
        reply=reply,
        latency_seconds=latency,
        proposed_tools=proposed,
        executed_tools=executed,
        llm_call_count=len(llm_started),
        llm_seconds=llm_seconds,
        request_succeeded=request_succeeded,
        corrections_triggered=corrections,
        error=error,
    )


def run_cases(client: TestClient, cases: list[EvalCase], trials: int) -> list[CaseExecutionResult]:
    """Run every case for `trials` repeated trials, sequentially (this
    codebase's production request handling is itself synchronous — see
    app/main.py's `def chat(...)`, not `async def` — so sequential
    TestClient calls exercise the exact same code path a real deployment
    would under one worker)."""
    results: list[CaseExecutionResult] = []
    for case in cases:
        for trial in range(trials):
            results.append(run_case_once(client, case, trial))
    return results
