"""Observability primitives (Milestone 19, Phase 1): a small, typed event
model plus an injectable sink, following the Phase-1 design report's
approved architecture.

    ChatService.ask()
        request_id = uuid4()          <-- generated HERE, once per request
        emitter = EventEmitter(self.event_sink, request_id, session_id)
            |
            v  (explicit constructor injection, never global/contextvar state)
    AgentOrchestrator -> AgentLoop -> LLMDecisionMaker
            |
            v
    EventEmitter.emit(event_type, **fields) -> builds an AgentEvent,
                                                 forwards to EventSink.emit()

--------------------------------------------------------------------------
Observational only — never a participant
--------------------------------------------------------------------------
Every rule below is a hard constraint, not a suggestion:
- Nothing in this module calls into ToolRegistry, PermissionPolicy,
  ToolExecutionGate, or an LLM. It has no way to.
- `EventEmitter.emit()` is fire-and-forget: it returns None, so no caller
  can branch on what emission "decided" — there is nothing to branch on.
- A user-supplied `EventSink` that raises is NEVER allowed to fail the
  request it is describing. `EventEmitter` always emits through a
  `SafeEventSink` wrapper (constructing one automatically if the sink
  given to it isn't already one) — see `SafeEventSink` below. This is a
  structural guarantee, verified in tests/test_telemetry.py and
  tests/test_telemetry_integration.py by a sink whose `emit` always
  raises, run all the way through AgentLoop/ToolExecutionGate.
- `AgentEvent` has a closed, typed field set — deliberately no
  `payload: dict`, no `message: str`, no `detail`. There is nowhere to put
  raw prompts, raw model output, raw tool input/output, or memory content,
  so none of it can leak through this module by construction. The one
  string-valued field, `tool_name`, is always a validated ToolRegistry key
  (see app/agent/tool_execution.py's identity-binding check) — never
  arbitrary text.
- Nothing here is read back by the agent. `EventEmitter`/`EventSink` are
  write-only from the agent's perspective; the only readers are test code
  and app/agent/execution_evaluation.py, both of which consume a finished
  execution's events after the fact.

--------------------------------------------------------------------------
Why request_id lives on EventEmitter, not on AgentState or a contextvar
--------------------------------------------------------------------------
`ChatService.ask()` already builds a fresh `AgentOrchestrator` per call
specifically to avoid two concurrent requests racing on shared mutable
state (FastAPI runs sync handlers in a thread pool — see
app/services/chat.py's module docstring). A module-level telemetry
singleton or a `contextvars`-based ambient request id would reintroduce
exactly that hazard. `EventEmitter` is instead a small, per-request value
object, passed down through ordinary constructor parameters — the same
explicit-DI pattern this codebase already uses for `correction_policy`,
`tool_execution_gate`, `execution_context`, and every other injectable
collaborator. `session_id` is carried for correlation only and is never
used as (or in place of) `request_id`: it is often None, and it is shared
across many requests by design (app/agent/permissions.py's module
docstring on why `session_id` confers no privilege) — using it as a
request identity would silently merge unrelated requests' event streams.

--------------------------------------------------------------------------
Sequence, event_id, timestamp
--------------------------------------------------------------------------
`EventEmitter` — not any individual `EventSink` — is the single owner of
sequence assignment: one emitter instance is constructed per request and
its internal counter is the one true order for that request, independent
of how many sinks eventually receive the event or how they are combined.
`event_id` is a fresh uuid4 per event (for external correlation, e.g. a
future log aggregator); `timestamp` is wall-clock UTC, for human display
only — duration measurements elsewhere in this codebase use
`time.perf_counter()` instead (see `monotonic_start`/`elapsed_ms` below),
since wall-clock time is not safe to subtract for a duration (clock
adjustments, DST, leap seconds).
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Protocol, runtime_checkable

from app.agent.permissions import PermissionDecision
from app.agent.reliability import FailureCategory

logger = logging.getLogger(__name__)


class EventType(Enum):
    """The closed set of lifecycle events this milestone emits. Deliberately
    NOT the full brainstormed list from the Phase-1 design brief —
    STEP_COMPLETED and CORRECTION_COMPLETED were considered and dropped
    (see the Phase-1 design report §4): both are fully derivable from
    adjacent events already in this set, and adding them would only
    double event volume for zero reconstructive value.
    """

    REQUEST_STARTED = "request.started"
    REQUEST_COMPLETED = "request.completed"
    REQUEST_FAILED = "request.failed"
    PLAN_CREATED = "plan.created"
    STEP_STARTED = "step.started"
    LLM_CALL_STARTED = "llm.call.started"
    LLM_CALL_COMPLETED = "llm.call.completed"
    LLM_CALL_FAILED = "llm.call.failed"
    TOOL_PROPOSED = "tool.proposed"
    TOOL_AUTHORIZED = "tool.authorized"
    TOOL_DENIED = "tool.denied"
    TOOL_CONFIRMATION_REQUIRED = "tool.confirmation_required"
    TOOL_INPUT_REJECTED = "tool.input.rejected"
    TOOL_EXECUTION_COMPLETED = "tool.execution.completed"
    CORRECTION_TRIGGERED = "correction.triggered"
    CORRECTION_DECLINED = "correction.declined"


class LLMPurpose(Enum):
    """Why a given LLM call was made — read by the event, never by the
    model. `ROUTE` is defined for completeness but is not emitted by any
    current call site: `Router.classify_hint()` (app/agent/router.py) is
    explicitly LLM-free (see app/agent/decision_maker.py's module
    docstring) — nothing in this codebase today makes a real LLM call to
    route. A future Router.decide() call site could emit it without
    changing this enum.
    """

    DECIDE = "decide"
    PLAN = "plan"
    EXTRACT = "extract"
    ROUTE = "route"


@dataclass(frozen=True)
class AgentEvent:
    """One immutable, structured fact about a running request.

    Every field is an id, enum, bool, count, or duration — there is no
    free-form string field that could ever hold a prompt, a model
    response, tool input/output, or memory content. `tool_name` is the
    one exception, and it is constrained to a validated registry key by
    every call site that sets it (see app/agent/loop.py) — never raw
    model-invented text (an unregistered tool name is rejected before any
    event referencing it is ever built).
    """

    event_id: str
    request_id: str
    event_type: EventType
    sequence: int
    timestamp: datetime
    session_id: str | None = None
    step: int | None = None
    tool_name: str | None = None
    duration_ms: float | None = None
    success: bool | None = None
    failure_category: FailureCategory | None = None
    permission_decision: PermissionDecision | None = None
    llm_purpose: LLMPurpose | None = None
    plan_step_count: int | None = None
    item_count: int | None = None
    output_chars: int | None = None


@runtime_checkable
class EventSink(Protocol):
    """Whatever receives finished `AgentEvent`s. A Protocol, matching every
    other injectable collaborator in this codebase (Tool, DecisionMaker,
    CorrectionPolicy, PermissionPolicy, ...): no base class required.

    `emit` MUST NOT raise in a well-behaved implementation, but nothing in
    this codebase trusts that: see `SafeEventSink` and `EventEmitter`,
    which together guarantee a raising sink can never affect execution.
    """

    def emit(self, event: AgentEvent) -> None:
        ...


class SafeEventSink:
    """Wraps another `EventSink` and swallows every exception it raises,
    logging the exception's TYPE NAME only (never its message, which could
    echo whatever a buggy sink implementation was trying to do with event
    data) at WARNING.

    This is what makes "telemetry exceptions must never fail the request"
    a structural property of the sink itself, not merely a convention
    callers are expected to follow. `EventEmitter` (below) applies this
    wrapper automatically to whatever sink it is given, so constructing
    one directly is optional — it exists as its own class because the
    Milestone 19 design calls for it explicitly as a reusable, testable
    primitive, and because a caller composing sinks by hand (e.g. wrapping
    only one branch of a multiplexing sink) may want it independently of
    `EventEmitter`.
    """

    def __init__(self, wrapped: EventSink) -> None:
        self._wrapped = wrapped

    def emit(self, event: AgentEvent) -> None:
        try:
            self._wrapped.emit(event)
        except Exception as exc:  # noqa: BLE001 - deliberately broad: telemetry must never propagate
            logger.warning("telemetry.sink.failed sink=%s error_type=%s", type(self._wrapped).__name__, type(exc).__name__)


class ListEventSink:
    """A bounded, in-memory collector — the sink tests and
    app/agent/execution_evaluation.py use to make an execution's event
    stream directly inspectable.

    Bounded (Phase-1 design report §12: "avoid unbounded event storage"):
    once `max_events` is reached, the OLDEST event is dropped to make room
    for the newest, and `dropped_count` records how many were discarded —
    so truncation is visible rather than silent. A single request's event
    count is already bounded by `max_iterations` (see app/agent/loop.py),
    so `max_events` is a second, independent, much larger safety net, not
    the primary bound.
    """

    def __init__(self, max_events: int = 10_000) -> None:
        if not isinstance(max_events, int) or max_events <= 0:
            raise ValueError("max_events must be a positive integer.")
        self.max_events = max_events
        self.events: list[AgentEvent] = []
        self.dropped_count = 0

    def emit(self, event: AgentEvent) -> None:
        if len(self.events) >= self.max_events:
            self.events.pop(0)
            self.dropped_count += 1
        self.events.append(event)

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self):
        return iter(self.events)


# Event types whose occurrence represents a rejected/failed/declined
# outcome, rendered at WARNING by LoggingEventSink; every other event type
# is rendered at INFO. A closed set, not a name-based heuristic ("contains
# 'fail'"), so a future EventType addition must explicitly choose a level.
_WARNING_EVENT_TYPES = frozenset(
    {
        EventType.REQUEST_FAILED,
        EventType.LLM_CALL_FAILED,
        EventType.TOOL_DENIED,
        EventType.TOOL_CONFIRMATION_REQUIRED,
        EventType.TOOL_INPUT_REJECTED,
        EventType.CORRECTION_DECLINED,
    }
)


class LoggingEventSink:
    """Renders each `AgentEvent` as one human-readable line on the existing
    stdlib logging pipeline — the "derived human-readable view" the
    Phase-1 design calls for, so structured events and logs never drift
    apart by being maintained as two independent hand-written things.

    The rendered line follows the SAME dotted-event-name + key=value
    convention already used by app/agent/tool_execution.py and
    app/agent/loop.py's pre-existing log calls (e.g.
    "tool.authorization.denied tool=%s") — this formalizes that
    convention rather than inventing a new one. Only the closed,
    non-sensitive `AgentEvent` fields are ever rendered; there is nothing
    else on the event that could be rendered.
    """

    def __init__(self, logger_: logging.Logger | None = None) -> None:
        self._logger = logger_ or logger

    def emit(self, event: AgentEvent) -> None:
        parts = [event.event_type.value, f"request_id={event.request_id}"]
        if event.session_id is not None:
            parts.append(f"session_id={event.session_id}")
        if event.step is not None:
            parts.append(f"step={event.step}")
        if event.tool_name is not None:
            parts.append(f"tool={event.tool_name}")
        if event.llm_purpose is not None:
            parts.append(f"purpose={event.llm_purpose.value}")
        if event.permission_decision is not None:
            parts.append(f"decision={event.permission_decision.value}")
        if event.failure_category is not None:
            parts.append(f"category={event.failure_category.value}")
        if event.success is not None:
            parts.append(f"success={event.success}")
        if event.duration_ms is not None:
            parts.append(f"duration_ms={event.duration_ms:.1f}")
        if event.plan_step_count is not None:
            parts.append(f"plan_step_count={event.plan_step_count}")
        if event.item_count is not None:
            parts.append(f"item_count={event.item_count}")
        if event.output_chars is not None:
            parts.append(f"output_chars={event.output_chars}")

        level = logging.WARNING if event.event_type in _WARNING_EVENT_TYPES else logging.INFO
        self._logger.log(level, " ".join(parts))


class EventEmitter:
    """The per-request handle every component receives (or doesn't — the
    parameter defaults to `None` everywhere it's threaded through). Owns
    `request_id`/`session_id` and sequence assignment for exactly one
    request; never shared across requests, never global.

    Constructing one with `sink=None` is deliberately unsupported — a
    caller with no sink should pass `event_emitter=None` all the way
    through instead (see every call site in app/agent/loop.py,
    app/agent/decision_maker.py, app/agent/orchestrator.py), so that
    disabled telemetry constructs neither an `EventEmitter` nor a single
    `AgentEvent`. This class exists only for the case where a real sink
    IS present.
    """

    def __init__(self, sink: EventSink, request_id: str, session_id: str | None = None) -> None:
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError("request_id must be a non-empty string.")
        # Always emit through a SafeEventSink, so a raising sink can never
        # affect the request being described — a structural guarantee, not
        # a convention every caller must remember. Avoids double-wrapping
        # if the caller already passed one.
        self._sink: EventSink = sink if isinstance(sink, SafeEventSink) else SafeEventSink(sink)
        self.request_id = request_id
        self.session_id = session_id
        self._sequence = 0

    def emit(self, event_type: EventType, **fields: object) -> None:
        self._sequence += 1
        event = AgentEvent(
            event_id=str(uuid.uuid4()),
            request_id=self.request_id,
            event_type=event_type,
            sequence=self._sequence,
            timestamp=datetime.now(timezone.utc),
            session_id=self.session_id,
            **fields,  # type: ignore[arg-type]
        )
        self._sink.emit(event)


def monotonic_start() -> float:
    """A monotonic timer start value — use with `elapsed_ms` to compute a
    duration immune to wall-clock adjustments. Never use this value for
    anything but subtraction; it has no meaning as an absolute time."""
    return time.perf_counter()


def elapsed_ms(start: float) -> float:
    """Milliseconds elapsed since `start` (from `monotonic_start()`)."""
    return (time.perf_counter() - start) * 1000.0
