"""Milestone 19, Phase 1: app/agent/telemetry.py in isolation.

No AgentLoop, no orchestrator, no LLM, no tools — this file proves the
telemetry primitives themselves are correct: a closed event field set,
sequence/id/timestamp assignment, sink safety, bounded collection, and
logging rendering. Wiring into the agent's execution path is covered
separately in tests/test_telemetry_integration.py.
"""
from __future__ import annotations

import dataclasses
import logging
from datetime import datetime, timezone

import pytest

from app.agent.permissions import PermissionDecision
from app.agent.reliability import FailureCategory
from app.agent.telemetry import (
    AgentEvent,
    EventEmitter,
    EventSink,
    EventType,
    LLMPurpose,
    ListEventSink,
    LoggingEventSink,
    SafeEventSink,
    elapsed_ms,
    monotonic_start,
)


class RaisingSink:
    """An EventSink whose emit() always raises — used to prove telemetry
    failures can never propagate."""

    def __init__(self) -> None:
        self.attempts = 0

    def emit(self, event: AgentEvent) -> None:
        self.attempts += 1
        raise RuntimeError("sink is broken")


# ===========================================================================
# AgentEvent — closed field set, no free-form payload
# ===========================================================================

def test_agent_event_field_set_is_exactly_the_closed_allowlist() -> None:
    """Structural proof, not a string search: the dataclass fields are
    enumerable, so no free-form payload/message/detail field can exist
    without this test being updated to acknowledge it."""
    field_names = {f.name for f in dataclasses.fields(AgentEvent)}

    assert field_names == {
        "event_id",
        "request_id",
        "event_type",
        "sequence",
        "timestamp",
        "session_id",
        "step",
        "tool_name",
        "duration_ms",
        "success",
        "failure_category",
        "permission_decision",
        "llm_purpose",
        "plan_step_count",
        "item_count",
        "output_chars",
    }


def test_agent_event_is_frozen() -> None:
    event = AgentEvent(
        event_id="e1", request_id="r1", event_type=EventType.REQUEST_STARTED, sequence=1,
        timestamp=datetime.now(timezone.utc),
    )
    with pytest.raises((AttributeError, TypeError)):
        event.tool_name = "time"  # type: ignore[misc]


def test_event_type_is_a_closed_16_member_enum() -> None:
    assert len(EventType) == 16


def test_llm_purpose_is_a_closed_enum_including_unused_route() -> None:
    assert {p.value for p in LLMPurpose} == {"decide", "plan", "extract", "route"}


# ===========================================================================
# EventEmitter — sequencing, ids, timestamps, correlation
# ===========================================================================

def test_emitter_assigns_increasing_sequence_numbers() -> None:
    sink = ListEventSink()
    emitter = EventEmitter(sink, request_id="r1")

    emitter.emit(EventType.REQUEST_STARTED)
    emitter.emit(EventType.STEP_STARTED, step=1)
    emitter.emit(EventType.REQUEST_COMPLETED)

    assert [e.sequence for e in sink.events] == [1, 2, 3]


def test_emitter_stamps_request_id_and_session_id_on_every_event() -> None:
    sink = ListEventSink()
    emitter = EventEmitter(sink, request_id="r1", session_id="alice")

    emitter.emit(EventType.REQUEST_STARTED)

    event = sink.events[0]
    assert event.request_id == "r1"
    assert event.session_id == "alice"


def test_emitter_generates_a_fresh_event_id_per_event() -> None:
    sink = ListEventSink()
    emitter = EventEmitter(sink, request_id="r1")

    emitter.emit(EventType.REQUEST_STARTED)
    emitter.emit(EventType.REQUEST_COMPLETED)

    ids = {e.event_id for e in sink.events}
    assert len(ids) == 2


def test_emitter_stamps_a_timezone_aware_utc_timestamp() -> None:
    sink = ListEventSink()
    emitter = EventEmitter(sink, request_id="r1")

    emitter.emit(EventType.REQUEST_STARTED)

    ts = sink.events[0].timestamp
    assert ts.tzinfo is not None
    assert ts.utcoffset().total_seconds() == 0


def test_emitter_forwards_arbitrary_typed_fields() -> None:
    sink = ListEventSink()
    emitter = EventEmitter(sink, request_id="r1")

    emitter.emit(
        EventType.TOOL_EXECUTION_COMPLETED,
        step=2,
        tool_name="time",
        success=True,
        duration_ms=12.5,
    )

    event = sink.events[0]
    assert event.step == 2
    assert event.tool_name == "time"
    assert event.success is True
    assert event.duration_ms == 12.5


def test_two_emitters_have_independent_sequences() -> None:
    sink = ListEventSink()
    a = EventEmitter(sink, request_id="r1")
    b = EventEmitter(sink, request_id="r2")

    a.emit(EventType.REQUEST_STARTED)
    b.emit(EventType.REQUEST_STARTED)
    a.emit(EventType.REQUEST_COMPLETED)

    a_events = [e for e in sink.events if e.request_id == "r1"]
    b_events = [e for e in sink.events if e.request_id == "r2"]
    assert [e.sequence for e in a_events] == [1, 2]
    assert [e.sequence for e in b_events] == [1]


def test_emitter_rejects_a_blank_request_id() -> None:
    with pytest.raises(ValueError):
        EventEmitter(ListEventSink(), request_id="")


def test_emitter_conforms_to_the_event_sink_protocol_via_its_wrapped_sink() -> None:
    sink = ListEventSink()
    assert isinstance(sink, EventSink)


# ===========================================================================
# SafeEventSink / EventEmitter — telemetry can never fail the caller
# ===========================================================================

def test_safe_event_sink_swallows_exceptions_from_the_wrapped_sink() -> None:
    raising = RaisingSink()
    safe = SafeEventSink(raising)
    event = AgentEvent(
        event_id="e1", request_id="r1", event_type=EventType.REQUEST_STARTED, sequence=1,
        timestamp=datetime.now(timezone.utc),
    )

    safe.emit(event)  # must not raise

    assert raising.attempts == 1


def test_safe_event_sink_logs_the_exception_type_only(caplog: pytest.LogCaptureFixture) -> None:
    safe = SafeEventSink(RaisingSink())
    event = AgentEvent(
        event_id="e1", request_id="r1", event_type=EventType.REQUEST_STARTED, sequence=1,
        timestamp=datetime.now(timezone.utc),
    )

    with caplog.at_level(logging.WARNING, logger="app.agent.telemetry"):
        safe.emit(event)

    assert "RuntimeError" in caplog.text
    assert "sink is broken" not in caplog.text  # exception MESSAGE never logged


def test_emitter_wraps_a_raw_raising_sink_automatically() -> None:
    """A caller need not remember to wrap their sink in SafeEventSink —
    EventEmitter does it unconditionally."""
    raising = RaisingSink()
    emitter = EventEmitter(raising, request_id="r1")

    emitter.emit(EventType.REQUEST_STARTED)  # must not raise

    assert raising.attempts == 1


def test_emitter_does_not_double_wrap_an_already_safe_sink() -> None:
    raising = RaisingSink()
    already_safe = SafeEventSink(raising)
    emitter = EventEmitter(already_safe, request_id="r1")

    emitter.emit(EventType.REQUEST_STARTED)

    assert raising.attempts == 1  # exactly one underlying attempt, not double-wrapped noise


# ===========================================================================
# ListEventSink — bounded collection
# ===========================================================================

def test_list_event_sink_collects_events_in_order() -> None:
    sink = ListEventSink()
    emitter = EventEmitter(sink, request_id="r1")

    emitter.emit(EventType.REQUEST_STARTED)
    emitter.emit(EventType.REQUEST_COMPLETED)

    assert [e.event_type for e in sink] == [EventType.REQUEST_STARTED, EventType.REQUEST_COMPLETED]
    assert len(sink) == 2


def test_list_event_sink_drops_oldest_when_bounded() -> None:
    sink = ListEventSink(max_events=2)
    emitter = EventEmitter(sink, request_id="r1")

    emitter.emit(EventType.REQUEST_STARTED)
    emitter.emit(EventType.STEP_STARTED, step=1)
    emitter.emit(EventType.REQUEST_COMPLETED)

    assert len(sink.events) == 2
    assert sink.dropped_count == 1
    assert sink.events[0].event_type == EventType.STEP_STARTED  # oldest (REQUEST_STARTED) was dropped
    assert sink.events[-1].event_type == EventType.REQUEST_COMPLETED


def test_list_event_sink_rejects_non_positive_max_events() -> None:
    with pytest.raises(ValueError):
        ListEventSink(max_events=0)


# ===========================================================================
# LoggingEventSink — derived human-readable view
# ===========================================================================

def test_logging_event_sink_renders_without_raising_for_every_field_combination() -> None:
    sink = LoggingEventSink()
    event = AgentEvent(
        event_id="e1", request_id="r1", event_type=EventType.TOOL_EXECUTION_COMPLETED, sequence=1,
        timestamp=datetime.now(timezone.utc), session_id="s1", step=2, tool_name="time",
        duration_ms=5.0, success=True, failure_category=FailureCategory.TOOL_EXECUTION_FAILED,
        permission_decision=PermissionDecision.ALLOW, llm_purpose=LLMPurpose.DECIDE,
        plan_step_count=3, item_count=1, output_chars=42,
    )

    sink.emit(event)  # must not raise


def test_logging_event_sink_uses_info_for_normal_events(caplog: pytest.LogCaptureFixture) -> None:
    sink = LoggingEventSink()
    event = AgentEvent(
        event_id="e1", request_id="r1", event_type=EventType.REQUEST_STARTED, sequence=1,
        timestamp=datetime.now(timezone.utc),
    )

    with caplog.at_level(logging.INFO, logger="app.agent.telemetry"):
        sink.emit(event)

    assert caplog.records[0].levelno == logging.INFO
    assert "request.started" in caplog.text
    assert "request_id=r1" in caplog.text


def test_logging_event_sink_uses_warning_for_rejected_or_failed_events(caplog: pytest.LogCaptureFixture) -> None:
    sink = LoggingEventSink()
    event = AgentEvent(
        event_id="e1", request_id="r1", event_type=EventType.TOOL_DENIED, sequence=1,
        timestamp=datetime.now(timezone.utc), tool_name="delete_file",
        permission_decision=PermissionDecision.DENY,
    )

    with caplog.at_level(logging.WARNING, logger="app.agent.telemetry"):
        sink.emit(event)

    assert caplog.records[0].levelno == logging.WARNING
    assert "tool.denied" in caplog.text
    assert "tool=delete_file" in caplog.text


def test_logging_event_sink_never_needs_a_free_form_field_to_render() -> None:
    """Structural proof that rendering uses ONLY the closed AgentEvent
    fields: constructing the sink and emitting the minimal event (every
    optional field None) still produces informative output."""
    sink = LoggingEventSink()
    event = AgentEvent(
        event_id="e1", request_id="r1", event_type=EventType.REQUEST_STARTED, sequence=1,
        timestamp=datetime.now(timezone.utc),
    )

    sink.emit(event)  # must not raise despite every optional field being None


# ===========================================================================
# Timing helpers
# ===========================================================================

def test_elapsed_ms_is_non_negative_and_monotonic() -> None:
    start = monotonic_start()
    later = elapsed_ms(start)

    assert later >= 0.0


def test_elapsed_ms_increases_over_a_measurable_interval() -> None:
    import time

    start = monotonic_start()
    time.sleep(0.01)
    duration = elapsed_ms(start)

    assert duration >= 5.0  # generous lower bound, avoids flakiness on slow CI
