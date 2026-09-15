from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.agent.episodic_memory import EpisodicMemory, EpisodicMemoryRecord, InMemoryEpisodicMemory


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _record(**overrides: object) -> EpisodicMemoryRecord:
    fields: dict[str, object] = {
        "event_id": "evt-1",
        "session_id": "A",
        "event_type": "conversation_completed",
        "summary": "User discussed AI agents.",
        "timestamp": _now(),
        "metadata": {"steps": 1},
    }
    fields.update(overrides)
    return EpisodicMemoryRecord(**fields)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 1-2: record creation, immutability.
# ---------------------------------------------------------------------------

def test_record_creation_with_valid_fields() -> None:
    ts = _now()
    record = _record(timestamp=ts)

    assert record.event_id == "evt-1"
    assert record.session_id == "A"
    assert record.event_type == "conversation_completed"
    assert record.summary == "User discussed AI agents."
    assert record.timestamp == ts
    assert dict(record.metadata) == {"steps": 1}


def test_record_is_immutable() -> None:
    record = _record()

    with pytest.raises(Exception):
        record.summary = "changed"  # type: ignore[misc]


def test_record_metadata_cannot_be_mutated_after_construction() -> None:
    record = _record(metadata={"steps": 1})

    with pytest.raises(Exception):
        record.metadata["steps"] = 999  # type: ignore[index]


# ---------------------------------------------------------------------------
# 3-6: field validation.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_value", ["", "   ", None, 123])
def test_event_id_validation(bad_value: object) -> None:
    with pytest.raises(ValueError):
        _record(event_id=bad_value)


@pytest.mark.parametrize("bad_value", ["", "   ", None, 123])
def test_session_id_validation(bad_value: object) -> None:
    with pytest.raises(ValueError):
        _record(session_id=bad_value)


@pytest.mark.parametrize("bad_value", ["", "   ", None, 123])
def test_event_type_validation(bad_value: object) -> None:
    with pytest.raises(ValueError):
        _record(event_type=bad_value)


@pytest.mark.parametrize("bad_value", ["", "   ", None, 123])
def test_summary_validation(bad_value: object) -> None:
    with pytest.raises(ValueError):
        _record(summary=bad_value)


# ---------------------------------------------------------------------------
# 7: timezone-aware timestamp validation.
# ---------------------------------------------------------------------------

def test_naive_timestamp_is_rejected() -> None:
    with pytest.raises(ValueError):
        _record(timestamp=datetime.now())  # naive, no tzinfo


def test_non_datetime_timestamp_is_rejected() -> None:
    with pytest.raises(ValueError):
        _record(timestamp="2026-01-01T00:00:00Z")  # type: ignore[arg-type]


def test_timezone_aware_timestamp_is_accepted() -> None:
    record = _record(timestamp=_now())
    assert record.timestamp.tzinfo is not None


# ---------------------------------------------------------------------------
# 8: metadata isolation — mutating the caller's original dict afterward
# must not affect the stored record.
# ---------------------------------------------------------------------------

def test_metadata_isolation_from_caller_dict() -> None:
    caller_metadata = {"steps": 1}
    record = _record(metadata=caller_metadata)

    caller_metadata["steps"] = 999
    caller_metadata["new_key"] = "sneaky"

    assert dict(record.metadata) == {"steps": 1}


def test_default_metadata_is_an_empty_mapping() -> None:
    record = EpisodicMemoryRecord(
        event_id="evt-1", session_id="A", event_type="t", summary="s", timestamp=_now()
    )
    assert dict(record.metadata) == {}


# ---------------------------------------------------------------------------
# 9-10: empty memory, add record.
# ---------------------------------------------------------------------------

def test_empty_memory_returns_no_records_for_any_session() -> None:
    memory = InMemoryEpisodicMemory()

    assert memory.get_recent("A") == []


def test_add_record_makes_it_retrievable() -> None:
    memory = InMemoryEpisodicMemory()
    record = _record()

    memory.add(record)

    assert memory.get_recent("A") == [record]


# ---------------------------------------------------------------------------
# 11-12: retrieve recent records, insertion ordering (newest-first).
# ---------------------------------------------------------------------------

def test_get_recent_returns_newest_first() -> None:
    memory = InMemoryEpisodicMemory()
    first = _record(event_id="evt-1", summary="first")
    second = _record(event_id="evt-2", summary="second")
    third = _record(event_id="evt-3", summary="third")

    memory.add(first)
    memory.add(second)
    memory.add(third)

    assert memory.get_recent("A") == [third, second, first]


def test_get_recent_respects_limit() -> None:
    memory = InMemoryEpisodicMemory()
    for i in range(5):
        memory.add(_record(event_id=f"evt-{i}", summary=f"summary {i}"))

    recent = memory.get_recent("A", limit=2)

    assert [r.event_id for r in recent] == ["evt-4", "evt-3"]


# ---------------------------------------------------------------------------
# 13-14: limit validation, oldest eviction.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_limit", [0, -1, -100])
def test_invalid_limit_is_rejected(bad_limit: int) -> None:
    memory = InMemoryEpisodicMemory()

    with pytest.raises(ValueError):
        memory.get_recent("A", limit=bad_limit)


def test_oldest_records_are_evicted_when_max_records_per_session_exceeded() -> None:
    memory = InMemoryEpisodicMemory(max_records_per_session=2)

    memory.add(_record(event_id="evt-1"))
    memory.add(_record(event_id="evt-2"))
    memory.add(_record(event_id="evt-3"))  # evicts evt-1

    remaining_ids = {r.event_id for r in memory.get_recent("A", limit=10)}
    assert remaining_ids == {"evt-2", "evt-3"}


@pytest.mark.parametrize("bad_max", [0, -1, -100])
def test_max_records_per_session_must_be_at_least_one(bad_max: int) -> None:
    with pytest.raises(ValueError):
        InMemoryEpisodicMemory(max_records_per_session=bad_max)


# ---------------------------------------------------------------------------
# 15-16: clear session, session isolation.
# ---------------------------------------------------------------------------

def test_clear_removes_only_that_sessions_records() -> None:
    memory = InMemoryEpisodicMemory()
    memory.add(_record(event_id="evt-a", session_id="A"))
    memory.add(_record(event_id="evt-b", session_id="B"))

    memory.clear("A")

    assert memory.get_recent("A") == []
    assert [r.event_id for r in memory.get_recent("B")] == ["evt-b"]


def test_session_a_and_session_b_never_cross_contaminate() -> None:
    memory = InMemoryEpisodicMemory()
    memory.add(_record(event_id="evt-a", session_id="A", summary="User discussed AI agents"))
    memory.add(_record(event_id="evt-b", session_id="B", summary="User discussed databases"))

    records_a = memory.get_recent("A")
    records_b = memory.get_recent("B")

    assert [r.summary for r in records_a] == ["User discussed AI agents"]
    assert [r.summary for r in records_b] == ["User discussed databases"]


# ---------------------------------------------------------------------------
# 17: missing session returns empty list.
# ---------------------------------------------------------------------------

def test_get_recent_for_never_created_session_returns_empty_list() -> None:
    memory = InMemoryEpisodicMemory()
    memory.add(_record(session_id="A"))

    assert memory.get_recent("never-existed") == []


def test_clear_on_a_never_created_session_is_a_safe_no_op() -> None:
    memory = InMemoryEpisodicMemory()

    memory.clear("never-existed")  # must not raise


# ---------------------------------------------------------------------------
# 18: invalid limit already covered above; also cover invalid session_id.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_session_id", ["", "   ", None, 123])
def test_invalid_session_id_rejected_by_get_recent(bad_session_id: object) -> None:
    memory = InMemoryEpisodicMemory()

    with pytest.raises(ValueError):
        memory.get_recent(bad_session_id)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_session_id", ["", "   ", None, 123])
def test_invalid_session_id_rejected_by_clear(bad_session_id: object) -> None:
    memory = InMemoryEpisodicMemory()

    with pytest.raises(ValueError):
        memory.clear(bad_session_id)  # type: ignore[arg-type]


def test_add_rejects_a_non_record_object() -> None:
    memory = InMemoryEpisodicMemory()

    with pytest.raises(ValueError):
        memory.add({"not": "a record"})  # type: ignore[arg-type]


def test_session_id_whitespace_is_normalized_consistently() -> None:
    memory = InMemoryEpisodicMemory()
    memory.add(_record(session_id="A"))

    assert memory.get_recent("  A  ") == memory.get_recent("A")


# ---------------------------------------------------------------------------
# 19: independent memory instances.
# ---------------------------------------------------------------------------

def test_independent_memory_instances_do_not_share_state() -> None:
    memory_1 = InMemoryEpisodicMemory()
    memory_2 = InMemoryEpisodicMemory()

    memory_1.add(_record(session_id="A"))

    assert memory_1.get_recent("A") != []
    assert memory_2.get_recent("A") == []


# ---------------------------------------------------------------------------
# 20: no module-level global state; Protocol conformance.
# ---------------------------------------------------------------------------

def test_in_memory_episodic_memory_conforms_to_the_protocol() -> None:
    assert isinstance(InMemoryEpisodicMemory(), EpisodicMemory)


# ---------------------------------------------------------------------------
# Step 15, Part 3: explicit retention-boundary tests (A-J).
# ---------------------------------------------------------------------------

def test_exactly_max_records_per_session_keeps_all_records() -> None:
    """Part 3.A: adding exactly max_records_per_session records evicts
    nothing."""
    memory = InMemoryEpisodicMemory(max_records_per_session=3)

    for i in range(3):
        memory.add(_record(event_id=f"evt-{i}"))

    assert len(memory.get_recent("A", limit=10)) == 3


def test_max_records_per_session_plus_one_evicts_exactly_one_oldest() -> None:
    """Part 3.B."""
    memory = InMemoryEpisodicMemory(max_records_per_session=3)

    for i in range(4):  # one over the limit
        memory.add(_record(event_id=f"evt-{i}"))

    remaining_ids = {r.event_id for r in memory.get_recent("A", limit=10)}
    assert remaining_ids == {"evt-1", "evt-2", "evt-3"}  # evt-0 evicted
    assert len(remaining_ids) == 3


def test_multiple_overflows_evict_deterministically() -> None:
    """Part 3.C: adding well past the limit always keeps exactly the
    newest `max_records_per_session` records, oldest evicted first."""
    memory = InMemoryEpisodicMemory(max_records_per_session=3)

    for i in range(10):
        memory.add(_record(event_id=f"evt-{i}"))

    remaining_ids = [r.event_id for r in memory.get_recent("A", limit=10)]
    assert remaining_ids == ["evt-9", "evt-8", "evt-7"]  # newest-first


def test_session_isolation_is_preserved_while_another_session_overflows() -> None:
    """Part 3.E: session A overflowing its limit must never evict or
    otherwise touch session B's records."""
    memory = InMemoryEpisodicMemory(max_records_per_session=2)
    memory.add(_record(event_id="b-evt-1", session_id="B"))

    for i in range(5):  # session A overflows repeatedly
        memory.add(_record(event_id=f"a-evt-{i}", session_id="A"))

    records_b = memory.get_recent("B", limit=10)
    assert [r.event_id for r in records_b] == ["b-evt-1"]  # untouched
    assert len(memory.get_recent("A", limit=10)) == 2  # A correctly capped


def test_clear_then_add_starts_the_session_cleanly() -> None:
    """Part 3.G."""
    memory = InMemoryEpisodicMemory()
    memory.add(_record(event_id="evt-old", session_id="A"))

    memory.clear("A")
    memory.add(_record(event_id="evt-new", session_id="A"))

    records = memory.get_recent("A", limit=10)
    assert [r.event_id for r in records] == ["evt-new"]


def test_get_recent_with_limit_one_returns_only_the_newest_record() -> None:
    """Part 3.H."""
    memory = InMemoryEpisodicMemory()
    memory.add(_record(event_id="evt-1"))
    memory.add(_record(event_id="evt-2"))
    memory.add(_record(event_id="evt-3"))

    records = memory.get_recent("A", limit=1)

    assert [r.event_id for r in records] == ["evt-3"]


def test_get_recent_with_limit_larger_than_stored_returns_all_without_error() -> None:
    """Part 3.I."""
    memory = InMemoryEpisodicMemory()
    memory.add(_record(event_id="evt-1"))
    memory.add(_record(event_id="evt-2"))

    records = memory.get_recent("A", limit=1000)

    assert [r.event_id for r in records] == ["evt-2", "evt-1"]


# ---------------------------------------------------------------------------
# Step 15, Part 5: defensive immutability of the RETURNED LIST specifically
# (record-level immutability is already covered above).
# ---------------------------------------------------------------------------

def test_mutating_the_returned_list_does_not_affect_internal_storage() -> None:
    memory = InMemoryEpisodicMemory()
    memory.add(_record(event_id="evt-1"))

    returned = memory.get_recent("A")
    returned.append(_record(event_id="evt-fabricated"))
    returned.clear()

    fresh = memory.get_recent("A")
    assert [r.event_id for r in fresh] == ["evt-1"]


def test_two_calls_to_get_recent_return_independent_list_objects() -> None:
    memory = InMemoryEpisodicMemory()
    memory.add(_record(event_id="evt-1"))

    first_call = memory.get_recent("A")
    second_call = memory.get_recent("A")

    assert first_call == second_call
    assert first_call is not second_call


# ---------------------------------------------------------------------------
# Step 15, Part 6: duplicate event_id is an intentionally ALLOWED case —
# the store never enforces id uniqueness (see the module docstring's
# "Duplicate event_id" section for the full reasoning). This test pins
# down and preserves that decision.
# ---------------------------------------------------------------------------

def test_duplicate_event_id_within_the_same_session_is_allowed_and_both_are_retained() -> None:
    memory = InMemoryEpisodicMemory()
    first = _record(event_id="evt-dup", summary="first event")
    second = _record(event_id="evt-dup", summary="second event")

    memory.add(first)
    memory.add(second)

    records = memory.get_recent("A", limit=10)
    assert len(records) == 2
    assert {r.summary for r in records} == {"first event", "second event"}
    assert all(r.event_id == "evt-dup" for r in records)


def test_duplicate_event_id_across_different_sessions_is_allowed() -> None:
    memory = InMemoryEpisodicMemory()
    memory.add(_record(event_id="evt-shared", session_id="A", summary="in A"))
    memory.add(_record(event_id="evt-shared", session_id="B", summary="in B"))

    assert memory.get_recent("A")[0].summary == "in A"
    assert memory.get_recent("B")[0].summary == "in B"


def test_many_sessions_remain_mutually_isolated() -> None:
    memory = InMemoryEpisodicMemory()
    session_ids = [f"session-{i}" for i in range(20)]

    for session_id in session_ids:
        memory.add(_record(event_id=f"evt-{session_id}", session_id=session_id, summary=f"event for {session_id}"))

    for session_id in session_ids:
        records = memory.get_recent(session_id)
        assert [r.summary for r in records] == [f"event for {session_id}"]
