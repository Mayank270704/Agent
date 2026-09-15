from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.agent.semantic_memory import InMemorySemanticMemory, SemanticMemoryRecord, SemanticMemoryStore


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _record(**overrides: object) -> SemanticMemoryRecord:
    fields: dict[str, object] = {
        "memory_id": "mem-1",
        "session_id": "A",
        "content": "User prefers Python.",
        "created_at": _now(),
        "source_event_ids": ("evt-1",),
        "confidence": 1.0,
        "active": True,
    }
    fields.update(overrides)
    return SemanticMemoryRecord(**fields)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# A: valid record creation.
# ---------------------------------------------------------------------------

def test_valid_record_creation() -> None:
    ts = _now()
    record = _record(created_at=ts)

    assert record.memory_id == "mem-1"
    assert record.session_id == "A"
    assert record.content == "User prefers Python."
    assert record.created_at == ts
    assert record.source_event_ids == ("evt-1",)
    assert record.confidence == 1.0
    assert record.active is True


def test_confidence_defaults_to_one_and_active_defaults_to_true() -> None:
    record = SemanticMemoryRecord(
        memory_id="mem-1", session_id="A", content="User prefers Python.",
        created_at=_now(), source_event_ids=("evt-1",),
    )

    assert record.confidence == 1.0
    assert record.active is True


# ---------------------------------------------------------------------------
# B: blank/whitespace content rejected (and the other required strings).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_value", ["", "   ", None, 123])
def test_content_validation(bad_value: object) -> None:
    with pytest.raises(ValueError):
        _record(content=bad_value)


@pytest.mark.parametrize("bad_value", ["", "   ", None, 123])
def test_memory_id_validation(bad_value: object) -> None:
    with pytest.raises(ValueError):
        _record(memory_id=bad_value)


# ---------------------------------------------------------------------------
# C: invalid scope/session_id rejected.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_value", ["", "   ", None, 123])
def test_session_id_validation(bad_value: object) -> None:
    with pytest.raises(ValueError):
        _record(session_id=bad_value)


# ---------------------------------------------------------------------------
# D: naive datetime rejected.
# ---------------------------------------------------------------------------

def test_naive_created_at_is_rejected() -> None:
    with pytest.raises(ValueError):
        _record(created_at=datetime.now())  # naive, no tzinfo


def test_non_datetime_created_at_is_rejected() -> None:
    with pytest.raises(ValueError):
        _record(created_at="2026-01-01T00:00:00Z")  # type: ignore[arg-type]


def test_timezone_aware_created_at_is_accepted() -> None:
    record = _record(created_at=_now())
    assert record.created_at.tzinfo is not None


# ---------------------------------------------------------------------------
# E: invalid confidence rejected.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_confidence", [-0.01, 1.01, -1, 2, "0.5", None, [0.5]])
def test_confidence_validation(bad_confidence: object) -> None:
    with pytest.raises(ValueError):
        _record(confidence=bad_confidence)


def test_confidence_rejects_bool_even_though_bool_is_an_int_subclass() -> None:
    with pytest.raises(ValueError):
        _record(confidence=True)


@pytest.mark.parametrize("boundary", [0.0, 1.0, 0.5])
def test_confidence_boundaries_are_accepted(boundary: float) -> None:
    record = _record(confidence=boundary)
    assert record.confidence == boundary


def test_integer_confidence_is_accepted_and_stored_as_float() -> None:
    record = _record(confidence=1)
    assert record.confidence == 1.0
    assert isinstance(record.confidence, float)


# ---------------------------------------------------------------------------
# F: source provenance preserved / validated.
# ---------------------------------------------------------------------------

def test_source_event_ids_preserved_in_order() -> None:
    record = _record(source_event_ids=("evt-1", "evt-2"))
    assert record.source_event_ids == ("evt-1", "evt-2")


def test_source_event_ids_accepts_a_list_and_stores_as_tuple() -> None:
    record = _record(source_event_ids=["evt-1", "evt-2"])
    assert record.source_event_ids == ("evt-1", "evt-2")
    assert isinstance(record.source_event_ids, tuple)


def test_empty_source_event_ids_is_rejected() -> None:
    """Every semantic memory must trace back to at least one episodic
    event -- see the class docstring's provenance-requirement reasoning."""
    with pytest.raises(ValueError):
        _record(source_event_ids=())


@pytest.mark.parametrize("bad_value", ["evt-1", None, 123])
def test_source_event_ids_must_be_a_tuple_or_list(bad_value: object) -> None:
    with pytest.raises(ValueError):
        _record(source_event_ids=bad_value)


@pytest.mark.parametrize("bad_item", ["", "   ", None, 123])
def test_source_event_ids_items_must_be_non_empty_strings(bad_item: object) -> None:
    with pytest.raises(ValueError):
        _record(source_event_ids=("evt-1", bad_item))


def test_provenance_is_not_folded_into_content() -> None:
    record = _record(content="User prefers Python.", source_event_ids=("evt-42",))
    assert "evt-42" not in record.content


# ---------------------------------------------------------------------------
# active field validation.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_value", [1, 0, "true", None])
def test_active_must_be_a_bool(bad_value: object) -> None:
    with pytest.raises(ValueError):
        _record(active=bad_value)


def test_active_can_be_explicitly_false() -> None:
    record = _record(active=False)
    assert record.active is False


# ---------------------------------------------------------------------------
# G: immutable record behavior.
# ---------------------------------------------------------------------------

def test_record_is_immutable() -> None:
    record = _record()

    with pytest.raises(Exception):
        record.content = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# H: immutable/copy-safe collections (source_event_ids).
# ---------------------------------------------------------------------------

def test_source_event_ids_isolation_from_callers_original_list() -> None:
    caller_list = ["evt-1", "evt-2"]
    record = _record(source_event_ids=caller_list)

    caller_list.append("evt-sneaky")
    caller_list[0] = "tampered"

    assert record.source_event_ids == ("evt-1", "evt-2")


def test_source_event_ids_tuple_has_no_mutating_methods() -> None:
    record = _record(source_event_ids=("evt-1",))

    with pytest.raises(AttributeError):
        record.source_event_ids.append("evt-2")  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# I: session/scope isolation.
# ---------------------------------------------------------------------------

def test_session_a_and_session_b_are_isolated() -> None:
    store = InMemorySemanticMemory()
    store.add(_record(memory_id="mem-a", session_id="A", content="User prefers Python."))
    store.add(_record(memory_id="mem-b", session_id="B", content="User prefers Java."))

    records_a = store.list_recent("A")
    records_b = store.list_recent("B")

    assert [r.content for r in records_a] == ["User prefers Python."]
    assert [r.content for r in records_b] == ["User prefers Java."]


def test_session_id_whitespace_is_normalized_consistently() -> None:
    store = InMemorySemanticMemory()
    store.add(_record(memory_id="mem-1", session_id="A"))

    assert store.list_recent("  A  ") == store.list_recent("A")


def test_many_sessions_remain_mutually_isolated() -> None:
    store = InMemorySemanticMemory()
    session_ids = [f"session-{i}" for i in range(20)]

    for session_id in session_ids:
        store.add(_record(memory_id=f"mem-{session_id}", session_id=session_id, content=f"fact for {session_id}"))

    for session_id in session_ids:
        records = store.list_recent(session_id)
        assert [r.content for r in records] == [f"fact for {session_id}"]


# ---------------------------------------------------------------------------
# J: add/get/list behavior.
# ---------------------------------------------------------------------------

def test_add_then_get_returns_the_same_record() -> None:
    store = InMemorySemanticMemory()
    record = _record(memory_id="mem-1")

    store.add(record)

    assert store.get("mem-1") is record


def test_get_for_unknown_memory_id_returns_none() -> None:
    store = InMemorySemanticMemory()

    assert store.get("never-added") is None


@pytest.mark.parametrize("bad_value", ["", "   ", None, 123])
def test_get_rejects_invalid_memory_id(bad_value: object) -> None:
    store = InMemorySemanticMemory()

    with pytest.raises(ValueError):
        store.get(bad_value)  # type: ignore[arg-type]


def test_list_recent_returns_newest_first() -> None:
    store = InMemorySemanticMemory()
    first = _record(memory_id="mem-1", content="first fact")
    second = _record(memory_id="mem-2", content="second fact")
    third = _record(memory_id="mem-3", content="third fact")

    store.add(first)
    store.add(second)
    store.add(third)

    assert store.list_recent("A") == [third, second, first]


def test_list_recent_respects_limit() -> None:
    store = InMemorySemanticMemory()
    for i in range(5):
        store.add(_record(memory_id=f"mem-{i}", content=f"fact {i}"))

    recent = store.list_recent("A", limit=2)

    assert [r.memory_id for r in recent] == ["mem-4", "mem-3"]


def test_list_recent_with_limit_larger_than_stored_returns_all() -> None:
    store = InMemorySemanticMemory()
    store.add(_record(memory_id="mem-1"))
    store.add(_record(memory_id="mem-2"))

    records = store.list_recent("A", limit=1000)

    assert [r.memory_id for r in records] == ["mem-2", "mem-1"]


def test_empty_session_returns_empty_list() -> None:
    store = InMemorySemanticMemory()

    assert store.list_recent("never-existed") == []


@pytest.mark.parametrize("bad_limit", [0, -1, -100])
def test_invalid_limit_is_rejected(bad_limit: int) -> None:
    store = InMemorySemanticMemory()

    with pytest.raises(ValueError):
        store.list_recent("A", limit=bad_limit)


def test_add_rejects_a_non_record_object() -> None:
    store = InMemorySemanticMemory()

    with pytest.raises(ValueError):
        store.add({"not": "a record"})  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_value", ["", "   ", None, 123])
def test_list_recent_rejects_invalid_session_id(bad_value: object) -> None:
    store = InMemorySemanticMemory()

    with pytest.raises(ValueError):
        store.list_recent(bad_value)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# K: clear behavior.
# ---------------------------------------------------------------------------

def test_clear_removes_only_that_sessions_records() -> None:
    store = InMemorySemanticMemory()
    store.add(_record(memory_id="mem-a", session_id="A"))
    store.add(_record(memory_id="mem-b", session_id="B"))

    store.clear("A")

    assert store.list_recent("A") == []
    assert [r.memory_id for r in store.list_recent("B")] == ["mem-b"]


def test_clear_makes_the_memory_id_unreachable_via_get() -> None:
    store = InMemorySemanticMemory()
    store.add(_record(memory_id="mem-1", session_id="A"))

    store.clear("A")

    assert store.get("mem-1") is None


def test_clear_then_add_starts_the_session_cleanly() -> None:
    store = InMemorySemanticMemory()
    store.add(_record(memory_id="mem-old", session_id="A"))

    store.clear("A")
    store.add(_record(memory_id="mem-new", session_id="A", content="new fact"))

    records = store.list_recent("A")
    assert [r.memory_id for r in records] == ["mem-new"]


def test_clear_on_a_never_created_session_is_a_safe_no_op() -> None:
    store = InMemorySemanticMemory()

    store.clear("never-existed")  # must not raise


@pytest.mark.parametrize("bad_value", ["", "   ", None, 123])
def test_clear_rejects_invalid_session_id(bad_value: object) -> None:
    store = InMemorySemanticMemory()

    with pytest.raises(ValueError):
        store.clear(bad_value)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# L: duplicate behavior.
# ---------------------------------------------------------------------------

def test_duplicate_memory_id_is_rejected() -> None:
    """Unlike EpisodicMemoryRecord's event_id, memory_id IS a lookup key
    here (via get()), so the store enforces its uniqueness -- a
    deliberate, documented difference from episodic memory's decision."""
    store = InMemorySemanticMemory()
    store.add(_record(memory_id="mem-1", content="first"))

    with pytest.raises(ValueError):
        store.add(_record(memory_id="mem-1", content="second"))


def test_duplicate_content_with_different_memory_ids_is_allowed() -> None:
    """Content-level deduplication needs semantic comparison (embeddings)
    and is explicitly deferred -- two independently-derived facts with the
    same wording are both retained."""
    store = InMemorySemanticMemory()
    store.add(_record(memory_id="mem-1", content="User prefers Python.", source_event_ids=("evt-1",)))
    store.add(_record(memory_id="mem-2", content="User prefers Python.", source_event_ids=("evt-9",)))

    records = store.list_recent("A", limit=10)
    assert len(records) == 2
    assert {r.memory_id for r in records} == {"mem-1", "mem-2"}
    assert all(r.content == "User prefers Python." for r in records)


def test_duplicate_memory_id_across_different_sessions_is_still_rejected() -> None:
    """memory_id uniqueness is enforced store-wide, not per-session --
    get() has no session_id parameter, so a per-session key would make
    get() ambiguous."""
    store = InMemorySemanticMemory()
    store.add(_record(memory_id="mem-shared", session_id="A"))

    with pytest.raises(ValueError):
        store.add(_record(memory_id="mem-shared", session_id="B"))


# ---------------------------------------------------------------------------
# M: retention behavior (optional cap; unbounded by default).
# ---------------------------------------------------------------------------

def test_default_store_is_unbounded() -> None:
    store = InMemorySemanticMemory()

    for i in range(150):
        store.add(_record(memory_id=f"mem-{i}", content=f"fact {i}"))

    assert len(store.list_recent("A", limit=1000)) == 150


def test_max_records_per_session_evicts_oldest_first_when_set() -> None:
    store = InMemorySemanticMemory(max_records_per_session=2)

    store.add(_record(memory_id="mem-1"))
    store.add(_record(memory_id="mem-2"))
    store.add(_record(memory_id="mem-3"))  # evicts mem-1

    remaining_ids = {r.memory_id for r in store.list_recent("A", limit=10)}
    assert remaining_ids == {"mem-2", "mem-3"}


def test_eviction_also_removes_the_record_from_get_index() -> None:
    store = InMemorySemanticMemory(max_records_per_session=1)

    store.add(_record(memory_id="mem-1"))
    store.add(_record(memory_id="mem-2"))  # evicts mem-1

    assert store.get("mem-1") is None
    assert store.get("mem-2") is not None


@pytest.mark.parametrize("bad_max", [0, -1, -100])
def test_max_records_per_session_must_be_at_least_one_if_provided(bad_max: int) -> None:
    with pytest.raises(ValueError):
        InMemorySemanticMemory(max_records_per_session=bad_max)


def test_eviction_in_one_session_never_touches_another_session() -> None:
    store = InMemorySemanticMemory(max_records_per_session=1)
    store.add(_record(memory_id="mem-b", session_id="B"))

    store.add(_record(memory_id="mem-a1", session_id="A"))
    store.add(_record(memory_id="mem-a2", session_id="A"))  # evicts mem-a1, not mem-b

    assert [r.memory_id for r in store.list_recent("B")] == ["mem-b"]


# ---------------------------------------------------------------------------
# Independent instances, no module-level global state, Protocol conformance.
# ---------------------------------------------------------------------------

def test_independent_store_instances_do_not_share_state() -> None:
    store_1 = InMemorySemanticMemory()
    store_2 = InMemorySemanticMemory()

    store_1.add(_record(memory_id="mem-1", session_id="A"))

    assert store_1.list_recent("A") != []
    assert store_2.list_recent("A") == []
    assert store_2.get("mem-1") is None


def test_in_memory_semantic_memory_conforms_to_the_protocol() -> None:
    assert isinstance(InMemorySemanticMemory(), SemanticMemoryStore)
