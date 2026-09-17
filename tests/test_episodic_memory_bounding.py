"""Milestone 23, item 1: bounded session count (LRU) on
`InMemoryEpisodicMemory` (app/agent/episodic_memory.py) — the same
policy applied to `InMemorySessionMemoryStore`, extended here for
consistency since the Production Readiness Audit flagged both stores.

Fully offline.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.agent.episodic_memory import EpisodicMemoryRecord, InMemoryEpisodicMemory


def _record(session_id: str, event_id: str = "e1") -> EpisodicMemoryRecord:
    return EpisodicMemoryRecord(
        event_id=event_id,
        session_id=session_id,
        event_type="conversation_completed",
        summary="a test event",
        timestamp=datetime.now(timezone.utc),
    )


def test_sessions_below_the_limit_are_all_retained() -> None:
    store = InMemoryEpisodicMemory(max_sessions=5)
    for sid in ("a", "b", "c"):
        store.add(_record(sid))

    for sid in ("a", "b", "c"):
        assert len(store.get_recent(sid)) == 1


def test_exceeding_the_limit_evicts_the_least_recently_used_session() -> None:
    store = InMemoryEpisodicMemory(max_sessions=2)
    store.add(_record("a"))
    store.add(_record("b"))

    store.add(_record("c"))  # evicts 'a'

    assert store.get_recent("a") == []
    assert len(store.get_recent("b")) == 1
    assert len(store.get_recent("c")) == 1


def test_get_recent_also_refreshes_recency() -> None:
    store = InMemoryEpisodicMemory(max_sessions=2)
    store.add(_record("a"))
    store.add(_record("b"))
    store.get_recent("a")  # touch 'a' via a READ, not just add()

    store.add(_record("c"))  # 'b' is now the LRU entry, not 'a'

    assert len(store.get_recent("a")) == 1
    assert store.get_recent("b") == []


def test_eviction_is_deterministic() -> None:
    def run() -> bool:
        store = InMemoryEpisodicMemory(max_sessions=2)
        store.add(_record("a"))
        store.add(_record("b"))
        store.add(_record("c"))
        return store.get_recent("a") == []

    assert run() is True
    assert run() is True


def test_per_session_record_bound_is_unchanged() -> None:
    store = InMemoryEpisodicMemory(max_records_per_session=3, max_sessions=10)
    for i in range(10):
        store.add(_record("a", event_id=f"e{i}"))

    assert len(store.get_recent("a", limit=100)) == 3


def test_surviving_sessions_are_unaffected_by_another_sessions_eviction() -> None:
    store = InMemoryEpisodicMemory(max_sessions=2)
    store.add(_record("a"))
    store.add(_record("b"))

    store.add(_record("c"))  # evicts 'a'

    assert len(store.get_recent("b")) == 1
    assert len(store.get_recent("c")) == 1


def test_max_sessions_must_be_at_least_one() -> None:
    with pytest.raises(ValueError):
        InMemoryEpisodicMemory(max_sessions=0)


def test_default_capacity_does_not_affect_ordinary_small_scale_use() -> None:
    store = InMemoryEpisodicMemory()
    for i in range(20):
        store.add(_record(f"session-{i}"))

    for i in range(20):
        assert len(store.get_recent(f"session-{i}")) == 1
