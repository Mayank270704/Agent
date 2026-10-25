"""Step 37: SQLiteEpisodicMemory (app/persistence/sqlite_episodic_memory.py)
-- the persistent counterpart to InMemoryEpisodicMemory
(app/agent/episodic_memory.py).

Every test uses a pytest `tmp_path` database file. Fully offline: no
Ollama, no network, no LLM.

Categories, matching the Step 37 brief exactly:
    A. Basic write/read                I. Persistence across instances
    B. Multiple event types             J. Persistence across processes
    C. Metadata round-trip              K. Concurrency
    D. Ordering                         L. Schema migration (see
    E. Per-session record cap              tests/test_persistence_schema.py)
    F. Session isolation                M. Unsupported version (ditto)
    G. Multiple sessions                N. Security
    H. Session clear                    O. In-memory parity
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.agent.episodic_memory import EpisodicMemory, EpisodicMemoryRecord, InMemoryEpisodicMemory
from app.persistence.sqlite_episodic_memory import SQLiteEpisodicMemory

REPO_ROOT = Path(__file__).resolve().parent.parent


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _record(**overrides: object) -> EpisodicMemoryRecord:
    fields: dict[str, object] = {
        "event_id": "evt-1",
        "session_id": "A",
        "event_type": "conversation_completed",
        "summary": "User discussed AI agents.",
        "timestamp": _now(),
        "metadata": {"steps": 1, "tool_count": 0},
    }
    fields.update(overrides)
    return EpisodicMemoryRecord(**fields)  # type: ignore[arg-type]


def _store(tmp_path, name: str = "episodic.db", **kwargs) -> SQLiteEpisodicMemory:
    return SQLiteEpisodicMemory(db_path=str(tmp_path / name), **kwargs)


# ===========================================================================
# A — basic write/read
# ===========================================================================

def test_empty_session_has_no_records(tmp_path) -> None:
    store = _store(tmp_path)
    assert store.get_recent("s") == []


def test_add_and_retrieve_a_single_record(tmp_path) -> None:
    store = _store(tmp_path)
    record = _record(session_id="s")

    store.add(record)
    results = store.get_recent("s")

    assert len(results) == 1
    assert results[0].event_id == record.event_id
    assert results[0].session_id == "s"
    assert results[0].summary == record.summary


def test_add_rejects_non_record_objects(tmp_path) -> None:
    store = _store(tmp_path)
    with pytest.raises(ValueError):
        store.add("not a record")  # type: ignore[arg-type]


def test_get_recent_rejects_invalid_limit(tmp_path) -> None:
    store = _store(tmp_path)
    store.add(_record(session_id="s"))
    for bad_limit in (0, -1, "5", 1.5, True):
        with pytest.raises(ValueError):
            store.get_recent("s", limit=bad_limit)  # type: ignore[arg-type]


def test_conforms_to_the_episodic_memory_protocol(tmp_path) -> None:
    assert isinstance(_store(tmp_path), EpisodicMemory)


def test_max_records_per_session_must_be_at_least_one(tmp_path) -> None:
    with pytest.raises(ValueError):
        _store(tmp_path, max_records_per_session=0)


def test_max_sessions_must_be_at_least_one(tmp_path) -> None:
    with pytest.raises(ValueError):
        _store(tmp_path, max_sessions=0)


def test_session_id_must_be_a_non_empty_string(tmp_path) -> None:
    store = _store(tmp_path)
    with pytest.raises(ValueError):
        store.get_recent("")
    with pytest.raises(ValueError):
        store.get_recent("   ")


# ===========================================================================
# B — multiple event types
# ===========================================================================

def test_different_event_types_are_preserved(tmp_path) -> None:
    store = _store(tmp_path)
    store.add(_record(event_id="e1", session_id="s", event_type="conversation_completed"))
    store.add(_record(event_id="e2", session_id="s", event_type="tool_failure_noted"))
    store.add(_record(event_id="e3", session_id="s", event_type="preference_learned"))

    results = store.get_recent("s", limit=10)
    event_types = {r.event_type for r in results}

    assert event_types == {"conversation_completed", "tool_failure_noted", "preference_learned"}


def test_event_type_is_free_form_not_constrained_to_a_fixed_set(tmp_path) -> None:
    """event_type is documented as a validated free-form string, not an
    enum (app/agent/episodic_memory.py) -- the schema must not reject an
    unrecognized value via a CHECK constraint."""
    store = _store(tmp_path)
    store.add(_record(session_id="s", event_type="some_brand_new_event_type_nobody_anticipated"))

    assert store.get_recent("s")[0].event_type == "some_brand_new_event_type_nobody_anticipated"


# ===========================================================================
# C — metadata round-trip
# ===========================================================================

def test_metadata_round_trips_exactly(tmp_path) -> None:
    store = _store(tmp_path)
    store.add(_record(session_id="s", metadata={"steps": 7, "tool_count": 2}))

    retrieved = store.get_recent("s")[0]

    assert dict(retrieved.metadata) == {"steps": 7, "tool_count": 2}


def test_empty_metadata_round_trips(tmp_path) -> None:
    store = _store(tmp_path)
    store.add(_record(session_id="s", metadata={}))

    assert dict(store.get_recent("s")[0].metadata) == {}


def test_retrieved_record_metadata_is_immutable(tmp_path) -> None:
    store = _store(tmp_path)
    store.add(_record(session_id="s", metadata={"steps": 1}))

    retrieved = store.get_recent("s")[0]
    with pytest.raises(Exception):
        retrieved.metadata["steps"] = 999  # type: ignore[index]


def test_timestamp_round_trips_with_timezone_preserved(tmp_path) -> None:
    store = _store(tmp_path)
    ts = datetime(2026, 3, 1, 12, 30, 0, tzinfo=timezone.utc)
    store.add(_record(session_id="s", timestamp=ts))

    retrieved = store.get_recent("s")[0]

    assert retrieved.timestamp == ts
    assert retrieved.timestamp.tzinfo is not None


# ===========================================================================
# D — ordering: newest first, matching InMemoryEpisodicMemory exactly
# ===========================================================================

def test_get_recent_returns_newest_first(tmp_path) -> None:
    store = _store(tmp_path)
    for i in range(5):
        store.add(_record(event_id=f"e{i}", session_id="s", summary=f"event {i}"))

    results = store.get_recent("s", limit=5)

    assert [r.summary for r in results] == ["event 4", "event 3", "event 2", "event 1", "event 0"]


def test_get_recent_respects_limit(tmp_path) -> None:
    store = _store(tmp_path)
    for i in range(10):
        store.add(_record(event_id=f"e{i}", session_id="s", summary=f"event {i}"))

    results = store.get_recent("s", limit=3)

    assert [r.summary for r in results] == ["event 9", "event 8", "event 7"]


def test_get_recent_default_limit_is_10(tmp_path) -> None:
    store = _store(tmp_path)
    for i in range(15):
        store.add(_record(event_id=f"e{i}", session_id="s", summary=f"event {i}"))

    assert len(store.get_recent("s")) == 10


# ===========================================================================
# E — per-session record cap (default 100), preserved exactly
# ===========================================================================

def test_default_max_records_per_session_is_100(tmp_path) -> None:
    store = _store(tmp_path)
    for i in range(150):
        store.add(_record(event_id=f"e{i}", session_id="s", summary=f"event {i}"))

    row = store._conn.execute(
        "SELECT COUNT(*) AS c FROM episodic_records WHERE session_id = 's'"
    ).fetchone()
    assert row["c"] == 100


def test_oldest_records_are_evicted_once_the_cap_is_exceeded(tmp_path) -> None:
    store = _store(tmp_path, max_records_per_session=3)
    for i in range(5):
        store.add(_record(event_id=f"e{i}", session_id="s", summary=f"event {i}"))

    results = store.get_recent("s", limit=10)

    assert [r.summary for r in results] == ["event 4", "event 3", "event 2"]


def test_eviction_is_scoped_to_the_overflowing_session_only(tmp_path) -> None:
    store = _store(tmp_path, max_records_per_session=2)
    store.add(_record(event_id="a1", session_id="a", summary="a1"))
    store.add(_record(event_id="b1", session_id="b", summary="b1"))
    store.add(_record(event_id="a2", session_id="a", summary="a2"))
    store.add(_record(event_id="a3", session_id="a", summary="a3"))  # evicts a1 only

    assert [r.summary for r in store.get_recent("a", limit=10)] == ["a3", "a2"]
    assert [r.summary for r in store.get_recent("b", limit=10)] == ["b1"]


# ===========================================================================
# F — session isolation
# ===========================================================================

def test_session_a_cannot_see_session_bs_records(tmp_path) -> None:
    store = _store(tmp_path)
    store.add(_record(event_id="a1", session_id="a", summary="only in A"))
    store.add(_record(event_id="b1", session_id="b", summary="only in B"))

    assert [r.summary for r in store.get_recent("a")] == ["only in A"]
    assert [r.summary for r in store.get_recent("b")] == ["only in B"]


# ===========================================================================
# G — multiple sessions, including session-count cap/LRU (default 1000)
# ===========================================================================

def test_many_sessions_remain_mutually_isolated(tmp_path) -> None:
    store = _store(tmp_path, max_sessions=50)
    for i in range(20):
        store.add(_record(event_id=f"e{i}", session_id=f"session-{i}", summary=f"event {i}"))

    for i in range(20):
        assert [r.summary for r in store.get_recent(f"session-{i}")] == [f"event {i}"]


def test_exceeding_max_sessions_evicts_the_least_recently_used_session(tmp_path) -> None:
    store = _store(tmp_path, max_sessions=2)
    store.add(_record(event_id="a1", session_id="a", summary="a"))
    store.add(_record(event_id="b1", session_id="b", summary="b"))

    store.add(_record(event_id="c1", session_id="c", summary="c"))  # evicts 'a'

    ids = {row["session_id"] for row in store._conn.execute("SELECT session_id FROM episodic_sessions")}
    assert ids == {"b", "c"}
    assert store.get_recent("a") == []


def test_get_recent_also_refreshes_session_recency(tmp_path) -> None:
    """Matches InMemoryEpisodicMemory.get_recent's move_to_end call: reading
    a session also protects it from LRU eviction."""
    store = _store(tmp_path, max_sessions=2)
    store.add(_record(event_id="a1", session_id="a", summary="a"))
    store.add(_record(event_id="b1", session_id="b", summary="b"))
    store.get_recent("a")  # touch 'a' -> 'b' becomes the LRU entry

    store.add(_record(event_id="c1", session_id="c", summary="c"))

    ids = {row["session_id"] for row in store._conn.execute("SELECT session_id FROM episodic_sessions")}
    assert ids == {"a", "c"}


def test_get_recent_never_creates_a_session_entry_for_a_nonexistent_session(tmp_path) -> None:
    store = _store(tmp_path)
    store.get_recent("never-written-to")

    row = store._conn.execute(
        "SELECT COUNT(*) AS c FROM episodic_sessions WHERE session_id = 'never-written-to'"
    ).fetchone()
    assert row["c"] == 0


def test_a_brand_new_session_is_never_evicted_by_its_own_creation(tmp_path) -> None:
    store = _store(tmp_path, max_sessions=1)
    store.add(_record(session_id="only-session"))

    assert len(store.get_recent("only-session")) == 1


# ===========================================================================
# H — session clear
# ===========================================================================

def test_clear_removes_all_of_a_sessions_records(tmp_path) -> None:
    store = _store(tmp_path)
    store.add(_record(session_id="a", summary="x"))

    store.clear("a")

    assert store.get_recent("a") == []


def test_clear_does_not_affect_other_sessions(tmp_path) -> None:
    store = _store(tmp_path)
    store.add(_record(session_id="a", summary="from a"))
    store.add(_record(session_id="b", summary="from b"))

    store.clear("a")

    assert store.get_recent("a") == []
    assert [r.summary for r in store.get_recent("b")] == ["from b"]


def test_clear_is_a_no_op_for_a_never_created_session(tmp_path) -> None:
    store = _store(tmp_path)
    store.clear("never-existed")  # must not raise


def test_session_can_be_used_again_after_clear(tmp_path) -> None:
    store = _store(tmp_path)
    store.add(_record(session_id="a", summary="old"))
    store.clear("a")

    store.add(_record(session_id="a", summary="new"))

    assert [r.summary for r in store.get_recent("a")] == ["new"]


# ===========================================================================
# I — persistence across instances
# ===========================================================================

def test_records_survive_a_simulated_process_restart(tmp_path) -> None:
    db_path = str(tmp_path / "restart.db")

    store1 = SQLiteEpisodicMemory(db_path=db_path)
    store1.add(_record(event_id="e1", session_id="s", summary="before restart"))
    store1.close()

    store2 = SQLiteEpisodicMemory(db_path=db_path)
    results = store2.get_recent("s")
    store2.close()

    assert len(results) == 1
    assert results[0].summary == "before restart"
    assert results[0].event_id == "e1"


def test_session_cap_state_survives_a_restart(tmp_path) -> None:
    db_path = str(tmp_path / "restart_cap.db")

    store1 = SQLiteEpisodicMemory(db_path=db_path, max_sessions=2)
    store1.add(_record(session_id="a"))
    store1.add(_record(session_id="b"))
    store1.close()

    store2 = SQLiteEpisodicMemory(db_path=db_path, max_sessions=2)
    store2.add(_record(session_id="c"))  # must evict the pre-restart LRU session ('a')
    ids = {row["session_id"] for row in store2._conn.execute("SELECT session_id FROM episodic_sessions")}
    store2.close()

    assert ids == {"b", "c"}


# ===========================================================================
# J — persistence across separate interpreter processes, through the real app
# ===========================================================================

def test_episodic_memory_persists_through_the_real_app_across_processes(tmp_path) -> None:
    db_path = tmp_path / "e2e_episodic.db"
    write_probe = (
        "import app.main; "
        "from datetime import datetime, timezone; "
        "from app.agent.episodic_memory import EpisodicMemoryRecord; "
        "record = EpisodicMemoryRecord("
        "  event_id='e2e-1', session_id='e2e-session', event_type='conversation_completed', "
        "  summary='persisted episode', timestamp=datetime.now(timezone.utc), metadata={'steps': 1}"
        "); "
        "app.main.chat_service.episodic_memory.add(record); "
        "app.main.chat_service.episodic_memory.close(); "
        "print('WRITE_OK')"
    )
    env = dict(os.environ)
    env["MEMORY_BACKEND"] = "sqlite"
    env["SQLITE_DB_PATH"] = str(db_path)
    write_result = subprocess.run(
        [sys.executable, "-c", write_probe], capture_output=True, text=True, cwd=str(REPO_ROOT), env=env
    )
    assert write_result.returncode == 0, write_result.stderr
    assert "WRITE_OK" in write_result.stdout

    read_probe = (
        "import app.main; "
        "records = app.main.chat_service.episodic_memory.get_recent('e2e-session'); "
        "assert len(records) == 1, records; "
        "assert records[0].summary == 'persisted episode', records[0].summary; "
        "print('READ_OK')"
    )
    read_result = subprocess.run(
        [sys.executable, "-c", read_probe], capture_output=True, text=True, cwd=str(REPO_ROOT), env=env
    )
    assert read_result.returncode == 0, read_result.stderr
    assert "READ_OK" in read_result.stdout


def test_default_process_startup_uses_in_memory_episodic_backend() -> None:
    probe = (
        "import app.main; "
        "from app.agent.episodic_memory import InMemoryEpisodicMemory; "
        "assert isinstance(app.main.chat_service.episodic_memory, InMemoryEpisodicMemory); "
        "print('EPISODIC_DEFAULT_OK')"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, cwd=str(REPO_ROOT)
    )
    assert result.returncode == 0, result.stderr
    assert "EPISODIC_DEFAULT_OK" in result.stdout


def test_sqlite_backend_env_selects_sqlite_episodic_memory(tmp_path) -> None:
    db_path = tmp_path / "probe_episodic.db"
    probe = (
        "import app.main; "
        "from app.persistence.sqlite_episodic_memory import SQLiteEpisodicMemory; "
        "assert isinstance(app.main.chat_service.episodic_memory, SQLiteEpisodicMemory); "
        "print('EPISODIC_SQLITE_OK')"
    )
    env = dict(os.environ)
    env["MEMORY_BACKEND"] = "sqlite"
    env["SQLITE_DB_PATH"] = str(db_path)
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, cwd=str(REPO_ROOT), env=env
    )
    assert result.returncode == 0, result.stderr
    assert "EPISODIC_SQLITE_OK" in result.stdout


# ===========================================================================
# K — concurrency
# ===========================================================================

def test_concurrent_writes_to_the_same_session_lose_no_records(tmp_path) -> None:
    store = _store(tmp_path, max_records_per_session=100)
    barrier = threading.Barrier(30)

    def worker(index: int) -> None:
        barrier.wait()
        store.add(_record(event_id=f"e{index}", session_id="race", summary=f"event {index}"))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(30)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    summaries = {r.summary for r in store.get_recent("race", limit=100)}
    assert summaries == {f"event {i}" for i in range(30)}


def test_concurrent_first_write_to_new_sessions_creates_no_duplicate_session_rows(tmp_path) -> None:
    store = _store(tmp_path)
    barrier = threading.Barrier(40)

    def worker(index: int) -> None:
        barrier.wait()
        store.add(_record(event_id=f"e{index}", session_id=f"session-{index % 10}", summary=f"e{index}"))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(40)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    row = store._conn.execute("SELECT COUNT(*) AS c FROM episodic_sessions").fetchone()
    assert row["c"] == 10
    for session_index in range(10):
        assert len(store.get_recent(f"session-{session_index}", limit=100)) == 4


# ===========================================================================
# N — security
# ===========================================================================

def test_sql_injection_attempt_in_session_id_is_treated_as_data(tmp_path) -> None:
    store = _store(tmp_path)
    malicious_id = "a'; DROP TABLE episodic_records; --"

    store.add(_record(session_id=malicious_id, summary="hello"))

    assert [r.summary for r in store.get_recent(malicious_id)] == ["hello"]
    store.add(_record(session_id="another-session"))  # table wasn't dropped


def test_sql_injection_attempt_in_summary_and_event_type_is_treated_as_data(tmp_path) -> None:
    store = _store(tmp_path)
    malicious_summary = "'; DROP TABLE episodic_records; --"
    malicious_event_type = "'; DROP TABLE episodic_sessions; --"

    store.add(_record(session_id="s", summary=malicious_summary, event_type=malicious_event_type))
    result = store.get_recent("s")[0]

    assert result.summary == malicious_summary
    assert result.event_type == malicious_event_type


def test_sql_injection_attempt_in_metadata_values_is_treated_as_data(tmp_path) -> None:
    store = _store(tmp_path)
    store.add(_record(session_id="s", metadata={"note": "'; DROP TABLE episodic_records; --"}))

    assert dict(store.get_recent("s")[0].metadata) == {"note": "'; DROP TABLE episodic_records; --"}


def test_episodic_records_table_has_no_tool_or_agent_state_columns(tmp_path) -> None:
    """Structural guard: only what EpisodicMemoryRecord represents is ever
    persisted -- never raw tool calls/observations/plan/model-output
    fields."""
    store = _store(tmp_path)
    store.add(_record(session_id="s"))

    columns = {row[1] for row in store._conn.execute("PRAGMA table_info(episodic_records)")}
    assert columns == {"id", "event_id", "session_id", "event_type", "summary", "timestamp", "metadata_json"}


def test_event_id_is_not_a_unique_constraint(tmp_path) -> None:
    """Deliberately preserves InMemoryEpisodicMemory's documented
    architectural decision: duplicate event_ids are accepted, not
    rejected (see app/agent/episodic_memory.py, "Step 15, Part 6")."""
    store = _store(tmp_path)
    store.add(_record(event_id="dup", session_id="s", summary="first"))
    store.add(_record(event_id="dup", session_id="s", summary="second"))  # must not raise

    summaries = {r.summary for r in store.get_recent("s", limit=10)}
    assert summaries == {"first", "second"}


# ===========================================================================
# O — SQLite vs in-memory behavioral parity
# ===========================================================================

def test_parity_fifo_eviction_order_matches_in_memory(tmp_path) -> None:
    in_memory = InMemoryEpisodicMemory(max_records_per_session=3)
    sqlite_store = _store(tmp_path, max_records_per_session=3)

    for i in range(5):
        record = _record(event_id=f"e{i}", session_id="s", summary=f"event {i}")
        in_memory.add(record)
        sqlite_store.add(record)

    in_memory_summaries = [r.summary for r in in_memory.get_recent("s", limit=10)]
    sqlite_summaries = [r.summary for r in sqlite_store.get_recent("s", limit=10)]

    assert in_memory_summaries == sqlite_summaries == ["event 4", "event 3", "event 2"]


def test_parity_session_isolation_matches_in_memory(tmp_path) -> None:
    in_memory = InMemoryEpisodicMemory()
    sqlite_store = _store(tmp_path)

    for backend in (in_memory, sqlite_store):
        backend.add(_record(event_id="a1", session_id="a", summary="only in A"))
        backend.add(_record(event_id="b1", session_id="b", summary="only in B"))

    assert [r.summary for r in in_memory.get_recent("a")] == [r.summary for r in sqlite_store.get_recent("a")]
    assert [r.summary for r in in_memory.get_recent("b")] == [r.summary for r in sqlite_store.get_recent("b")]


def test_parity_clear_behavior_matches_in_memory(tmp_path) -> None:
    in_memory = InMemoryEpisodicMemory()
    sqlite_store = _store(tmp_path)

    for backend in (in_memory, sqlite_store):
        backend.add(_record(session_id="a", summary="x"))
        backend.clear("a")

    assert in_memory.get_recent("a") == [] == sqlite_store.get_recent("a")


def test_parity_default_constructor_values_match(tmp_path) -> None:
    import inspect

    in_memory_defaults = {
        k: v.default
        for k, v in inspect.signature(InMemoryEpisodicMemory.__init__).parameters.items()
        if v.default is not inspect.Parameter.empty
    }
    sqlite_defaults = {
        k: v.default
        for k, v in inspect.signature(SQLiteEpisodicMemory.__init__).parameters.items()
        if v.default is not inspect.Parameter.empty
    }

    assert in_memory_defaults == sqlite_defaults == {"max_records_per_session": 100, "max_sessions": 1000}
