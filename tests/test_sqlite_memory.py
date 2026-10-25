"""Step 36: SQLiteConversationMemory / SQLiteSessionMemoryStore
(app/persistence/sqlite_memory.py) — the persistent counterpart to
InMemoryConversationMemory/InMemorySessionMemoryStore (app/agent/memory.py).

Every test uses a pytest `tmp_path` database file -- no real development
database is ever touched, and no two tests share a file. Fully offline: no
Ollama, no network, no LLM.

Categories, matching the Step 36 brief exactly:
    A. Basic CRUD                      E. Persistence across instances
    B. FIFO eviction                   F. Concurrency
    C. Session isolation               G. Transactions/recovery
    D. Session lifecycle               J. Security
(H — schema/migration — lives in tests/test_persistence_schema.py, since it
exercises app/persistence/db.py directly, a different unit than the two
classes here. I — configuration — lives in tests/test_memory_backend_config.py.)
"""
from __future__ import annotations

import sqlite3
import threading

import pytest

from app.agent.memory import ConversationMemory, SessionMemoryStore
from app.persistence.sqlite_memory import SQLiteConversationMemory, SQLiteSessionMemoryStore


def _store(tmp_path, name: str = "test.db", **kwargs) -> SQLiteSessionMemoryStore:
    return SQLiteSessionMemoryStore(db_path=str(tmp_path / name), **kwargs)


# ===========================================================================
# A — basic CRUD
# ===========================================================================

def test_empty_session_has_no_messages(tmp_path) -> None:
    store = _store(tmp_path)

    assert store.get_memory("s").get_messages() == []


def test_add_user_message(tmp_path) -> None:
    store = _store(tmp_path)

    store.get_memory("s").add_user_message("hello")

    assert store.get_memory("s").get_messages() == [{"role": "user", "content": "hello"}]


def test_add_assistant_message(tmp_path) -> None:
    store = _store(tmp_path)

    store.get_memory("s").add_assistant_message("hi there")

    assert store.get_memory("s").get_messages() == [{"role": "assistant", "content": "hi there"}]


def test_insertion_order_is_preserved(tmp_path) -> None:
    store = _store(tmp_path)
    memory = store.get_memory("s")

    memory.add_user_message("first")
    memory.add_assistant_message("second")
    memory.add_user_message("third")

    assert store.get_memory("s").get_messages() == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
        {"role": "user", "content": "third"},
    ]


@pytest.mark.parametrize("blank", ["", "   "])
def test_reject_empty_user_message(tmp_path, blank: str) -> None:
    store = _store(tmp_path)

    with pytest.raises(ValueError):
        store.get_memory("s").add_user_message(blank)


@pytest.mark.parametrize("bad_value", [None, 123, ["not", "a", "string"], {"content": "x"}])
def test_reject_non_string_message(tmp_path, bad_value: object) -> None:
    store = _store(tmp_path)

    with pytest.raises(ValueError):
        store.get_memory("s").add_user_message(bad_value)  # type: ignore[arg-type]


def test_clear_removes_all_messages(tmp_path) -> None:
    store = _store(tmp_path)
    store.get_memory("s").add_user_message("hello")

    store.get_memory("s").clear()

    assert store.get_memory("s").get_messages() == []


def test_memory_can_be_used_again_after_clear(tmp_path) -> None:
    store = _store(tmp_path)
    store.get_memory("s").add_user_message("hello")
    store.get_memory("s").clear()

    store.get_memory("s").add_user_message("new conversation")

    assert store.get_memory("s").get_messages() == [{"role": "user", "content": "new conversation"}]


def test_conforms_to_the_conversation_memory_protocol(tmp_path) -> None:
    store = _store(tmp_path)
    assert isinstance(store.get_memory("s"), ConversationMemory)


def test_conforms_to_the_session_memory_store_protocol(tmp_path) -> None:
    assert isinstance(_store(tmp_path), SessionMemoryStore)


# ===========================================================================
# B — FIFO eviction, preserving max_messages=20 behavior exactly
# ===========================================================================

def test_default_max_messages_per_session_is_20(tmp_path) -> None:
    store = _store(tmp_path)
    memory = store.get_memory("s")

    for i in range(25):
        memory.add_user_message(f"m{i}")

    assert len(store.get_memory("s").get_messages()) == 20


def test_oldest_messages_are_evicted_once_the_limit_is_exceeded(tmp_path) -> None:
    store = _store(tmp_path, max_messages_per_session=3)
    memory = store.get_memory("s")

    memory.add_user_message("m1")
    memory.add_assistant_message("m2")
    memory.add_user_message("m3")
    memory.add_assistant_message("m4")  # exceeds the limit -> m1 evicted

    assert [m["content"] for m in store.get_memory("s").get_messages()] == ["m2", "m3", "m4"]


def test_newest_messages_remain_after_repeated_eviction(tmp_path) -> None:
    store = _store(tmp_path, max_messages_per_session=2)
    memory = store.get_memory("s")

    for i in range(1, 6):
        memory.add_user_message(f"m{i}")

    assert [m["content"] for m in store.get_memory("s").get_messages()] == ["m4", "m5"]


def test_max_messages_must_be_at_least_one(tmp_path) -> None:
    with pytest.raises(ValueError):
        _store(tmp_path, max_messages_per_session=0)


# ===========================================================================
# C — session isolation
# ===========================================================================

def test_session_a_cannot_see_session_bs_messages(tmp_path) -> None:
    store = _store(tmp_path)
    store.get_memory("a").add_user_message("only in A")
    store.get_memory("b").add_user_message("only in B")

    assert [m["content"] for m in store.get_memory("a").get_messages()] == ["only in A"]
    assert [m["content"] for m in store.get_memory("b").get_messages()] == ["only in B"]


def test_clearing_session_a_does_not_affect_session_b(tmp_path) -> None:
    store = _store(tmp_path)
    store.get_memory("a").add_user_message("from a")
    store.get_memory("b").add_user_message("from b")

    store.clear_session("a")

    assert store.get_memory("a").get_messages() == []
    assert store.get_memory("b").get_messages() == [{"role": "user", "content": "from b"}]


def test_many_sessions_remain_mutually_isolated(tmp_path) -> None:
    store = _store(tmp_path, max_sessions=50)
    for i in range(20):
        store.get_memory(f"session-{i}").add_user_message(f"message {i}")

    for i in range(20):
        assert store.get_memory(f"session-{i}").get_messages() == [
            {"role": "user", "content": f"message {i}"}
        ]


# ===========================================================================
# D — session lifecycle: create, retrieve existing, touch, max_sessions=1000
# ===========================================================================

def test_default_max_sessions_is_generous_enough_for_ordinary_use(tmp_path) -> None:
    store = _store(tmp_path)
    for i in range(20):
        store.get_memory(f"session-{i}")

    row = store._conn.execute("SELECT COUNT(*) AS c FROM sessions").fetchone()
    assert row["c"] == 20


def test_exceeding_max_sessions_evicts_exactly_one_session(tmp_path) -> None:
    store = _store(tmp_path, max_sessions=2)
    store.get_memory("a")
    store.get_memory("b")

    store.get_memory("c")  # third distinct session, capacity is 2

    row = store._conn.execute("SELECT COUNT(*) AS c FROM sessions").fetchone()
    assert row["c"] == 2


def test_the_least_recently_used_session_is_evicted(tmp_path) -> None:
    store = _store(tmp_path, max_sessions=2)
    store.get_memory("a")
    store.get_memory("b")

    store.get_memory("c")  # evicts 'a' (never touched again since creation)

    ids = {row["session_id"] for row in store._conn.execute("SELECT session_id FROM sessions")}
    assert ids == {"b", "c"}


def test_re_accessing_a_session_protects_it_from_eviction(tmp_path) -> None:
    store = _store(tmp_path, max_sessions=2)
    store.get_memory("a")
    store.get_memory("b")
    store.get_memory("a")  # re-touch 'a' -> 'b' becomes the LRU entry

    store.get_memory("c")

    ids = {row["session_id"] for row in store._conn.execute("SELECT session_id FROM sessions")}
    assert ids == {"a", "c"}


def test_a_brand_new_session_is_never_evicted_by_its_own_creation(tmp_path) -> None:
    store = _store(tmp_path, max_sessions=1)

    store.get_memory("only-session").add_user_message("hi")

    assert store.get_memory("only-session").get_messages() == [{"role": "user", "content": "hi"}]


def test_evicting_a_session_removes_its_conversation_messages_too(tmp_path) -> None:
    """ON DELETE CASCADE: eviction must not leave orphaned message rows."""
    store = _store(tmp_path, max_sessions=2)
    store.get_memory("a").add_user_message("old history")
    store.get_memory("b")
    store.get_memory("c")  # evicts 'a'

    row = store._conn.execute(
        "SELECT COUNT(*) AS c FROM conversation_messages WHERE session_id = 'a'"
    ).fetchone()
    assert row["c"] == 0
    assert store.get_memory("a").get_messages() == []  # a fresh, genuinely empty session


def test_max_sessions_must_be_at_least_one(tmp_path) -> None:
    with pytest.raises(ValueError):
        _store(tmp_path, max_sessions=0)


def test_session_id_must_be_a_non_empty_string(tmp_path) -> None:
    store = _store(tmp_path)
    with pytest.raises(ValueError):
        store.get_memory("")
    with pytest.raises(ValueError):
        store.get_memory("   ")


def test_session_id_is_normalized_by_stripping_whitespace(tmp_path) -> None:
    store = _store(tmp_path)
    store.get_memory(" a ").add_user_message("hello")

    assert store.get_memory("a").get_messages() == [{"role": "user", "content": "hello"}]


def test_db_path_must_be_a_non_empty_string(tmp_path) -> None:
    with pytest.raises(ValueError):
        SQLiteSessionMemoryStore(db_path="")


# ===========================================================================
# E — persistence across instances (mandatory: proves actual persistence,
# not merely a long-lived object)
# ===========================================================================

def test_conversation_survives_a_simulated_process_restart(tmp_path) -> None:
    db_path = str(tmp_path / "restart.db")

    store1 = SQLiteSessionMemoryStore(db_path=db_path)
    store1.get_memory("session-x").add_user_message("hello")
    store1.get_memory("session-x").add_assistant_message("hi there")
    store1.close()  # simulate process end: release the connection cleanly

    store2 = SQLiteSessionMemoryStore(db_path=db_path)  # a genuinely new instance
    messages = store2.get_memory("session-x").get_messages()
    store2.close()

    assert messages == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]


def test_session_metadata_also_survives_a_restart(tmp_path) -> None:
    """Session existence/recency, not just message content, must survive --
    otherwise a restarted process would treat every session as brand new."""
    db_path = str(tmp_path / "restart_meta.db")

    store1 = SQLiteSessionMemoryStore(db_path=db_path, max_sessions=2)
    store1.get_memory("a")
    store1.get_memory("b")
    store1.close()

    store2 = SQLiteSessionMemoryStore(db_path=db_path, max_sessions=2)
    store2.get_memory("c")  # must evict the least-recently-used PRE-RESTART session ('a')
    ids = {row["session_id"] for row in store2._conn.execute("SELECT session_id FROM sessions")}
    store2.close()

    assert ids == {"b", "c"}


def test_multiple_sessions_all_survive_a_restart(tmp_path) -> None:
    db_path = str(tmp_path / "restart_multi.db")

    store1 = SQLiteSessionMemoryStore(db_path=db_path)
    for i in range(5):
        store1.get_memory(f"s{i}").add_user_message(f"message {i}")
    store1.close()

    store2 = SQLiteSessionMemoryStore(db_path=db_path)
    for i in range(5):
        assert store2.get_memory(f"s{i}").get_messages() == [
            {"role": "user", "content": f"message {i}"}
        ]
    store2.close()


# ===========================================================================
# F — concurrency: deterministic, no duplicate/corrupt state
# ===========================================================================

def test_concurrent_first_access_to_a_new_session_creates_exactly_one_row(tmp_path) -> None:
    store = _store(tmp_path)
    barrier = threading.Barrier(30)

    def worker() -> None:
        barrier.wait()
        store.get_memory("shared-new-session")

    threads = [threading.Thread(target=worker) for _ in range(30)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    row = store._conn.execute(
        "SELECT COUNT(*) AS c FROM sessions WHERE session_id = 'shared-new-session'"
    ).fetchone()
    assert row["c"] == 1


def test_concurrent_writes_to_the_same_session_lose_no_messages(tmp_path) -> None:
    store = _store(tmp_path, max_messages_per_session=100)
    barrier = threading.Barrier(30)

    def worker(index: int) -> None:
        barrier.wait()
        store.get_memory("race-session").add_user_message(f"msg-{index}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(30)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    final = {m["content"] for m in store.get_memory("race-session").get_messages()}
    assert final == {f"msg-{i}" for i in range(30)}


def test_concurrent_access_across_many_distinct_sessions_creates_no_duplicates(tmp_path) -> None:
    store = _store(tmp_path, max_sessions=1000)
    barrier = threading.Barrier(40)

    def worker(index: int) -> None:
        barrier.wait()
        store.get_memory(f"session-{index % 10}").add_user_message(f"from-{index}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(40)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    row = store._conn.execute("SELECT COUNT(*) AS c FROM sessions").fetchone()
    assert row["c"] == 10
    for session_index in range(10):
        messages = store.get_memory(f"session-{session_index}").get_messages()
        assert len(messages) == 4  # 40 threads / 10 sessions


# ===========================================================================
# G — transactions/recovery: a failed atomic operation leaves no partial
# conversation mutation
# ===========================================================================

class _FlakyConnection:
    """Wraps a real sqlite3.Connection and raises on the Nth `execute()`
    call, to simulate a failure partway through a logically-atomic
    operation. Only the subset of the connection interface
    SQLiteConversationMemory actually uses is implemented."""

    def __init__(self, real: sqlite3.Connection, fail_on_call: int):
        self._real = real
        self._calls = 0
        self._fail_on = fail_on_call

    def execute(self, sql, params=()):
        self._calls += 1
        if self._calls == self._fail_on:
            raise sqlite3.OperationalError("simulated failure")
        return self._real.execute(sql, params)

    def __enter__(self):
        self._real.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        return self._real.__exit__(exc_type, exc, tb)


def test_a_failure_during_eviction_leaves_the_insert_rolled_back_too(tmp_path) -> None:
    store = _store(tmp_path, max_messages_per_session=1)
    store.get_memory("a").add_user_message("first")  # 1 message stored, at the cap

    # _append's transaction body for the NEXT message is: INSERT (call 1),
    # SELECT COUNT (call 2), DELETE for eviction (call 3) -- fail exactly on
    # the eviction DELETE.
    flaky = _FlakyConnection(store._conn, fail_on_call=3)
    flaky_memory = SQLiteConversationMemory(flaky, store._lock, "a", max_messages=1)

    with pytest.raises(sqlite3.OperationalError):
        flaky_memory.add_assistant_message("second")

    # If the transaction were not atomic, "second" would be inserted even
    # though its eviction failed, leaving 2 messages (violating the cap).
    # The whole operation must roll back instead.
    assert store.get_memory("a").get_messages() == [{"role": "user", "content": "first"}]


def test_a_failure_after_insert_leaves_no_session_row(tmp_path) -> None:
    """New-session call order is: (1) existence check, (2) next-seq query,
    (3) INSERT, (4) post-insert COUNT. Failing on call 4 lets the INSERT
    execute within the transaction, then fails before commit -- the whole
    transaction, including the INSERT, must roll back."""
    store = _store(tmp_path)
    flaky = _FlakyConnection(store._conn, fail_on_call=4)
    broken_store = SQLiteSessionMemoryStore.__new__(SQLiteSessionMemoryStore)
    broken_store._conn = flaky
    broken_store._lock = store._lock
    broken_store._max_messages_per_session = 20
    broken_store._max_sessions = 1000

    with pytest.raises(sqlite3.OperationalError):
        broken_store.get_memory("new-session")

    row = store._conn.execute(
        "SELECT COUNT(*) AS c FROM sessions WHERE session_id = 'new-session'"
    ).fetchone()
    assert row["c"] == 0


# ===========================================================================
# J — security
# ===========================================================================

def test_sql_injection_attempt_in_session_id_is_treated_as_data(tmp_path) -> None:
    store = _store(tmp_path)
    malicious_id = "a'; DROP TABLE sessions; --"

    store.get_memory(malicious_id).add_user_message("hello")

    assert store.get_memory(malicious_id).get_messages() == [{"role": "user", "content": "hello"}]
    # the sessions table must still exist -- an injection would have dropped it.
    store.get_memory("another-session")


def test_sql_injection_attempt_in_message_content_is_treated_as_data(tmp_path) -> None:
    store = _store(tmp_path)
    malicious_content = "'; DROP TABLE conversation_messages; --"

    store.get_memory("s").add_user_message(malicious_content)

    assert store.get_memory("s").get_messages() == [{"role": "user", "content": malicious_content}]
    store.get_memory("s").add_assistant_message("still works")  # table wasn't dropped


def test_conversation_messages_table_has_no_tool_or_agent_state_columns(tmp_path) -> None:
    """Structural guard: only what ConversationMemory represents (role,
    content) is ever persisted -- never a tool call, observation, plan, or
    raw model output field."""
    store = _store(tmp_path)
    store.get_memory("s").add_user_message("hi")

    columns = {row[1] for row in store._conn.execute("PRAGMA table_info(conversation_messages)")}
    assert columns == {"id", "session_id", "role", "content"}


def test_sessions_table_has_no_secret_or_identity_columns(tmp_path) -> None:
    store = _store(tmp_path)
    store.get_memory("s")

    columns = {row[1] for row in store._conn.execute("PRAGMA table_info(sessions)")}
    assert columns == {"session_id", "created_at", "last_accessed_at", "last_accessed_seq"}
