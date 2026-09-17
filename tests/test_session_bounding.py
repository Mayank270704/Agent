"""Milestone 23, items 1+2: bounded session count (LRU) and atomic
get-or-create on `InMemorySessionMemoryStore` (app/agent/memory.py).

Fully offline: no Ollama, no network, no LLM.
"""
from __future__ import annotations

import threading

import pytest

from app.agent.memory import InMemoryConversationMemory, InMemorySessionMemoryStore

# ===========================================================================
# 1a — normal retention below the limit
# ===========================================================================

def test_sessions_below_the_limit_are_all_retained() -> None:
    store = InMemorySessionMemoryStore(max_sessions=5)

    memories = {sid: store.get_memory(sid) for sid in ("a", "b", "c")}

    for sid, memory in memories.items():
        assert store.get_memory(sid) is memory  # same instance, nothing evicted


def test_default_capacity_is_generous_enough_for_ordinary_use() -> None:
    """Matches the existing test_session_memory.py::
    test_many_sessions_remain_mutually_isolated's scale (20 sessions) —
    the default must not regress ordinary usage."""
    store = InMemorySessionMemoryStore()
    for i in range(20):
        store.get_memory(f"session-{i}")

    assert len(store._sessions) == 20


# ===========================================================================
# 1b — deterministic eviction once the limit is exceeded
# ===========================================================================

def test_exceeding_the_limit_evicts_exactly_one_session() -> None:
    store = InMemorySessionMemoryStore(max_sessions=2)
    store.get_memory("a")
    store.get_memory("b")

    store.get_memory("c")  # third distinct session, capacity is 2

    assert len(store._sessions) == 2


def test_eviction_is_deterministic_across_repeated_runs() -> None:
    def run() -> list[str]:
        store = InMemorySessionMemoryStore(max_sessions=2)
        store.get_memory("a")
        store.get_memory("b")
        store.get_memory("c")
        return list(store._sessions.keys())

    assert run() == run()


# ===========================================================================
# 1c — the CORRECT session is evicted: least-recently-used, never an
# arbitrary/random choice, and never a session just touched
# ===========================================================================

def test_the_least_recently_used_session_is_evicted() -> None:
    store = InMemorySessionMemoryStore(max_sessions=2)
    store.get_memory("a")
    store.get_memory("b")

    store.get_memory("c")  # evicts 'a' (never touched again since creation)

    assert list(store._sessions.keys()) == ["b", "c"]


def test_re_accessing_a_session_protects_it_from_eviction() -> None:
    """The policy is LRU, not FIFO: touching 'a' again before 'c' is
    created must make 'b' the least-recently-used one instead."""
    store = InMemorySessionMemoryStore(max_sessions=2)
    store.get_memory("a")
    store.get_memory("b")
    store.get_memory("a")  # re-touch 'a' -> 'b' is now the LRU entry

    store.get_memory("c")

    assert list(store._sessions.keys()) == ["a", "c"]


def test_a_brand_new_session_is_never_evicted_by_its_own_creation() -> None:
    store = InMemorySessionMemoryStore(max_sessions=1)

    memory = store.get_memory("only-session")

    assert store.get_memory("only-session") is memory


def test_eviction_is_not_silent_on_arbitrary_heuristics_only_recency() -> None:
    """Structural proof the policy is recency-based, not e.g. session_id
    ordering or hash order: reversing access order reverses which
    session survives."""
    store_1 = InMemorySessionMemoryStore(max_sessions=2)
    store_1.get_memory("a")
    store_1.get_memory("b")
    store_1.get_memory("c")
    assert "a" not in store_1._sessions  # 'a' touched first -> evicted first

    store_2 = InMemorySessionMemoryStore(max_sessions=2)
    store_2.get_memory("c")
    store_2.get_memory("b")
    store_2.get_memory("a")
    assert "c" not in store_2._sessions  # now 'c' touched first -> evicted first


# ===========================================================================
# 1d — explicit session isolation and per-session bounds are unchanged
# ===========================================================================

def test_surviving_sessions_keep_their_own_history_after_an_eviction_elsewhere() -> None:
    store = InMemorySessionMemoryStore(max_sessions=2)
    store.get_memory("a").add_user_message("from a")
    store.get_memory("b").add_user_message("from b")

    store.get_memory("c").add_user_message("from c")  # evicts 'a'

    assert store.get_memory("b").get_messages() == [{"role": "user", "content": "from b"}]
    assert store.get_memory("c").get_messages() == [{"role": "user", "content": "from c"}]


def test_an_evicted_sessions_new_lookup_starts_genuinely_fresh() -> None:
    store = InMemorySessionMemoryStore(max_sessions=2)
    store.get_memory("a").add_user_message("old history")
    store.get_memory("b")
    store.get_memory("c")  # evicts 'a'

    fresh = store.get_memory("a")

    assert fresh.get_messages() == []


def test_per_session_message_bound_is_unchanged_by_session_count_bounding() -> None:
    store = InMemorySessionMemoryStore(max_messages_per_session=3, max_sessions=10)
    memory = store.get_memory("a")

    for i in range(10):
        memory.add_user_message(f"message {i}")

    assert len(memory.get_messages()) == 3  # unaffected by the new max_sessions concept


def test_max_sessions_must_be_at_least_one() -> None:
    with pytest.raises(ValueError):
        InMemorySessionMemoryStore(max_sessions=0)


def test_clear_session_still_works_normally_with_bounding_enabled() -> None:
    store = InMemorySessionMemoryStore(max_sessions=5)
    store.get_memory("a").add_user_message("hi")

    store.clear_session("a")

    assert store.get_memory("a").get_messages() == []


# ===========================================================================
# 1e — anonymous (request-scoped) memory is entirely untouched by this
# ===========================================================================

def test_request_scoped_anonymous_memory_never_touches_the_session_store() -> None:
    """Milestone 23 only bounds the SESSION store. A request-scoped
    InMemoryConversationMemory (session_id=None path, app/services/
    chat.py) is constructed directly, never through
    InMemorySessionMemoryStore, and is therefore entirely unaffected by
    max_sessions -- proven structurally: it has no session_id concept at
    all."""
    memory = InMemoryConversationMemory()

    memory.add_user_message("anonymous request")

    assert memory.get_messages() == [{"role": "user", "content": "anonymous request"}]
    assert not hasattr(memory, "session_id")


# ===========================================================================
# 2 — atomic get-or-create under concurrency
# ===========================================================================

def test_concurrent_first_access_to_a_new_session_yields_one_shared_memory() -> None:
    store = InMemorySessionMemoryStore()
    results: list[object] = [None] * 50
    barrier = threading.Barrier(50)

    def worker(index: int) -> None:
        barrier.wait()  # maximize the chance of a genuine race
        results[index] = store.get_memory("shared-new-session")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    first = results[0]
    assert all(result is first for result in results)  # exactly one instance was ever created
    assert len(store._sessions) == 1


def test_concurrent_access_never_loses_a_message_written_after_creation() -> None:
    """Every thread that raced to create the session, then immediately
    writes one message to whatever it got back. If initialization were
    ever lost (two different underlying objects silently created), some
    writes would land on an orphaned object and vanish from the final
    session's history."""
    store = InMemorySessionMemoryStore(max_messages_per_session=100)
    barrier = threading.Barrier(30)

    def worker(index: int) -> None:
        barrier.wait()
        memory = store.get_memory("race-session")
        memory.add_user_message(f"msg-{index}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(30)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    final_messages = {m["content"] for m in store.get_memory("race-session").get_messages()}
    assert final_messages == {f"msg-{i}" for i in range(30)}


def test_concurrent_access_across_many_distinct_new_sessions_creates_no_duplicates() -> None:
    store = InMemorySessionMemoryStore(max_sessions=1000)
    barrier = threading.Barrier(40)
    results: dict[int, object] = {}
    results_lock = threading.Lock()

    def worker(index: int) -> None:
        barrier.wait()
        memory = store.get_memory(f"session-{index % 10}")  # 40 threads, 10 distinct sessions
        with results_lock:
            results[index] = memory

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(40)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # every thread targeting the same session_id must have received the SAME object
    for session_index in range(10):
        instances = {id(results[i]) for i in range(40) if i % 10 == session_index}
        assert len(instances) == 1


def test_existing_session_behavior_is_unchanged_after_the_concurrency_fix() -> None:
    """Regression guard: the lock must not change ordinary single-threaded
    semantics in any observable way."""
    store = InMemorySessionMemoryStore()

    first = store.get_memory("a")
    second = store.get_memory("a")
    different = store.get_memory("b")

    assert first is second
    assert first is not different
