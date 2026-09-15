from __future__ import annotations

import pytest

from app.agent.memory import InMemorySessionMemoryStore, SessionMemoryStore


# ---------------------------------------------------------------------------
# 1-4: basic get-or-create semantics.
# ---------------------------------------------------------------------------

def test_empty_store_has_no_sessions() -> None:
    store = InMemorySessionMemoryStore()

    # No public way to enumerate sessions (deliberately minimal) — the
    # observable fact of "empty" is that every session starts fresh.
    assert store.get_memory("A").get_messages() == []


def test_create_session_returns_a_conversation_memory() -> None:
    store = InMemorySessionMemoryStore()

    memory = store.get_memory("A")

    assert memory.get_messages() == []
    memory.add_user_message("hello")
    assert memory.get_messages() == [{"role": "user", "content": "hello"}]


def test_same_session_id_returns_the_same_memory_instance() -> None:
    store = InMemorySessionMemoryStore()

    memory_a1 = store.get_memory("A")
    memory_a2 = store.get_memory("A")

    assert memory_a1 is memory_a2


def test_different_session_ids_return_different_memory_instances() -> None:
    store = InMemorySessionMemoryStore()

    memory_a = store.get_memory("A")
    memory_b = store.get_memory("B")

    assert memory_a is not memory_b


# ---------------------------------------------------------------------------
# 5: isolation between sessions.
# ---------------------------------------------------------------------------

def test_session_a_history_is_isolated_from_session_b() -> None:
    store = InMemorySessionMemoryStore()

    store.get_memory("A").add_user_message("My name is Alice.")
    store.get_memory("A").add_assistant_message("Nice to meet you, Alice.")
    store.get_memory("B").add_user_message("My name is Bob.")
    store.get_memory("B").add_assistant_message("Nice to meet you, Bob.")

    assert store.get_memory("A").get_messages() == [
        {"role": "user", "content": "My name is Alice."},
        {"role": "assistant", "content": "Nice to meet you, Alice."},
    ]
    assert store.get_memory("B").get_messages() == [
        {"role": "user", "content": "My name is Bob."},
        {"role": "assistant", "content": "Nice to meet you, Bob."},
    ]


# ---------------------------------------------------------------------------
# 6-8: clear_session().
# ---------------------------------------------------------------------------

def test_clear_session_removes_history() -> None:
    store = InMemorySessionMemoryStore()
    store.get_memory("A").add_user_message("hello")

    store.clear_session("A")

    assert store.get_memory("A").get_messages() == []


def test_cleared_session_gets_a_genuinely_fresh_memory_object() -> None:
    store = InMemorySessionMemoryStore()
    memory = store.get_memory("A")
    memory.add_user_message("hello")

    store.clear_session("A")
    new_memory = store.get_memory("A")

    assert new_memory is not memory
    assert new_memory.get_messages() == []


def test_old_memory_object_is_unaffected_by_clearing_and_does_not_reappear() -> None:
    store = InMemorySessionMemoryStore()
    memory = store.get_memory("A")
    memory.add_user_message("hello")

    store.clear_session("A")

    # The caller's old reference still has its own (now orphaned) history —
    # clearing doesn't retroactively wipe an object someone still holds —
    # but the store itself never hands it out again.
    assert memory.get_messages() == [{"role": "user", "content": "hello"}]
    assert store.get_memory("A") is not memory


def test_clear_session_on_a_never_created_session_is_a_safe_no_op() -> None:
    store = InMemorySessionMemoryStore()

    store.clear_session("never-existed")  # must not raise

    assert store.get_memory("never-existed").get_messages() == []


# ---------------------------------------------------------------------------
# 9: invalid session IDs.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_session_id", ["", "   ", None, 123, ["A"]])
def test_invalid_session_id_rejected_by_get_memory(bad_session_id: object) -> None:
    store = InMemorySessionMemoryStore()

    with pytest.raises(ValueError):
        store.get_memory(bad_session_id)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_session_id", ["", "   ", None])
def test_invalid_session_id_rejected_by_clear_session(bad_session_id: object) -> None:
    store = InMemorySessionMemoryStore()

    with pytest.raises(ValueError):
        store.clear_session(bad_session_id)  # type: ignore[arg-type]


def test_session_id_whitespace_is_normalized_consistently() -> None:
    store = InMemorySessionMemoryStore()

    store.get_memory("A").add_user_message("hello")

    # Leading/trailing whitespace must resolve to the SAME session.
    assert store.get_memory("  A  ") is store.get_memory("A")
    assert store.get_memory(" A") .get_messages() == [{"role": "user", "content": "hello"}]


# ---------------------------------------------------------------------------
# 10: multiple independent store instances.
# ---------------------------------------------------------------------------

def test_multiple_independent_store_instances_do_not_share_state() -> None:
    store_a = InMemorySessionMemoryStore()
    store_b = InMemorySessionMemoryStore()

    store_a.get_memory("A").add_user_message("only in store_a")

    assert store_b.get_memory("A").get_messages() == []


# ---------------------------------------------------------------------------
# 11: configurable max_messages is respected.
# ---------------------------------------------------------------------------

def test_configurable_max_messages_per_session_is_respected() -> None:
    store = InMemorySessionMemoryStore(max_messages_per_session=2)

    memory = store.get_memory("A")
    memory.add_user_message("m1")
    memory.add_assistant_message("m2")
    memory.add_user_message("m3")  # exceeds the limit -> m1 evicted

    assert [m["content"] for m in memory.get_messages()] == ["m2", "m3"]


@pytest.mark.parametrize("bad_max", [0, -1, -100])
def test_max_messages_per_session_must_be_at_least_one(bad_max: int) -> None:
    with pytest.raises(ValueError):
        InMemorySessionMemoryStore(max_messages_per_session=bad_max)


# ---------------------------------------------------------------------------
# 12: many sessions remain isolated.
# ---------------------------------------------------------------------------

def test_many_sessions_remain_mutually_isolated() -> None:
    store = InMemorySessionMemoryStore()
    session_ids = [f"session-{i}" for i in range(20)]

    for session_id in session_ids:
        store.get_memory(session_id).add_user_message(f"hello from {session_id}")

    for session_id in session_ids:
        messages = store.get_memory(session_id).get_messages()
        assert messages == [{"role": "user", "content": f"hello from {session_id}"}]


# ---------------------------------------------------------------------------
# Structural: the concrete implementation satisfies the Protocol.
# ---------------------------------------------------------------------------

def test_in_memory_session_memory_store_conforms_to_the_protocol() -> None:
    assert isinstance(InMemorySessionMemoryStore(), SessionMemoryStore)
