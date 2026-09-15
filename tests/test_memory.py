from __future__ import annotations

import pytest

from app.agent.memory import ConversationMemory, InMemoryConversationMemory


# ---------------------------------------------------------------------------
# 1-4: basic construction, adding messages, insertion order.
# ---------------------------------------------------------------------------

def test_empty_memory_has_no_messages() -> None:
    memory = InMemoryConversationMemory()

    assert memory.get_messages() == []


def test_add_user_message() -> None:
    memory = InMemoryConversationMemory()

    memory.add_user_message("hello")

    assert memory.get_messages() == [{"role": "user", "content": "hello"}]


def test_add_assistant_message() -> None:
    memory = InMemoryConversationMemory()

    memory.add_assistant_message("hi there")

    assert memory.get_messages() == [{"role": "assistant", "content": "hi there"}]


def test_insertion_order_is_preserved() -> None:
    memory = InMemoryConversationMemory()

    memory.add_user_message("first")
    memory.add_assistant_message("second")
    memory.add_user_message("third")

    assert memory.get_messages() == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
        {"role": "user", "content": "third"},
    ]


# ---------------------------------------------------------------------------
# 5-6: get_messages returns a copy; internal state can't be mutated through it.
# ---------------------------------------------------------------------------

def test_get_messages_returns_a_new_list_each_time() -> None:
    memory = InMemoryConversationMemory()
    memory.add_user_message("hello")

    first_call = memory.get_messages()
    second_call = memory.get_messages()

    assert first_call == second_call
    assert first_call is not second_call


def test_mutating_the_returned_list_does_not_affect_memory() -> None:
    memory = InMemoryConversationMemory()
    memory.add_user_message("hello")

    returned = memory.get_messages()
    returned.append({"role": "user", "content": "smuggled in"})
    returned[0]["content"] = "corrupted"

    assert memory.get_messages() == [{"role": "user", "content": "hello"}]


# ---------------------------------------------------------------------------
# 7-9: validation.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("blank", ["", "   "])
def test_reject_empty_user_message(blank: str) -> None:
    memory = InMemoryConversationMemory()

    with pytest.raises(ValueError):
        memory.add_user_message(blank)


@pytest.mark.parametrize("blank", ["", "   "])
def test_reject_empty_assistant_message(blank: str) -> None:
    memory = InMemoryConversationMemory()

    with pytest.raises(ValueError):
        memory.add_assistant_message(blank)


@pytest.mark.parametrize("bad_value", [None, 123, ["not", "a", "string"], {"content": "x"}])
def test_reject_non_string_message(bad_value: object) -> None:
    memory = InMemoryConversationMemory()

    with pytest.raises(ValueError):
        memory.add_user_message(bad_value)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        memory.add_assistant_message(bad_value)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 10-12: max_messages validation and eviction.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_max", [0, -1, -100])
def test_max_messages_must_be_at_least_one(bad_max: int) -> None:
    with pytest.raises(ValueError):
        InMemoryConversationMemory(max_messages=bad_max)


def test_max_messages_of_one_is_allowed() -> None:
    memory = InMemoryConversationMemory(max_messages=1)

    memory.add_user_message("hello")

    assert memory.get_messages() == [{"role": "user", "content": "hello"}]


def test_oldest_messages_are_evicted_once_the_limit_is_exceeded() -> None:
    memory = InMemoryConversationMemory(max_messages=3)

    memory.add_user_message("m1")
    memory.add_assistant_message("m2")
    memory.add_user_message("m3")
    memory.add_assistant_message("m4")  # exceeds the limit -> m1 evicted

    assert [m["content"] for m in memory.get_messages()] == ["m2", "m3", "m4"]


def test_newest_messages_remain_after_repeated_eviction() -> None:
    memory = InMemoryConversationMemory(max_messages=2)

    for i in range(1, 6):
        memory.add_user_message(f"m{i}")

    assert [m["content"] for m in memory.get_messages()] == ["m4", "m5"]
    assert len(memory.get_messages()) == 2


# ---------------------------------------------------------------------------
# 13: clear().
# ---------------------------------------------------------------------------

def test_clear_removes_all_messages() -> None:
    memory = InMemoryConversationMemory()
    memory.add_user_message("hello")
    memory.add_assistant_message("hi")

    memory.clear()

    assert memory.get_messages() == []


def test_memory_can_be_used_again_after_clear() -> None:
    memory = InMemoryConversationMemory()
    memory.add_user_message("hello")
    memory.clear()

    memory.add_user_message("new conversation")

    assert memory.get_messages() == [{"role": "user", "content": "new conversation"}]


# ---------------------------------------------------------------------------
# 14: role correctness — the API, not the caller, controls the role.
# ---------------------------------------------------------------------------

def test_role_is_determined_by_which_method_is_called() -> None:
    memory = InMemoryConversationMemory()

    memory.add_user_message("a user message")
    memory.add_assistant_message("an assistant message")

    messages = memory.get_messages()
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"


def test_add_user_message_has_no_role_parameter_to_smuggle_a_different_role() -> None:
    import inspect

    signature = inspect.signature(InMemoryConversationMemory.add_user_message)
    assert list(signature.parameters) == ["self", "message"]


# ---------------------------------------------------------------------------
# 15: multiple turns.
# ---------------------------------------------------------------------------

def test_multiple_conversation_turns_accumulate_correctly() -> None:
    memory = InMemoryConversationMemory()

    memory.add_user_message("What is Python?")
    memory.add_assistant_message("Python is a programming language.")
    memory.add_user_message("What about JavaScript?")
    memory.add_assistant_message("JavaScript is also a programming language.")

    assert memory.get_messages() == [
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a programming language."},
        {"role": "user", "content": "What about JavaScript?"},
        {"role": "assistant", "content": "JavaScript is also a programming language."},
    ]


# ---------------------------------------------------------------------------
# Structural: the concrete implementation satisfies the Protocol.
# ---------------------------------------------------------------------------

def test_in_memory_conversation_memory_conforms_to_the_protocol() -> None:
    assert isinstance(InMemoryConversationMemory(), ConversationMemory)


# ---------------------------------------------------------------------------
# Part 12: isolation — two independent instances never share state.
# ---------------------------------------------------------------------------

def test_two_independent_memory_instances_do_not_share_state() -> None:
    memory_a = InMemoryConversationMemory()
    memory_b = InMemoryConversationMemory()

    memory_a.add_user_message("only in A")

    assert memory_a.get_messages() == [{"role": "user", "content": "only in A"}]
    assert memory_b.get_messages() == []


def test_default_max_messages_does_not_leak_across_instances() -> None:
    """Guards against a classic Python bug: a mutable default argument
    shared across instances. Each instance must own its own list."""
    memory_a = InMemoryConversationMemory()
    memory_b = InMemoryConversationMemory()

    for i in range(25):
        memory_a.add_user_message(f"msg {i}")

    assert memory_b.get_messages() == []
