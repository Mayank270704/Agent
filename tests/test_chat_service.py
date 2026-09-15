"""Tests for ChatService (app/services/chat.py), including Step 13's
session-scoped memory.

Every test uses a FakeLLM injected via ChatService(llm_client=...) — the
REAL AgentOrchestrator -> AgentLoop -> LLMDecisionMaker pipeline runs
underneath (only the LLM boundary is faked), so these tests exercise the
actual production code path, not a shortcut around it. No network, no
Ollama, no Tavily.
"""
from __future__ import annotations

import json

import pytest

from app.agent.episodic_memory import InMemoryEpisodicMemory
from app.services.chat import ChatService


class FakeLLM:
    """Replays a fixed sequence of raw text responses, one per generate()
    call."""

    def __init__(self, responses: list[str]):
        self.responses = iter(responses)
        self.calls: list[dict[str, object]] = []

    def generate(self, messages: list[dict[str, str]], *, json_mode: bool = False) -> str:
        self.calls.append({"messages": messages, "json_mode": json_mode})
        try:
            return next(self.responses)
        except StopIteration:
            raise AssertionError("FakeLLM ran out of scripted responses") from None


def _final_json(answer: str) -> str:
    return json.dumps({"action_type": "final", "final_answer": answer})


# ---------------------------------------------------------------------------
# 1: ask() still works (no session_id) — backward compatibility.
# ---------------------------------------------------------------------------

def test_ask_without_session_id_still_works() -> None:
    chat_service = ChatService(llm_client=FakeLLM([_final_json("Python is a programming language.")]))

    reply = chat_service.ask("What is Python?")

    assert reply == "Python is a programming language."


def test_ask_without_session_id_preserves_the_legacy_single_conversation_across_calls() -> None:
    chat_service = ChatService(llm_client=FakeLLM([
        _final_json("Nice to meet you, Alice."),
        _final_json("Your name is Alice."),
    ]))

    chat_service.ask("My name is Alice")
    reply = chat_service.ask("What is my name?")

    assert reply == "Your name is Alice."


# ---------------------------------------------------------------------------
# 2/3: explicit session ID uses session memory and preserves conversation.
# ---------------------------------------------------------------------------

def test_explicit_session_id_uses_session_memory_and_preserves_conversation() -> None:
    chat_service = ChatService(llm_client=FakeLLM([
        _final_json("Nice to meet you, Alice."),
        _final_json("Your name is Alice."),
    ]))

    first = chat_service.ask("My name is Alice", session_id="A")
    second = chat_service.ask("What is my name?", session_id="A")

    assert first == "Nice to meet you, Alice."
    assert second == "Your name is Alice."
    assert chat_service.session_store.get_memory("A").get_messages() == [
        {"role": "user", "content": "My name is Alice"},
        {"role": "assistant", "content": "Nice to meet you, Alice."},
        {"role": "user", "content": "What is my name?"},
        {"role": "assistant", "content": "Your name is Alice."},
    ]


# ---------------------------------------------------------------------------
# 4-8: two sessions remain isolated; each second request sees only its own
# session's history.
# ---------------------------------------------------------------------------

def test_two_sessions_remain_isolated_and_each_sees_only_its_own_history() -> None:
    chat_service = ChatService(llm_client=FakeLLM([
        _final_json("Nice to meet you, Alice."),
        _final_json("Nice to meet you, Bob."),
        _final_json("Your name is Alice."),
        _final_json("Your name is Bob."),
    ]))

    chat_service.ask("My name is Alice", session_id="A")
    chat_service.ask("My name is Bob", session_id="B")
    reply_a = chat_service.ask("What is my name?", session_id="A")
    reply_b = chat_service.ask("What is my name?", session_id="B")

    assert reply_a == "Your name is Alice."
    assert reply_b == "Your name is Bob."
    assert chat_service.session_store.get_memory("A").get_messages() == [
        {"role": "user", "content": "My name is Alice"},
        {"role": "assistant", "content": "Nice to meet you, Alice."},
        {"role": "user", "content": "What is my name?"},
        {"role": "assistant", "content": "Your name is Alice."},
    ]
    assert chat_service.session_store.get_memory("B").get_messages() == [
        {"role": "user", "content": "My name is Bob"},
        {"role": "assistant", "content": "Nice to meet you, Bob."},
        {"role": "user", "content": "What is my name?"},
        {"role": "assistant", "content": "Your name is Bob."},
    ]
    # Cross-check: neither session's memory ever mentions the other name.
    for message in chat_service.session_store.get_memory("A").get_messages():
        assert "Bob" not in message["content"]
    for message in chat_service.session_store.get_memory("B").get_messages():
        assert "Alice" not in message["content"]


# ---------------------------------------------------------------------------
# 9: invalid session ID rejected.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_session_id", ["", "   "])
def test_invalid_session_id_rejected(bad_session_id: str) -> None:
    chat_service = ChatService(llm_client=FakeLLM([]))

    with pytest.raises(ValueError):
        chat_service.ask("hello", session_id=bad_session_id)


def test_empty_user_message_is_still_rejected_regardless_of_session_id() -> None:
    chat_service = ChatService(llm_client=FakeLLM([]))

    with pytest.raises(ValueError):
        chat_service.ask("   ", session_id="A")


# ---------------------------------------------------------------------------
# 10: separate ChatService instances do not share session state.
# ---------------------------------------------------------------------------

def test_separate_chat_service_instances_do_not_share_session_state() -> None:
    chat_service_1 = ChatService(llm_client=FakeLLM([_final_json("Nice to meet you, Alice.")]))
    chat_service_2 = ChatService(llm_client=FakeLLM([]))

    chat_service_1.ask("My name is Alice", session_id="A")

    assert chat_service_1.session_store is not chat_service_2.session_store
    assert chat_service_2.session_store.get_memory("A").get_messages() == []


# ---------------------------------------------------------------------------
# Step 14 correction: episodic memory is now LIVE through the real
# ChatService wiring (Part 11's 10 items), not just at the
# AgentOrchestrator level.
# ---------------------------------------------------------------------------

def test_chat_service_owns_an_episodic_memory_instance() -> None:
    """Part 11 item 1."""
    chat_service = ChatService(llm_client=FakeLLM([]))

    assert isinstance(chat_service.episodic_memory, InMemoryEpisodicMemory)


def test_explicit_session_a_creates_an_episodic_record() -> None:
    """Part 11 item 2 / Part 8."""
    chat_service = ChatService(llm_client=FakeLLM([_final_json("Nice to meet you, Alice.")]))

    chat_service.ask("My name is Alice", session_id="A")

    records = chat_service.episodic_memory.get_recent("A")
    assert len(records) == 1
    assert records[0].session_id == "A"
    assert records[0].event_type == "conversation_completed"


def test_explicit_session_b_creates_an_episodic_record() -> None:
    """Part 11 item 3 / Part 8."""
    chat_service = ChatService(llm_client=FakeLLM([_final_json("Nice to meet you, Bob.")]))

    chat_service.ask("My name is Bob", session_id="B")

    records = chat_service.episodic_memory.get_recent("B")
    assert len(records) == 1
    assert records[0].session_id == "B"


def test_sessions_a_and_b_remain_isolated_in_episodic_memory() -> None:
    """Part 11 item 4 / Part 10."""
    chat_service = ChatService(llm_client=FakeLLM([
        _final_json("Nice to meet you, Alice."),
        _final_json("Nice to meet you, Bob."),
    ]))

    chat_service.ask("My name is Alice", session_id="A")
    chat_service.ask("My name is Bob", session_id="B")

    records_a = chat_service.episodic_memory.get_recent("A")
    records_b = chat_service.episodic_memory.get_recent("B")
    assert len(records_a) == 1 and len(records_b) == 1
    assert "Alice" in records_a[0].summary
    assert "Bob" not in records_a[0].summary
    assert "Bob" in records_b[0].summary
    assert "Alice" not in records_b[0].summary

    chat_service.episodic_memory.clear("A")

    assert chat_service.episodic_memory.get_recent("A") == []
    assert len(chat_service.episodic_memory.get_recent("B")) == 1  # B untouched


def test_same_session_creates_multiple_episodic_events() -> None:
    """Part 11 item 5."""
    chat_service = ChatService(llm_client=FakeLLM([
        _final_json("Nice to meet you, Alice."),
        _final_json("Your name is Alice."),
    ]))

    chat_service.ask("My name is Alice", session_id="A")
    chat_service.ask("What is my name?", session_id="A")

    assert len(chat_service.episodic_memory.get_recent("A", limit=10)) == 2


def test_failed_request_creates_no_episode() -> None:
    """Part 11 item 6 / Part 9: malformed model output fails the execution
    gracefully (AgentStatus.FAILED) rather than raising, and must not
    create an episode nor disturb conversation memory."""
    chat_service = ChatService(llm_client=FakeLLM(["this is not valid JSON"]))

    reply = chat_service.ask("do something", session_id="A")

    assert "could not complete this request" in reply
    assert chat_service.episodic_memory.get_recent("A") == []
    assert chat_service.session_store.get_memory("A").get_messages() == []


def test_session_id_none_does_not_create_an_episodic_record() -> None:
    """Part 11 item 7 / Part 6: the legacy no-session path never invents a
    fake session identity, so episodic recording stays off entirely."""
    chat_service = ChatService(llm_client=FakeLLM([_final_json("Python is a programming language.")]))

    chat_service.ask("What is Python?")

    assert chat_service.episodic_memory.get_recent("A") == []
    # No public way to enumerate "all sessions" (deliberately minimal store)
    # -- the observable fact is that not a single explicit session gained a
    # record from this call.
    for probe_session_id in ("A", "B", "default", "anonymous", "None"):
        assert chat_service.episodic_memory.get_recent(probe_session_id) == []


def test_separate_chat_service_instances_have_separate_episodic_memory() -> None:
    """Part 11 item 8 / Part 16."""
    chat_service_1 = ChatService(llm_client=FakeLLM([_final_json("Nice to meet you, Alice.")]))
    chat_service_2 = ChatService(llm_client=FakeLLM([]))

    chat_service_1.ask("My name is Alice", session_id="A")

    assert chat_service_1.episodic_memory is not chat_service_2.episodic_memory
    assert chat_service_2.episodic_memory.get_recent("A") == []


def test_conversation_memory_behavior_remains_correct_alongside_episodic_memory() -> None:
    """Part 11 item 9 / Part 15: ConversationMemory (Step 12/13 behavior)
    stays exactly as it was, unaffected by the new episodic wiring."""
    chat_service = ChatService(llm_client=FakeLLM([
        _final_json("Nice to meet you, Alice."),
        _final_json("Your name is Alice."),
    ]))

    chat_service.ask("My name is Alice", session_id="A")
    chat_service.ask("What is my name?", session_id="A")

    assert chat_service.session_store.get_memory("A").get_messages() == [
        {"role": "user", "content": "My name is Alice"},
        {"role": "assistant", "content": "Nice to meet you, Alice."},
        {"role": "user", "content": "What is my name?"},
        {"role": "assistant", "content": "Your name is Alice."},
    ]


def test_ask_without_session_id_still_works_after_episodic_wiring() -> None:
    """Part 11 item 10 (regression)."""
    chat_service = ChatService(llm_client=FakeLLM([_final_json("Python is a programming language.")]))

    reply = chat_service.ask("What is Python?")

    assert reply == "Python is a programming language."


def test_same_service_two_sessions_remain_isolated_in_episodic_memory() -> None:
    """Part 16: Service A / Session A != Service A / Session B."""
    chat_service = ChatService(llm_client=FakeLLM([
        _final_json("Nice to meet you, Alice."),
        _final_json("Nice to meet you, Bob."),
    ]))

    chat_service.ask("My name is Alice", session_id="A")
    chat_service.ask("My name is Bob", session_id="B")

    records_a = chat_service.episodic_memory.get_recent("A")
    records_b = chat_service.episodic_memory.get_recent("B")
    assert records_a != records_b
    assert "Alice" in records_a[0].summary
    assert "Bob" in records_b[0].summary
