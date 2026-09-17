"""Anonymous request memory isolation.

Capability Assessment finding #3: `session_id=None` resolved to a single
`ChatService._default_memory` instance attribute, so every anonymous
caller shared one ever-growing conversation — reproduced live, where an
unrelated anonymous request was answered with content carried over from an
earlier one. Anonymous requests are now request-scoped.

Assertions here are made on the PROMPT each call actually built (and on
the session store's own contents), never on scripted reply text: a FakeLLM
returns whatever it was scripted to return regardless of context, so only
the prompt proves what the model was really given.

Fully offline: no Ollama, no network.
"""
from __future__ import annotations

import json

import pytest

from app.agent.memory import InMemoryConversationMemory
from app.services.chat import ChatService


class FakeLLM:
    def __init__(self, responses: list[str]):
        self.responses = iter(responses)
        self.prompts: list[str] = []

    def generate(self, messages: list[dict[str, str]], *, json_mode: bool = False) -> str:
        self.prompts.append(messages[-1]["content"])
        try:
            return next(self.responses)
        except StopIteration:
            raise AssertionError("FakeLLM ran out of scripted responses") from None


def _final_json(answer: str) -> str:
    return json.dumps({"action_type": "final", "final_answer": answer})


# ===========================================================================
# 1/2/3 — anonymous requests are isolated from each other
# ===========================================================================

def test_two_anonymous_requests_do_not_share_conversation_history() -> None:
    llm = FakeLLM([_final_json("Nice to meet you, Alice."), _final_json("I don't know.")])
    service = ChatService(llm_client=llm)

    service.ask("My name is Alice")
    service.ask("What is my name?")

    assert "Alice" not in llm.prompts[-1]


def test_anonymous_request_b_cannot_see_anonymous_request_a_messages() -> None:
    llm = FakeLLM([_final_json("ok"), _final_json("ok")])
    service = ChatService(llm_client=llm)

    service.ask("SECRET-ALPHA-TOKEN please remember this")
    service.ask("what did I just say?")

    second_prompt = llm.prompts[-1]
    assert "SECRET-ALPHA-TOKEN" not in second_prompt
    assert "what did I just say?" in second_prompt  # its OWN message is present


@pytest.mark.parametrize("call_count", [3, 5])
def test_multiple_anonymous_requests_remain_mutually_isolated(call_count: int) -> None:
    llm = FakeLLM([_final_json("ok")] * call_count)
    service = ChatService(llm_client=llm)

    for index in range(call_count):
        service.ask(f"ANON-MARKER-{index}")

    # Every prompt contains only its own marker, never an earlier one.
    for index, prompt in enumerate(llm.prompts):
        assert f"ANON-MARKER-{index}" in prompt
        for earlier in range(index):
            assert f"ANON-MARKER-{earlier}" not in prompt


def test_anonymous_memory_is_not_retained_on_the_service_after_a_request() -> None:
    """Structural proof: no attribute on ChatService holds a conversation
    for anonymous traffic — the shared attribute WAS the defect."""
    llm = FakeLLM([_final_json("ok")])
    service = ChatService(llm_client=llm)

    service.ask("ANON-RETENTION-PROBE")

    assert not hasattr(service, "_default_memory")
    for value in vars(service).values():
        if isinstance(value, InMemoryConversationMemory):
            assert value.get_messages() == []


def test_each_anonymous_request_gets_a_distinct_memory_instance() -> None:
    service = ChatService(llm_client=FakeLLM([]))

    first = service._resolve_memory(None)
    second = service._resolve_memory(None)

    assert first is not second
    assert isinstance(first, InMemoryConversationMemory)


# ===========================================================================
# 4/5 — explicit sessions are unchanged
# ===========================================================================

def test_explicit_session_still_persists_across_requests() -> None:
    llm = FakeLLM([_final_json("Nice to meet you, Alice."), _final_json("Your name is Alice.")])
    service = ChatService(llm_client=llm)

    service.ask("My name is Alice", session_id="A")
    service.ask("What is my name?", session_id="A")

    assert "Alice" in llm.prompts[-1]  # the session's history WAS carried forward
    assert len(service.session_store.get_memory("A").get_messages()) == 4


def test_two_explicit_sessions_remain_isolated_from_each_other() -> None:
    llm = FakeLLM([_final_json("ok")] * 3)
    service = ChatService(llm_client=llm)

    service.ask("SESSION-A-SECRET", session_id="A")
    service.ask("SESSION-B-SECRET", session_id="B")
    service.ask("what did I say?", session_id="B")

    third_prompt = llm.prompts[-1]
    assert "SESSION-B-SECRET" in third_prompt
    assert "SESSION-A-SECRET" not in third_prompt


def test_the_same_explicit_session_returns_the_same_memory_instance() -> None:
    service = ChatService(llm_client=FakeLLM([]))

    assert service._resolve_memory("A") is service._resolve_memory("A")
    assert service._resolve_memory("A") is not service._resolve_memory("B")


# ===========================================================================
# 6/7 — anonymous and explicit traffic do not affect each other
# ===========================================================================

def test_anonymous_requests_do_not_affect_an_explicit_session() -> None:
    llm = FakeLLM([_final_json("ok")] * 3)
    service = ChatService(llm_client=llm)

    service.ask("SESSION-A-FIRST", session_id="A")
    service.ask("ANON-INTERLOPER", session_id=None)
    service.ask("what did I say?", session_id="A")

    third_prompt = llm.prompts[-1]
    assert "SESSION-A-FIRST" in third_prompt  # session A intact
    assert "ANON-INTERLOPER" not in third_prompt  # anonymous traffic never joined it


def test_an_explicit_session_does_not_leak_into_an_anonymous_request() -> None:
    llm = FakeLLM([_final_json("ok")] * 2)
    service = ChatService(llm_client=llm)

    service.ask("SESSION-A-SECRET", session_id="A")
    service.ask("anonymous follow-up", session_id=None)

    assert "SESSION-A-SECRET" not in llm.prompts[-1]


def test_anonymous_traffic_creates_no_session_store_entry() -> None:
    """An anonymous request must not invent a session identity — not
    "default", not "anonymous", not the request_id."""
    llm = FakeLLM([_final_json("ok")])
    service = ChatService(llm_client=llm)

    service.ask("ANON-NO-IDENTITY")

    assert service.session_store._sessions == {}


def test_anonymous_requests_still_record_no_episode() -> None:
    """Pre-existing rule, preserved: request-scoped conversation memory
    must NOT be mistaken for a durable cross-request identity, so
    anonymous traffic still creates no episodic record."""
    llm = FakeLLM([_final_json("ok")])
    service = ChatService(llm_client=llm)

    service.ask("ANON-EPISODE-PROBE")

    for probe in ("A", "default", "anonymous", "None", ""):
        if probe:
            assert service.episodic_memory.get_recent(probe) == []


# ===========================================================================
# 8 — existing memory bounds still work
# ===========================================================================

def test_anonymous_memory_is_bounded() -> None:
    memory = ChatService(llm_client=FakeLLM([]))._resolve_memory(None)

    for index in range(50):
        memory.add_user_message(f"message {index}")

    assert len(memory.get_messages()) == 20  # InMemoryConversationMemory's default bound


def test_explicit_session_memory_bound_is_unchanged() -> None:
    memory = ChatService(llm_client=FakeLLM([]))._resolve_memory("A")

    for index in range(50):
        memory.add_user_message(f"message {index}")

    assert len(memory.get_messages()) == 20


# ===========================================================================
# 9 — production ChatService behavior
# ===========================================================================

def test_production_chat_service_isolates_anonymous_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.main as main_module

    llm = FakeLLM([_final_json("ok"), _final_json("ok")])
    monkeypatch.setattr(main_module.chat_service, "llm", llm)

    main_module.chat_service.ask("PROD-ANON-SECRET")
    main_module.chat_service.ask("what did I just say?")

    assert "PROD-ANON-SECRET" not in llm.prompts[-1]


def test_production_chat_service_still_persists_explicit_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.main as main_module

    llm = FakeLLM([_final_json("ok"), _final_json("ok")])
    monkeypatch.setattr(main_module.chat_service, "llm", llm)
    session_id = "prod-isolation-probe-session"

    main_module.chat_service.ask("PROD-SESSION-SECRET", session_id=session_id)
    main_module.chat_service.ask("what did I say?", session_id=session_id)

    assert "PROD-SESSION-SECRET" in llm.prompts[-1]
    main_module.chat_service.session_store.clear_session(session_id)


def test_production_chat_service_has_no_shared_anonymous_memory_attribute() -> None:
    import app.main as main_module

    assert not hasattr(main_module.chat_service, "_default_memory")


# ===========================================================================
# 10 — no security / permission / routing / telemetry behavior change
# ===========================================================================

def test_security_and_routing_wiring_is_untouched() -> None:
    import app.main as main_module
    from app.agent.reliability import BudgetedCorrectionPolicy
    from app.agent.tool_execution import ToolExecutionGate

    assert isinstance(main_module.chat_service.tool_execution_gate, ToolExecutionGate)
    assert main_module.chat_service.tool_execution_gate is main_module._tool_execution_gate
    assert isinstance(main_module._correction_policy, BudgetedCorrectionPolicy)
    assert main_module.chat_service.correction_policy is main_module._correction_policy
    assert main_module.chat_service.deterministic_temporal_routing is True
    assert sorted(main_module._permission_policy.allowed_tools) == ["date", "time", "web_search"]


def test_request_id_is_still_per_request_and_never_a_conversation_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """request_id remains telemetry-only: two anonymous requests get two
    different request_ids, and neither becomes a session store key."""
    import app.main as main_module
    from app.agent.telemetry import ListEventSink

    sink = ListEventSink()
    llm = FakeLLM([_final_json("ok"), _final_json("ok")])
    monkeypatch.setattr(main_module.chat_service, "llm", llm)
    monkeypatch.setattr(main_module.chat_service, "event_sink", sink)

    main_module.chat_service.ask("first anonymous")
    main_module.chat_service.ask("second anonymous")

    request_ids = {event.request_id for event in sink.events}
    assert len(request_ids) == 2
    for request_id in request_ids:
        assert request_id not in main_module.chat_service.session_store._sessions
    assert all(event.session_id is None for event in sink.events)
