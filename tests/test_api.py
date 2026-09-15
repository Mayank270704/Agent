"""API-level tests for POST /chat (app/main.py), including Step 13's
session_id support.

Uses FastAPI's TestClient (httpx-based) for genuine HTTP-level testing —
real Pydantic request validation, real status codes, real JSON response
shape — not a shortcut around the HTTP layer.

The module-level `chat_service` singleton's LLM is swapped for a FakeLLM
via `monkeypatch` in each test (auto-restored afterward). No network, no
Ollama, no Tavily.
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import app.main as main_module


class FakeLLM:
    def __init__(self, responses: list[str]):
        self.responses = iter(responses)

    def generate(self, messages: list[dict[str, str]], *, json_mode: bool = False) -> str:
        try:
            return next(self.responses)
        except StopIteration:
            raise AssertionError("FakeLLM ran out of scripted responses") from None


def _final_json(answer: str) -> str:
    return json.dumps({"action_type": "final", "final_answer": answer})


@pytest.fixture
def client() -> TestClient:
    return TestClient(main_module.app)


def _use_fake_llm(monkeypatch: pytest.MonkeyPatch, responses: list[str]) -> None:
    monkeypatch.setattr(main_module.chat_service, "llm", FakeLLM(responses))


# ---------------------------------------------------------------------------
# Backward compatibility: POST /chat without session_id still works.
# ---------------------------------------------------------------------------

def test_chat_without_session_id_still_works(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_fake_llm(monkeypatch, [_final_json("Python is a programming language.")])

    response = client.post("/chat", json={"message": "What is Python?"})

    assert response.status_code == 200
    assert response.json() == {"reply": "Python is a programming language."}


def test_get_root_still_works(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200


# ---------------------------------------------------------------------------
# POST /chat with session_id works.
# ---------------------------------------------------------------------------

def test_chat_with_session_id_works(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_fake_llm(monkeypatch, [_final_json("Nice to meet you, Alice.")])

    response = client.post("/chat", json={"message": "My name is Alice", "session_id": "A"})

    assert response.status_code == 200
    assert response.json() == {"reply": "Nice to meet you, Alice."}


# ---------------------------------------------------------------------------
# Session A conversation remains isolated from session B, through the API.
# ---------------------------------------------------------------------------

def test_session_a_and_session_b_remain_isolated_via_the_api(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_fake_llm(monkeypatch, [
        _final_json("Nice to meet you, Alice."),
        _final_json("Nice to meet you, Bob."),
        _final_json("Your name is Alice."),
        _final_json("Your name is Bob."),
    ])

    client.post("/chat", json={"message": "My name is Alice", "session_id": "A"})
    client.post("/chat", json={"message": "My name is Bob", "session_id": "B"})
    response_a = client.post("/chat", json={"message": "What is my name?", "session_id": "A"})
    response_b = client.post("/chat", json={"message": "What is my name?", "session_id": "B"})

    assert response_a.status_code == 200
    assert response_b.status_code == 200
    assert response_a.json() == {"reply": "Your name is Alice."}
    assert response_b.json() == {"reply": "Your name is Bob."}


# ---------------------------------------------------------------------------
# Malformed / invalid session IDs.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_session_id", ["", "   "])
def test_chat_with_blank_session_id_returns_400(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, bad_session_id: str
) -> None:
    _use_fake_llm(monkeypatch, [])

    response = client.post("/chat", json={"message": "hello", "session_id": bad_session_id})

    assert response.status_code == 400


def test_chat_with_empty_message_returns_400_regardless_of_session_id(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_fake_llm(monkeypatch, [])

    response = client.post("/chat", json={"message": "   ", "session_id": "A"})

    assert response.status_code == 400


def test_chat_with_missing_message_field_returns_422_pydantic_validation_error(client: TestClient) -> None:
    response = client.post("/chat", json={"session_id": "A"})

    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Response schema remains exactly {"reply": "..."}.
# ---------------------------------------------------------------------------

def test_response_schema_contains_only_reply(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_fake_llm(monkeypatch, [_final_json("ok")])

    response = client.post("/chat", json={"message": "hello", "session_id": "A"})

    assert set(response.json().keys()) == {"reply"}


def test_response_schema_unaffected_by_omitting_session_id(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_fake_llm(monkeypatch, [_final_json("ok")])

    response = client.post("/chat", json={"message": "hello"})

    assert set(response.json().keys()) == {"reply"}


# ---------------------------------------------------------------------------
# Step 14 correction (Part 12): a successful request with an explicit
# session_id actually reaches ChatService's live episodic memory, through
# the real HTTP layer -- no new /memory or /episodes endpoint is added; the
# response shape stays exactly {"reply": ...}.
# ---------------------------------------------------------------------------

def test_chat_with_session_id_reaches_chat_services_episodic_memory(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_fake_llm(monkeypatch, [_final_json("Nice to meet you, Alice.")])
    # A session_id unique to this test -- main_module.chat_service is a
    # module-level singleton shared by every test in this file, so reusing
    # "A"/"B" here would pick up episodic records left behind by other
    # tests that also use those session IDs.
    session_id = "episodic-api-test-session"

    response = client.post("/chat", json={"message": "My name is Alice", "session_id": session_id})

    assert response.status_code == 200
    assert response.json() == {"reply": "Nice to meet you, Alice."}
    records = main_module.chat_service.episodic_memory.get_recent(session_id)
    assert len(records) == 1
    assert records[0].session_id == session_id
    assert records[0].event_type == "conversation_completed"
