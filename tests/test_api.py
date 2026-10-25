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


# ---------------------------------------------------------------------------
# Milestone 41: request-size validation (ChatRequest.message/session_id).
# ---------------------------------------------------------------------------

def test_oversized_message_is_rejected_with_422(client: TestClient) -> None:
    response = client.post("/chat", json={"message": "x" * (main_module.MAX_MESSAGE_LENGTH + 1)})

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "message"]


def test_oversized_session_id_is_rejected_with_422(client: TestClient) -> None:
    response = client.post(
        "/chat", json={"message": "hi", "session_id": "x" * (main_module.MAX_SESSION_ID_LENGTH + 1)}
    )

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "session_id"]


def test_message_exactly_at_the_length_limit_is_accepted(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_fake_llm(monkeypatch, [_final_json("ok")])

    response = client.post("/chat", json={"message": "x" * main_module.MAX_MESSAGE_LENGTH})

    assert response.status_code == 200
    assert response.json() == {"reply": "ok"}


def test_session_id_exactly_at_the_length_limit_is_accepted(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_fake_llm(monkeypatch, [_final_json("ok")])

    response = client.post(
        "/chat", json={"message": "hi", "session_id": "s" * main_module.MAX_SESSION_ID_LENGTH}
    )

    assert response.status_code == 200


def test_malformed_json_body_returns_422(client: TestClient) -> None:
    response = client.post("/chat", content=b"{not valid json", headers={"Content-Type": "application/json"})

    assert response.status_code == 422


def test_unexpected_extra_fields_are_silently_ignored_not_rejected(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unknown fields must not break a forward-compatible client -- they
    are ignored, never used for anything (ChatRequest only ever reads
    .message/.session_id)."""
    _use_fake_llm(monkeypatch, [_final_json("ok")])

    response = client.post(
        "/chat", json={"message": "hi", "session_id": "A", "client_request_id": "abc-123", "debug": True}
    )

    assert response.status_code == 200
    assert response.json() == {"reply": "ok"}


# ---------------------------------------------------------------------------
# Milestone 41: error-contract audit (tool/model/unexpected failure).
# ---------------------------------------------------------------------------

def test_model_runtime_error_returns_502_with_no_internal_leak_beyond_the_existing_message(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FailingLLM:
        def generate(self, messages, *, json_mode: bool = False) -> str:
            raise RuntimeError("Ollama connection failed at http://localhost:11434: test")

    monkeypatch.setattr(main_module.chat_service, "llm", FailingLLM())

    response = client.post("/chat", json={"message": "hello"})

    assert response.status_code == 502
    body = response.json()
    assert set(body.keys()) == {"detail"}
    # No secret, no stack trace, no file path -- only the pre-existing,
    # already-reviewed RuntimeError text (Milestone 40 Security Audit §8).
    assert "Traceback" not in body["detail"]
    assert "TAVILY_API_KEY" not in body["detail"]
    assert "OPENAI_API_KEY" not in body["detail"]


def test_missing_tavily_key_tool_failure_never_exposes_anything_to_the_client(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Corrects an initial assumption made while writing this test: a tool
    precondition failure (missing TAVILY_API_KEY, a ValueError raised
    INSIDE WebSearchTool.execute()) is NOT the same ValueError path /chat's
    own `except ValueError` branch handles (that branch only ever sees a
    ValueError from ChatService.ask()'s OWN input validation, e.g. an
    empty message -- raised before AgentOrchestrator is even built).

    A tool-level ValueError is instead caught by AgentLoop, classified
    INVALID_TOOL_INPUT, and run through the production
    BudgetedCorrectionPolicy (max_corrections=2): here the SAME failure
    repeats identically on retry, so the "repeated" rule terminates after
    attempt 2, the execution ends FAILED, and AgentOrchestrator's
    `_failure_answer` (Step 17 hardening, F8) returns the fixed, generic,
    safe string -- never the ValueError's own text -- as a NORMAL 200
    reply. This is MORE conservative than a 400 with the var name would
    have been: the TAVILY_API_KEY name itself is logged server-side only
    (WARNING level) and never appears anywhere in the HTTP response."""
    web_search_tool = main_module._tool_registry.get("web_search")
    monkeypatch.setattr(web_search_tool, "api_key", "")
    _use_fake_llm(
        monkeypatch,
        [
            json.dumps({"action_type": "tool", "tool_name": "web_search", "tool_input": "x"}),
            json.dumps({"action_type": "tool", "tool_name": "web_search", "tool_input": "x"}),
        ],
    )

    response = client.post("/chat", json={"message": "search the web for x"})

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"reply"}
    assert "could not complete this request" in body["reply"]
    assert "TAVILY_API_KEY" not in body["reply"]
    assert "tvly-" not in body["reply"]


def test_unexpected_exception_type_returns_a_consistent_json_500(monkeypatch: pytest.MonkeyPatch) -> None:
    """Milestone 41: the new catch-all handler keeps the error response
    SHAPE consistent (JSON {"detail": ...}) for whatever escapes /chat's
    own ValueError/RuntimeError/MemorySessionIsolationError handling --
    previously this fell through to Starlette's default plain-text 500.

    Uses a dedicated client with raise_server_exceptions=False: TestClient's
    DEFAULT (True, used by every other test in this file via the `client`
    fixture) deliberately re-raises an exception a handler did NOT catch,
    so test failures surface loudly -- but here the exception IS handled
    (by the new catch-all), and this test needs to see what a REAL
    deployed server would actually return to a REAL client, not pytest's
    own re-raised traceback."""
    import sqlite3

    class BrokenSessionStore:
        def get_memory(self, session_id):
            raise sqlite3.OperationalError("disk I/O error")

        def clear_session(self, session_id):
            pass

    original_store = main_module.chat_service.session_store
    monkeypatch.setattr(main_module.chat_service, "session_store", BrokenSessionStore())
    try:
        no_raise_client = TestClient(main_module.app, raise_server_exceptions=False)
        response = no_raise_client.post("/chat", json={"message": "hello", "session_id": "x"})
    finally:
        main_module.chat_service.session_store = original_store

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error."}
    assert response.headers["content-type"].startswith("application/json")


def test_existing_http_exception_paths_are_unaffected_by_the_catch_all_handler(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The new global Exception handler must never intercept an
    HTTPException a route raises on purpose -- FastAPI resolves the more
    specific HTTPException handler first (verified directly)."""
    _use_fake_llm(monkeypatch, [])

    response = client.post("/chat", json={"message": "   "})  # the EXISTING 400 path

    assert response.status_code == 400
    assert response.json() == {"detail": "User message cannot be empty."}


# ---------------------------------------------------------------------------
# Milestone 41: timeout, through the real HTTP layer.
# ---------------------------------------------------------------------------

def test_an_expired_deadline_fails_gracefully_through_the_real_http_layer(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Existing behavior (tests/test_request_timeout.py already proves this
    at the ChatService level) re-confirmed through the ACTUAL HTTP
    endpoint: a timeout returns HTTP 200 with a generic, safe answer --
    never an exception, never a 5xx, never raw deadline/internal text.
    This is deliberate, pre-existing AgentOrchestrator behavior
    (AgentResult.answer is always a usable string) -- not changed here."""

    class NeverCalledLLM:
        def generate(self, messages, *, json_mode: bool = False) -> str:
            raise AssertionError("LLM must never be called once the deadline has already passed")

    # Force an already-expired deadline the same way the loop itself checks
    # it (time.monotonic()) by patching settings.request_timeout_seconds to
    # 0 for this one call via the already-established pattern
    # (tests/test_request_timeout.py uses the identical approach).
    import app.services.chat as chat_module
    from app.config import Settings

    monkeypatch.setenv("REQUEST_TIMEOUT_SECONDS", "0")
    monkeypatch.setattr(chat_module, "settings", Settings())
    monkeypatch.setattr(main_module.chat_service, "llm", NeverCalledLLM())

    response = client.post("/chat", json={"message": "hello"})

    assert response.status_code == 200
    assert "could not complete this request" in response.json()["reply"]


# ---------------------------------------------------------------------------
# Milestone 41: OpenAPI surface.
# ---------------------------------------------------------------------------

def test_openapi_exposes_the_three_endpoints_with_descriptions(client: TestClient) -> None:
    spec = client.get("/openapi.json").json()

    assert set(spec["paths"].keys()) == {"/", "/health", "/chat"}
    assert spec["paths"]["/chat"]["post"].get("description")
    assert spec["paths"]["/health"]["get"].get("description")
    assert spec["info"].get("description")


def test_openapi_chat_schema_documents_message_and_session_id(client: TestClient) -> None:
    spec = client.get("/openapi.json").json()
    chat_request_schema = spec["components"]["schemas"]["ChatRequest"]["properties"]

    assert "message" in chat_request_schema
    assert "session_id" in chat_request_schema
    assert chat_request_schema["message"].get("description")
    assert chat_request_schema["session_id"].get("description")


def test_docs_endpoint_is_reachable(client: TestClient) -> None:
    response = client.get("/docs")

    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Milestone 42: CORS -- configuration-backed, naming the actual known
# frontend origin (frontend/, `npm run dev`, port 3000). Milestone 41
# deliberately deferred CORS because no frontend origin was known yet; this
# supersedes that state now that one exists. See
# docs/MILESTONE_42_FRONTEND.md for the full decision.
# ---------------------------------------------------------------------------

def test_default_cors_origins_include_the_local_frontend_dev_server() -> None:
    assert "http://localhost:3000" in main_module.settings.cors_allowed_origins
    assert "http://127.0.0.1:3000" in main_module.settings.cors_allowed_origins


def test_cors_allows_the_configured_frontend_origin(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_fake_llm(monkeypatch, [_final_json("ok")])

    response = client.post(
        "/chat",
        json={"message": "hi"},
        headers={"Origin": "http://localhost:3000"},
    )

    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_cors_preflight_request_succeeds_for_the_allowed_origin(client: TestClient) -> None:
    response = client.options(
        "/chat",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )

    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"
    allowed_methods = response.headers.get("access-control-allow-methods", "")
    assert "POST" in allowed_methods


def test_cors_does_not_allow_an_unrecognized_origin(client: TestClient) -> None:
    response = client.get("/health", headers={"Origin": "http://evil.example.com"})

    assert "access-control-allow-origin" not in {k.lower() for k in response.headers}


def test_cors_is_never_a_wildcard() -> None:
    """Structural guard: the configured origin list must never contain
    "*" -- Milestone 42 explicitly forbids allow_origins=["*"]."""
    assert "*" not in main_module.settings.cors_allowed_origins


def test_cors_does_not_allow_credentials() -> None:
    """This API uses no cookies/auth (session_id travels in the JSON
    body) -- allow_credentials stays False, the more restrictive choice,
    confirmed directly against the live middleware stack rather than
    just the settings value."""
    cors_middleware = next(
        m for m in main_module.app.user_middleware if m.cls.__name__ == "CORSMiddleware"
    )
    assert cors_middleware.kwargs.get("allow_credentials") is False


def test_cors_origins_are_configurable_via_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.config import Settings

    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "https://example.com, https://other.example.com")

    assert Settings().cors_allowed_origins == ("https://example.com", "https://other.example.com")


def test_cors_origins_default_when_env_var_is_unset_or_blank(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.config import Settings

    monkeypatch.delenv("CORS_ALLOWED_ORIGINS", raising=False)
    assert Settings().cors_allowed_origins == ("http://localhost:3000", "http://127.0.0.1:3000")

    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "   ")
    assert Settings().cors_allowed_origins == ("http://localhost:3000", "http://127.0.0.1:3000")
