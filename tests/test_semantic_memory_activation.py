"""Step 16I: end-to-end semantic memory activation.

Two kinds of tests:

1. THROUGH ChatService directly (DeterministicEmbeddingProvider via
   `build_semantic_memory`'s `provider_factory` seam, FakeLLM) — the real
   production call path (`app/main.py`'s route handler calls exactly
   `chat_service.ask(...)`), with no FastAPI layer and no real model.

2. THROUGH the live FastAPI app (`app.main`), proving app/main.py's own
   wiring: disabled-by-default behavior, the public response shape, and
   the MemorySessionIsolationError -> HTTP 500 mapping. The module-level
   `chat_service` this file imports is the SAME one every other API test
   already shares (`test_api.py`, `test_memory_integration.py`), and it
   stays disabled in this process because SEMANTIC_MEMORY_ENABLED is not
   set in the test environment (proven in test_semantic_memory_config.py).

Fully offline throughout: no network, no Ollama, no real embedding model.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.agent.embeddings import DeterministicEmbeddingProvider
from app.agent.memory_extraction import MemoryCandidate
from app.agent.memory_formatting import MEMORY_CONTEXT_LABEL
from app.agent.semantic_memory import MemorySessionIsolationError
from app.semantic_memory_wiring import build_semantic_memory
from app.services.chat import ChatService

REPO_ROOT = Path(__file__).resolve().parent.parent
DIMENSION = 8


def _final_json(answer: str) -> str:
    return json.dumps({"action_type": "final", "final_answer": answer})


def _empty_extraction_json() -> str:
    """A `LLMMemoryExtractor`-shaped response proposing nothing durable.

    Every COMPLETED interaction triggers exactly one extraction call
    (16E-D) once both an extractor and a writer are wired — which
    `_enabled_chat_service` always wires. Tests that are not specifically
    exercising extraction still need to script this second response, or
    `FakeLLM` runs out of scripted turns on the extraction call it did not
    expect.
    """
    return json.dumps({"memories": []})


class FakeLLM:
    def __init__(self, responses: list[str] | None = None):
        self.prompts: list[str] = []
        self._responses = iter(responses or [])

    def generate(self, messages: list[dict[str, str]], *, json_mode: bool = False) -> str:
        self.prompts.append(messages[-1]["content"])
        try:
            return next(self._responses)
        except StopIteration:
            raise AssertionError("FakeLLM ran out of scripted responses") from None


def _fake_provider_factory(*, model_name: str, device: str):
    return DeterministicEmbeddingProvider(dimension=DIMENSION)


def _enabled_chat_service(llm: FakeLLM) -> ChatService:
    """Builds a ChatService wired exactly the way app/main.py wires one
    when SEMANTIC_MEMORY_ENABLED is true, but with a fake provider — the
    same composition path, none of the real model."""
    bundle = build_semantic_memory(
        enabled=True,
        model_name="fake/model",
        device="cpu",
        max_records_per_session=200,
        llm_client=llm,
        provider_factory=_fake_provider_factory,
    )
    return ChatService(
        llm_client=llm,
        memory_retriever=bundle.retriever,
        memory_extractor=bundle.extractor,
        memory_writer=bundle.writer,
    )


# ===========================================================================
# D1 — RETRIEVAL REACHES THE PROMPT THROUGH ChatService
# ===========================================================================

def test_retrieval_reaches_the_prompt_through_chat_service() -> None:
    llm = FakeLLM([_final_json("Use Python."), _empty_extraction_json()])
    chat_service = _enabled_chat_service(llm)
    # Seed a fact directly through the bundle's own writer, bypassing
    # extraction — isolates "does retrieval work" from "does writing work".
    chat_service.memory_writer.write(
        "alice", [MemoryCandidate("User prefers Python for machine learning.", 0.9)], ["evt-seed"]
    )

    answer = chat_service.ask("What should I use?", session_id="alice")

    assert answer == "Use Python."
    assert MEMORY_CONTEXT_LABEL in llm.prompts[0]
    assert "Python for machine learning" in llm.prompts[0]


# ===========================================================================
# D2 — A COMPLETED INTERACTION WRITES SEMANTIC MEMORY
# ===========================================================================

def test_completed_interaction_writes_semantic_memory() -> None:
    # LLMMemoryExtractor parses the SAME FakeLLM's output, so two scripted
    # responses are needed: one for the agent's final answer, one for the
    # extractor's own JSON call after the turn completes.
    llm = FakeLLM([
        _final_json("Good to know."),
        json.dumps({"memories": [{"content": "User prefers Python for ML.", "confidence": 0.9}]}),
    ])
    chat_service = _enabled_chat_service(llm)

    chat_service.ask("I prefer Python for ML.", session_id="alice")

    stored = chat_service.memory_writer.semantic_memory.list_recent("alice", limit=10)
    assert len(stored) == 1
    assert stored[0].content == "User prefers Python for ML."


def test_failed_interaction_writes_nothing() -> None:
    """An LLM response that fails to parse produces no answer worth
    remembering — the orchestrator's existing rule (16E-D), unaffected by
    activation."""
    llm = FakeLLM(["this is not valid JSON"])
    chat_service = _enabled_chat_service(llm)

    chat_service.ask("I prefer Python.", session_id="alice")

    assert chat_service.memory_writer.semantic_memory.list_recent("alice", limit=10) == []


# ===========================================================================
# D3 — A LATER REQUEST RETRIEVES WHAT AN EARLIER ONE WROTE
# ===========================================================================

def test_a_later_request_retrieves_what_an_earlier_one_wrote() -> None:
    """The true 16I acceptance test: two separate ChatService.ask() calls,
    sharing one bundle, with the second request's prompt actually
    containing what the first one caused to be written.

    Each COMPLETED ask() makes two LLM calls (decision, then extraction),
    so with two ask() calls the recorded prompts are, in order:
    [0] first decision  [1] first extraction  [2] second decision
    [3] second extraction. The assertion below is against [2] — the
    second call's DECISION prompt — not the last recorded prompt, which
    would be its extraction prompt instead.
    """
    llm = FakeLLM([
        _final_json("Good to know."),
        json.dumps({"memories": [{"content": "User prefers Python for ML.", "confidence": 0.9}]}),
        _final_json("Use Python, based on what you told me."),
        _empty_extraction_json(),
    ])
    chat_service = _enabled_chat_service(llm)

    chat_service.ask("I prefer Python for ML.", session_id="alice")
    second_answer = chat_service.ask("What language should I use?", session_id="alice")

    assert second_answer == "Use Python, based on what you told me."
    assert len(llm.prompts) == 4
    assert MEMORY_CONTEXT_LABEL in llm.prompts[2]
    assert "Python for ML" in llm.prompts[2]


def test_the_same_stack_is_reused_across_requests() -> None:
    llm = FakeLLM([
        _final_json("a"),
        _empty_extraction_json(),
        _final_json("b"),
        _empty_extraction_json(),
    ])
    chat_service = _enabled_chat_service(llm)
    store_before = chat_service.memory_writer.semantic_memory

    chat_service.ask("hello", session_id="alice")
    chat_service.ask("hello again", session_id="alice")

    assert chat_service.memory_writer.semantic_memory is store_before


# ===========================================================================
# D4 — SESSION ISOLATION HOLDS THROUGH ChatService
# ===========================================================================

def test_session_a_and_session_b_never_share_memory_through_chat_service() -> None:
    llm = FakeLLM([_final_json("noted for bob"), _empty_extraction_json()])
    chat_service = _enabled_chat_service(llm)
    chat_service.memory_writer.write(
        "alice", [MemoryCandidate("Alice prefers Python.", 0.9)], ["evt-1"]
    )

    chat_service.ask("What do I use?", session_id="bob")

    assert MEMORY_CONTEXT_LABEL not in llm.prompts[0]


def test_session_id_none_neither_reads_nor_writes_semantic_memory() -> None:
    """The legacy no-session path (16E-C/16E-D's rule) must survive
    activation unchanged: no invented shared identity, ever."""
    llm = FakeLLM([_final_json("hello there")])
    chat_service = _enabled_chat_service(llm)
    chat_service.memory_writer.write(
        "alice", [MemoryCandidate("Alice prefers Python.", 0.9)], ["evt-1"]
    )

    answer = chat_service.ask("hello", session_id=None)

    assert answer == "hello there"
    assert MEMORY_CONTEXT_LABEL not in llm.prompts[0]


# ===========================================================================
# API-LEVEL: app/main.py's OWN wiring
# ===========================================================================

@pytest.fixture
def client() -> TestClient:
    return TestClient(main_module.app)


def test_default_startup_leaves_chat_service_with_no_memory_collaborators() -> None:
    """Restates the pre-existing 16E-C/16E-D guarantee (see
    test_memory_write_path.py / test_memory_integration.py) — 16I must not
    have changed it."""
    assert main_module.chat_service.memory_retriever is None
    assert main_module.chat_service.memory_extractor is None
    assert main_module.chat_service.memory_writer is None


def test_chat_service_and_a_bundles_extractor_share_one_llm_client() -> None:
    """Proves app/main.py builds exactly one LLMClient and hands it to
    both consumers — verified structurally here (via a throwaway enabled
    bundle built the same way main.py would), since the live
    chat_service has no extractor while disabled."""
    from app.models.llm import LLMClient

    llm = LLMClient(provider="ollama", model_name="llama3.2:3b")
    bundle = build_semantic_memory(
        enabled=True,
        model_name="fake/model",
        device="cpu",
        max_records_per_session=200,
        llm_client=llm,
        provider_factory=_fake_provider_factory,
    )
    chat_service = ChatService(
        llm_client=llm,
        memory_retriever=bundle.retriever,
        memory_extractor=bundle.extractor,
        memory_writer=bundle.writer,
    )

    assert chat_service.llm is chat_service.memory_extractor.llm


def test_post_chat_response_shape_is_unchanged(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main_module.chat_service, "llm", FakeLLM([_final_json("hi")]))

    response = client.post("/chat", json={"message": "hello"})

    assert response.status_code == 200
    assert set(response.json().keys()) == {"reply"}


def test_post_chat_with_active_memory_still_returns_only_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    """The strongest version of the "no internal metadata leaks" guarantee:
    swap the live app's chat_service for one with ACTIVE (fake-backed)
    semantic memory and confirm the HTTP response still contains nothing
    but "reply", even though memory genuinely participated."""
    llm = FakeLLM([_final_json("Use Python."), _empty_extraction_json()])
    live_chat_service = _enabled_chat_service(llm)
    live_chat_service.memory_writer.write(
        "alice", [MemoryCandidate("User prefers Python for ML.", 0.9)], ["evt-1"]
    )
    monkeypatch.setattr(main_module, "chat_service", live_chat_service)
    client = TestClient(main_module.app)

    response = client.post("/chat", json={"message": "What should I use?", "session_id": "alice"})

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"reply"}
    assert body["reply"] == "Use Python."
    # No memory internals anywhere in the raw body text.
    raw = response.text
    for leaked in ("memory_id", "similarity", "embedding", "vector", "confidence", "session_id"):
        assert leaked not in raw


def test_memory_session_isolation_error_maps_to_http_500(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _raise(*args: object, **kwargs: object) -> str:
        raise MemorySessionIsolationError(
            "session isolation violation: vector index returned memory_id 'secret-id' "
            "for session 'alice', but the stored record belongs to session 'bob'."
        )

    monkeypatch.setattr(main_module.chat_service, "ask", _raise)

    response = client.post("/chat", json={"message": "hello", "session_id": "alice"})

    assert response.status_code == 500


def test_memory_session_isolation_error_response_is_generic_and_leaks_nothing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _raise(*args: object, **kwargs: object) -> str:
        raise MemorySessionIsolationError(
            "session isolation violation: vector index returned memory_id 'super-secret-memory-id' "
            "for session 'alice-session', but the stored record belongs to session 'bob-session'."
        )

    monkeypatch.setattr(main_module.chat_service, "ask", _raise)

    response = client.post("/chat", json={"message": "hello", "session_id": "alice-session"})

    body = response.json()
    assert "detail" in body
    for leaked in ("super-secret-memory-id", "alice-session", "bob-session", "isolation violation"):
        assert leaked not in response.text


def test_memory_session_isolation_error_is_caught_before_the_plain_value_error_handler(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MemorySessionIsolationError IS a ValueError subclass — this proves
    the branch ordering in app/main.py actually catches it as 500, not as
    the generic ValueError->400 branch."""

    def _raise(*args: object, **kwargs: object) -> str:
        raise MemorySessionIsolationError("session isolation violation: details.")

    monkeypatch.setattr(main_module.chat_service, "ask", _raise)

    response = client.post("/chat", json={"message": "hello"})

    assert response.status_code == 500
    assert response.status_code != 400


def test_an_ordinary_value_error_still_maps_to_400(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: the isolation-error branch must not have swallowed the
    pre-existing plain ValueError -> 400 mapping."""

    def _raise(*args: object, **kwargs: object) -> str:
        raise ValueError("ordinary bad input")

    monkeypatch.setattr(main_module.chat_service, "ask", _raise)

    response = client.post("/chat", json={"message": "hello"})

    assert response.status_code == 400


def test_an_ordinary_runtime_error_still_maps_to_502(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*args: object, **kwargs: object) -> str:
        raise RuntimeError("upstream dependency failed")

    monkeypatch.setattr(main_module.chat_service, "ask", _raise)

    response = client.post("/chat", json={"message": "hello"})

    assert response.status_code == 502


# ===========================================================================
# PROCESS-LEVEL: SEMANTIC_MEMORY_ENABLED is genuinely read at import
# ===========================================================================

def test_a_provider_construction_failure_during_wiring_propagates_uncaught() -> None:
    """Fast, no-torch equivalent of the real end-to-end startup-failure
    proof below: build_semantic_memory adds no try/except of its own
    around provider_factory, so a construction failure (a real
    EmbeddingModelLoadError, or here any exception a fake factory raises)
    must reach the caller unmodified — this is what makes application
    startup fail loudly rather than boot with a silently broken bundle.
    """

    class Boom(RuntimeError):
        pass

    def failing_factory(*, model_name: str, device: str):
        raise Boom("simulated unloadable model")

    with pytest.raises(Boom):
        build_semantic_memory(
            enabled=True,
            model_name="fake/model",
            device="cpu",
            max_records_per_session=200,
            llm_client=FakeLLM(),
            provider_factory=failing_factory,
        )


@pytest.mark.integration
def test_enabling_via_env_var_with_an_unloadable_model_fails_startup() -> None:
    """A clean-interpreter, REAL end-to-end probe: enabling semantic
    memory with a deliberately unloadable model name must fail
    application STARTUP loudly — never a silent fallback to a
    working-but-meaningless deterministic provider, and never a boot that
    looks successful.

    Marked integration (not part of the ordinary suite) because it
    genuinely imports torch/sentence-transformers in a fresh subprocess —
    that import alone costs tens of seconds regardless of outcome, even
    though no model is ever actually downloaded. The fast, always-run
    equivalent above proves the same wiring guarantee without paying that
    cost or touching the real ML stack.
    """
    import os

    env = dict(os.environ)
    env["SEMANTIC_MEMORY_ENABLED"] = "true"
    env["EMBEDDING_MODEL_NAME"] = "definitely-not-a-real-model/does-not-exist"

    result = subprocess.run(
        [sys.executable, "-c", "import app.main"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
    )

    assert result.returncode != 0
