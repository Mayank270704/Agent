"""Step 16I: AgentOrchestrator's degradation policy for semantic memory
failures.

Covers the two narrow `except EmbeddingError` catches added to
`_retrieve_memory_context_or_none` (read) and `_maybe_write_semantic_memory`
(write). Every failure mode is exercised with a FAKE retriever/writer that
raises the EXACT exception type under test — never a real embedding
provider, never a real model, fully offline.

The two negative tests in this file (`bare RuntimeError` and `bare
ValueError` propagate) are the most important ones: they prove the catch
is narrow by TYPE, not merely by which exception happens to be raised
first, so `MemorySessionIsolationError` (a `ValueError` subclass) can
never be caught by an overly broad handler.
"""
from __future__ import annotations

import json
import logging

import pytest

from app.agent.embeddings import DeterministicEmbeddingProvider
from app.agent.episodic_memory import InMemoryEpisodicMemory
from app.agent.local_embeddings import EmbeddingComputeError, EmbeddingModelLoadError
from app.agent.memory_context import build_memory_context
from app.agent.memory_extraction import MemoryCandidate, MemoryExtractionError
from app.agent.memory_retriever import RetrievedMemory
from app.agent.orchestrator import AgentOrchestrator
from app.agent.semantic_memory import (
    InMemorySemanticMemory,
    MemorySessionIsolationError,
    SemanticMemoryRecord,
)
from app.agent.tool_registry import ToolRegistry
from app.agent.vector_index import InMemoryVectorIndex
from app.agent.memory_writer import SemanticMemoryWriter

DIMENSION = 8


def _final_json(answer: str) -> str:
    return json.dumps({"action_type": "final", "final_answer": answer})


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


class FakeExtractor:
    def __init__(self, candidates: list[MemoryCandidate] | None = None):
        self._candidates = candidates or []

    def extract(self, user_message: str, assistant_answer: str) -> list[MemoryCandidate]:
        return list(self._candidates)


class RaisingRetriever:
    """A MemoryRetriever fake that raises whatever exception is configured
    on every `retrieve()` call, rather than actually retrieving anything."""

    def __init__(self, exc: Exception):
        self._exc = exc
        self.calls = 0

    def retrieve(self, session_id: str, query: str, top_k: int = 5):
        self.calls += 1
        raise self._exc


class RaisingWriter:
    """A MemoryWriter fake that raises whatever exception is configured on
    every `write()` call."""

    def __init__(self, exc: Exception):
        self._exc = exc
        self.calls = 0

    def write(self, session_id, candidates, source_event_ids):
        self.calls += 1
        raise self._exc


def _orchestrator(llm, *, retriever=None, extractor=None, writer=None, session_id="alice"):
    return AgentOrchestrator(
        llm_client=llm,
        tool_registry=ToolRegistry(),
        episodic_memory=InMemoryEpisodicMemory(),
        memory_retriever=retriever,
        memory_extractor=extractor,
        memory_writer=writer,
        session_id=session_id,
    )


def _record(memory_id: str = "m1", session_id: str = "alice") -> SemanticMemoryRecord:
    from datetime import datetime, timezone

    return SemanticMemoryRecord(
        memory_id=memory_id,
        session_id=session_id,
        content="a fact",
        created_at=datetime.now(timezone.utc),
        source_event_ids=("evt-1",),
    )


# ===========================================================================
# E1 — READ PATH: EmbeddingError degrades to "no memory," answer preserved
# ===========================================================================

def test_embedding_model_load_error_during_retrieval_degrades_to_no_memory() -> None:
    retriever = RaisingRetriever(EmbeddingModelLoadError("model unavailable"))
    orchestrator = _orchestrator(FakeLLM([_final_json("Hello.")]), retriever=retriever)

    result = orchestrator.process("hi")

    assert result.answer == "Hello."
    assert retriever.calls == 1


def test_embedding_compute_error_during_retrieval_degrades_to_no_memory() -> None:
    retriever = RaisingRetriever(EmbeddingComputeError("transient encode failure"))
    orchestrator = _orchestrator(FakeLLM([_final_json("Hello.")]), retriever=retriever)

    result = orchestrator.process("hi")

    assert result.answer == "Hello."


def test_retrieval_degradation_is_logged_at_warning_with_type_only(caplog: pytest.LogCaptureFixture) -> None:
    retriever = RaisingRetriever(EmbeddingComputeError("do not log this text"))
    orchestrator = _orchestrator(FakeLLM([_final_json("Hello.")]), retriever=retriever)

    with caplog.at_level(logging.WARNING, logger="app.agent.orchestrator"):
        orchestrator.process("hi")

    assert "EmbeddingComputeError" in caplog.text
    assert "do not log this text" not in caplog.text
    assert "hi" not in caplog.text  # the query text itself must not be logged


def test_retrieval_degradation_prompt_is_identical_to_no_retriever_at_all() -> None:
    """Degrading must produce a prompt byte-identical to the "no retriever
    injected" case — the model never sees a partial or broken memory
    section."""
    retriever = RaisingRetriever(EmbeddingModelLoadError("boom"))
    with_failure = _orchestrator(FakeLLM([_final_json("Hello.")]), retriever=retriever)
    without_retriever = _orchestrator(FakeLLM([_final_json("Hello.")]))

    with_failure.process("hi")
    without_retriever.process("hi")

    assert with_failure.decision_maker.llm.prompts[0] == without_retriever.decision_maker.llm.prompts[0]


# ===========================================================================
# E2 — WRITE PATH: EmbeddingError degrades, answer already produced is kept
# ===========================================================================

def test_embedding_model_load_error_during_write_preserves_the_answer() -> None:
    writer = RaisingWriter(EmbeddingModelLoadError("model unavailable"))
    extractor = FakeExtractor([MemoryCandidate("User prefers Python.", 0.9)])
    orchestrator = _orchestrator(FakeLLM([_final_json("Good to know.")]), extractor=extractor, writer=writer)

    result = orchestrator.process("I prefer Python.")

    assert result.answer == "Good to know."
    assert writer.calls == 1


def test_embedding_compute_error_during_write_preserves_the_answer() -> None:
    writer = RaisingWriter(EmbeddingComputeError("transient encode failure"))
    extractor = FakeExtractor([MemoryCandidate("User prefers Python.", 0.9)])
    orchestrator = _orchestrator(FakeLLM([_final_json("Good to know.")]), extractor=extractor, writer=writer)

    result = orchestrator.process("I prefer Python.")

    assert result.answer == "Good to know."


def test_write_degradation_is_logged_at_warning_with_type_and_count_only(
    caplog: pytest.LogCaptureFixture,
) -> None:
    writer = RaisingWriter(EmbeddingComputeError("do not log this"))
    extractor = FakeExtractor(
        [MemoryCandidate("User prefers Python.", 0.9), MemoryCandidate("User prefers ML.", 0.9)]
    )
    orchestrator = _orchestrator(FakeLLM([_final_json("noted")]), extractor=extractor, writer=writer)

    with caplog.at_level(logging.WARNING, logger="app.agent.orchestrator"):
        orchestrator.process("hi")

    assert "EmbeddingComputeError" in caplog.text
    assert "2 candidate" in caplog.text
    assert "do not log this" not in caplog.text
    assert "User prefers Python" not in caplog.text  # candidate content never logged


def test_write_degradation_does_not_retry_or_partially_write() -> None:
    """A single write() call either fully raises or fully returns — this
    orchestrator call site must not attempt any retry."""
    writer = RaisingWriter(EmbeddingModelLoadError("boom"))
    extractor = FakeExtractor([MemoryCandidate("User prefers Python.", 0.9)])
    orchestrator = _orchestrator(FakeLLM([_final_json("noted")]), extractor=extractor, writer=writer)

    orchestrator.process("hi")

    assert writer.calls == 1


# ===========================================================================
# E3 — DIMENSION MISMATCH: a wiring-time failure, never silently absorbed
# ===========================================================================

def test_dimension_mismatch_between_provider_and_index_raises_at_construction() -> None:
    provider = DeterministicEmbeddingProvider(dimension=DIMENSION)
    mismatched_index = InMemoryVectorIndex(dimension=DIMENSION + 1)

    with pytest.raises(ValueError, match="dimension"):
        SemanticMemoryWriter(InMemorySemanticMemory(), provider, mismatched_index)


# ===========================================================================
# E4 — MemorySessionIsolationError NEVER degrades, on read or write
# ===========================================================================

def test_session_isolation_error_during_retrieval_propagates_uncaught() -> None:
    retriever = RaisingRetriever(
        MemorySessionIsolationError("session isolation violation: details.")
    )
    orchestrator = _orchestrator(FakeLLM([_final_json("Hello.")]), retriever=retriever)

    with pytest.raises(MemorySessionIsolationError):
        orchestrator.process("hi")


def test_session_isolation_error_during_write_propagates_uncaught() -> None:
    writer = RaisingWriter(MemorySessionIsolationError("session isolation violation: details."))
    extractor = FakeExtractor([MemoryCandidate("User prefers Python.", 0.9)])
    orchestrator = _orchestrator(FakeLLM([_final_json("noted")]), extractor=extractor, writer=writer)

    with pytest.raises(MemorySessionIsolationError):
        orchestrator.process("hi")


def test_build_memory_context_isolation_error_is_not_an_embedding_error() -> None:
    """MemorySessionIsolationError is a ValueError subclass, never an
    EmbeddingError — the type hierarchies are disjoint by construction, so
    no catch ordering trick is needed to keep them apart."""
    from app.agent.local_embeddings import EmbeddingError

    assert not issubclass(MemorySessionIsolationError, EmbeddingError)

    with pytest.raises(MemorySessionIsolationError):
        build_memory_context(
            "alice", [RetrievedMemory(memory=_record(session_id="bob"), similarity=1.0)]
        )


# ===========================================================================
# E5 — a bare ValueError propagates (never mistaken for EmbeddingError)
# ===========================================================================

def test_a_plain_value_error_during_retrieval_propagates_uncaught() -> None:
    retriever = RaisingRetriever(ValueError("query must be a non-empty string."))
    orchestrator = _orchestrator(FakeLLM([_final_json("Hello.")]), retriever=retriever)

    with pytest.raises(ValueError):
        orchestrator.process("hi")


def test_a_plain_value_error_during_write_propagates_uncaught() -> None:
    writer = RaisingWriter(ValueError("candidates must be a list or tuple of MemoryCandidate."))
    extractor = FakeExtractor([MemoryCandidate("User prefers Python.", 0.9)])
    orchestrator = _orchestrator(FakeLLM([_final_json("noted")]), extractor=extractor, writer=writer)

    with pytest.raises(ValueError):
        orchestrator.process("hi")


# ===========================================================================
# E6 — a bare RuntimeError is NEVER caught by the EmbeddingError handler
# ===========================================================================

def test_a_bare_runtime_error_during_retrieval_is_not_swallowed() -> None:
    """The single most important negative test in this file: proves the
    catch is `except EmbeddingError`, never `except RuntimeError` or
    `except Exception` — a plain RuntimeError from any other cause (e.g. a
    genuinely broken retriever, or a future infrastructure failure that is
    NOT embedding-related) must still surface as a real failure."""
    retriever = RaisingRetriever(RuntimeError("unrelated infrastructure failure"))
    orchestrator = _orchestrator(FakeLLM([_final_json("Hello.")]), retriever=retriever)

    with pytest.raises(RuntimeError) as exc_info:
        orchestrator.process("hi")

    assert not isinstance(exc_info.value, (EmbeddingModelLoadError, EmbeddingComputeError))


def test_a_bare_runtime_error_during_write_is_not_swallowed() -> None:
    writer = RaisingWriter(RuntimeError("unrelated infrastructure failure"))
    extractor = FakeExtractor([MemoryCandidate("User prefers Python.", 0.9)])
    orchestrator = _orchestrator(FakeLLM([_final_json("noted")]), extractor=extractor, writer=writer)

    with pytest.raises(RuntimeError) as exc_info:
        orchestrator.process("hi")

    assert not isinstance(exc_info.value, (EmbeddingModelLoadError, EmbeddingComputeError))


# ===========================================================================
# E7 — MemoryExtractionError behavior remains unchanged (16E-D regression)
# ===========================================================================

class RaisingExtractor:
    def __init__(self, exc: Exception):
        self._exc = exc

    def extract(self, user_message: str, assistant_answer: str) -> list[MemoryCandidate]:
        raise self._exc


def test_memory_extraction_error_still_degrades_to_nothing_extracted() -> None:
    """Regression: 16I's new EmbeddingError catches must not have changed
    the pre-existing MemoryExtractionError handling in any way — different
    exception type, different call site (extract(), not write())."""
    extractor = RaisingExtractor(MemoryExtractionError("could not parse extractor output"))
    store = InMemorySemanticMemory()
    provider = DeterministicEmbeddingProvider(dimension=DIMENSION)
    index = InMemoryVectorIndex(dimension=DIMENSION)
    writer = SemanticMemoryWriter(store, provider, index)
    orchestrator = _orchestrator(FakeLLM([_final_json("Hello.")]), extractor=extractor, writer=writer)

    result = orchestrator.process("hi")

    assert result.answer == "Hello."
    assert store.list_recent("alice", limit=10) == []


def test_memory_extraction_error_does_not_call_the_writer() -> None:
    extractor = RaisingExtractor(MemoryExtractionError("bad output"))
    writer = RaisingWriter(RuntimeError("must never be called"))
    orchestrator = _orchestrator(FakeLLM([_final_json("Hello.")]), extractor=extractor, writer=writer)

    orchestrator.process("hi")

    assert writer.calls == 0
