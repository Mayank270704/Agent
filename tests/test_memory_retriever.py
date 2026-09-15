"""Tests for SemanticMemoryRetriever (Step 16D).

Everything here is deterministic and fully offline: no Ollama, no Tavily,
no model download, no network. The DeterministicEmbeddingProvider from
Step 16B is used where a real provider's SHAPE is needed, and hand-built
vectors are used where an OBVIOUS mathematical result is needed.

Important: the deterministic provider is not semantically intelligent, so
no test here asserts that "I prefer Python" retrieves for a query like
"what language do I prefer?". These tests verify the retrieval PLUMBING —
that the right collaborator is called, that IDs resolve to records, that
sessions stay isolated, that ordering and top-K are preserved.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.agent.embeddings import DeterministicEmbeddingProvider
from app.agent.memory_retriever import (
    MemoryRetriever,
    RetrievedMemory,
    SemanticMemoryRetriever,
)
from app.agent.semantic_memory import InMemorySemanticMemory, SemanticMemoryRecord
from app.agent.vector_index import InMemoryVectorIndex

DIMENSION = 2


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _record(memory_id: str, session_id: str = "A", **overrides: object) -> SemanticMemoryRecord:
    fields: dict[str, object] = {
        "memory_id": memory_id,
        "session_id": session_id,
        "content": f"fact {memory_id}",
        "created_at": _now(),
        "source_event_ids": ("evt-1",),
    }
    fields.update(overrides)
    return SemanticMemoryRecord(**fields)  # type: ignore[arg-type]


def _build(dimension: int = DIMENSION):
    """Returns (retriever, semantic_memory, vector_index)."""
    semantic_memory = InMemorySemanticMemory()
    vector_index = InMemoryVectorIndex(dimension=dimension)
    provider = DeterministicEmbeddingProvider(dimension=dimension)
    retriever = SemanticMemoryRetriever(
        semantic_memory=semantic_memory,
        embedding_provider=provider,
        vector_index=vector_index,
    )
    return retriever, semantic_memory, vector_index


class FixedVectorProvider:
    """An EmbeddingProvider that returns a caller-chosen vector, so tests
    can control the query's position in vector space exactly and reason
    about similarity by hand. Also records every call, which is how the
    "retriever does not implement embedding itself" tests verify
    delegation."""

    def __init__(self, vector: tuple[float, ...]):
        self._vector = vector
        self.calls: list[str] = []

    @property
    def dimension(self) -> int:
        return len(self._vector)

    def embed(self, text: str) -> tuple[float, ...]:
        self.calls.append(text)
        return self._vector

    def embed_many(self, texts):  # pragma: no cover - retrieval never batches
        return [self.embed(text) for text in texts]


def _build_with_query_vector(query_vector: tuple[float, ...]):
    """Returns (retriever, semantic_memory, vector_index, provider)."""
    semantic_memory = InMemorySemanticMemory()
    vector_index = InMemoryVectorIndex(dimension=len(query_vector))
    provider = FixedVectorProvider(query_vector)
    retriever = SemanticMemoryRetriever(
        semantic_memory=semantic_memory,
        embedding_provider=provider,
        vector_index=vector_index,
    )
    return retriever, semantic_memory, vector_index, provider


# ---------------------------------------------------------------------------
# 1: basic retrieval -- memory content and similarity both come back.
# ---------------------------------------------------------------------------

def test_basic_retrieval_returns_record_and_similarity() -> None:
    retriever, semantic_memory, vector_index, _ = _build_with_query_vector((1.0, 0.0))
    semantic_memory.add(_record("m1", content="User prefers Python for ML."))
    vector_index.add("m1", "A", (1.0, 0.0))  # identical to the query vector

    results = retriever.retrieve("A", "any query text", top_k=5)

    assert len(results) == 1
    assert isinstance(results[0], RetrievedMemory)
    assert results[0].memory.memory_id == "m1"
    assert results[0].memory.content == "User prefers Python for ML."
    assert results[0].similarity == pytest.approx(1.0)


def test_retrieved_memory_carries_the_full_record_not_just_the_id() -> None:
    retriever, semantic_memory, vector_index, _ = _build_with_query_vector((1.0, 0.0))
    semantic_memory.add(_record("m1", confidence=0.8, source_event_ids=("evt-7", "evt-9")))
    vector_index.add("m1", "A", (1.0, 0.0))

    record = retriever.retrieve("A", "q", top_k=1)[0].memory

    assert record.confidence == 0.8
    assert record.source_event_ids == ("evt-7", "evt-9")
    assert record.session_id == "A"


# ---------------------------------------------------------------------------
# 2/3: top-K semantics.
# ---------------------------------------------------------------------------

def test_top_k_limits_the_number_of_results() -> None:
    retriever, semantic_memory, vector_index, _ = _build_with_query_vector((1.0, 0.0))
    for memory_id, vector in [("m1", (1.0, 0.0)), ("m2", (0.9, 0.1)), ("m3", (0.0, 1.0))]:
        semantic_memory.add(_record(memory_id))
        vector_index.add(memory_id, "A", vector)

    results = retriever.retrieve("A", "q", top_k=2)

    assert len(results) == 2


def test_fewer_memories_than_top_k_returns_what_exists_without_padding() -> None:
    retriever, semantic_memory, vector_index, _ = _build_with_query_vector((1.0, 0.0))
    for memory_id in ["m1", "m2"]:
        semantic_memory.add(_record(memory_id))
        vector_index.add(memory_id, "A", (1.0, 0.0))

    results = retriever.retrieve("A", "q", top_k=5)

    assert len(results) == 2


def test_default_top_k_is_five() -> None:
    retriever, semantic_memory, vector_index, _ = _build_with_query_vector((1.0, 0.0))
    for i in range(8):
        semantic_memory.add(_record(f"m{i}"))
        vector_index.add(f"m{i}", "A", (1.0, 0.0))

    assert len(retriever.retrieve("A", "q")) == 5


# ---------------------------------------------------------------------------
# 4: empty session.
# ---------------------------------------------------------------------------

def test_retrieval_from_an_empty_session_returns_empty_list() -> None:
    retriever, _, _ = _build()

    assert retriever.retrieve("A", "any query", top_k=5) == []


def test_retrieval_from_a_session_with_no_vectors_returns_empty_list() -> None:
    """Records exist in the semantic store, but nothing was indexed --
    retrieval is driven by the index, so there is nothing to find."""
    retriever, semantic_memory, _, _ = _build_with_query_vector((1.0, 0.0))
    semantic_memory.add(_record("m1"))

    assert retriever.retrieve("A", "q", top_k=5) == []


# ---------------------------------------------------------------------------
# 5: session isolation -- NON-NEGOTIABLE.
# ---------------------------------------------------------------------------

def test_session_a_retrieval_never_returns_session_b_memories() -> None:
    retriever, semantic_memory, vector_index, _ = _build_with_query_vector((1.0, 0.0))
    semantic_memory.add(_record("m-a", session_id="A", content="Alice's fact"))
    semantic_memory.add(_record("m-b", session_id="B", content="Bob's fact"))
    # Identical vectors in both sessions: if isolation were filter-based
    # and buggy, B would be just as "similar" as A and would surface.
    vector_index.add("m-a", "A", (1.0, 0.0))
    vector_index.add("m-b", "B", (1.0, 0.0))

    results_a = retriever.retrieve("A", "q", top_k=10)

    assert [r.memory.memory_id for r in results_a] == ["m-a"]
    assert all("Bob" not in r.memory.content for r in results_a)


def test_session_b_retrieval_never_returns_session_a_memories() -> None:
    retriever, semantic_memory, vector_index, _ = _build_with_query_vector((1.0, 0.0))
    semantic_memory.add(_record("m-a", session_id="A", content="Alice's fact"))
    semantic_memory.add(_record("m-b", session_id="B", content="Bob's fact"))
    vector_index.add("m-a", "A", (1.0, 0.0))
    vector_index.add("m-b", "B", (1.0, 0.0))

    results_b = retriever.retrieve("B", "q", top_k=10)

    assert [r.memory.memory_id for r in results_b] == ["m-b"]


def test_many_sessions_remain_mutually_isolated() -> None:
    retriever, semantic_memory, vector_index, _ = _build_with_query_vector((1.0, 0.0))
    session_ids = [f"s{i}" for i in range(10)]
    for session_id in session_ids:
        semantic_memory.add(_record(f"m-{session_id}", session_id=session_id))
        vector_index.add(f"m-{session_id}", session_id, (1.0, 0.0))

    for session_id in session_ids:
        results = retriever.retrieve(session_id, "q", top_k=10)
        assert [r.memory.memory_id for r in results] == [f"m-{session_id}"]


def test_session_id_whitespace_is_normalized() -> None:
    retriever, semantic_memory, vector_index, _ = _build_with_query_vector((1.0, 0.0))
    semantic_memory.add(_record("m1", session_id="A"))
    vector_index.add("m1", "A", (1.0, 0.0))

    assert retriever.retrieve("  A  ", "q", top_k=5) == retriever.retrieve("A", "q", top_k=5)


def test_cross_session_record_mismatch_raises_rather_than_leaking() -> None:
    """Defense in depth: if the index and the store ever disagree about
    which session owns a memory, that is a data-integrity bug with a
    leak implication -- it must fail loudly, not be silently filtered."""
    retriever, semantic_memory, vector_index, _ = _build_with_query_vector((1.0, 0.0))
    # The record says it belongs to B, but its vector was indexed under A.
    semantic_memory.add(_record("m-rogue", session_id="B"))
    vector_index.add("m-rogue", "A", (1.0, 0.0))

    with pytest.raises(ValueError, match="session isolation violation"):
        retriever.retrieve("A", "q", top_k=5)


# ---------------------------------------------------------------------------
# 6: stale index entry -- skipped, not fatal, not invented.
# ---------------------------------------------------------------------------

def test_stale_index_entry_is_skipped_not_fabricated() -> None:
    retriever, semantic_memory, vector_index, _ = _build_with_query_vector((1.0, 0.0))
    vector_index.add("ghost", "A", (1.0, 0.0))  # indexed, but never stored

    results = retriever.retrieve("A", "q", top_k=5)

    assert results == []


def test_stale_entry_does_not_prevent_other_results_from_returning() -> None:
    retriever, semantic_memory, vector_index, _ = _build_with_query_vector((1.0, 0.0))
    semantic_memory.add(_record("m1"))
    vector_index.add("m1", "A", (1.0, 0.0))
    vector_index.add("ghost", "A", (1.0, 0.0))  # higher/equal similarity, but stale

    results = retriever.retrieve("A", "q", top_k=5)

    assert [r.memory.memory_id for r in results] == ["m1"]


def test_stale_entry_is_logged_as_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    retriever, _, vector_index, _ = _build_with_query_vector((1.0, 0.0))
    vector_index.add("ghost", "A", (1.0, 0.0))

    with caplog.at_level("WARNING"):
        retriever.retrieve("A", "q", top_k=5)

    assert "ghost" in caplog.text


def test_retrieval_does_not_repair_the_vector_index() -> None:
    """retrieve() is a pure read: a stale entry is skipped but NOT removed
    from the index -- reconciliation belongs to the write path."""
    retriever, _, vector_index, _ = _build_with_query_vector((1.0, 0.0))
    vector_index.add("ghost", "A", (1.0, 0.0))

    retriever.retrieve("A", "q", top_k=5)

    # Still present in the index; retrieval had no side effect on it.
    assert [hit.memory_id for hit in vector_index.search("A", (1.0, 0.0), top_k=5)] == ["ghost"]


# ---------------------------------------------------------------------------
# Inactive records are excluded (documented Step 16D decision).
# ---------------------------------------------------------------------------

def test_inactive_memories_are_excluded_from_retrieval() -> None:
    retriever, semantic_memory, vector_index, _ = _build_with_query_vector((1.0, 0.0))
    semantic_memory.add(_record("m-inactive", active=False))
    vector_index.add("m-inactive", "A", (1.0, 0.0))

    assert retriever.retrieve("A", "q", top_k=5) == []


def test_inactive_memories_are_excluded_but_active_ones_still_return() -> None:
    retriever, semantic_memory, vector_index, _ = _build_with_query_vector((1.0, 0.0))
    semantic_memory.add(_record("m-active", active=True))
    semantic_memory.add(_record("m-inactive", active=False))
    vector_index.add("m-active", "A", (1.0, 0.0))
    vector_index.add("m-inactive", "A", (1.0, 0.0))

    results = retriever.retrieve("A", "q", top_k=5)

    assert [r.memory.memory_id for r in results] == ["m-active"]


def test_excluded_entries_mean_fewer_than_top_k_without_padding() -> None:
    retriever, semantic_memory, vector_index, _ = _build_with_query_vector((1.0, 0.0))
    semantic_memory.add(_record("m1"))
    semantic_memory.add(_record("m2", active=False))
    vector_index.add("m1", "A", (1.0, 0.0))
    vector_index.add("m2", "A", (1.0, 0.0))
    vector_index.add("ghost", "A", (1.0, 0.0))

    results = retriever.retrieve("A", "q", top_k=3)

    assert len(results) == 1  # 3 hits, 2 dropped, nothing padded


# ---------------------------------------------------------------------------
# 7/8/9: input validation.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_session_id", ["", "   ", None, 123])
def test_invalid_session_id_is_rejected(bad_session_id: object) -> None:
    retriever, _, _ = _build()

    with pytest.raises(ValueError):
        retriever.retrieve(bad_session_id, "query", top_k=5)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_query", ["", "   ", "\t\n", None, 123])
def test_invalid_query_is_rejected(bad_query: object) -> None:
    retriever, _, _ = _build()

    with pytest.raises(ValueError):
        retriever.retrieve("A", bad_query, top_k=5)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_top_k", [0, -1, -100, True, 1.5, "5", None])
def test_invalid_top_k_is_rejected_not_silently_defaulted(bad_top_k: object) -> None:
    retriever, _, _ = _build()

    with pytest.raises(ValueError):
        retriever.retrieve("A", "query", top_k=bad_top_k)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 10: dependency validation.
# ---------------------------------------------------------------------------

def test_none_dependencies_are_rejected() -> None:
    semantic_memory = InMemorySemanticMemory()
    provider = DeterministicEmbeddingProvider(dimension=DIMENSION)
    index = InMemoryVectorIndex(dimension=DIMENSION)

    with pytest.raises(ValueError):
        SemanticMemoryRetriever(None, provider, index)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        SemanticMemoryRetriever(semantic_memory, None, index)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        SemanticMemoryRetriever(semantic_memory, provider, None)  # type: ignore[arg-type]


def test_dependencies_not_satisfying_their_protocols_are_rejected() -> None:
    semantic_memory = InMemorySemanticMemory()
    provider = DeterministicEmbeddingProvider(dimension=DIMENSION)
    index = InMemoryVectorIndex(dimension=DIMENSION)

    with pytest.raises(ValueError):
        SemanticMemoryRetriever("not a store", provider, index)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        SemanticMemoryRetriever(semantic_memory, object(), index)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        SemanticMemoryRetriever(semantic_memory, provider, object())  # type: ignore[arg-type]


def test_retriever_conforms_to_the_memory_retriever_protocol() -> None:
    retriever, _, _ = _build()

    assert isinstance(retriever, MemoryRetriever)


# ---------------------------------------------------------------------------
# 11: dimension mismatch, detected eagerly at construction.
# ---------------------------------------------------------------------------

def test_provider_index_dimension_mismatch_is_rejected_at_construction() -> None:
    semantic_memory = InMemorySemanticMemory()
    provider = DeterministicEmbeddingProvider(dimension=8)
    index = InMemoryVectorIndex(dimension=16)

    with pytest.raises(ValueError, match="dimension"):
        SemanticMemoryRetriever(semantic_memory, provider, index)


def test_matching_dimensions_construct_successfully() -> None:
    semantic_memory = InMemorySemanticMemory()
    provider = DeterministicEmbeddingProvider(dimension=16)
    index = InMemoryVectorIndex(dimension=16)

    retriever = SemanticMemoryRetriever(semantic_memory, provider, index)

    assert retriever.embedding_provider.dimension == retriever.vector_index.dimension


# ---------------------------------------------------------------------------
# 12: ordering from the VectorIndex is preserved exactly.
# ---------------------------------------------------------------------------

def test_similarity_ordering_from_the_vector_index_is_preserved() -> None:
    retriever, semantic_memory, vector_index, _ = _build_with_query_vector((1.0, 0.0))
    # Deliberately added in an order that does NOT match expected output.
    for memory_id, vector in [
        ("opposite", (-1.0, 0.0)),   # similarity -1.0
        ("exact", (1.0, 0.0)),       # similarity  1.0
        ("orthogonal", (0.0, 1.0)),  # similarity  0.0
    ]:
        semantic_memory.add(_record(memory_id))
        vector_index.add(memory_id, "A", vector)

    results = retriever.retrieve("A", "q", top_k=3)

    assert [r.memory.memory_id for r in results] == ["exact", "orthogonal", "opposite"]
    assert [r.similarity for r in results] == pytest.approx([1.0, 0.0, -1.0])


def test_ordering_is_preserved_when_entries_are_dropped() -> None:
    """Dropping a stale/inactive hit must not reshuffle the survivors."""
    retriever, semantic_memory, vector_index, _ = _build_with_query_vector((1.0, 0.0))
    semantic_memory.add(_record("best"))
    semantic_memory.add(_record("worst"))
    vector_index.add("best", "A", (1.0, 0.0))
    vector_index.add("middle-ghost", "A", (0.0, 1.0))  # stale, would rank 2nd
    vector_index.add("worst", "A", (-1.0, 0.0))

    results = retriever.retrieve("A", "q", top_k=5)

    assert [r.memory.memory_id for r in results] == ["best", "worst"]


def test_similarity_values_are_passed_through_unmodified() -> None:
    retriever, semantic_memory, vector_index, _ = _build_with_query_vector((1.0, 0.0))
    semantic_memory.add(_record("m1"))
    vector_index.add("m1", "A", (0.0, 1.0))

    retrieved = retriever.retrieve("A", "q", top_k=1)[0]
    index_hit = vector_index.search("A", (1.0, 0.0), top_k=1)[0]

    assert retrieved.similarity == index_hit.similarity


# ---------------------------------------------------------------------------
# 13/14: no LLM, and embedding is delegated to the provider.
# ---------------------------------------------------------------------------

def test_query_is_embedded_through_the_embedding_provider() -> None:
    retriever, semantic_memory, vector_index, provider = _build_with_query_vector((1.0, 0.0))
    semantic_memory.add(_record("m1"))
    vector_index.add("m1", "A", (1.0, 0.0))

    retriever.retrieve("A", "what do I prefer?", top_k=5)

    assert provider.calls == ["what do I prefer?"]


def test_only_the_query_is_embedded_never_the_stored_memories() -> None:
    """Read path embeds exactly one thing: the query. Stored memory
    vectors must already be in the index."""
    retriever, semantic_memory, vector_index, provider = _build_with_query_vector((1.0, 0.0))
    for memory_id in ["m1", "m2", "m3"]:
        semantic_memory.add(_record(memory_id))
        vector_index.add(memory_id, "A", (1.0, 0.0))

    retriever.retrieve("A", "single query", top_k=5)

    assert len(provider.calls) == 1


def test_retrieval_is_deterministic_across_repeated_calls() -> None:
    retriever, semantic_memory, vector_index, _ = _build_with_query_vector((1.0, 0.0))
    for memory_id, vector in [("m1", (1.0, 0.0)), ("m2", (0.0, 1.0))]:
        semantic_memory.add(_record(memory_id))
        vector_index.add(memory_id, "A", vector)

    first = retriever.retrieve("A", "q", top_k=5)
    second = retriever.retrieve("A", "q", top_k=5)

    assert first == second


def test_retrieval_works_with_the_deterministic_provider_end_to_end() -> None:
    """Uses the real Step 16B provider rather than a fake, to prove the
    pieces compose. Note: similarity here carries no semantic meaning --
    only the plumbing is under test."""
    retriever, semantic_memory, vector_index = _build(dimension=16)
    provider = retriever.embedding_provider

    semantic_memory.add(_record("m1", content="User prefers Python for ML."))
    vector_index.add("m1", "A", provider.embed("User prefers Python for ML."))

    # Querying with the exact same text reproduces the exact same vector
    # (SHA-256 determinism), so similarity is 1.0 -- an identity check on
    # the pipeline, NOT a claim about language understanding.
    results = retriever.retrieve("A", "User prefers Python for ML.", top_k=5)

    assert [r.memory.memory_id for r in results] == ["m1"]
    assert results[0].similarity == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# RetrievedMemory result type.
# ---------------------------------------------------------------------------

def test_retrieved_memory_is_immutable() -> None:
    result = RetrievedMemory(memory=_record("m1"), similarity=0.5)

    with pytest.raises(Exception):
        result.similarity = 0.9  # type: ignore[misc]


@pytest.mark.parametrize("bad_similarity", [1.5, -1.5, "0.5", True, None])
def test_retrieved_memory_validates_similarity(bad_similarity: object) -> None:
    with pytest.raises(ValueError):
        RetrievedMemory(memory=_record("m1"), similarity=bad_similarity)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_memory", [None, "m1", {"memory_id": "m1"}])
def test_retrieved_memory_validates_the_record(bad_memory: object) -> None:
    with pytest.raises(ValueError):
        RetrievedMemory(memory=bad_memory, similarity=0.5)  # type: ignore[arg-type]


def test_retrieved_memory_has_no_speculative_ranking_fields() -> None:
    """Step 16D deliberately ships only `memory` + `similarity`."""
    result = RetrievedMemory(memory=_record("m1"), similarity=0.5)

    for absent in ["final_score", "recency_score", "rerank_score", "distance", "embedding_model"]:
        assert not hasattr(result, absent)


# ---------------------------------------------------------------------------
# Independent instances share no state.
# ---------------------------------------------------------------------------

def test_independent_retrievers_do_not_share_state() -> None:
    retriever_1, semantic_memory_1, vector_index_1, _ = _build_with_query_vector((1.0, 0.0))
    retriever_2, _, _, _ = _build_with_query_vector((1.0, 0.0))

    semantic_memory_1.add(_record("m1"))
    vector_index_1.add("m1", "A", (1.0, 0.0))

    assert len(retriever_1.retrieve("A", "q", top_k=5)) == 1
    assert retriever_2.retrieve("A", "q", top_k=5) == []
