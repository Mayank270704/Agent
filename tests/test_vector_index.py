from __future__ import annotations

import math

import pytest

from app.agent.vector_index import (
    InMemoryVectorIndex,
    VectorIndex,
    VectorSearchResult,
    cosine_similarity,
)


# ---------------------------------------------------------------------------
# A-D, 21: cosine similarity mathematics, using hand-crafted vectors whose
# expected results are mathematically obvious.
# ---------------------------------------------------------------------------

def test_cosine_similarity_identical_vectors_is_one() -> None:
    assert cosine_similarity((1.0, 0.0), (1.0, 0.0)) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_is_zero() -> None:
    assert cosine_similarity((1.0, 0.0), (0.0, 1.0)) == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors_is_negative_one() -> None:
    assert cosine_similarity((1.0, 0.0), (-1.0, 0.0)) == pytest.approx(-1.0)


def test_cosine_similarity_is_symmetric() -> None:
    a = (1.0, 2.0, 3.0)
    b = (4.0, -1.0, 0.5)

    assert cosine_similarity(a, b) == pytest.approx(cosine_similarity(b, a))


def test_cosine_similarity_unnormalized_vectors_matches_normalized() -> None:
    """Similarity must be scale-invariant -- scaling a vector must not
    change its cosine similarity to another vector (Part 2, option D)."""
    a = (3.0, 4.0)  # norm = 5
    scaled_a = (6.0, 8.0)  # same direction, norm = 10
    b = (1.0, 0.0)

    assert cosine_similarity(a, b) == pytest.approx(cosine_similarity(scaled_a, b))


def test_cosine_similarity_of_45_degree_vectors() -> None:
    assert cosine_similarity((1.0, 1.0), (1.0, 0.0)) == pytest.approx(math.sqrt(2) / 2)


# ---------------------------------------------------------------------------
# F/G/H: dimension mismatch, empty vector, non-finite value rejection.
# ---------------------------------------------------------------------------

def test_cosine_similarity_rejects_dimension_mismatch() -> None:
    with pytest.raises(ValueError):
        cosine_similarity((1.0, 0.0), (1.0, 0.0, 0.0))


def test_cosine_similarity_rejects_empty_vectors() -> None:
    with pytest.raises(ValueError):
        cosine_similarity((), (1.0,))


def test_cosine_similarity_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError):
        cosine_similarity((1.0, float("nan")), (1.0, 0.0))

    with pytest.raises(ValueError):
        cosine_similarity((1.0, float("inf")), (1.0, 0.0))


def test_cosine_similarity_rejects_zero_norm_vector() -> None:
    with pytest.raises(ValueError):
        cosine_similarity((0.0, 0.0), (1.0, 0.0))


def test_cosine_similarity_rejects_non_numeric_components() -> None:
    with pytest.raises(ValueError):
        cosine_similarity(("a", "b"), (1.0, 0.0))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# I: numerical clamping keeps the result within [-1.0, 1.0].
# ---------------------------------------------------------------------------

def test_cosine_similarity_result_is_always_within_valid_range() -> None:
    import random

    rng = random.Random(42)
    for _ in range(200):
        a = tuple(rng.uniform(-5, 5) for _ in range(6))
        b = tuple(rng.uniform(-5, 5) for _ in range(6))
        try:
            result = cosine_similarity(a, b)
        except ValueError:
            continue  # zero-norm draw; not the property under test
        assert -1.0 <= result <= 1.0


def test_cosine_similarity_of_a_vector_with_itself_never_exceeds_one() -> None:
    """A floating-point-drift regression check: identical vectors must
    clamp to exactly 1.0, never 1.0000000000000002."""
    a = (0.1, 0.2, 0.3, 0.4, 0.5)
    assert cosine_similarity(a, a) == 1.0


# ---------------------------------------------------------------------------
# J: index creation.
# ---------------------------------------------------------------------------

def test_index_creation_with_valid_dimension() -> None:
    index = InMemoryVectorIndex(dimension=4)
    assert index.dimension == 4


@pytest.mark.parametrize("bad_dimension", [0, -1, -100, 1.5, "4", True, None])
def test_index_creation_rejects_invalid_dimension(bad_dimension: object) -> None:
    with pytest.raises(ValueError):
        InMemoryVectorIndex(dimension=bad_dimension)  # type: ignore[arg-type]


def test_index_conforms_to_the_protocol() -> None:
    assert isinstance(InMemoryVectorIndex(dimension=3), VectorIndex)


# ---------------------------------------------------------------------------
# K/L: valid/invalid add.
# ---------------------------------------------------------------------------

def test_valid_add_makes_vector_searchable() -> None:
    index = InMemoryVectorIndex(dimension=2)

    index.add("mem-1", "A", (1.0, 0.0))

    results = index.search("A", (1.0, 0.0), top_k=1)
    assert [r.memory_id for r in results] == ["mem-1"]


@pytest.mark.parametrize("bad_memory_id", ["", "   ", None, 123])
def test_add_rejects_invalid_memory_id(bad_memory_id: object) -> None:
    index = InMemoryVectorIndex(dimension=2)
    with pytest.raises(ValueError):
        index.add(bad_memory_id, "A", (1.0, 0.0))  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_session_id", ["", "   ", None, 123])
def test_add_rejects_invalid_session_id(bad_session_id: object) -> None:
    index = InMemoryVectorIndex(dimension=2)
    with pytest.raises(ValueError):
        index.add("mem-1", bad_session_id, (1.0, 0.0))  # type: ignore[arg-type]


def test_add_rejects_wrong_dimension_vector() -> None:
    index = InMemoryVectorIndex(dimension=3)
    with pytest.raises(ValueError):
        index.add("mem-1", "A", (1.0, 0.0))  # 2 components, index wants 3


def test_add_rejects_non_finite_vector_components() -> None:
    index = InMemoryVectorIndex(dimension=2)
    with pytest.raises(ValueError):
        index.add("mem-1", "A", (1.0, float("nan")))


def test_add_rejects_empty_vector() -> None:
    index = InMemoryVectorIndex(dimension=2)
    with pytest.raises(ValueError):
        index.add("mem-1", "A", ())


# ---------------------------------------------------------------------------
# M-R: search / top-k semantics.
# ---------------------------------------------------------------------------

def test_valid_search_returns_scored_results() -> None:
    index = InMemoryVectorIndex(dimension=2)
    index.add("mem-1", "A", (1.0, 0.0))
    index.add("mem-2", "A", (0.0, 1.0))

    results = index.search("A", (1.0, 0.0), top_k=2)

    assert results[0].memory_id == "mem-1"
    assert results[0].similarity == pytest.approx(1.0)
    assert results[1].memory_id == "mem-2"
    assert results[1].similarity == pytest.approx(0.0)


def test_top_k_one_returns_only_the_best_match() -> None:
    index = InMemoryVectorIndex(dimension=2)
    index.add("mem-1", "A", (1.0, 0.0))
    index.add("mem-2", "A", (0.9, 0.1))
    index.add("mem-3", "A", (0.0, 1.0))

    results = index.search("A", (1.0, 0.0), top_k=1)

    assert len(results) == 1
    assert results[0].memory_id == "mem-1"


def test_top_k_larger_than_candidate_count_returns_all_available() -> None:
    index = InMemoryVectorIndex(dimension=2)
    index.add("mem-1", "A", (1.0, 0.0))
    index.add("mem-2", "A", (0.0, 1.0))

    results = index.search("A", (1.0, 0.0), top_k=1000)

    assert len(results) == 2


@pytest.mark.parametrize("bad_top_k", [0, -1, -100, True])
def test_invalid_top_k_is_rejected_not_silently_defaulted(bad_top_k: object) -> None:
    index = InMemoryVectorIndex(dimension=2)
    index.add("mem-1", "A", (1.0, 0.0))

    with pytest.raises(ValueError):
        index.search("A", (1.0, 0.0), top_k=bad_top_k)  # type: ignore[arg-type]


def test_search_on_empty_index_returns_empty_list() -> None:
    index = InMemoryVectorIndex(dimension=2)

    assert index.search("A", (1.0, 0.0), top_k=5) == []


def test_search_rejects_wrong_dimension_query_vector() -> None:
    index = InMemoryVectorIndex(dimension=3)
    index.add("mem-1", "A", (1.0, 0.0, 0.0))

    with pytest.raises(ValueError):
        index.search("A", (1.0, 0.0), top_k=1)


# ---------------------------------------------------------------------------
# S/T: deterministic ordering, exact ties.
# ---------------------------------------------------------------------------

def test_search_results_are_ordered_by_similarity_descending() -> None:
    index = InMemoryVectorIndex(dimension=2)
    index.add("far", "A", (0.0, 1.0))
    index.add("near", "A", (0.99, 0.14))
    index.add("exact", "A", (1.0, 0.0))

    results = index.search("A", (1.0, 0.0), top_k=3)

    assert [r.memory_id for r in results] == ["exact", "near", "far"]
    similarities = [r.similarity for r in results]
    assert similarities == sorted(similarities, reverse=True)


def test_exact_ties_break_by_memory_id_ascending() -> None:
    index = InMemoryVectorIndex(dimension=2)
    # Added in an order that would NOT match the expected tie-break order,
    # so this test cannot pass by accidentally relying on insertion order.
    index.add("zebra", "A", (1.0, 0.0))
    index.add("mango", "A", (1.0, 0.0))
    index.add("apple", "A", (1.0, 0.0))

    results = index.search("A", (1.0, 0.0), top_k=3)

    assert [r.memory_id for r in results] == ["apple", "mango", "zebra"]


def test_ordering_is_stable_across_repeated_searches() -> None:
    index = InMemoryVectorIndex(dimension=2)
    index.add("mem-1", "A", (1.0, 0.0))
    index.add("mem-2", "A", (0.0, 1.0))

    first = index.search("A", (1.0, 0.0), top_k=2)
    second = index.search("A", (1.0, 0.0), top_k=2)

    assert first == second


# ---------------------------------------------------------------------------
# U/V, 8/24: session isolation and cross-session leakage regression.
# ---------------------------------------------------------------------------

def test_search_for_session_a_never_returns_session_b_vectors() -> None:
    index = InMemoryVectorIndex(dimension=2)
    index.add("mem-a", "A", (1.0, 0.0))
    index.add("mem-b", "B", (1.0, 0.0))  # identical vector, different session

    results_a = index.search("A", (1.0, 0.0), top_k=10)

    assert [r.memory_id for r in results_a] == ["mem-a"]


def test_session_b_search_never_returns_session_a_vectors() -> None:
    index = InMemoryVectorIndex(dimension=2)
    index.add("mem-a", "A", (1.0, 0.0))
    index.add("mem-b", "B", (1.0, 0.0))

    results_b = index.search("B", (1.0, 0.0), top_k=10)

    assert [r.memory_id for r in results_b] == ["mem-b"]


def test_many_sessions_remain_mutually_isolated_under_search() -> None:
    index = InMemoryVectorIndex(dimension=2)
    session_ids = [f"session-{i}" for i in range(15)]

    for i, session_id in enumerate(session_ids):
        index.add(f"mem-{session_id}", session_id, (1.0, float(i)))

    for i, session_id in enumerate(session_ids):
        results = index.search(session_id, (1.0, float(i)), top_k=10)
        assert [r.memory_id for r in results] == [f"mem-{session_id}"]


def test_session_id_whitespace_is_normalized_consistently() -> None:
    index = InMemoryVectorIndex(dimension=2)
    index.add("mem-1", "A", (1.0, 0.0))

    assert index.search("  A  ", (1.0, 0.0), top_k=5) == index.search("A", (1.0, 0.0), top_k=5)


# ---------------------------------------------------------------------------
# W/X: memory_id duplicate / update behavior.
# ---------------------------------------------------------------------------

def test_duplicate_memory_id_within_the_same_session_is_rejected() -> None:
    index = InMemoryVectorIndex(dimension=2)
    index.add("mem-1", "A", (1.0, 0.0))

    with pytest.raises(ValueError):
        index.add("mem-1", "A", (0.0, 1.0))


def test_add_after_reject_leaves_the_original_vector_intact() -> None:
    index = InMemoryVectorIndex(dimension=2)
    index.add("mem-1", "A", (1.0, 0.0))

    with pytest.raises(ValueError):
        index.add("mem-1", "A", (0.0, 1.0))

    results = index.search("A", (1.0, 0.0), top_k=1)
    assert results[0].similarity == pytest.approx(1.0)  # unchanged, original vector


def test_same_memory_id_across_different_sessions_is_allowed() -> None:
    """Part 9: the identity key is (session_id, memory_id), not memory_id
    alone -- a deliberate, documented difference from
    SemanticMemoryStore's global uniqueness decision."""
    index = InMemoryVectorIndex(dimension=2)
    index.add("mem-shared", "A", (1.0, 0.0))
    index.add("mem-shared", "B", (0.0, 1.0))  # must not raise

    results_a = index.search("A", (1.0, 0.0), top_k=1)
    results_b = index.search("B", (0.0, 1.0), top_k=1)
    assert results_a[0].memory_id == "mem-shared"
    assert results_b[0].memory_id == "mem-shared"


def test_remove_then_add_allows_replacing_a_vector() -> None:
    index = InMemoryVectorIndex(dimension=2)
    index.add("mem-1", "A", (1.0, 0.0))

    index.remove("mem-1", "A")
    index.add("mem-1", "A", (0.0, 1.0))  # must not raise now

    results = index.search("A", (0.0, 1.0), top_k=1)
    assert results[0].memory_id == "mem-1"
    assert results[0].similarity == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Y/Z: remove behavior, wrong-session removal.
# ---------------------------------------------------------------------------

def test_remove_existing_vector_makes_it_unsearchable() -> None:
    index = InMemoryVectorIndex(dimension=2)
    index.add("mem-1", "A", (1.0, 0.0))

    index.remove("mem-1", "A")

    assert index.search("A", (1.0, 0.0), top_k=10) == []


def test_remove_unknown_memory_id_is_a_safe_no_op() -> None:
    index = InMemoryVectorIndex(dimension=2)

    index.remove("never-added", "A")  # must not raise


def test_remove_twice_is_a_safe_no_op() -> None:
    index = InMemoryVectorIndex(dimension=2)
    index.add("mem-1", "A", (1.0, 0.0))

    index.remove("mem-1", "A")
    index.remove("mem-1", "A")  # must not raise the second time


def test_removing_from_the_wrong_session_does_not_affect_the_real_owner() -> None:
    """Part 11/8: remove(memory_id, session_id) for a session that never
    held that memory_id must be a no-op, and must NEVER delete another
    session's real entry sharing the same memory_id."""
    index = InMemoryVectorIndex(dimension=2)
    index.add("mem-shared", "A", (1.0, 0.0))

    index.remove("mem-shared", "B")  # B never had this id

    results_a = index.search("A", (1.0, 0.0), top_k=10)
    assert [r.memory_id for r in results_a] == ["mem-shared"]  # untouched


@pytest.mark.parametrize("bad_value", ["", "   ", None, 123])
def test_remove_rejects_invalid_memory_id(bad_value: object) -> None:
    index = InMemoryVectorIndex(dimension=2)
    with pytest.raises(ValueError):
        index.remove(bad_value, "A")  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_value", ["", "   ", None, 123])
def test_remove_rejects_invalid_session_id(bad_value: object) -> None:
    index = InMemoryVectorIndex(dimension=2)
    with pytest.raises(ValueError):
        index.remove("mem-1", bad_value)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# AA: vector defensive copying.
# ---------------------------------------------------------------------------

def test_mutating_the_callers_original_list_after_add_does_not_affect_the_index() -> None:
    index = InMemoryVectorIndex(dimension=3)
    original = [1.0, 0.0, 0.0]

    index.add("mem-1", "A", original)
    original[0] = 999.0
    original.append(42.0)

    results = index.search("A", (1.0, 0.0, 0.0), top_k=1)
    assert results[0].similarity == pytest.approx(1.0)  # unaffected by the mutation


def test_stored_vector_is_a_real_tuple_even_when_a_list_was_passed() -> None:
    index = InMemoryVectorIndex(dimension=2)
    index.add("mem-1", "A", [1.0, 0.0])

    # No public accessor for the raw stored vector (deliberately minimal
    # Protocol) -- observable proof is that search still works correctly
    # and repeatedly, which a corrupted/aliased internal state would not
    # survive.
    for _ in range(3):
        results = index.search("A", (1.0, 0.0), top_k=1)
        assert results[0].similarity == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# AB: result immutability.
# ---------------------------------------------------------------------------

def test_search_result_is_immutable() -> None:
    index = InMemoryVectorIndex(dimension=2)
    index.add("mem-1", "A", (1.0, 0.0))

    result = index.search("A", (1.0, 0.0), top_k=1)[0]

    with pytest.raises(Exception):
        result.similarity = 0.0  # type: ignore[misc]


def test_result_list_mutation_does_not_affect_the_index() -> None:
    index = InMemoryVectorIndex(dimension=2)
    index.add("mem-1", "A", (1.0, 0.0))

    results = index.search("A", (1.0, 0.0), top_k=1)
    results.append(VectorSearchResult(memory_id="fabricated", similarity=1.0))
    results.clear()

    fresh = index.search("A", (1.0, 0.0), top_k=1)
    assert [r.memory_id for r in fresh] == ["mem-1"]


@pytest.mark.parametrize("bad_similarity", [1.5, -1.5, float("nan"), float("inf"), "1.0", True])
def test_vector_search_result_validates_similarity(bad_similarity: object) -> None:
    with pytest.raises(ValueError):
        VectorSearchResult(memory_id="mem-1", similarity=bad_similarity)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_memory_id", ["", "   ", None, 123])
def test_vector_search_result_validates_memory_id(bad_memory_id: object) -> None:
    with pytest.raises(ValueError):
        VectorSearchResult(memory_id=bad_memory_id, similarity=0.5)  # type: ignore[arg-type]


def test_vector_search_result_does_not_carry_session_id() -> None:
    """Part 5: session_id is deliberately omitted from the result model."""
    result = VectorSearchResult(memory_id="mem-1", similarity=0.5)
    assert not hasattr(result, "session_id")


# ---------------------------------------------------------------------------
# AC: index dimension enforcement is independent of any embedding provider.
# ---------------------------------------------------------------------------

def test_index_dimension_is_independent_of_any_embedding_provider_dimension() -> None:
    """The index's dimension is whatever it was constructed with -- it has
    no notion of, or dependency on, DeterministicEmbeddingProvider's own
    (unrelated) dimension."""
    index = InMemoryVectorIndex(dimension=384)  # e.g. a real model's size
    vector_384 = (1.0,) + (0.0,) * 383

    index.add("mem-1", "A", vector_384)


def test_a_vector_matching_a_different_indexs_dimension_is_rejected_here() -> None:
    small_index = InMemoryVectorIndex(dimension=2)
    sixteen_dim_vector = tuple(0.1 for _ in range(16))

    with pytest.raises(ValueError):
        small_index.add("mem-1", "A", sixteen_dim_vector)


# ---------------------------------------------------------------------------
# AD: multiple sessions exercised together (search, add, remove interleaved).
# ---------------------------------------------------------------------------

def test_interleaved_operations_across_multiple_sessions_stay_correct() -> None:
    index = InMemoryVectorIndex(dimension=2)

    index.add("mem-a1", "A", (1.0, 0.0))
    index.add("mem-b1", "B", (0.0, 1.0))
    index.add("mem-a2", "A", (0.9, 0.1))
    index.remove("mem-b1", "B")
    index.add("mem-b2", "B", (1.0, 0.0))

    results_a = index.search("A", (1.0, 0.0), top_k=10)
    results_b = index.search("B", (1.0, 0.0), top_k=10)

    assert [r.memory_id for r in results_a] == ["mem-a1", "mem-a2"]
    assert [r.memory_id for r in results_b] == ["mem-b2"]


# ---------------------------------------------------------------------------
# AE: no module-level global state; independent instances.
# ---------------------------------------------------------------------------

def test_independent_index_instances_do_not_share_state() -> None:
    index_1 = InMemoryVectorIndex(dimension=2)
    index_2 = InMemoryVectorIndex(dimension=2)

    index_1.add("mem-1", "A", (1.0, 0.0))

    assert index_1.search("A", (1.0, 0.0), top_k=10) != []
    assert index_2.search("A", (1.0, 0.0), top_k=10) == []
