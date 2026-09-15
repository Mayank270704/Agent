from __future__ import annotations

import math

import pytest

from app.agent.embeddings import DeterministicEmbeddingProvider, EmbeddingProvider


# ---------------------------------------------------------------------------
# A: Protocol conformance.
# ---------------------------------------------------------------------------

def test_deterministic_provider_conforms_to_the_protocol() -> None:
    assert isinstance(DeterministicEmbeddingProvider(), EmbeddingProvider)


# ---------------------------------------------------------------------------
# B: valid text produces a vector.
# ---------------------------------------------------------------------------

def test_valid_text_produces_a_vector() -> None:
    provider = DeterministicEmbeddingProvider()

    vector = provider.embed("hello world")

    assert isinstance(vector, tuple)
    assert len(vector) == provider.dimension
    assert all(isinstance(component, float) for component in vector)


# ---------------------------------------------------------------------------
# C/D: blank / whitespace-only text rejected.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_text", ["", "   ", "\t\n", None, 123])
def test_blank_or_invalid_text_is_rejected_by_embed(bad_text: object) -> None:
    provider = DeterministicEmbeddingProvider()

    with pytest.raises(ValueError):
        provider.embed(bad_text)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_text", ["", "   ", None, 123])
def test_blank_or_invalid_text_is_rejected_within_embed_many(bad_text: object) -> None:
    provider = DeterministicEmbeddingProvider()

    with pytest.raises(ValueError):
        provider.embed_many(["valid text", bad_text])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# E: vector is non-empty.
# ---------------------------------------------------------------------------

def test_vector_is_never_empty() -> None:
    provider = DeterministicEmbeddingProvider()

    vector = provider.embed("some text")

    assert len(vector) > 0


# ---------------------------------------------------------------------------
# F: vector values are finite.
# ---------------------------------------------------------------------------

def test_vector_values_are_all_finite() -> None:
    provider = DeterministicEmbeddingProvider()

    for text in ["a", "hello world", "x" * 500, "unicode: café ☃"]:
        vector = provider.embed(text)
        assert all(math.isfinite(component) for component in vector)


# ---------------------------------------------------------------------------
# G: same input produces identical vector (determinism).
# ---------------------------------------------------------------------------

def test_same_text_produces_identical_vector() -> None:
    provider = DeterministicEmbeddingProvider()

    first = provider.embed("consistent text")
    second = provider.embed("consistent text")

    assert first == second


def test_same_text_produces_identical_vector_across_separate_instances() -> None:
    """Determinism must not depend on any per-instance or per-process
    state -- two independently constructed providers must agree."""
    provider_1 = DeterministicEmbeddingProvider()
    provider_2 = DeterministicEmbeddingProvider()

    assert provider_1.embed("shared text") == provider_2.embed("shared text")


# ---------------------------------------------------------------------------
# H: different inputs can produce different vectors.
# ---------------------------------------------------------------------------

def test_different_texts_generally_produce_different_vectors() -> None:
    provider = DeterministicEmbeddingProvider()

    assert provider.embed("apple") != provider.embed("banana")


def test_a_single_changed_character_changes_the_vector() -> None:
    provider = DeterministicEmbeddingProvider()

    assert provider.embed("cat") != provider.embed("bat")


# ---------------------------------------------------------------------------
# I: dimensionality is fixed for a given provider configuration.
# ---------------------------------------------------------------------------

def test_dimension_is_fixed_across_many_calls() -> None:
    provider = DeterministicEmbeddingProvider()

    for text in ["a", "a much longer piece of text than the last one", "x"]:
        assert len(provider.embed(text)) == provider.dimension


def test_dimension_is_configurable() -> None:
    provider = DeterministicEmbeddingProvider(dimension=8)

    assert provider.dimension == 8
    assert len(provider.embed("hello")) == 8


def test_default_dimension_is_documented_test_only_value() -> None:
    provider = DeterministicEmbeddingProvider()
    assert provider.dimension == 16


# ---------------------------------------------------------------------------
# J: invalid provider dimension rejected.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_dimension", [0, -1, -100, 1.5, "16", True, None])
def test_invalid_dimension_is_rejected_at_construction(bad_dimension: object) -> None:
    with pytest.raises(ValueError):
        DeterministicEmbeddingProvider(dimension=bad_dimension)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# K: no network/model dependency -- this is really a design assertion, but
# it is testable indirectly: many calls complete instantly and without any
# external resource, which would not be true of a real model.
# ---------------------------------------------------------------------------

def test_provider_requires_no_network_or_model_and_runs_fast() -> None:
    import time

    provider = DeterministicEmbeddingProvider()
    start = time.monotonic()
    for i in range(200):
        provider.embed(f"text number {i}")
    elapsed = time.monotonic() - start

    assert elapsed < 2.0  # generous bound; a real model load alone would blow this


# ---------------------------------------------------------------------------
# L/M/N: batch behavior, empty batch, ordering.
# ---------------------------------------------------------------------------

def test_embed_many_returns_one_vector_per_input_in_order() -> None:
    provider = DeterministicEmbeddingProvider()
    texts = ["apple", "banana", "cherry"]

    vectors = provider.embed_many(texts)

    assert len(vectors) == len(texts)
    for text, vector in zip(texts, vectors):
        assert vector == provider.embed(text)  # same order, same values


def test_embed_many_matches_individual_embed_calls_for_each_text() -> None:
    """embed(t) and embed_many([t])[0] must be equal -- the documented
    Protocol invariant that callers never see a difference between the
    singular and batch paths."""
    provider = DeterministicEmbeddingProvider()

    for text in ["one", "two", "three"]:
        assert provider.embed(text) == provider.embed_many([text])[0]


def test_embed_many_empty_input_returns_empty_list() -> None:
    provider = DeterministicEmbeddingProvider()

    assert provider.embed_many([]) == []


def test_embed_many_rejects_non_list_input() -> None:
    provider = DeterministicEmbeddingProvider()

    with pytest.raises(ValueError):
        provider.embed_many("not a list")  # type: ignore[arg-type]


def test_embed_many_output_count_matches_input_count() -> None:
    provider = DeterministicEmbeddingProvider()
    texts = [f"text {i}" for i in range(10)]

    vectors = provider.embed_many(texts)

    assert len(vectors) == len(texts)


def test_embed_many_accepts_a_tuple_as_well_as_a_list() -> None:
    provider = DeterministicEmbeddingProvider()

    vectors = provider.embed_many(("a", "b"))

    assert len(vectors) == 2


# ---------------------------------------------------------------------------
# O: vector representation cannot be accidentally mutated.
# ---------------------------------------------------------------------------

def test_vector_cannot_be_mutated_in_place() -> None:
    provider = DeterministicEmbeddingProvider()
    vector = provider.embed("immutable please")

    with pytest.raises(TypeError):
        vector[0] = 999.0  # type: ignore[index]


def test_vector_has_no_append_or_extend() -> None:
    provider = DeterministicEmbeddingProvider()
    vector = provider.embed("immutable please")

    assert not hasattr(vector, "append")
    assert not hasattr(vector, "extend")


# ---------------------------------------------------------------------------
# Documented invariants: unit normalization.
# ---------------------------------------------------------------------------

def test_vectors_are_unit_normalized() -> None:
    provider = DeterministicEmbeddingProvider()

    for text in ["a", "a longer sentence used to check normalization", "z" * 50]:
        vector = provider.embed(text)
        norm = math.sqrt(sum(component * component for component in vector))
        assert math.isclose(norm, 1.0, rel_tol=1e-9, abs_tol=1e-9)


def test_batch_vectors_are_also_unit_normalized() -> None:
    provider = DeterministicEmbeddingProvider()

    for vector in provider.embed_many(["one", "two", "three"]):
        norm = math.sqrt(sum(component * component for component in vector))
        assert math.isclose(norm, 1.0, rel_tol=1e-9, abs_tol=1e-9)
