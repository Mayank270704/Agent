"""Step 16I: app/semantic_memory_wiring.py — the composition factory.

Fully offline. Every test drives `build_semantic_memory` with
`provider_factory` overridden to a fake that ignores `model_name`/`device`
and returns a `DeterministicEmbeddingProvider` — no real model, no torch,
no network, ever, in this file. The one place the real
`LocalEmbeddingProvider` default is checked is a signature/identity test
that never actually calls it.
"""
from __future__ import annotations

import sys

import pytest

from app.agent.embeddings import DeterministicEmbeddingProvider
from app.agent.local_embeddings import LocalEmbeddingProvider
from app.agent.memory_extraction import LLMMemoryExtractor
from app.agent.memory_retriever import SemanticMemoryRetriever
from app.agent.memory_writer import SemanticMemoryWriter
from app.agent.semantic_memory import InMemorySemanticMemory
from app.agent.vector_index import InMemoryVectorIndex
from app.semantic_memory_wiring import SemanticMemoryBundle, build_semantic_memory

FAKE_DIMENSION = 8


class FakeLLM:
    def generate(self, messages: list[dict[str, str]], *, json_mode: bool = False) -> str:
        raise AssertionError("FakeLLM should never be called by wiring tests")


def _fake_provider_factory(*, model_name: str = "", device: str = "", dimension: int = FAKE_DIMENSION):
    """Returns a provider_factory that ignores model_name/device and hands
    back a DeterministicEmbeddingProvider — exercises the wiring logic with
    no real model."""

    def factory(*, model_name: str, device: str):
        return DeterministicEmbeddingProvider(dimension=dimension)

    return factory


def _build(**overrides):
    kwargs = dict(
        enabled=True,
        model_name="fake/model",
        device="cpu",
        max_records_per_session=200,
        llm_client=FakeLLM(),
        provider_factory=_fake_provider_factory(),
    )
    kwargs.update(overrides)
    return build_semantic_memory(**kwargs)


# ===========================================================================
# B1 — DISABLED
# ===========================================================================

def test_disabled_returns_none() -> None:
    assert (
        build_semantic_memory(
            enabled=False,
            model_name="fake/model",
            device="cpu",
            max_records_per_session=200,
            llm_client=FakeLLM(),
        )
        is None
    )


def test_disabled_never_calls_the_provider_factory() -> None:
    calls: list[object] = []

    def spy_factory(*, model_name: str, device: str):
        calls.append((model_name, device))
        return DeterministicEmbeddingProvider(dimension=FAKE_DIMENSION)

    build_semantic_memory(
        enabled=False,
        model_name="fake/model",
        device="cpu",
        max_records_per_session=200,
        llm_client=FakeLLM(),
        provider_factory=spy_factory,
    )

    assert calls == []


def test_disabled_does_not_import_the_ml_stack() -> None:
    """Uses the REAL default provider_factory (LocalEmbeddingProvider) to
    prove disabled mode never even references it, in THIS process's
    sys.modules — a stricter check than the spy-based test above, which
    only proves the callable was not invoked."""
    result = build_semantic_memory(
        enabled=False,
        model_name="fake/model",
        device="cpu",
        max_records_per_session=200,
        llm_client=FakeLLM(),
        # provider_factory omitted -> uses the real LocalEmbeddingProvider default
    )

    assert result is None
    assert "sentence_transformers" not in sys.modules
    assert "torch" not in sys.modules


@pytest.mark.parametrize("bad_enabled", [None, "true", 1, 0, "false"])
def test_enabled_must_be_a_real_bool(bad_enabled: object) -> None:
    with pytest.raises(ValueError, match="enabled"):
        build_semantic_memory(
            enabled=bad_enabled,  # type: ignore[arg-type]
            model_name="fake/model",
            device="cpu",
            max_records_per_session=200,
            llm_client=FakeLLM(),
            provider_factory=_fake_provider_factory(),
        )


# ===========================================================================
# B2 — ENABLED: exactly one provider, exactly one bundle
# ===========================================================================

def test_enabled_returns_a_bundle() -> None:
    bundle = _build()

    assert isinstance(bundle, SemanticMemoryBundle)
    assert isinstance(bundle.retriever, SemanticMemoryRetriever)
    assert isinstance(bundle.writer, SemanticMemoryWriter)
    assert isinstance(bundle.extractor, LLMMemoryExtractor)


def test_enabled_constructs_exactly_one_provider() -> None:
    calls: list[tuple[str, str]] = []

    def spy_factory(*, model_name: str, device: str):
        calls.append((model_name, device))
        return DeterministicEmbeddingProvider(dimension=FAKE_DIMENSION)

    _build(provider_factory=spy_factory)

    assert len(calls) == 1


def test_provider_factory_receives_the_configured_model_name_and_device() -> None:
    received: dict[str, str] = {}

    def spy_factory(*, model_name: str, device: str):
        received["model_name"] = model_name
        received["device"] = device
        return DeterministicEmbeddingProvider(dimension=FAKE_DIMENSION)

    _build(model_name="org/some-model", device="cuda", provider_factory=spy_factory)

    assert received == {"model_name": "org/some-model", "device": "cuda"}


def test_default_provider_factory_is_the_real_local_embedding_provider() -> None:
    """Signature/identity only — never actually calls it, so this test
    never loads a model."""
    import inspect

    sig = inspect.signature(build_semantic_memory)
    assert sig.parameters["provider_factory"].default is LocalEmbeddingProvider


# ===========================================================================
# B3 — SHARING INVARIANT: writer and retriever share the SAME instances
# ===========================================================================

def test_writer_and_retriever_share_the_same_provider() -> None:
    bundle = _build()

    assert bundle.writer.embedding_provider is bundle.retriever.embedding_provider


def test_writer_and_retriever_share_the_same_store() -> None:
    bundle = _build()

    assert bundle.writer.semantic_memory is bundle.retriever.semantic_memory


def test_writer_and_retriever_share_the_same_index() -> None:
    bundle = _build()

    assert bundle.writer.vector_index is bundle.retriever.vector_index


def test_the_shared_stack_actually_works_end_to_end() -> None:
    """The behavioral consequence of the sharing invariant: a fact written
    through the bundle's writer is retrievable through its retriever — the
    real risk a silently mismatched stack would produce."""
    from app.agent.memory_extraction import MemoryCandidate

    bundle = _build()
    bundle.writer.write("alice", [MemoryCandidate("User prefers Python for ML.", 0.9)], ["evt-1"])

    results = bundle.retriever.retrieve("alice", "User prefers Python for ML.", top_k=5)

    assert [r.memory.content for r in results] == ["User prefers Python for ML."]


# ===========================================================================
# B4 — DIMENSION: never hard-coded
# ===========================================================================

def test_index_dimension_equals_provider_dimension() -> None:
    bundle = _build(provider_factory=_fake_provider_factory(dimension=FAKE_DIMENSION))

    assert bundle.writer.vector_index.dimension == FAKE_DIMENSION
    assert bundle.writer.embedding_provider.dimension == FAKE_DIMENSION


def test_index_dimension_tracks_whatever_the_provider_reports() -> None:
    """A different provider dimension must produce a different index
    dimension — proving it is READ, not a hard-coded 384 or 8."""
    bundle_small = _build(provider_factory=_fake_provider_factory(dimension=4))
    bundle_large = _build(provider_factory=_fake_provider_factory(dimension=64))

    assert bundle_small.writer.vector_index.dimension == 4
    assert bundle_large.writer.vector_index.dimension == 64


# ===========================================================================
# B5 — RETENTION: the writer is the sole owner
# ===========================================================================

def test_store_is_constructed_uncapped() -> None:
    bundle = _build(max_records_per_session=5)

    # InMemorySemanticMemory has no public way to read its own cap other
    # than behavior, so this asserts the DOCUMENTED internal (mirrors the
    # 16G/16F tests' own style of asserting internals for exactly this
    # kind of structural guarantee).
    assert isinstance(bundle.writer.semantic_memory, InMemorySemanticMemory)
    assert bundle.writer.semantic_memory._max_records_per_session is None


def test_writer_receives_the_configured_retention_cap() -> None:
    bundle = _build(max_records_per_session=5)

    assert bundle.writer.max_records_per_session == 5


def test_retention_evicts_both_the_record_and_its_vector() -> None:
    """The behavioral proof that one owner cannot strand a vector: writing
    past the cap must shrink the store AND the index together."""
    from app.agent.memory_extraction import MemoryCandidate

    bundle = _build(max_records_per_session=3)
    for i in range(5):
        bundle.writer.write("alice", [MemoryCandidate(f"User distinct fact {i}.", 0.9)], [f"evt-{i}"])

    stored = bundle.writer.semantic_memory.list_recent("alice", limit=100)
    assert len(stored) == 3

    provider = bundle.writer.embedding_provider
    for i in range(5):
        hits = bundle.writer.vector_index.search("alice", provider.embed(f"User distinct fact {i}."), top_k=1)
        matched = hits and hits[0].similarity > 0.999
        if matched:
            assert hits[0].memory_id in {r.memory_id for r in stored}


@pytest.mark.parametrize("bad_cap", [0, -1])
def test_invalid_retention_cap_propagates_from_the_writer(bad_cap: int) -> None:
    """build_semantic_memory adds no validation of its own here — it relies
    on SemanticMemoryWriter's existing (16F) validation."""
    with pytest.raises(ValueError):
        _build(max_records_per_session=bad_cap)


# ===========================================================================
# B6 — EXTRACTOR: receives the exact supplied LLMClient
# ===========================================================================

def test_extractor_receives_the_exact_supplied_llm_client() -> None:
    llm = FakeLLM()

    bundle = _build(llm_client=llm)

    assert bundle.extractor.llm is llm


def test_a_different_llm_client_produces_a_different_extractor_binding() -> None:
    llm_a = FakeLLM()
    llm_b = FakeLLM()

    bundle_a = _build(llm_client=llm_a)
    bundle_b = _build(llm_client=llm_b)

    assert bundle_a.extractor.llm is llm_a
    assert bundle_b.extractor.llm is llm_b
    assert bundle_a.extractor.llm is not bundle_b.extractor.llm


def test_a_none_llm_client_raises() -> None:
    """LLMMemoryExtractor already rejects None; this function adds no
    try/except, so that failure propagates unchanged."""
    with pytest.raises(ValueError):
        _build(llm_client=None)


# ===========================================================================
# B7 — INDEPENDENCE: separate calls build separate stacks
# ===========================================================================

def test_separate_factory_calls_create_separate_stacks() -> None:
    bundle_1 = _build()
    bundle_2 = _build()

    assert bundle_1.writer.semantic_memory is not bundle_2.writer.semantic_memory
    assert bundle_1.writer.vector_index is not bundle_2.writer.vector_index
    assert bundle_1.writer.embedding_provider is not bundle_2.writer.embedding_provider


def test_separate_stacks_do_not_share_written_memory() -> None:
    from app.agent.memory_extraction import MemoryCandidate

    bundle_1 = _build()
    bundle_2 = _build()

    bundle_1.writer.write("alice", [MemoryCandidate("User prefers Python.", 0.9)], ["evt-1"])

    assert bundle_2.retriever.retrieve("alice", "User prefers Python.", top_k=5) == []


# ===========================================================================
# B8 — NO HIDDEN GLOBAL STATE
# ===========================================================================

def test_build_semantic_memory_has_no_module_level_singleton() -> None:
    """There must be no cached bundle at module scope that a second call
    could accidentally reuse."""
    import app.semantic_memory_wiring as wiring_module

    module_level_bundles = [
        name
        for name, value in vars(wiring_module).items()
        if isinstance(value, SemanticMemoryBundle)
    ]

    assert module_level_bundles == []


def test_bundle_is_frozen() -> None:
    bundle = _build()

    with pytest.raises((AttributeError, TypeError)):
        bundle.writer = None  # type: ignore[misc]
