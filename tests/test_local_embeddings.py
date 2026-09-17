"""Step 16H: the real local embedding provider.

Two halves, kept strictly apart:

1. UNIT tests (the whole file except the bottom section). These never
   import sentence-transformers, never load a model and never touch the
   network. They drive LocalEmbeddingProvider against a fake in-process
   model injected through `sys.modules`, which is what lets the contract,
   the batching, the validation, the normalization check and every failure
   path be tested on a machine that has no torch installed at all.

2. INTEGRATION tests, all marked `@pytest.mark.integration` (bottom
   section). These load the real model and are the only tests that prove
   the vectors are actually semantic. `pytest -m "not integration"` skips
   them entirely, so the normal suite never depends on a download.

The deterministic provider is untouched by this milestone and is asserted
so below — it remains the provider every other test file uses.
"""
from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import pytest

from app.agent.embeddings import DeterministicEmbeddingProvider, EmbeddingProvider
from app.agent.local_embeddings import (
    DEFAULT_EMBEDDING_MODEL_NAME,
    EmbeddingComputeError,
    EmbeddingError,
    EmbeddingModelLoadError,
    LocalEmbeddingProvider,
)
from app.agent.memory_extraction import MemoryCandidate
from app.agent.memory_retriever import SemanticMemoryRetriever
from app.agent.memory_writer import SemanticMemoryWriter
from app.agent.semantic_memory import InMemorySemanticMemory
from app.agent.vector_index import InMemoryVectorIndex

FAKE_DIMENSION = 8


# ===========================================================================
# Fake model plumbing — stands in for sentence_transformers.SentenceTransformer
# ===========================================================================


class FakeSentenceTransformer:
    """An in-process stand-in with the same surface LocalEmbeddingProvider
    uses: construction, `get_sentence_embedding_dimension()`, `encode()`.

    It produces normalized vectors from a trivial character-code hash. Like
    DeterministicEmbeddingProvider, this is NOT semantic — it exists to
    exercise plumbing, not meaning.
    """

    # Per-class knobs the tests flip to drive failure paths.
    dimension_override: object = None
    encode_result_override: object = None
    raise_on_load: Exception | None = None
    raise_on_encode: Exception | None = None

    def __init__(self, model_name: str, device: str = "cpu", **kwargs: object):
        if type(self).raise_on_load is not None:
            raise type(self).raise_on_load
        self.model_name = model_name
        self.device = device
        self.encode_calls: list[list[str]] = []
        self.encode_kwargs: list[dict[str, object]] = []
        self.dimension_accessor_used: str | None = None

    def get_embedding_dimension(self):
        """The name sentence-transformers >= 6.0 uses."""
        self.dimension_accessor_used = "get_embedding_dimension"
        if type(self).dimension_override is not None:
            return type(self).dimension_override
        return FAKE_DIMENSION

    def get_sentence_embedding_dimension(self):
        """The pre-6.0 name, kept in 6.x as a deprecated alias."""
        self.dimension_accessor_used = "get_sentence_embedding_dimension"
        if type(self).dimension_override is not None:
            return type(self).dimension_override
        return FAKE_DIMENSION

    def encode(self, texts, **kwargs):
        self.encode_calls.append(list(texts))
        self.encode_kwargs.append(dict(kwargs))
        if type(self).raise_on_encode is not None:
            raise type(self).raise_on_encode
        if type(self).encode_result_override is not None:
            return type(self).encode_result_override
        return [self._one(text) for text in texts]

    @staticmethod
    def _one(text: str) -> list[float]:
        raw = [float((ord(ch) * (i + 7)) % 97) + 1.0 for i, ch in enumerate(text[:FAKE_DIMENSION])]
        raw += [1.0] * (FAKE_DIMENSION - len(raw))
        norm = math.sqrt(sum(v * v for v in raw))
        return [v / norm for v in raw]


@pytest.fixture
def fake_st(monkeypatch: pytest.MonkeyPatch):
    """Installs a fake `sentence_transformers` module for the duration of a
    test. monkeypatch removes it afterwards, so no test leaks a fake module
    into another."""
    FakeSentenceTransformer.dimension_override = None
    FakeSentenceTransformer.encode_result_override = None
    FakeSentenceTransformer.raise_on_load = None
    FakeSentenceTransformer.raise_on_encode = None

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    yield FakeSentenceTransformer

    FakeSentenceTransformer.dimension_override = None
    FakeSentenceTransformer.encode_result_override = None
    FakeSentenceTransformer.raise_on_load = None
    FakeSentenceTransformer.raise_on_encode = None


@pytest.fixture
def no_st(monkeypatch: pytest.MonkeyPatch):
    """Simulates sentence-transformers not being installed, even on a
    machine where it is.

    A `None` entry in `sys.modules` is the standard way to do this: the
    import machinery raises ImportError for it. Patching
    `builtins.__import__` would also work but would intercept every import
    in the test, including pytest's own.
    """
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    return None


# ===========================================================================
# 1 — CONSTRUCTION
# ===========================================================================

def test_provider_constructs_and_satisfies_the_embedding_provider_protocol(fake_st) -> None:
    provider = LocalEmbeddingProvider()

    assert isinstance(provider, EmbeddingProvider)


def test_the_model_is_loaded_once_at_construction_not_per_call(fake_st) -> None:
    provider = LocalEmbeddingProvider()
    loaded_model = provider._model

    provider.embed("one")
    provider.embed("two")
    provider.embed_many(["three", "four"])

    assert provider._model is loaded_model  # same object throughout


def test_there_is_no_hidden_global_model_singleton(fake_st) -> None:
    """Two providers hold two models. A process-wide cache would make
    "which model is this session using?" unanswerable."""
    first = LocalEmbeddingProvider()
    second = LocalEmbeddingProvider()

    assert first._model is not second._model


def test_importing_the_package_does_not_import_sentence_transformers() -> None:
    """The heavy import lives inside __init__, so `import app.agent` stays
    cheap for every process that never builds this provider.

    Checked in a SUBPROCESS with a clean interpreter: asserting against this
    process's `sys.modules` would prove nothing, since the integration tests
    in this same file may already have loaded the library.
    """
    import subprocess

    probe = (
        "import sys; import app.agent; "
        "assert 'sentence_transformers' not in sys.modules, 'imported at module scope'; "
        "assert 'torch' not in sys.modules, 'imported at module scope'; "
        "print('clean')"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent.parent),
    )

    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


# ===========================================================================
# 2 — MODEL CONFIGURATION
# ===========================================================================

def test_the_default_model_is_the_documented_minilm(fake_st) -> None:
    assert DEFAULT_EMBEDDING_MODEL_NAME == "sentence-transformers/all-MiniLM-L6-v2"
    assert LocalEmbeddingProvider().model_name == DEFAULT_EMBEDDING_MODEL_NAME


def test_the_model_name_is_passed_through_to_the_library(fake_st) -> None:
    provider = LocalEmbeddingProvider("some-org/some-other-model")

    assert provider._model.model_name == "some-org/some-other-model"
    assert provider.model_name == "some-org/some-other-model"


def test_cpu_is_the_default_device_and_no_gpu_is_requested(fake_st) -> None:
    assert LocalEmbeddingProvider()._model.device == "cpu"


def test_the_device_is_configurable(fake_st) -> None:
    assert LocalEmbeddingProvider(device="cuda")._model.device == "cuda"


def test_config_default_matches_the_module_default() -> None:
    """Guards the one duplicated literal: app/config.py's env default and
    the provider's own default must not drift apart."""
    from app.config import Settings

    assert Settings().embedding_model_name == DEFAULT_EMBEDDING_MODEL_NAME


def test_no_api_key_is_read_for_embeddings() -> None:
    from app.config import Settings

    settings = Settings()
    assert "key" not in settings.embedding_model_name.lower()
    assert not hasattr(settings, "embedding_api_key")


@pytest.mark.parametrize("bad", ["", "   ", None, 42, True])
def test_invalid_model_name_is_rejected_as_a_configuration_error(fake_st, bad: object) -> None:
    with pytest.raises(ValueError, match="model_name"):
        LocalEmbeddingProvider(bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", ["", "   ", None, 1])
def test_invalid_device_is_rejected_as_a_configuration_error(fake_st, bad: object) -> None:
    with pytest.raises(ValueError, match="device"):
        LocalEmbeddingProvider(device=bad)  # type: ignore[arg-type]


# ===========================================================================
# 3 — DIMENSION
# ===========================================================================

def test_dimension_comes_from_the_model_not_a_hard_coded_constant(fake_st) -> None:
    fake_st.dimension_override = 512

    assert LocalEmbeddingProvider().dimension == 512


def test_dimension_is_available_before_any_embed_call(fake_st) -> None:
    """The retriever and writer read it at construction, so it must not be
    inferred from the first embedding."""
    provider = LocalEmbeddingProvider()

    assert provider.dimension == FAKE_DIMENSION
    assert provider._model.encode_calls == []  # nothing was embedded to learn it


def test_dimension_is_stable_across_many_calls(fake_st) -> None:
    provider = LocalEmbeddingProvider()
    before = provider.dimension

    for text in ("a", "bb", "ccc", "a much longer sentence than the others"):
        assert len(provider.embed(text)) == before

    assert provider.dimension == before


def test_a_declared_expected_dimension_that_matches_is_accepted(fake_st) -> None:
    assert LocalEmbeddingProvider(expected_dimension=FAKE_DIMENSION).dimension == FAKE_DIMENSION


def test_a_declared_expected_dimension_that_mismatches_raises(fake_st) -> None:
    """Guards against a silently switched model: vectors from a different
    model are not a degraded search, they are a meaningless one."""
    with pytest.raises(EmbeddingModelLoadError, match="expected_dimension"):
        LocalEmbeddingProvider(expected_dimension=384)


@pytest.mark.parametrize("bad", [0, -1, 1.5, "384", True])
def test_invalid_expected_dimension_is_a_configuration_error(fake_st, bad: object) -> None:
    with pytest.raises(ValueError, match="expected_dimension"):
        LocalEmbeddingProvider(expected_dimension=bad)  # type: ignore[arg-type]


def test_the_non_deprecated_dimension_accessor_is_preferred(fake_st) -> None:
    """sentence-transformers 6.0 renamed the accessor and kept the old name
    as a deprecated alias. Using the old one emits a FutureWarning on every
    load."""
    provider = LocalEmbeddingProvider()

    assert provider._model.dimension_accessor_used == "get_embedding_dimension"


def test_the_legacy_dimension_accessor_still_works(fake_st, monkeypatch) -> None:
    """requirements-embeddings.txt allows sentence-transformers >= 3.0.0,
    where only the pre-6.0 name exists."""
    monkeypatch.delattr(FakeSentenceTransformer, "get_embedding_dimension")

    provider = LocalEmbeddingProvider()

    assert provider.dimension == FAKE_DIMENSION
    assert provider._model.dimension_accessor_used == "get_sentence_embedding_dimension"


def test_a_model_with_no_dimension_accessor_is_rejected(fake_st, monkeypatch) -> None:
    monkeypatch.delattr(FakeSentenceTransformer, "get_embedding_dimension")
    monkeypatch.delattr(FakeSentenceTransformer, "get_sentence_embedding_dimension")

    with pytest.raises(EmbeddingModelLoadError, match="no embedding-dimension accessor"):
        LocalEmbeddingProvider()


@pytest.mark.parametrize("bad", [0, -5, "384", 12.5])
def test_a_model_reporting_a_nonsense_dimension_raises(fake_st, bad: object) -> None:
    fake_st.dimension_override = bad

    with pytest.raises(EmbeddingModelLoadError, match="dimension"):
        LocalEmbeddingProvider()


# ===========================================================================
# 4 — embed()
# ===========================================================================

def test_embed_returns_a_vector_of_the_declared_dimension(fake_st) -> None:
    vector = LocalEmbeddingProvider().embed("User prefers Python.")

    assert isinstance(vector, tuple)
    assert len(vector) == FAKE_DIMENSION


def test_embed_returns_plain_python_floats_not_backend_scalars(fake_st) -> None:
    vector = LocalEmbeddingProvider().embed("User prefers Python.")

    assert all(type(component) is float for component in vector)


def test_embed_output_is_immutable(fake_st) -> None:
    vector = LocalEmbeddingProvider().embed("text")

    with pytest.raises(TypeError):
        vector[0] = 0.0  # type: ignore[index]


def test_embed_calls_the_model_exactly_once(fake_st) -> None:
    provider = LocalEmbeddingProvider()
    provider.embed("text")

    assert len(provider._model.encode_calls) == 1


def test_embed_is_deterministic_for_the_same_text(fake_st) -> None:
    provider = LocalEmbeddingProvider()

    assert provider.embed("User prefers Python.") == provider.embed("User prefers Python.")


def test_embed_and_embed_many_agree_for_the_same_text(fake_st) -> None:
    """A caller must never be able to tell the singular and batch paths
    apart — here that holds by construction, since embed routes through the
    batch path."""
    provider = LocalEmbeddingProvider()

    assert provider.embed("User prefers Python.") == provider.embed_many(["User prefers Python."])[0]


# ===========================================================================
# 5/6/7 — embed_many(): batching, order, empty input
# ===========================================================================

def test_embed_many_returns_one_vector_per_input(fake_st) -> None:
    texts = ["alpha", "beta", "gamma", "delta"]

    vectors = LocalEmbeddingProvider().embed_many(texts)

    assert len(vectors) == len(texts)
    assert all(len(v) == FAKE_DIMENSION for v in vectors)


def test_embed_many_preserves_input_order(fake_st) -> None:
    provider = LocalEmbeddingProvider()
    texts = ["alpha", "beta", "gamma"]

    batched = provider.embed_many(texts)

    assert batched == [provider.embed(text) for text in texts]


def test_embed_many_uses_one_batched_model_call_not_one_per_item(fake_st) -> None:
    """A real transformer is dramatically more efficient batched; this is
    the reason embed_many exists in the Protocol at all."""
    provider = LocalEmbeddingProvider()

    provider.embed_many(["a", "b", "c", "d", "e"])

    assert len(provider._model.encode_calls) == 1
    assert provider._model.encode_calls[0] == ["a", "b", "c", "d", "e"]


def test_embed_many_empty_input_returns_empty_list_without_calling_the_model(fake_st) -> None:
    provider = LocalEmbeddingProvider()

    assert provider.embed_many([]) == []
    assert provider._model.encode_calls == []


def test_embed_many_accepts_a_tuple(fake_st) -> None:
    assert len(LocalEmbeddingProvider().embed_many(("a", "b"))) == 2


def test_embed_many_rejects_a_bare_string(fake_st) -> None:
    """A string is a sequence of characters; accepting it would silently
    embed each letter."""
    with pytest.raises(ValueError, match="list or tuple"):
        LocalEmbeddingProvider().embed_many("not a list")  # type: ignore[arg-type]


def test_embed_many_does_not_silently_truncate_or_pad(fake_st) -> None:
    provider = LocalEmbeddingProvider()
    fake_st.encode_result_override = [FakeSentenceTransformer._one("only one")]

    with pytest.raises(EmbeddingModelLoadError, match="embeddings for"):
        provider.embed_many(["a", "b", "c"])


def test_a_wrong_width_row_from_the_model_is_rejected(fake_st) -> None:
    provider = LocalEmbeddingProvider()
    fake_st.encode_result_override = [(1.0, 0.0, 0.0)]  # 3 wide, declared 8

    with pytest.raises(EmbeddingModelLoadError, match="dimensional"):
        provider.embed("text")


# ===========================================================================
# 8 — INPUT VALIDATION (shared with the deterministic provider)
# ===========================================================================

@pytest.mark.parametrize("bad", ["", "   ", "\n\t", None, 42, True, [], b"bytes"])
def test_embed_rejects_invalid_text(fake_st, bad: object) -> None:
    with pytest.raises(ValueError, match="text"):
        LocalEmbeddingProvider().embed(bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", ["", "   ", None, 42, True])
def test_embed_many_rejects_an_invalid_item_and_names_its_index(fake_st, bad: object) -> None:
    with pytest.raises(ValueError, match=r"texts\[1\]"):
        LocalEmbeddingProvider().embed_many(["fine", bad])  # type: ignore[list-item]


def test_invalid_input_is_rejected_before_the_model_is_called(fake_st) -> None:
    provider = LocalEmbeddingProvider()

    with pytest.raises(ValueError):
        provider.embed_many(["fine", ""])

    assert provider._model.encode_calls == []


def test_both_providers_apply_the_same_text_rule(fake_st) -> None:
    """Validation is shared, not duplicated: one definition of "valid
    text", so the two implementations cannot drift apart."""
    local = LocalEmbeddingProvider()
    deterministic = DeterministicEmbeddingProvider()

    for bad in ("", "   ", None, 42):
        with pytest.raises(ValueError):
            local.embed(bad)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            deterministic.embed(bad)  # type: ignore[arg-type]


# ===========================================================================
# 9/10 — FINITENESS AND NORMALIZATION
# ===========================================================================

def test_vectors_are_unit_normalized(fake_st) -> None:
    provider = LocalEmbeddingProvider()

    for text in ("short", "a considerably longer piece of text than the other one"):
        norm = math.sqrt(sum(c * c for c in provider.embed(text)))
        assert norm == pytest.approx(1.0, abs=1e-6)


def test_normalization_is_requested_explicitly_not_assumed_of_the_model(fake_st) -> None:
    """The default model normalizes itself, but that is a property of one
    model's config file — a swapped model that stopped would otherwise
    corrupt every similarity score silently."""
    provider = LocalEmbeddingProvider()
    provider.embed("text")

    assert provider._model.encode_kwargs[0]["normalize_embeddings"] is True


def test_an_unnormalized_vector_from_the_model_is_rejected(fake_st) -> None:
    provider = LocalEmbeddingProvider()
    fake_st.encode_result_override = [tuple([3.0] * FAKE_DIMENSION)]  # norm ~8.49

    with pytest.raises(EmbeddingModelLoadError, match="unit-normalized"):
        provider.embed("text")


@pytest.mark.parametrize("bad_component", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_components_are_rejected(fake_st, bad_component: float) -> None:
    provider = LocalEmbeddingProvider()
    fake_st.encode_result_override = [(bad_component,) + (0.0,) * (FAKE_DIMENSION - 1)]

    with pytest.raises(EmbeddingModelLoadError, match="non-finite"):
        provider.embed("text")


def test_all_batch_vectors_are_finite_and_normalized(fake_st) -> None:
    vectors = LocalEmbeddingProvider().embed_many(["alpha", "beta", "gamma"])

    for vector in vectors:
        assert all(math.isfinite(c) for c in vector)
        assert math.sqrt(sum(c * c for c in vector)) == pytest.approx(1.0, abs=1e-6)


# ===========================================================================
# 11/12 — MODEL LOADING FAILURE, NEVER A SILENT FALLBACK
# ===========================================================================

def test_a_missing_library_raises_with_actionable_guidance(no_st) -> None:
    with pytest.raises(EmbeddingModelLoadError) as exc_info:
        LocalEmbeddingProvider()

    message = str(exc_info.value)
    assert "sentence-transformers is not installed" in message
    assert "requirements-embeddings.txt" in message


def test_a_missing_library_does_not_fall_back_to_the_deterministic_provider(no_st) -> None:
    """The single most important assertion in this file: meaningless
    vectors served confidently are worse than a loud failure."""
    with pytest.raises(EmbeddingModelLoadError):
        provider = LocalEmbeddingProvider()
        assert not isinstance(provider, DeterministicEmbeddingProvider)


def test_a_failed_model_load_is_reported_not_swallowed(fake_st) -> None:
    fake_st.raise_on_load = OSError("model 'nope/nope' not found on the hub")

    with pytest.raises(EmbeddingModelLoadError, match="failed to load embedding model"):
        LocalEmbeddingProvider("nope/nope")


def test_the_original_load_failure_is_chained_for_diagnosis(fake_st) -> None:
    cause = OSError("connection refused")
    fake_st.raise_on_load = cause

    with pytest.raises(EmbeddingModelLoadError) as exc_info:
        LocalEmbeddingProvider()

    assert exc_info.value.__cause__ is cause


def test_load_errors_are_runtime_errors_not_value_errors() -> None:
    """app/main.py maps ValueError to 400 and RuntimeError to 502. A model
    that will not load is infrastructure, not a bad request."""
    assert issubclass(EmbeddingModelLoadError, RuntimeError)
    assert not issubclass(EmbeddingModelLoadError, ValueError)


def test_a_load_failure_leaves_no_half_built_provider(fake_st) -> None:
    fake_st.raise_on_load = OSError("boom")

    with pytest.raises(EmbeddingModelLoadError):
        LocalEmbeddingProvider()
    # Nothing to assert on the object -- it was never returned. The point is
    # that construction raises rather than yielding a provider whose embed()
    # would fail later at an arbitrary user request.


def test_no_text_or_vector_is_logged(fake_st, caplog: pytest.LogCaptureFixture) -> None:
    secret = "User's passphrase is hunter2-correct-horse."

    with caplog.at_level("DEBUG", logger="app.agent.local_embeddings"):
        LocalEmbeddingProvider().embed(secret)

    assert secret not in caplog.text
    assert "hunter2" not in caplog.text


# ===========================================================================
# 12b — EMBEDDING ERROR TAXONOMY (Step 16I)
# ===========================================================================

def test_embedding_model_load_error_is_an_embedding_error() -> None:
    """Step 16I introduces the EmbeddingError base after the fact — every
    16H test asserting EmbeddingModelLoadError's ValueError/RuntimeError
    relationship must still hold unchanged."""
    assert issubclass(EmbeddingModelLoadError, EmbeddingError)
    assert issubclass(EmbeddingError, RuntimeError)
    assert not issubclass(EmbeddingModelLoadError, ValueError)


def test_embedding_compute_error_is_an_embedding_error() -> None:
    assert issubclass(EmbeddingComputeError, EmbeddingError)
    assert issubclass(EmbeddingComputeError, RuntimeError)
    assert not issubclass(EmbeddingComputeError, ValueError)


def test_load_and_compute_errors_are_siblings_not_each_other() -> None:
    """The two failure modes are distinguishable by type — a caller that
    wants to react to load failure differently from compute failure can."""
    assert not issubclass(EmbeddingModelLoadError, EmbeddingComputeError)
    assert not issubclass(EmbeddingComputeError, EmbeddingModelLoadError)


def test_a_failure_inside_encode_itself_is_a_compute_error_not_a_load_error(fake_st) -> None:
    """A provider that already loaded successfully hitting a transient
    inference fault is a DIFFERENT failure from the model never loading at
    all — the two must not be conflated under one type."""
    provider = LocalEmbeddingProvider()  # loads successfully
    fake_st.raise_on_encode = RuntimeError("CUDA out of memory (simulated)")

    with pytest.raises(EmbeddingComputeError) as exc_info:
        provider.embed("text")

    assert not isinstance(exc_info.value, EmbeddingModelLoadError)


def test_compute_error_chains_the_original_exception(fake_st) -> None:
    provider = LocalEmbeddingProvider()
    cause = RuntimeError("internal encode failure (simulated)")
    fake_st.raise_on_encode = cause

    with pytest.raises(EmbeddingComputeError) as exc_info:
        provider.embed("text")

    assert exc_info.value.__cause__ is cause


def test_compute_error_is_raised_for_embed_many_too(fake_st) -> None:
    provider = LocalEmbeddingProvider()
    fake_st.raise_on_encode = OSError("simulated batch failure")

    with pytest.raises(EmbeddingComputeError):
        provider.embed_many(["a", "b", "c"])


def test_a_compute_error_does_not_prevent_a_later_successful_call(fake_st) -> None:
    """The failure is per-call, not a poisoned provider — the model already
    proved it loads; one bad batch does not make the instance unusable."""
    provider = LocalEmbeddingProvider()
    fake_st.raise_on_encode = RuntimeError("transient (simulated)")

    with pytest.raises(EmbeddingComputeError):
        provider.embed("first call fails")

    fake_st.raise_on_encode = None
    vector = provider.embed("second call succeeds")

    assert len(vector) == FAKE_DIMENSION


def test_existing_post_load_validation_still_raises_model_load_error_not_compute_error(fake_st) -> None:
    """Row-count mismatches, wrong-width rows, non-finite components, and
    non-unit-norm vectors are validation of what a WORKING model returned,
    not an encode() failure — Step 16I must not have reclassified them."""
    provider = LocalEmbeddingProvider()
    fake_st.encode_result_override = [(1.0, 0.0, 0.0)]  # wrong width

    with pytest.raises(EmbeddingModelLoadError) as exc_info:
        provider.embed("text")

    assert not isinstance(exc_info.value, EmbeddingComputeError)


def test_no_text_or_vector_is_logged_on_a_compute_failure(fake_st, caplog: pytest.LogCaptureFixture) -> None:
    provider = LocalEmbeddingProvider()
    fake_st.raise_on_encode = RuntimeError("boom")
    secret = "User's passphrase is hunter2-correct-horse."

    with caplog.at_level("DEBUG", logger="app.agent.local_embeddings"):
        with pytest.raises(EmbeddingComputeError):
            provider.embed(secret)

    assert secret not in caplog.text
    assert "hunter2" not in caplog.text


# ===========================================================================
# 13 — THE DETERMINISTIC PROVIDER IS UNCHANGED
# ===========================================================================

def test_deterministic_provider_still_exists_and_conforms() -> None:
    provider = DeterministicEmbeddingProvider()

    assert isinstance(provider, EmbeddingProvider)
    assert provider.dimension == 16


def test_deterministic_provider_still_needs_no_library(no_st) -> None:
    """It must keep working on a machine with no ML stack at all — that is
    what makes deterministic CI possible."""
    vector = DeterministicEmbeddingProvider().embed("User prefers Python.")

    assert len(vector) == 16
    assert math.sqrt(sum(c * c for c in vector)) == pytest.approx(1.0, abs=1e-9)


def test_the_two_providers_are_independent_implementations(fake_st) -> None:
    local = LocalEmbeddingProvider()
    deterministic = DeterministicEmbeddingProvider(dimension=FAKE_DIMENSION)

    assert local.embed("User prefers Python.") != deterministic.embed("User prefers Python.")


# ===========================================================================
# 14 — VECTOR INDEX / PIPELINE COMPATIBILITY
# ===========================================================================

def test_an_index_built_from_the_provider_dimension_accepts_its_vectors(fake_st) -> None:
    provider = LocalEmbeddingProvider()
    index = InMemoryVectorIndex(dimension=provider.dimension)

    index.add("m1", "A", provider.embed("User prefers Python."))

    assert index.search("A", provider.embed("User prefers Python."), top_k=1)[0].memory_id == "m1"


def test_the_writer_and_retriever_accept_the_local_provider(fake_st) -> None:
    provider = LocalEmbeddingProvider()
    store = InMemorySemanticMemory()
    index = InMemoryVectorIndex(dimension=provider.dimension)

    writer = SemanticMemoryWriter(store, provider, index)
    retriever = SemanticMemoryRetriever(store, provider, index)

    assert writer.embedding_provider is provider
    assert retriever.embedding_provider is provider


def test_a_dimension_mismatch_is_still_caught_at_wiring_time(fake_st) -> None:
    """16D/16G's eager compatibility check must keep working with the real
    provider — which is only possible because dimension is known at
    construction."""
    provider = LocalEmbeddingProvider()
    wrong_index = InMemoryVectorIndex(dimension=provider.dimension + 1)

    with pytest.raises(ValueError, match="dimension"):
        SemanticMemoryRetriever(InMemorySemanticMemory(), provider, wrong_index)


def test_the_full_write_read_pipeline_runs_on_the_local_provider(fake_st) -> None:
    """The pipeline is unchanged by 16H; only the injected provider is
    different."""
    provider = LocalEmbeddingProvider()
    store = InMemorySemanticMemory()
    index = InMemoryVectorIndex(dimension=provider.dimension)
    writer = SemanticMemoryWriter(store, provider, index)
    retriever = SemanticMemoryRetriever(store, provider, index)

    writer.write("alice", [MemoryCandidate("User prefers Python for ML.", 0.9)], ["evt-1"])
    results = retriever.retrieve("alice", "User prefers Python for ML.", top_k=5)

    assert [r.memory.content for r in results] == ["User prefers Python for ML."]


def test_no_memory_component_imports_the_concrete_provider() -> None:
    """The architecture rule: everything above depends on the Protocol."""
    root = Path(__file__).resolve().parent.parent / "app"
    for name in (
        "agent/semantic_memory.py",
        "agent/vector_index.py",
        "agent/memory_retriever.py",
        "agent/memory_writer.py",
        "agent/memory_context.py",
        "agent/memory_formatting.py",
        "agent/memory_extraction.py",
        "agent/orchestrator.py",
        "services/chat.py",
    ):
        source = (root / name).read_text(encoding="utf-8")
        assert "LocalEmbeddingProvider" not in source, name
        assert "sentence_transformers" not in source, name


def test_the_live_application_does_not_construct_a_real_provider() -> None:
    """Nothing is forced on the deployed agent: no startup download, no
    torch import, no model load."""
    root = Path(__file__).resolve().parent.parent / "app"
    for name in ("main.py", "services/chat.py"):
        assert "LocalEmbeddingProvider" not in (root / name).read_text(encoding="utf-8")


# ===========================================================================
# 15/16 — REAL MODEL (INTEGRATION ONLY)
# ===========================================================================
#
# Everything below loads the real all-MiniLM-L6-v2 and is excluded from
# `pytest -m "not integration"`. The first run downloads ~90 MB; later runs
# are offline from the Hugging Face cache.

pytestmark_reason = "requires the real embedding model (pip install -r requirements-embeddings.txt)"


@pytest.fixture(scope="module")
def real_provider():
    """Loads the real model ONCE for every integration test in this module,
    and skips them all cleanly if it is unavailable rather than failing —
    an uninstalled library or an absent network is not a test failure."""
    try:
        return LocalEmbeddingProvider()
    except EmbeddingModelLoadError as exc:
        pytest.skip(f"{pytestmark_reason}: {exc}")


@pytest.mark.integration
def test_real_model_reports_the_expected_dimension(real_provider) -> None:
    assert real_provider.dimension == 384  # all-MiniLM-L6-v2's documented width
    assert len(real_provider.embed("User prefers Python.")) == 384


@pytest.mark.integration
def test_real_model_load_emits_no_deprecation_warning() -> None:
    """Loading the real library must not warn: a FutureWarning on every
    provider construction is noise the suite would learn to ignore."""
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            LocalEmbeddingProvider()
        except EmbeddingModelLoadError as exc:
            pytest.skip(f"{pytestmark_reason}: {exc}")

    offenders = [
        str(w.message)
        for w in caught
        if issubclass(w.category, (DeprecationWarning, FutureWarning))
        and "local_embeddings.py" in str(w.filename)
    ]
    assert offenders == []


@pytest.mark.integration
def test_real_model_output_is_unit_normalized(real_provider) -> None:
    for text in ("short", "a much longer sentence, with punctuation, and some length to it"):
        norm = math.sqrt(sum(c * c for c in real_provider.embed(text)))
        assert norm == pytest.approx(1.0, abs=1e-5)


@pytest.mark.integration
def test_real_model_is_actually_semantic(real_provider) -> None:
    """The point of the whole milestone. A paraphrase must score higher
    than an unrelated fact — a relative assertion, not a brittle absolute
    threshold."""
    from app.agent.vector_index import cosine_similarity

    a = real_provider.embed("User prefers Python for machine learning.")
    paraphrase = real_provider.embed("Python is the user's preferred language for ML.")
    unrelated = real_provider.embed("User prefers mountain biking.")

    assert cosine_similarity(a, paraphrase) > cosine_similarity(a, unrelated)


@pytest.mark.integration
def test_the_deterministic_provider_is_not_semantic_by_contrast(real_provider) -> None:
    """Proves the integration test above is measuring meaning rather than
    passing by luck: the same comparison on hashed vectors has no reason to
    hold, and the real provider's margin is far larger."""
    from app.agent.vector_index import cosine_similarity

    texts = (
        "User prefers Python for machine learning.",
        "Python is the user's preferred language for ML.",
        "User prefers mountain biking.",
    )
    real = [real_provider.embed(t) for t in texts]
    fake = [DeterministicEmbeddingProvider(dimension=384).embed(t) for t in texts]

    real_margin = cosine_similarity(real[0], real[1]) - cosine_similarity(real[0], real[2])
    fake_margin = cosine_similarity(fake[0], fake[1]) - cosine_similarity(fake[0], fake[2])

    assert real_margin > 0.1
    assert real_margin > fake_margin


@pytest.mark.integration
def test_real_embed_many_matches_individual_embeds_and_keeps_order(real_provider) -> None:
    texts = ["User prefers Python.", "User enjoys cycling.", "User lives in Berlin."]

    batched = real_provider.embed_many(texts)

    assert len(batched) == len(texts)
    for batch_vector, single in zip(batched, (real_provider.embed(t) for t in texts)):
        assert batch_vector == pytest.approx(single, abs=1e-5)


@pytest.mark.integration
def test_real_retrieval_finds_the_relevant_memory(real_provider) -> None:
    """16H's real goal: a query phrased nothing like the stored fact still
    retrieves it."""
    store = InMemorySemanticMemory()
    index = InMemoryVectorIndex(dimension=real_provider.dimension)
    writer = SemanticMemoryWriter(store, real_provider, index)
    retriever = SemanticMemoryRetriever(store, real_provider, index)

    writer.write("alice", [MemoryCandidate("User prefers Python for machine learning.", 0.9)], ["e1"])
    writer.write("alice", [MemoryCandidate("User enjoys mountain biking on weekends.", 0.9)], ["e2"])
    writer.write("alice", [MemoryCandidate("User lives in Berlin.", 0.9)], ["e3"])

    results = retriever.retrieve("alice", "What programming language do I like using for ML?", top_k=3)

    assert "Python" in results[0].memory.content


@pytest.mark.integration
def test_an_unrelated_query_does_not_outrank_the_relevant_memory(real_provider) -> None:
    store = InMemorySemanticMemory()
    index = InMemoryVectorIndex(dimension=real_provider.dimension)
    writer = SemanticMemoryWriter(store, real_provider, index)
    retriever = SemanticMemoryRetriever(store, real_provider, index)

    writer.write("alice", [MemoryCandidate("User prefers Python for machine learning.", 0.9)], ["e1"])
    writer.write("alice", [MemoryCandidate("User enjoys mountain biking on weekends.", 0.9)], ["e2"])

    biking_query = retriever.retrieve("alice", "What do I do for fun outdoors?", top_k=2)

    assert "biking" in biking_query[0].memory.content


@pytest.mark.integration
def test_real_provider_keeps_session_isolation(real_provider) -> None:
    """Every 16G guarantee must survive the provider swap: a semantically
    perfect match in another session is still invisible."""
    store = InMemorySemanticMemory()
    index = InMemoryVectorIndex(dimension=real_provider.dimension)
    writer = SemanticMemoryWriter(store, real_provider, index)
    retriever = SemanticMemoryRetriever(store, real_provider, index)

    writer.write("alice", [MemoryCandidate("User prefers Python for machine learning.", 0.9)], ["e1"])

    assert retriever.retrieve("bob", "What language do I like for ML?", top_k=5) == []
