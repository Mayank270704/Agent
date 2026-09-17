"""Real local semantic embeddings (Step 16H) — the second implementation
of the Step 16B `EmbeddingProvider` Protocol:

    EmbeddingProvider                      (app/agent/embeddings.py)
           ^                        ^
    DeterministicEmbeddingProvider   LocalEmbeddingProvider  (this module)

Everything 16A–16G built sits on the Protocol, never on a concrete
provider, so this milestone adds a class and changes no pipeline. The
flow is byte-for-byte the one 16E-D/16F/16G already established:

    MemoryCandidate -> SemanticMemoryWriter -> EmbeddingProvider.embed
        -> VectorIndex -> SemanticMemoryRetriever
        -> EmbeddingProvider.embed(query) -> vector search

Only WHICH provider is injected changes.

--------------------------------------------------------------------------
Why this is a separate module rather than a second class in embeddings.py
--------------------------------------------------------------------------
`app/agent/embeddings.py` states its own scope explicitly: "no
sentence-transformers, no torch, no Hugging Face, no network call, no
model download." That is not a stylistic note — it is what lets every
other memory component import the Protocol without pulling a machine
learning stack into the import graph. Adding a torch-backed class to that
file would quietly break the guarantee for every existing importer.

Here, the heavy import happens INSIDE `__init__` (never at module import
time), so `import app.agent` stays as cheap as it was before 16H even
though this module is re-exported from the package.

--------------------------------------------------------------------------
Model: sentence-transformers/all-MiniLM-L6-v2
--------------------------------------------------------------------------
- Dimension: 384 (read FROM the loaded model, never hard-coded — see
  `dimension`).
- Size on disk: ~183 MB in the Hugging Face cache after the first
  download. The weights themselves are ~90 MB; the Hub repo ships several
  formats (safetensors, pytorch_model.bin, ONNX) and the snapshot keeps
  them all.
- Runtime: CPU-only by default (`device="cpu"`). No GPU is required and
  none is requested unless a caller explicitly asks.
- Why this one: it is the smallest widely-used sentence-embedding model
  with genuinely good semantic quality, and it is the model the Step 16
  architecture review already named as the target. Larger models (mpnet,
  bge-large, e5-large) buy retrieval quality this prototype cannot yet
  measure, at several times the download and several times the per-query
  latency.

Measured on the development machine (Windows, CPU, Python 3.13,
sentence-transformers 6.0.1 / torch 2.14.0+cpu), with the model already
downloaded. Treat these as order-of-magnitude, not as a benchmark:

- `import sentence_transformers`: ~24 s. This is the torch import, paid
  once per PROCESS, and it dwarfs everything else — which is the main
  reason the library import lives inside `__init__` rather than at module
  scope, so no process pays it without asking for a provider.
- Constructing the provider once the library is imported: ~8 s.
- `embed()` on one short sentence: ~22 ms.
- `embed_many()` over 32 short sentences: ~118 ms total, ~3.7 ms each —
  roughly 6x cheaper per text than calling `embed()` in a loop. That ratio
  is the concrete reason `embed_many` issues ONE batched model call.

Semantic quality, same machine, cosine similarity between:
    "User prefers Python for machine learning."
  vs "Python is the user's preferred language for ML."   -> 0.77
  vs "User prefers mountain biking."                     -> 0.34
A margin of ~0.43 between a paraphrase and an unrelated fact is the whole
point of 16H; DeterministicEmbeddingProvider has no such structure.

--------------------------------------------------------------------------
Optional dependency, deliberately
--------------------------------------------------------------------------
`sentence-transformers` (and its torch dependency) are NOT in
requirements.txt. They live in requirements-embeddings.txt, and nothing in
the running application constructs this class: `ChatService()` and
`app/main.py` are untouched, so the deployed agent neither downloads a
model nor pays a startup cost it has no use for yet. A missing library is
therefore a normal, expected state — reported as a clear
`EmbeddingModelLoadError`, never as a silent fallback (see "Failure
semantics" below).

--------------------------------------------------------------------------
Failure semantics — no silent fallback, ever
--------------------------------------------------------------------------
If the library is missing, the model name is unknown, the download fails,
or the loaded model's dimension is not what the caller declared, this
class RAISES. It never falls back to DeterministicEmbeddingProvider and
never substitutes a different model.

That rule is the whole point of the milestone. A deterministic provider's
vectors carry no natural-language meaning, so a silent fallback would
turn "semantic retrieval is broken" into "semantic retrieval returns
confident nonsense" — a failure that surfaces as bad answers weeks later
rather than as a stack trace at startup. Likewise a silently substituted
model would leave an index full of vectors from one model being searched
with queries from another, which is not a degraded search but a
meaningless one.

`EmbeddingModelLoadError` is a RuntimeError, not a ValueError, matching
how this codebase already separates the two: `app/main.py` maps
ValueError to HTTP 400 (the caller sent something wrong) and RuntimeError
to 502 (a dependency this service needs is not working). A model that
will not load is the second kind. Genuine CONFIGURATION mistakes —
a non-string model name, a negative `expected_dimension` — stay
ValueError, because those are bugs in the wiring, not infrastructure
failures.

--------------------------------------------------------------------------
Security
--------------------------------------------------------------------------
Embedded text is arbitrary user content. It is passed to the model as
DATA and nothing here parses, executes, templates, or interprets it. No
text and no vector is ever logged at any level: the only things logged are
the model name, the resolved dimension, and load timing. The memory
layer's own security gates (16F's credential/instruction rejection) are
upstream of embedding and are unchanged.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from typing import Sequence

# `_require_valid_text` is module-private to embeddings.py but is imported
# here deliberately: it IS the EmbeddingProvider input contract, and every
# implementation must apply exactly the same rule. Re-implementing it would
# create two definitions of "valid text" free to drift apart, which is the
# duplication this codebase avoids elsewhere (see the retriever's note on
# not re-validating vectors the index already validates). Both classes live
# in one package, so this is an internal detail, not a public coupling.
from app.agent.embeddings import Vector, _require_valid_text

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
"""The model this project targets.

Kept as a module constant rather than a literal buried in the class so
that `app/config.py` can be checked against it (tests assert the two agree,
so the env default and the code default cannot drift apart), and so no
memory component ever has to name a model.
"""

# How far a vector's L2 norm may sit from 1.0 before we call it
# un-normalized. Generous enough for float32 accumulation over 384
# components, tight enough that a genuinely un-normalized vector (norm
# typically 3-10 for this model's raw pooled output) can never pass.
_NORM_TOLERANCE = 1e-3


class EmbeddingError(RuntimeError):
    """Base class for every failure this provider can raise (Step 16I).

    Introduced so a CALLER — the orchestrator's degradation logic, most
    concretely — can catch "something about real embeddings failed"
    without having to enumerate every concrete subtype, while still never
    catching bare `RuntimeError` (which would also swallow unrelated
    infrastructure failures, e.g. from `LLMClient`) or bare `Exception`
    (which would swallow `ValueError`/`MemorySessionIsolationError` too —
    exactly the integrity failures that must never be degraded away).

    Two siblings, split by WHEN the failure happens, because a caller
    reasonably cares about that distinction even if it reacts to both the
    same way today:

    - `EmbeddingModelLoadError` — the provider could not be MADE READY at
      all (missing library, bad model name, no network on first download,
      a dimension that does not match what was declared). This can only
      happen at construction time, i.e. at application wiring, so it is
      always a startup-time failure, never a per-request one.
    - `EmbeddingComputeError` — a WORKING, already-loaded provider failed
      to embed one particular batch of text (a transient inference fault
      inside `SentenceTransformer.encode`, e.g. a resource error). This
      can only happen per-call, from `embed`/`embed_many`, against a
      provider that loaded successfully.

    Nothing in this codebase catches this base class directly today; it
    exists as the stable supertype a future catch site is written against
    (`except EmbeddingError:`), so adding a third failure mode later
    extends the hierarchy without changing that catch site.
    """


class EmbeddingModelLoadError(EmbeddingError):
    """The real embedding model could not be made ready for use.

    Raised for a missing `sentence-transformers` install, an unknown or
    unreachable model, any other failure inside the library's load path,
    and a handful of POST-load sanity checks that are really about the
    model being unusable as loaded (a nonsense reported dimension, a
    declared `expected_dimension` that does not match, an output row of
    the wrong width, a non-finite or non-unit-norm component — see
    `_to_vector`). Deliberately ONE type for all of them: from a caller's
    point of view the actionable fact is identical in every case — there
    is no real embedding provider, so do not pretend there is — and the
    original exception is always chained (`raise ... from exc`) for
    anyone who needs the specific cause.
    """


class EmbeddingComputeError(EmbeddingError):
    """A loaded, working embedding model failed to embed a specific batch
    of text (Step 16I).

    Distinguished from `EmbeddingModelLoadError` because this is a
    PER-CALL failure against a provider that already proved itself usable
    at construction — a transient fault inside `SentenceTransformer.encode`
    itself (for example a resource exhaustion or an internal library
    error), not evidence the model or configuration is broken. Whether a
    caller treats "the model won't load" and "this one embed call failed"
    the same way (both currently degrade to "no memory" at the call site —
    see AgentOrchestrator) is a caller decision; this module only names
    the two cases distinctly so that decision can be made deliberately
    rather than by accident of which one happened to be defined.

    The original exception is always chained.
    """


class LocalEmbeddingProvider:
    """A real, locally-executed sentence-embedding model behind the
    Step 16B `EmbeddingProvider` Protocol.

    Unlike DeterministicEmbeddingProvider — which hashes text and whose
    similarities are meaningless — this provider produces vectors where
    cosine similarity actually tracks meaning, which is what makes the
    16D/16G retrieval stack useful rather than merely correct.

    --------------------------------------------------------------------
    Model lifecycle: loaded ONCE, at construction
    --------------------------------------------------------------------
    The model is loaded in `__init__` and held on the instance. It is
    never loaded per request, per call, or per embedded string.

    Eager rather than lazy, chosen deliberately:

    - `dimension` is a fact about a provider that the rest of the
      architecture reads at WIRING time — SemanticMemoryRetriever and
      SemanticMemoryWriter both compare `embedding_provider.dimension`
      against `vector_index.dimension` in their own constructors and
      refuse to be built on a mismatch. A lazily-loaded provider cannot
      answer `dimension` honestly before its first embed, so it would
      either have to guess (a hard-coded 384 that a swapped model would
      silently falsify) or defer the check to the first query — turning a
      wiring bug into a runtime one, which is precisely the trade this
      codebase has refused everywhere else.
    - It puts load failure at the moment of miswiring, where it is
      diagnosable, instead of at an arbitrary later user request.
    - The startup cost it implies is not paid by the running application:
      nothing in `ChatService` or `app/main.py` constructs this class, so
      the only processes that load a model are the ones that asked for
      one. Lazy loading would optimize a cost nobody is currently paying.

    There is NO module-level model cache and no singleton. Two instances
    load two models; an instance is passed where it is needed, exactly
    like every other collaborator in this codebase. A process-wide cache
    would make "which model is this session using?" un-answerable and is
    the kind of hidden global state the architecture has avoided
    throughout.

    --------------------------------------------------------------------
    Concurrency
    --------------------------------------------------------------------
    `SentenceTransformer.encode` is NOT documented as thread-safe, and
    this is a FastAPI application whose synchronous endpoints run in a
    threadpool — so concurrent calls into one provider instance are
    genuinely reachable. Rather than assert thread safety this project
    cannot verify about a library it does not own, every call into the
    model is serialized by a per-instance `threading.Lock`.

    What that costs, stated plainly: embedding calls do not run in
    parallel within one process. For a prototype embedding one query per
    request at a few milliseconds each, that is not a bottleneck, and
    correctness under concurrency is worth more than throughput that is
    not currently needed. The lock is per INSTANCE, not global, so two
    providers never contend with each other.

    Only the model call itself is inside the lock — validation and the
    numpy-to-tuple conversion are pure per-call work on local data and
    hold it for no reason.

    --------------------------------------------------------------------
    Normalization: requested explicitly AND verified
    --------------------------------------------------------------------
    The Protocol requires unit-normalized output, because 16C's cosine
    similarity and the whole retrieval stack above it are specified
    against it.

    all-MiniLM-L6-v2 ships a `Normalize` module in its own
    sentence-transformers configuration, so its output is already
    unit-length — but this class does not rely on that. It passes
    `normalize_embeddings=True` explicitly (normalizing an already-unit
    vector is a no-op, so this is safe and idempotent), and then CHECKS
    the resulting norm, raising if any vector is not unit-length within
    `_NORM_TOLERANCE`. Belt and braces, on purpose: "the model normalizes
    for us" is a property of one model's config file, and a swapped model
    that quietly stopped doing it would otherwise corrupt every
    similarity score in the system with no error anywhere.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
        *,
        device: str = "cpu",
        expected_dimension: int | None = None,
    ):
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string.")
        if not isinstance(device, str) or not device.strip():
            raise ValueError("device must be a non-empty string.")
        if expected_dimension is not None and (
            isinstance(expected_dimension, bool)
            or not isinstance(expected_dimension, int)
            or expected_dimension < 1
        ):
            raise ValueError("expected_dimension must be an integer >= 1 if provided.")

        self._model_name = model_name.strip()
        self._device = device.strip()
        # Guards the model call only — see the class docstring.
        self._lock = threading.Lock()

        self._model = self._load_model()
        self._dimension = self._resolve_dimension()

        if expected_dimension is not None and self._dimension != expected_dimension:
            # A caller that declared a dimension and got a different one is
            # looking at a different model than it thinks. Never silently
            # accept it: the vectors would be incomparable with anything
            # already indexed.
            raise EmbeddingModelLoadError(
                f"model {self._model_name!r} produces {self._dimension}-dimensional embeddings, "
                f"but expected_dimension={expected_dimension} was declared."
            )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _load_model(self):
        """Import the library and load the weights, or raise a single,
        explicit error.

        The import lives HERE rather than at module scope so that
        importing `app.agent` costs nothing for the (currently every)
        process that never builds this provider.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingModelLoadError(
                "sentence-transformers is not installed, so LocalEmbeddingProvider cannot be used. "
                "Install the optional embedding dependencies with "
                "`pip install -r requirements-embeddings.txt`. "
                "There is deliberately no fallback to DeterministicEmbeddingProvider: its vectors "
                "carry no semantic meaning, so falling back would silently produce useless retrieval."
            ) from exc

        started = time.monotonic()
        try:
            model = SentenceTransformer(self._model_name, device=self._device)
        except Exception as exc:  # noqa: BLE001 - see below
            # Deliberately broad: the library raises a wide and unstable
            # range of types for "could not load" (OSError, HFValidationError,
            # RepositoryNotFoundError, requests' connection errors, ...).
            # Enumerating them would couple this module to the internals of
            # two dependencies and would still miss one. Every case means
            # exactly the same thing to a caller, the cause is chained, and
            # nothing is swallowed — this converts to a clear error, it does
            # not continue.
            raise EmbeddingModelLoadError(
                f"failed to load embedding model {self._model_name!r} on device {self._device!r}: {exc}"
            ) from exc

        # Model name and timing only. Never the text, never the vectors.
        logger.info(
            "Loaded embedding model %r on device %r in %.2fs.",
            self._model_name,
            self._device,
            time.monotonic() - started,
        )
        return model

    def _resolve_dimension(self) -> int:
        """Read the dimension FROM the loaded model.

        Not hard-coded (384 is a fact about one model, and this class must
        stay correct when a caller names another) and not inferred from the
        first `embed()` call (the retriever and writer both need it at
        construction, before any embedding exists).
        """
        # sentence-transformers 6.0 renamed `get_sentence_embedding_dimension`
        # to `get_embedding_dimension` and left the old name in place as a
        # deprecated alias. Preferring the new name and falling back to the
        # old one keeps this working across the whole range
        # requirements-embeddings.txt allows (>=3.0.0, where only the old
        # name exists) without emitting a FutureWarning on 6.x. If a future
        # release drops the alias, the fallback simply stops being reached.
        accessor = getattr(self._model, "get_embedding_dimension", None)
        if accessor is None:
            accessor = getattr(self._model, "get_sentence_embedding_dimension", None)
        if accessor is None:
            raise EmbeddingModelLoadError(
                f"model {self._model_name!r} exposes no embedding-dimension accessor; "
                f"this sentence-transformers version is not supported."
            )

        try:
            dimension = accessor()
        except Exception as exc:  # noqa: BLE001 - same reasoning as _load_model
            raise EmbeddingModelLoadError(
                f"could not determine the embedding dimension of model {self._model_name!r}: {exc}"
            ) from exc

        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
            raise EmbeddingModelLoadError(
                f"model {self._model_name!r} reported an invalid embedding dimension {dimension!r}."
            )
        return dimension

    # ------------------------------------------------------------------
    # EmbeddingProvider contract
    # ------------------------------------------------------------------

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        """Which model these vectors came from.

        Exposed for diagnostics and for a future component that needs to
        refuse to search an index built by a different model. Not part of
        the EmbeddingProvider Protocol — nothing in the memory pipeline
        reads it, and nothing there should have to know a model exists.
        """
        return self._model_name

    def embed(self, text: str) -> Vector:
        _require_valid_text(text)
        # Routed through the batch path so `embed(t)` and
        # `embed_many([t])[0]` are identical BY CONSTRUCTION rather than by
        # two implementations agreeing — the Protocol requires a caller
        # never be able to tell them apart.
        return self._encode([text])[0]

    def embed_many(self, texts: Sequence[str]) -> list[Vector]:
        if not isinstance(texts, (list, tuple)):
            raise ValueError("texts must be a list or tuple of strings.")
        for index, text in enumerate(texts):
            _require_valid_text(text, label=f"texts[{index}]")

        if not texts:
            # An empty batch never reaches the model: some backends error on
            # a zero-length input and others return an awkwardly-shaped
            # array. "Nothing in, nothing out" is the contract
            # DeterministicEmbeddingProvider already honors.
            return []

        return self._encode(list(texts))

    # ------------------------------------------------------------------
    # The one place the model is called
    # ------------------------------------------------------------------

    def _encode(self, texts: list[str]) -> list[Vector]:
        """Run one batched forward pass and convert it to Vectors.

        ONE model call for the whole batch, never one per item: a real
        transformer is dramatically more efficient batched, which is the
        reason `embed_many` exists in the Protocol at all (16B).

        The call into `self._model.encode` is wrapped and reraised as
        `EmbeddingComputeError` (Step 16I) — distinct from
        `EmbeddingModelLoadError`, because a failure HERE means a model
        that already proved it loads and works has hit a transient
        per-call fault (e.g. a resource error inside torch), not that the
        model or configuration is broken. This is deliberately as broad as
        the load-time catch in `_load_model`, and for the identical
        reason: encode() can raise a wide and unstable range of
        exception types from its own internals, enumerating them would
        couple this module to two dependencies' internals and still miss
        one, and every case means the same actionable thing to a caller.
        The original exception is always chained.
        """
        try:
            with self._lock:
                raw = self._model.encode(
                    texts,
                    # Explicit rather than relying on this model's own
                    # Normalize module — see the class docstring. Idempotent.
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
        except Exception as exc:  # noqa: BLE001 - see the docstring above
            raise EmbeddingComputeError(
                f"model {self._model_name!r} failed to embed a batch of {len(texts)} text(s): {exc}"
            ) from exc

        if len(raw) != len(texts):
            # Cannot happen with a working model; asserted anyway because
            # silent truncation or padding here would misalign every vector
            # with its memory record, mislabeling facts rather than failing.
            raise EmbeddingModelLoadError(
                f"model {self._model_name!r} returned {len(raw)} embeddings for {len(texts)} inputs."
            )

        return [self._to_vector(row) for row in raw]

    def _to_vector(self, row) -> Vector:
        """One model output row -> a validated, immutable `Vector`.

        Converts to plain Python floats rather than leaving NumPy scalars
        in place: `Vector` is documented as a tuple of floats, tuples of
        `numpy.float32` would leak the backend into every downstream
        comparison and serialization, and float32 -> float is exactly the
        widening `cosine_similarity` already does internally.
        """
        vector = tuple(float(component) for component in row)

        if len(vector) != self._dimension:
            raise EmbeddingModelLoadError(
                f"model {self._model_name!r} returned a {len(vector)}-dimensional vector, "
                f"but its declared dimension is {self._dimension}."
            )
        if not all(math.isfinite(component) for component in vector):
            raise EmbeddingModelLoadError(
                f"model {self._model_name!r} produced a non-finite embedding component."
            )

        norm = math.sqrt(sum(component * component for component in vector))
        if not math.isclose(norm, 1.0, abs_tol=_NORM_TOLERANCE):
            raise EmbeddingModelLoadError(
                f"model {self._model_name!r} produced a vector with L2 norm {norm!r}; "
                f"EmbeddingProvider requires unit-normalized output."
            )

        return vector
