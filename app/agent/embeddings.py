"""Embedding abstraction (Step 16B) — the mathematical representation
boundary the Step 16 architecture review ("Semantic Memory Blueprint")
called for:

    TEXT  ->  EmbeddingProvider  ->  VECTOR

Nothing beyond that lives here. This module does not know about semantic
memory, retrieval, similarity, ranking, or the agent at all — see the
"Dependency direction" section below.

--------------------------------------------------------------------------
Step 16B scope — deliberately inert, same discipline as Step 16A
--------------------------------------------------------------------------
This module defines:
- ONE Protocol (EmbeddingProvider): the model-independent contract.
- ONE deterministic implementation (DeterministicEmbeddingProvider), which
  exists ONLY to exercise that contract in tests — it does NOT produce
  vectors with any real natural-language meaning (see its own docstring).

It deliberately contains NONE of the following (later, separately scoped
phases — see the Step 16 blueprint's phased roadmap, 16C onward):
- Cosine similarity, nearest-neighbor search, ranking, thresholds.
- A vector store, FAISS, Chroma, pgvector, or any index.
- A MemoryRetriever, or any connection to SemanticMemory/EpisodicMemory.
- A real embedding model — no sentence-transformers, no torch, no
  Hugging Face, no network call, no model download.

--------------------------------------------------------------------------
Dependency direction (Part 16)
--------------------------------------------------------------------------
    SemanticMemory  (app/agent/semantic_memory.py)
           ^
    EmbeddingProvider  (this module)
           ^
    future retrieval layer

SemanticMemory does not import this module, and this module does not
import SemanticMemory. Both are leaves a future MemoryRetriever will
depend on; they must never depend on each other, or on any concrete
implementation of the other — see app/agent/semantic_memory.py's own
"Step 16A scope" note ("semantic memory should NOT store embeddings yet").
"""
from __future__ import annotations

import hashlib
import math
from typing import Protocol, Sequence, runtime_checkable

Vector = tuple[float, ...]
"""A single embedding: a fixed-length, immutable sequence of floats.

Chosen over `list[float]` specifically for immutability (Part 3, Part 15
item O) — a caller cannot accidentally mutate a vector already stored or
already used in a computation, the same reasoning EpisodicMemoryRecord and
SemanticMemoryRecord already apply to their own fields. Chosen over a
dedicated wrapper class because a tuple of floats already satisfies every
requirement this step actually has: it is trivially comparable for the
deterministic tests below, it converts to a NumPy array or any future
vector-store's native type with zero ceremony (`numpy.array(vector)`), and
it needs no serialization support beyond what `tuple`/`list` already give
for free (e.g. `list(vector)` for JSON). Introducing NumPy now, before
there is a real similarity computation or a real model to justify it,
would add a dependency this project has consistently avoided adding ahead
of need (see the Step 16 blueprint's embedding section).
"""


def _require_valid_text(text: object, *, label: str = "text") -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{label} must be a non-empty string.")
    return text


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Model-independent contract for turning text into vectors.

    Deliberately minimal — no `fit`, no `load`, no model-specific
    configuration surface. Anything specific to a particular embedding
    model lives entirely inside that model's own concrete implementation,
    never in this Protocol (Part 11 — model independence).

    Two methods, not one, after evaluating both shapes for `embed` (Part
    2/10):

    - `embed(text)` returns a SINGLE vector for a SINGLE piece of text —
      the common case at query time (embedding one user message to search
      with). Returning a bare `Vector` here (rather than a one-element
      list) means every future call site that just wants "the" vector for
      "the" query never has to unwrap a list.
    - `embed_many(texts)` returns one vector PER input text, in order —
      the batch case (embedding many stored semantic memories at once,
      e.g. during a future backfill/indexing pass). Real embedding models
      (sentence-transformers included) are dramatically more efficient
      batched than called once per string, so building this in now avoids
      a forced interface change the day a real model arrives — exactly
      the "future replacement cost" this step is required to weigh.

    Keeping both, rather than only `embed_many` with single-item calls
    unwrapped by hand everywhere, avoids scattering `[0]` indexing across
    every future caller; keeping both, rather than only `embed`, avoids
    forcing a Python-level loop (with its per-call overhead) onto whatever
    real model eventually implements this.

    CONTRACT every implementation must uphold (Part 4) — enforced by
    DeterministicEmbeddingProvider below, and REQUIRED of any future real
    implementation even though a Protocol cannot enforce it at import
    time:
    - `text` (and every item of `texts`) must be a non-empty string once
      stripped; blank or whitespace-only input raises `ValueError`.
    - The returned vector is never empty: `len(vector) == self.dimension`,
      always, for this provider instance.
    - Every component is a finite float — never `NaN`, never `inf`/`-inf`.
    - Vectors are UNIT-NORMALIZED (L2 norm == 1.0, within floating-point
      tolerance) — see the module-level "Normalization" note below for
      why this is a provider-level guarantee rather than a similarity-
      layer concern.
    - `embed(t)` and `embed_many([t])[0]` are equal for the same `t` and
      the same provider instance/configuration — a caller must never be
      able to observe a difference between the singular and batch paths.
    - Same text, same provider configuration -> the SAME vector, every
      call, including across separate process runs (this is what makes a
      provider "deterministic"; DeterministicEmbeddingProvider satisfies
      it exactly, and is not required of every future provider — a real
      model is still deterministic in this sense given fixed weights, but
      that is that provider's responsibility to preserve, not this
      Protocol's to enforce).

    Normalization (Part 9): EmbeddingProvider implementations GUARANTEE
    unit-normalized output (Option A, not Option B — raw vectors
    normalized later by a similarity layer). This is a deliberate,
    forward-looking decision, already made in the Step 16 architecture
    review: normalizing once, inside the provider, means no downstream
    component can forget to do it, and it means a future cosine-similarity
    implementation reduces to a plain dot product (cosine of two unit
    vectors IS their dot product) rather than every caller re-deriving
    that normalization step. The consequence: every `EmbeddingProvider`,
    deterministic or real, takes on this small extra responsibility —
    which is exactly why it belongs in the Protocol's documented contract
    now, rather than being bolted on later once callers already assume
    raw output.
    """

    @property
    def dimension(self) -> int:
        """The fixed vector length this provider always returns.

        A property (Part 5), not a zero-argument method like
        `embedding_dimension()`: dimensionality is a static fact ABOUT a
        given provider instance/configuration, not a computation — it
        never changes between calls on the same instance, so reading it
        should look like reading a fact, not invoking an operation. A
        future retrieval layer needs this to reject or flag a query
        vector and a stored vector that came from different providers
        (mismatched dimensions) before ever attempting a similarity
        computation — that check is NOT implemented here (Part 5/13); this
        property only makes the information discoverable.
        """
        ...

    def embed(self, text: str) -> Vector:
        ...

    def embed_many(self, texts: Sequence[str]) -> list[Vector]:
        ...


class DeterministicEmbeddingProvider:
    """A dependency-free, deterministic stand-in for a real embedding
    model — used ONLY to exercise the EmbeddingProvider contract in tests
    and local development.

    IMPORTANT — this is NOT fake semantic intelligence (Part 6/8). The
    vectors this class returns carry NO natural-language meaning
    whatsoever: two texts that mean the same thing will generally produce
    UNRELATED vectors, and two texts that mean opposite things could just
    as easily land close together. It exists solely so that every other
    component in this codebase that will eventually depend on
    EmbeddingProvider (a future VectorStore, a future MemoryRetriever) can
    be built and tested against a real, working implementation of the
    contract, without downloading or running an actual model. Nothing
    about its output should ever be interpreted as "similar" or
    "relevant" in the way a real embedding model's output would be.

    --------------------------------------------------------------------
    How determinism is achieved (Part 7)
    --------------------------------------------------------------------
    Python's built-in `hash()` is NOT used: CPython randomizes `str`
    hashing per process by default (PYTHONHASHSEED) specifically to
    prevent hash-based attacks, which means the same string can hash
    differently across two runs of the same program — the opposite of
    what "same text -> same vector, including across separate process
    runs" requires. `hashlib.sha256` is used instead: it is part of the
    standard library, has no randomization, and produces the same digest
    for the same input bytes on every platform and every run, forever.

    Algorithm: the input text is encoded to UTF-8 and hashed with
    SHA-256. That 32-byte digest is expanded to `dimension` components by
    re-hashing `digest + counter` for successive `counter` values (0, 1,
    2, ...) whenever more bytes are needed than one digest provides, and
    interpreting each 4-byte chunk as an unsigned big-endian integer
    linearly mapped from [0, 2^32-1] to [-1.0, 1.0]. The resulting raw
    vector is then L2-normalized (see the Protocol's Normalization note).
    This is a hash-based pseudo-random projection, not a learned
    embedding — it is deterministic and text-sensitive (changing one
    character changes the whole digest, and so the whole vector), which
    is all the test contract in Part 6/7 actually requires.

    --------------------------------------------------------------------
    Dimension (Part 8)
    --------------------------------------------------------------------
    Defaults to 16. This dimension belongs to the deterministic TEST
    provider and must NOT be taken as appropriate for, or predictive of,
    any future real embedding model — a real local model such as
    all-MiniLM-L6-v2 (evaluated in the Step 16 architecture review)
    produces 384-dimensional vectors. 16 was chosen only because it is
    large enough to meaningfully exercise multi-dimensional vector logic
    in tests while staying cheap and easy to eyeball in test assertions.
    `dimension` is configurable at construction (validated `>= 1`) so
    tests can exercise dimension-mismatch scenarios deliberately.
    """

    def __init__(self, dimension: int = 16):
        if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 1:
            raise ValueError("dimension must be an integer >= 1.")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> Vector:
        _require_valid_text(text)
        return self._embed_one(text)

    def embed_many(self, texts: Sequence[str]) -> list[Vector]:
        if not isinstance(texts, (list, tuple)):
            raise ValueError("texts must be a list or tuple of strings.")
        for index, text in enumerate(texts):
            _require_valid_text(text, label=f"texts[{index}]")
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> Vector:
        raw = self._hash_to_raw_vector(text)
        return self._normalize(raw)

    def _hash_to_raw_vector(self, text: str) -> tuple[float, ...]:
        seed = text.encode("utf-8")
        values: list[float] = []
        counter = 0
        while len(values) < self._dimension:
            digest = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
            counter += 1
            for offset in range(0, len(digest) - 3, 4):
                if len(values) >= self._dimension:
                    break
                chunk = digest[offset : offset + 4]
                int_value = int.from_bytes(chunk, "big")
                # Linearly map [0, 2**32 - 1] -> [-1.0, 1.0].
                values.append((int_value / 0xFFFFFFFF) * 2.0 - 1.0)
        return tuple(values)

    def _normalize(self, raw: tuple[float, ...]) -> Vector:
        norm = math.sqrt(sum(component * component for component in raw))
        if norm == 0.0:
            # Astronomically unlikely with SHA-256-derived components, but
            # division by zero must never silently produce a NaN vector —
            # fail loudly instead (Part 4's finite-values invariant).
            raise ValueError("computed a zero-norm vector; cannot normalize.")
        normalized = tuple(component / norm for component in raw)
        if not all(math.isfinite(component) for component in normalized):
            raise ValueError("embedding provider produced a non-finite vector component.")
        return normalized
