"""Vector similarity and an in-memory vector index (Step 16C) — the
mathematical retrieval foundation the Step 16 architecture review called
for:

    VECTOR -> COSINE SIMILARITY -> SESSION-SCOPED EXACT SEARCH -> TOP-K

Nothing higher-level lives here. This module knows nothing about semantic
facts, episodic events, conversations, prompts, or the agent — see
"Dependency direction" below.

--------------------------------------------------------------------------
Step 16C scope — deliberately inert, same discipline as 16A/16B
--------------------------------------------------------------------------
This module defines:
- ONE pure function (cosine_similarity): the math, usable on its own.
- ONE immutable result type (VectorSearchResult).
- ONE Protocol (VectorIndex): store vectors, search them, nothing more.
- ONE brute-force implementation (InMemoryVectorIndex): exact, O(N) per
  query within the searched session's partition, a reference
  implementation this step is explicitly told NOT to optimize (no ANN,
  no trees, no clustering, no caching — see Part 19 of the Step 16C
  brief). It exists so a future PostgreSQL + pgvector implementation can
  replace it behind the SAME VectorIndex Protocol without the retrieval
  contract above it changing at all.

It deliberately contains NONE of the following (later, separately scoped
phases):
- A MemoryRetriever, or any composition of VectorIndex + EmbeddingProvider
  + SemanticMemoryStore.
- Ranking that combines similarity with recency/confidence.
- Relevance thresholds, filtering, or context construction.
- Any real vector database (FAISS, Chroma, pgvector, Pinecone, Qdrant,
  Weaviate) — this is pure Python / standard library only.
- Any connection to ChatService, AgentOrchestrator, AgentState, or the API.

--------------------------------------------------------------------------
Dependency direction (Part 15/16/24)
--------------------------------------------------------------------------
    SemanticMemoryStore   EmbeddingProvider   VectorIndex  (this module)
           \\                    |                    /
            \\___________________|___________________/
                                 v
                         future MemoryRetriever

This module imports NOTHING from app.agent.semantic_memory,
app.agent.episodic_memory, app.agent.memory, or app.agent.embeddings.
Vectors are represented by a Vector alias defined LOCALLY in this file
(see below) rather than imported from app.agent.embeddings — even though
the two aliases are structurally identical (`tuple[float, ...]`), keeping
zero imports between the vector-mechanics layer and the embedding-model
layer keeps VectorIndex genuinely independent of "how a vector was
produced," exactly as Part 16 recommends. A future MemoryRetriever is the
right place to compose "EmbeddingProvider produces this" with "VectorIndex
stores/searches this" — not this module.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

Vector = tuple[float, ...]
"""Deliberately re-declared here rather than imported from
app.agent.embeddings — see "Dependency direction" above."""


def _require_non_empty_str(field_name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")


def _validate_numeric_vector(vector: object, *, label: str) -> tuple[float, ...]:
    """Shared validation for any vector-shaped input: a non-empty
    tuple/list of finite, non-bool numbers. Used by both
    `cosine_similarity` and `InMemoryVectorIndex` so the two never drift
    into slightly different notions of "a valid vector"."""
    if not isinstance(vector, (tuple, list)):
        raise ValueError(f"{label} must be a tuple or list of numbers.")
    if len(vector) == 0:
        raise ValueError(f"{label} must not be empty.")
    values: list[float] = []
    for component in vector:
        if isinstance(component, bool) or not isinstance(component, (int, float)):
            raise ValueError(f"{label} components must be numbers.")
        if not math.isfinite(component):
            raise ValueError(f"{label} components must be finite (no NaN/inf).")
        values.append(float(component))
    return tuple(values)


def cosine_similarity(a: Vector, b: Vector) -> float:
    """cosine(a, b) = (a . b) / (||a|| * ||b||), clamped to [-1.0, 1.0].

    Design decision (Part 2) — the full formula is always computed,
    rather than either of the two cheaper-looking shortcuts:

    - Option A ("trust normalized vectors," i.e. skip the norm terms and
      return the bare dot product) was REJECTED: `EmbeddingProvider`
      (Step 16B) guarantees unit-normalized output, but this function has
      no way to verify that guarantee was honored by whatever produced
      `a`/`b` — a future real provider with a subtle normalization bug, or
      a raw vector reaching this function some other way, would silently
      produce a wrong, unbounded "similarity" instead of a safe error.
    - Option C (forcibly re-normalize `a` and `b` into new vectors before
      comparing) was REJECTED as wasted work for no extra safety: it
      allocates two new vectors just to arrive at the exact same division
      this function already performs inline.
    - Option D was chosen: compute norms as two scalars and divide. This
      is the SAME asymptotic cost as a bare dot product (one more O(n)
      pass, no new vector allocation) — so there is no meaningful
      "duplicate normalization" overhead being avoided by skipping it, and
      the function is correct whether or not its inputs happen to already
      be unit vectors.

    Validation (still required even though Step 16B's provider guarantees
    normalization — Part 2's explicit list): both vectors must be
    non-empty, the same length (dimension match), contain only finite
    numbers, and have non-zero norm — cosine similarity is mathematically
    undefined for a zero vector, so that raises rather than producing a
    NaN or ZeroDivisionError.

    Numerical precision (Part 3): the mathematical range of cosine
    similarity is exactly [-1.0, 1.0]. Floating-point summation can push
    the raw result a few ULPs outside that range (e.g. 1.0000000000000002)
    even for perfectly valid, correctly-computed inputs — this is a known
    artifact of the arithmetic, not a sign of a logic error, so the result
    is unconditionally clamped back into [-1.0, 1.0]. This does NOT hide
    real errors: the inputs are already validated above, so nothing that
    reaches the clamp could be masking a dimension mismatch, a non-finite
    component, or a zero-norm vector — those are rejected before the
    division ever runs.
    """
    values_a = _validate_numeric_vector(a, label="a")
    values_b = _validate_numeric_vector(b, label="b")
    if len(values_a) != len(values_b):
        raise ValueError(f"vector dimension mismatch: len(a)={len(values_a)} != len(b)={len(values_b)}.")

    dot = sum(x * y for x, y in zip(values_a, values_b))
    norm_a = math.sqrt(sum(x * x for x in values_a))
    norm_b = math.sqrt(sum(y * y for y in values_b))
    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("cosine similarity is undefined for a zero-norm vector.")

    raw = dot / (norm_a * norm_b)
    return max(-1.0, min(1.0, raw))


@dataclass(frozen=True)
class VectorSearchResult:
    """One scored search hit: which memory, how similar.

    `session_id` is deliberately NOT included (Part 5): the caller
    supplied the session_id to `search()` in the first place, every result
    it gets back is structurally guaranteed to belong to that same session
    (see InMemoryVectorIndex's session-isolation design below), and
    echoing it back on every result would be redundant information with
    no debugging or security value here — the boundary is enforced by
    construction, not by something a caller needs to re-check per result.

    Immutable (frozen dataclass), matching every other record type in this
    codebase (EpisodicMemoryRecord, SemanticMemoryRecord).
    """

    memory_id: str
    similarity: float

    def __post_init__(self) -> None:
        _require_non_empty_str("memory_id", self.memory_id)
        if isinstance(self.similarity, bool) or not isinstance(self.similarity, (int, float)):
            raise ValueError("similarity must be a number.")
        if not math.isfinite(self.similarity):
            raise ValueError("similarity must be finite.")
        if not (-1.0 <= float(self.similarity) <= 1.0):
            raise ValueError("similarity must be within [-1.0, 1.0].")
        object.__setattr__(self, "similarity", float(self.similarity))


@runtime_checkable
class VectorIndex(Protocol):
    """Stores vectors keyed by (session_id, memory_id) and searches them.

    Deliberately minimal (Part 4): add / remove / search, plus a
    `dimension` property. No `clear()` — nothing in this step's scope asks
    for "wipe an entire session's vectors" as a distinct operation, and
    `remove()` by memory_id already covers every tested lifecycle need;
    adding an unused method now would be exactly the kind of premature
    surface this codebase has consistently avoided.

    The index stores ONLY memory_id, session_id, and a vector — it knows
    nothing about SemanticMemoryRecord, episodic memory, conversations,
    LLMs, or prompts (Part 15). That separation is what lets a future
    `MemoryRetriever` compose `SemanticMemoryStore` (facts) +
    `EmbeddingProvider` (text -> vector) + `VectorIndex` (vector ->
    similarity search) without any of the three depending on either of
    the other two.
    """

    @property
    def dimension(self) -> int:
        """The fixed vector length this index accepts and searches
        against. Configured explicitly at construction (Part 12) — never
        inferred from the first vector added, and NOT tied to
        DeterministicEmbeddingProvider's own (unrelated) test dimension. A
        real embedding model will have its own, likely much larger,
        dimension (e.g. 384 for all-MiniLM-L6-v2); this property is what a
        future retrieval layer checks before ever comparing a query vector
        against this index.
        """
        ...

    def add(self, memory_id: str, session_id: str, vector: Vector) -> None:
        ...

    def remove(self, memory_id: str, session_id: str) -> None:
        ...

    def search(self, session_id: str, query_vector: Vector, top_k: int) -> list[VectorSearchResult]:
        ...


class InMemoryVectorIndex:
    """The only implementation for now: a plain in-process dict of
    session_id -> dict[memory_id, Vector].

    --------------------------------------------------------------------
    Session isolation (Part 8 — CRITICAL, both correctness and security)
    --------------------------------------------------------------------
    Vectors are partitioned STRUCTURALLY by session: `search(session_id,
    ...)` only ever iterates `self._sessions[normalize(session_id)]` — it
    is architecturally incapable of even looking at another session's
    dict, let alone returning one of its vectors. This is "retrieve from
    the correct partition," never "retrieve globally, then filter" — the
    latter was explicitly rejected because a filter is something that can
    be forgotten or buggy, while indexing into the right partition simply
    cannot leak the wrong session's data by construction. A future
    pgvector-backed implementation preserves this invariant with a
    mandatory `WHERE session_id = ...` (or, more robustly, a `session_id`
    column that is *part of the physical partition/index*), not an
    application-level filter applied after a broader query.

    --------------------------------------------------------------------
    memory_id uniqueness (Part 9) and update/replacement (Part 10)
    --------------------------------------------------------------------
    The identity key here is the PAIR (session_id, memory_id), NOT
    memory_id alone. This is a deliberate, DIFFERENT decision from
    SemanticMemoryStore's (Step 16A), which enforces memory_id uniqueness
    GLOBALLY — and the difference is justified, not an inconsistency:
    SemanticMemoryStore.get(memory_id) takes no session_id, so global
    uniqueness is the only way `get()` can have an unambiguous answer.
    VectorIndex has no such id-only lookup — `search`/`remove` both
    require a session_id — so nothing here forces global uniqueness, and
    this index must not assume SemanticMemoryStore's guarantee anyway
    (Part 15: no dependency between them). Concretely:
    - same memory_id + same session -> `add()` REJECTS (raises
      ValueError). The index must never silently hold two vectors for
      "the same" logical memory (Part 10) — a caller who wants to replace
      a vector must call `remove()` first, then `add()` again. This was
      chosen over silently replacing the old vector (a real alternative)
      because every other identity-collision decision in this codebase
      (SemanticMemoryStore's memory_id, SessionMemoryStore's session
      creation) favors an explicit reject over a silent overwrite that
      could mask a caller bug — consistency across the codebase's memory
      layers was weighed above the minor convenience of an implicit
      upsert.
    - same memory_id + DIFFERENT session -> fully ALLOWED. The two
      entries are entirely independent rows under this key model, exactly
      the shape a pgvector table with a composite (session_id, memory_id)
      unique index would naturally have for the same multi-tenant
      partitioning reason.

    --------------------------------------------------------------------
    remove() semantics (Part 11)
    --------------------------------------------------------------------
    `remove(memory_id, session_id)` is idempotent and safe: removing a
    vector that exists deletes it; removing a memory_id that was never
    added, or was added under a DIFFERENT session_id, is a silent no-op
    (matches this codebase's established "clearing something that was
    never there is not an error" convention — see EpisodicMemory.clear /
    SemanticMemoryStore.clear). Removing "from the wrong session" can
    therefore never delete another session's real entry — there is no
    global lookup step where that could happen; the lookup only ever
    touches the named session's own partition.

    --------------------------------------------------------------------
    Dimension contract (Part 12)
    --------------------------------------------------------------------
    `dimension` is a required constructor argument, validated `>= 1`, and
    is checked against EVERY vector passed to `add()` or `search()` —
    mismatches raise ValueError before any similarity computation runs.

    --------------------------------------------------------------------
    Determinism and tie-breaking (Part 7)
    --------------------------------------------------------------------
    `search()` results are sorted by similarity DESCENDING, then by
    `memory_id` ASCENDING as an explicit, stable tie-breaker — never left
    to rely on dict/insertion order (which Python dicts happen to
    preserve, but that is an implementation detail this code does not
    lean on).

    --------------------------------------------------------------------
    Complexity (Part 18/19)
    --------------------------------------------------------------------
    Exact brute-force search: O(N) cosine-similarity computations per
    query, where N is the number of vectors in the SEARCHED SESSION's
    partition only (not the total across all sessions — a direct benefit
    of structural partitioning, not a separate optimization). This is
    intentionally NOT approximate — no ANN, no trees, no clustering, no
    caching. It is a correctness-first reference implementation, meant to
    be replaced behind the same `VectorIndex` Protocol by a
    PostgreSQL + pgvector implementation once real scale requires it,
    without the retrieval contract above it changing.

    --------------------------------------------------------------------
    Immutability (Part 14)
    --------------------------------------------------------------------
    Every vector is converted to a real `tuple[float, ...]` before being
    stored, regardless of whether the caller passed a list or a tuple —
    so mutating the caller's original list afterward can never change
    what is stored. `VectorSearchResult` is a frozen dataclass, so
    returned results can't be mutated either.
    """

    def __init__(self, dimension: int):
        if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 1:
            raise ValueError("dimension must be an integer >= 1.")
        self._dimension = dimension
        self._sessions: dict[str, dict[str, Vector]] = {}

    @property
    def dimension(self) -> int:
        return self._dimension

    def add(self, memory_id: str, session_id: str, vector: Vector) -> None:
        _require_non_empty_str("memory_id", memory_id)
        validated_vector = self._validate_vector(vector)
        session_key = self._normalize(session_id)

        session_vectors = self._sessions.setdefault(session_key, {})
        if memory_id in session_vectors:
            raise ValueError(
                f"memory_id {memory_id!r} already exists in session {session_id!r}; call remove() first to replace it."
            )
        session_vectors[memory_id] = validated_vector

    def remove(self, memory_id: str, session_id: str) -> None:
        _require_non_empty_str("memory_id", memory_id)
        session_key = self._normalize(session_id)
        session_vectors = self._sessions.get(session_key)
        if session_vectors is not None:
            session_vectors.pop(memory_id, None)

    def search(self, session_id: str, query_vector: Vector, top_k: int) -> list[VectorSearchResult]:
        session_key = self._normalize(session_id)
        validated_query = self._validate_vector(query_vector)
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be an integer >= 1.")

        session_vectors = self._sessions.get(session_key, {})
        scored = [
            (memory_id, cosine_similarity(validated_query, vector))
            for memory_id, vector in session_vectors.items()
        ]
        scored.sort(key=lambda item: (-item[1], item[0]))  # similarity DESC, memory_id ASC
        return [VectorSearchResult(memory_id=memory_id, similarity=score) for memory_id, score in scored[:top_k]]

    def _validate_vector(self, vector: object) -> Vector:
        values = _validate_numeric_vector(vector, label="vector")
        if len(values) != self._dimension:
            raise ValueError(f"vector dimension {len(values)} does not match index dimension {self._dimension}.")
        return values

    def _normalize(self, session_id: str) -> str:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string.")
        return session_id.strip()
