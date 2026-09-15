"""Memory retrieval (Step 16D) — the component that COMPOSES the three
independent pieces built in 16A/16B/16C into one usable read path.

    query text
        |  EmbeddingProvider.embed(query)          (16B)
        v
    query vector
        |  VectorIndex.search(session_id, ..., k)  (16C)
        v
    VectorSearchResult[]  (memory_id + similarity, no content)
        |  SemanticMemoryStore.get(memory_id)      (16A)
        v
    RetrievedMemory[]     (the actual record + its similarity)

Why this module has to exist: a VectorIndex deliberately knows only
memory_id and similarity — it stores no content (16C). A
SemanticMemoryStore deliberately knows only facts — it stores no vectors
(16A). Neither one can answer "which of this session's remembered facts
are most similar to this query?" on its own, and neither should be
extended to, because that would collapse the separation that lets each be
swapped independently (a real embedding model for the deterministic one,
pgvector for the in-memory index). The bridge belongs in a third
component that depends on all three ABSTRACTIONS and none of their
concrete implementations — that component is this one.

--------------------------------------------------------------------------
What this component owns
--------------------------------------------------------------------------
- Validating retrieval inputs (session_id, query, top_k).
- Embedding the QUERY (and only the query — see the read/write split
  below).
- Delegating the actual search to a session-scoped VectorIndex.
- Resolving the returned memory_ids back into SemanticMemoryRecords.
- Enforcing, as defense in depth, that every record it returns really
  does belong to the requested session.
- Dropping index entries that no longer resolve to a stored record, and
  records explicitly marked inactive.

--------------------------------------------------------------------------
What it deliberately does NOT own (later, separately scoped milestones)
--------------------------------------------------------------------------
- Creating semantic memories, or extracting facts from episodes. Nothing
  here writes: `retrieve()` is a pure read.
- Populating the vector index. This module assumes the index is ALREADY
  populated — see the read/write split below.
- Any threshold beyond the optional `min_similarity` added in Step 16F-E.
  That one defaults to -1.0 (no filtering), deliberately: a useful cut-off
  is a property of the embedding model in use, and the only provider that
  exists is the deterministic test one, whose similarities carry no
  semantic meaning. Picking a non-zero default now would be a magic
  constant calibrated against nothing, so it must be set explicitly and
  recalibrated whenever the model changes.
- Any ranking beyond the VectorIndex's similarity ordering. The record's
  `created_at` and `confidence` fields are deliberately NOT combined with
  similarity here — recency/confidence weighting is future work, and
  implementing it now would bake a scoring formula in before there is a
  real embedding model to calibrate it against.
- Conflict resolution / supersession. This module READS the `active`
  flag (see SemanticMemoryRetriever's docstring) but never sets it, and
  implements no notion of one fact superseding another.
- Injecting anything into a prompt, touching AgentState, calling an LLM,
  or running a tool. There is no LLM call anywhere in this module, and no
  import from the agent's execution path — retrieval is fully
  deterministic given its three collaborators.
- Persistence, authentication, or user-level identity.

--------------------------------------------------------------------------
Read path vs write path
--------------------------------------------------------------------------
    WRITE (future milestone):   memory content -> embed -> VectorIndex.add
    READ  (this module):        query -> embed -> search -> store lookup

`retrieve()` embeds exactly one piece of text: the query. It never embeds
stored memories — those vectors must already be in the index, put there
by whatever future component owns the write path. Embedding stored
memories at read time would be both wasteful (re-deriving on every query
what should be computed once at write time) and wrong (the index, not a
freshly computed vector, is the authority on what is actually searchable).

--------------------------------------------------------------------------
Current limitation worth stating plainly
--------------------------------------------------------------------------
With the only EmbeddingProvider that exists today
(DeterministicEmbeddingProvider, 16B), similarity scores carry NO semantic
meaning: it hashes text with SHA-256, so "I prefer Python" and "what
language do I prefer?" are no more similar to each other than to anything
else. This module's retrieval PLUMBING is correct and fully testable
today; its RESULTS only become meaningful once a real embedding model is
substituted behind the same EmbeddingProvider Protocol. Nothing in this
module needs to change when that happens — which is the entire point of
depending on the Protocol rather than the implementation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.agent.embeddings import EmbeddingProvider
from app.agent.semantic_memory import SemanticMemoryRecord, SemanticMemoryStore
from app.agent.vector_index import VectorIndex

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5


@dataclass(frozen=True)
class RetrievedMemory:
    """One retrieved fact together with how similar it was to the query.

    Deliberately just two fields. No `final_score`, `recency_score`,
    `rerank_score`, `distance`, `token_count`, or `embedding_model` — each
    of those belongs to a ranking/threshold milestone that does not exist
    yet, and adding a field before the thing that computes it exists would
    mean shipping a value that is always None or always a duplicate of
    `similarity`.

    `similarity` is passed through UNCHANGED from the VectorIndex's
    VectorSearchResult — this module never rescales, re-weights, or
    recomputes it.
    """

    memory: SemanticMemoryRecord
    similarity: float

    def __post_init__(self) -> None:
        if not isinstance(self.memory, SemanticMemoryRecord):
            raise ValueError("memory must be a SemanticMemoryRecord.")
        if isinstance(self.similarity, bool) or not isinstance(self.similarity, (int, float)):
            raise ValueError("similarity must be a number.")
        if not (-1.0 <= float(self.similarity) <= 1.0):
            raise ValueError("similarity must be within [-1.0, 1.0].")
        object.__setattr__(self, "similarity", float(self.similarity))


@runtime_checkable
class MemoryRetriever(Protocol):
    """Answers: "which of this session's remembered facts are most
    similar to this query?"

    Deliberately one method. A future implementation backed by a real
    vector database satisfies this same Protocol without anything above it
    changing — which is why the signature mentions no index, no embedding
    model, and no storage detail.
    """

    def retrieve(self, session_id: str, query: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievedMemory]:
        ...


class SemanticMemoryRetriever:
    """The only implementation for now: composes a SemanticMemoryStore, an
    EmbeddingProvider, and a VectorIndex.

    Named for WHAT it retrieves rather than where things live — unlike
    InMemorySemanticMemory / InMemoryVectorIndex / InMemoryConversationMemory,
    this class stores nothing at all. It holds no dict, no list, no state
    beyond its three injected collaborators, so an `InMemory...` prefix
    would claim something untrue about it. Whether retrieval is in-memory
    or database-backed is entirely a property of the collaborators passed
    in, not of this class.

    --------------------------------------------------------------------
    Dependency injection (all three are Protocols, none are concretes)
    --------------------------------------------------------------------
    Constructor arguments are typed against SemanticMemoryStore,
    EmbeddingProvider, and VectorIndex — never against
    InMemorySemanticMemory, DeterministicEmbeddingProvider, or
    InMemoryVectorIndex. This class works with those three today and with
    a real embedding model and a pgvector-backed index tomorrow, with no
    change here. All three are required (there is no useful "retrieve with
    no index" degenerate mode, so none default to None — unlike the
    OPT-IN collaborators on AgentOrchestrator, which default to None
    precisely because the orchestrator is useful without them).

    --------------------------------------------------------------------
    Dimension compatibility is checked EAGERLY, at construction
    --------------------------------------------------------------------
    Both collaborators expose `dimension`, so a mismatch between the
    embedding provider and the vector index is detectable at wiring time.
    It is rejected there rather than being allowed to surface as a failure
    on the first `retrieve()` call: a provider/index dimension mismatch
    breaks EVERY retrieval without exception, so it is a wiring bug, and
    reporting it at the moment of miswiring is far clearer than reporting
    it later at an arbitrary query.

    --------------------------------------------------------------------
    Three ways a hit can fail to become a result (all deliberate)
    --------------------------------------------------------------------
    1. STALE INDEX ENTRY — the index returned a memory_id that the
       semantic store no longer has (e.g. the record was cleared but its
       vector was never removed). Such a hit is SKIPPED and logged at
       WARNING; it does not fail the whole retrieval. Rationale: this is
       benign, recoverable divergence between two independent stores, and
       one dangling vector should not make an otherwise good retrieval
       unusable. This module deliberately does NOT repair the index in
       response (no `vector_index.remove(...)` call here) — `retrieve()`
       is a pure read, and silently mutating an index during a read would
       be a surprising side effect; reconciliation belongs to whatever
       future component owns the write path.

    2. SESSION MISMATCH — the record resolved from a hit claims a
       DIFFERENT session_id than the one being searched. This RAISES
       rather than skipping, deliberately asymmetric to case 1. A hit came
       out of session A's partition, so a record claiming session B means
       the vector index and the semantic store disagree about ownership.
       That is never benign staleness: returning it would leak one
       session's fact into another's retrieval, across a boundary the
       architecture treats as non-negotiable. Skipping would hide the
       disagreement indefinitely; failing loudly surfaces it at once, and
       ordinary staleness cannot trigger it (a deleted record produces
       case 1, not this).

    3. INACTIVE RECORD — `record.active is False`. Such records are
       EXCLUDED. `active` already has exactly one documented meaning in
       the data model (16A: a fact superseded by a newer one), so honoring
       it is respecting an existing contract rather than inventing a
       lifecycle policy — and it is a VALIDITY filter, categorically
       different from the relevance thresholds and ranking signals this
       step is explicitly not implementing. The alternative (return them
       and let a future policy layer filter) was rejected because it makes
       "we returned a fact we already know is no longer true" the default
       behavior, which would have to be corrected as a behavior CHANGE
       once supersession lands rather than as a pure addition. Nothing in
       the codebase sets `active=False` yet, so this filter is currently
       unreachable in practice — it is tested by constructing an inactive
       record directly.

    Because any of the three can drop a hit, `retrieve()` may return FEWER
    than top_k results even when the index found top_k matches. Results
    are never padded to reach top_k, consistent with the VectorIndex's own
    contract.

    --------------------------------------------------------------------
    Session isolation and ordering
    --------------------------------------------------------------------
    Isolation comes from the VectorIndex's structural partitioning:
    `search(session_id, ...)` only ever examines that session's own
    partition, so another session's vectors are never even candidates —
    this is NOT "search globally, then filter." The per-record session
    check in case 2 above is an additional, independent assertion on top
    of that, not the primary mechanism.

    Ordering is the VectorIndex's, preserved exactly: hits are processed
    in the order returned (similarity descending, memory_id ascending as
    the tie-break) and never re-sorted here. Dropping a hit leaves the
    relative order of the rest untouched.
    """

    def __init__(
        self,
        semantic_memory: SemanticMemoryStore,
        embedding_provider: EmbeddingProvider,
        vector_index: VectorIndex,
        # -1.0, not 0.0, is the true "no filtering" value: cosine
        # similarity ranges [-1.0, 1.0], so a 0.0 floor would silently
        # discard every orthogonal-or-worse match rather than being a
        # no-op.
        min_similarity: float = -1.0,
    ):
        if not isinstance(semantic_memory, SemanticMemoryStore):
            raise ValueError("semantic_memory must implement the SemanticMemoryStore protocol.")
        if not isinstance(embedding_provider, EmbeddingProvider):
            raise ValueError("embedding_provider must implement the EmbeddingProvider protocol.")
        if not isinstance(vector_index, VectorIndex):
            raise ValueError("vector_index must implement the VectorIndex protocol.")
        if embedding_provider.dimension != vector_index.dimension:
            raise ValueError(
                f"embedding provider dimension {embedding_provider.dimension} does not match "
                f"vector index dimension {vector_index.dimension}."
            )

        if isinstance(min_similarity, bool) or not isinstance(min_similarity, (int, float)):
            raise ValueError("min_similarity must be a number.")
        if not (-1.0 <= float(min_similarity) <= 1.0):
            raise ValueError("min_similarity must be within [-1.0, 1.0].")

        self.semantic_memory = semantic_memory
        self.embedding_provider = embedding_provider
        self.vector_index = vector_index
        self.min_similarity = float(min_similarity)

    def retrieve(self, session_id: str, query: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievedMemory]:
        session_key = self._normalize(session_id)
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be an integer >= 1.")

        # The query — and ONLY the query — is embedded here. Stored memory
        # vectors are the index's business (see the module's read/write
        # split). Vector validity (dimension, finiteness) is NOT re-checked
        # here: VectorIndex.search already validates its query_vector and
        # raises a clear ValueError, so duplicating that check would give
        # two places to keep in sync for no extra safety — the same
        # reasoning that keeps session_id validation in exactly one place
        # (the store) rather than repeated up the call chain.
        query_vector = self.embedding_provider.embed(query.strip())
        hits = self.vector_index.search(session_key, query_vector, top_k)

        retrieved: list[RetrievedMemory] = []
        for hit in hits:
            # Threshold applied HERE rather than inside VectorIndex (Step
            # 16F-E): the index's job is "nearest K in this partition",
            # and what counts as "relevant enough to show a model" is a
            # retrieval-policy question that differs per embedding model.
            # Filtering before the store lookup also avoids resolving
            # records that were never going to be returned.
            if hit.similarity < self.min_similarity:
                continue

            record = self.semantic_memory.get(hit.memory_id)

            if record is None:
                logger.warning(
                    "Vector index returned memory_id %r for session %r, but no such semantic memory exists "
                    "— skipping stale index entry.",
                    hit.memory_id,
                    session_key,
                )
                continue

            if record.session_id.strip() != session_key:
                raise ValueError(
                    f"session isolation violation: vector index returned memory_id {hit.memory_id!r} "
                    f"for session {session_key!r}, but the stored record belongs to session "
                    f"{record.session_id!r}."
                )

            if not record.active:
                continue

            retrieved.append(RetrievedMemory(memory=record, similarity=hit.similarity))

        return retrieved

    def _normalize(self, session_id: str) -> str:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string.")
        return session_id.strip()
