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
- Any RELEVANCE threshold beyond the optional `min_similarity` added in
  Step 16F-E. That one defaults to -1.0 (no filtering), deliberately: a
  useful cut-off is a property of the embedding model in use, and the only
  provider that exists is the deterministic test one, whose similarities
  carry no semantic meaning. Picking a non-zero default now would be a
  magic constant calibrated against nothing, so it must be set explicitly
  and recalibrated whenever the model changes. Step 16G reviewed this and
  changed nothing about it — the default stays the true no-op.

  The SAFETY bounds added in 16G (`max_top_k`, `max_context_chars`) are a
  different kind of thing and do have real defaults, for a reason worth
  stating: a relevance threshold encodes a judgment about an embedding
  model nobody has calibrated yet, whereas a size bound encodes only "a
  prompt must be finite", which is true of every model that will ever sit
  behind this. One must be opted into; the other must be impossible to opt
  out of.
- Any ranking beyond the VectorIndex's similarity ordering. Cosine
  similarity remains the SOLE ordering signal, re-affirmed in 16G:

  * `confidence` stays METADATA. It is not a calibrated probability (see
    SemanticMemoryRecord), so multiplying it into a similarity score would
    produce a number with no meaning in either unit. The place a
    confidence policy legitimately lives is the WRITE path, where
    SemanticMemoryWriter already drops candidates below `min_confidence`
    (16F-D) at the one moment the value is actually decided; a second
    floor here would be the same policy in two places, free to drift
    apart, filtering a value that cannot change after the write.
  * `created_at` stays METADATA. A newer fact is not a more RELEVANT one,
    and blending recency into the score would quietly answer "what
    changed most recently?" when the caller asked "what is most similar?".
    Where recency genuinely matters — two remembered facts disagreeing —
    it is already served correctly: 16E-B renders each memory's date, so
    the MODEL weighs currency with the facts in front of it, rather than
    this layer silently deciding the question by reordering.
- Read-time duplicate suppression. 16F-A handles duplicates at WRITE time,
  merging a candidate at or above `duplicate_threshold` into the existing
  record and keeping one canonical fact with merged provenance. That is
  strictly the better place: it is decided once, with the full candidate in
  hand, and it leaves the store holding what it claims to hold. A
  read-time pass would have to re-decide it on every query, would have to
  pick a loser among records the store considers equally real, and — with
  identical content producing an identical vector, hence similarity 1.0 —
  would only ever fire on duplicates the write path already merges. It
  would therefore add a way to drop a distinct-but-similar fact while
  fixing nothing. Deliberately absent.
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
from app.agent.semantic_memory import (
    MemorySessionIsolationError,
    SemanticMemoryRecord,
    SemanticMemoryStore,
)
from app.agent.vector_index import VectorIndex

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5

# Hard ceiling on what a caller may ask for (Step 16G). `top_k` was
# previously validated only as ">= 1", so `retrieve(s, q, top_k=100_000)`
# was accepted and would have put every fact a session owns into a prompt.
# 20 is well above any plausible legitimate request (the agent asks for 5)
# and far below anything that could blow a context window.
#
# Exceeding it RAISES rather than being clamped down to the ceiling:
# silently serving 20 results to a caller who asked for 100 would hide a
# configuration bug behind behavior that looks like it worked.
DEFAULT_MAX_TOP_K = 20

# Total budget, in CHARACTERS of memory content, for one retrieval's
# results (Step 16G).
#
# Characters, not tokens, and this is an APPROXIMATION of a token budget,
# not a measurement of one. At the usual rough English heuristic of ~4
# characters per token, 4000 characters is on the order of 1000 tokens. A
# real tokenizer would be exact, but it would mean a model-specific
# dependency (tiktoken/transformers) in the retrieval layer for a bound
# whose only job is to stop an unbounded prompt — a character count does
# that with no dependency at all, and errs on the safe side because no
# tokenizer emits MORE than one token per character. Swapping in a
# tokenizer later changes only the accounting inside `retrieve`, not this
# module's contract.
#
# Why a budget is needed even with `max_top_k` in place: nothing bounds
# `SemanticMemoryRecord.content`. The 300-character cap lives on
# `MemoryCandidate` (16F), so it constrains only facts that came through
# an extractor — a record written straight into a SemanticMemoryStore can
# be any size. top_k alone therefore bounds the NUMBER of memories but not
# the SIZE of the block they render into; both bounds together are what
# make the memory section of the prompt provably finite.
DEFAULT_MAX_CONTEXT_CHARS = 4000


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
    contract. Step 16G adds a fourth, later reason (the character budget
    — see `_apply_context_budget`), which differs from all three above in
    that it drops a SUFFIX of otherwise-valid results rather than
    individual hits.

    --------------------------------------------------------------------
    Bounds (Step 16G) — why retrieval, not the index, owns them
    --------------------------------------------------------------------
    `max_top_k` and `max_context_chars` are enforced here rather than in
    VectorIndex for the same reason `min_similarity` is: the index's job
    is "the nearest K vectors in this partition", a question with a
    mathematically correct answer for any K. "How much of that is safe to
    put in front of a model" is a RETRIEVAL POLICY question — it depends
    on the consumer, not on the geometry — and this class is where that
    policy already lives. Capping the index instead would also make it
    impossible to ask the index a large diagnostic query (tests do), which
    is a legitimate use with no prompt attached.

    Together they make the memory block finite BY CONSTRUCTION rather than
    by convention: at most `max_top_k` records, totalling at most
    `max_context_chars` characters of content, with no configuration —
    accidental or deliberate — that removes either bound.

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
        max_top_k: int = DEFAULT_MAX_TOP_K,
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
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

        # Both bounds are CONFIGURATION, so they are validated eagerly here
        # rather than on the first retrieve() — same reasoning as the
        # dimension check above: a bad bound breaks every retrieval without
        # exception, so it is a wiring bug and belongs at wiring time.
        # Neither accepts None: "no limit" is not an option this class
        # offers, because the whole point of Step 16G is that there is no
        # configuration, accidental or deliberate, that produces an
        # unbounded prompt.
        if not isinstance(max_top_k, int) or isinstance(max_top_k, bool) or max_top_k < 1:
            raise ValueError("max_top_k must be an integer >= 1.")
        if (
            not isinstance(max_context_chars, int)
            or isinstance(max_context_chars, bool)
            or max_context_chars < 1
        ):
            raise ValueError("max_context_chars must be an integer >= 1.")

        self.semantic_memory = semantic_memory
        self.embedding_provider = embedding_provider
        self.vector_index = vector_index
        self.min_similarity = float(min_similarity)
        self.max_top_k = max_top_k
        self.max_context_chars = max_context_chars

    def retrieve(self, session_id: str, query: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievedMemory]:
        session_key = self._normalize(session_id)
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be an integer >= 1.")
        if top_k > self.max_top_k:
            raise ValueError(
                f"top_k {top_k} exceeds this retriever's max_top_k of {self.max_top_k}."
            )

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
                raise MemorySessionIsolationError(
                    f"session isolation violation: vector index returned memory_id {hit.memory_id!r} "
                    f"for session {session_key!r}, but the stored record belongs to session "
                    f"{record.session_id!r}."
                )

            if not record.active:
                continue

            retrieved.append(RetrievedMemory(memory=record, similarity=hit.similarity))

        return self._apply_context_budget(retrieved, session_key)

    def _apply_context_budget(
        self, retrieved: list[RetrievedMemory], session_key: str
    ) -> list[RetrievedMemory]:
        """Truncate the result list so its total content stays within
        `max_context_chars`.

        ONE rule, applied in the order results already have (similarity
        DESC): accumulate until an entry would push the running total past
        the budget, then stop and drop it and everything after it. The
        result is always a PREFIX of the unbudgeted result list, which is
        what makes the outcome explainable ("you got the best N that fit")
        and keeps the existing ordering guarantee intact — nothing is
        re-ordered, and nothing later is promoted over something earlier.

        The alternative — skipping an entry that does not fit and
        continuing to look for smaller ones further down — was rejected.
        It is a bin-packing policy dressed up as retrieval: it would let a
        weaker match outrank a stronger one purely because it was shorter,
        which is a ranking decision made on a signal (length) that has
        nothing to do with relevance.

        Consequence, stated plainly: a single record whose content alone
        exceeds the entire budget yields an EMPTY result. That is the
        honest outcome — it genuinely cannot be included without breaking
        the bound — and it is logged at WARNING so it is diagnosable
        rather than mysterious. It also cannot arise through the supported
        write path, where `MemoryCandidate` already caps content at 300
        characters (16F).

        Only `content` is counted. This layer deliberately knows nothing
        about how memories are later rendered (16E-B's JSON block, or
        anything that replaces it), so it budgets the only thing that is
        genuinely its own: the text it is handing over. The rendering
        layer adds a small FIXED overhead per entry, and the number of
        entries is itself bounded by `max_top_k`, so the rendered block is
        bounded by `max_context_chars + max_top_k * overhead` — finite by
        construction, without this module having to model the formatter.

        Truncation never touches a record's text: entries are dropped
        whole. Cutting a fact off mid-sentence would hand the model a
        mutilated claim it has no way to recognize as incomplete, and
        would break the guarantee that a returned `SemanticMemoryRecord`
        is exactly what is stored, provenance included.
        """
        budgeted: list[RetrievedMemory] = []
        used = 0
        for index, entry in enumerate(retrieved):
            cost = len(entry.memory.content)
            if used + cost > self.max_context_chars:
                logger.warning(
                    "Memory context budget of %d characters reached for session %r after %d of %d "
                    "result(s); dropping the remaining %d lower-similarity result(s).",
                    self.max_context_chars,
                    session_key,
                    index,
                    len(retrieved),
                    len(retrieved) - index,
                )
                break
            used += cost
            budgeted.append(entry)

        return budgeted

    def _normalize(self, session_id: str) -> str:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string.")
        return session_id.strip()
