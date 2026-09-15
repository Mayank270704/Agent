"""Memory context contract (Step 16E-A) — the boundary between the
RETRIEVAL layer and a future AGENT/LLM context layer.

    MemoryRetriever -> RetrievedMemory[] -> [this contract] -> future
    agent context layer -> future LLM prompt

Why this exists rather than passing RetrievedMemory straight through:
`RetrievedMemory` carries the whole `SemanticMemoryRecord`, which is a
STORAGE type. That record includes `active` (a supersession-lifecycle
flag), `source_event_ids` (a coupling into the episodic layer), and
`session_id` (a scope key). None of those are things an agent-context
layer needs, and handing the full record downstream means every future
consumer can reach into storage internals. This module is a deliberate
NARROWING PROJECTION: it answers "what memory information is available to
the agent?" and deliberately cannot answer "how was it retrieved?".

Exposed:  memory_id, content, similarity, created_at, confidence
Not exposed: vectors, embeddings, the embedding provider, the vector
index, storage dictionaries, `SemanticMemoryRecord` itself,
`source_event_ids`, `active`, or any other implementation object.

This module performs NO retrieval. It does not call EmbeddingProvider,
VectorIndex, SemanticMemoryStore, or an LLM; it does not touch AgentState
and it does not create memories. It is pure data plus one adapter.

Prompt formatting is deliberately NOT here (Step 16E-A's explicit
boundary). There are no template strings, no XML tags, no "Relevant
memories:" preamble, and no instructions to the model about how to treat
memories. This layer produces STRUCTURED APPLICATION DATA; turning that
into prompt text — including the security-critical framing of memories as
data rather than instructions — is a separate, later milestone, and
keeping the two apart means the rendering decision can change without
touching the contract (and can be tested independently of it).

The live agent remains untouched by this module: nothing in
AgentOrchestrator, ChatService, AgentState, or the API imports it. Like
16A-16D, this is a foundation milestone that is deliberately connected to
nothing yet.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from app.agent.memory_retriever import RetrievedMemory


def _require_non_empty_str(field_name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")


@dataclass(frozen=True)
class MemoryContextItem:
    """One remembered fact, projected down to what an agent-context layer
    legitimately needs.

    Why each field is here:

    - `content`: the fact itself. Without it there is nothing to give a
      model — this is the only field whose text is ever expected to reach
      a prompt.
    - `similarity`: the reason this item is in the context at all, passed
      through unchanged from retrieval. A future threshold/ranking layer
      consumes it directly, and it is a RELEVANCE signal, not a storage
      detail.
    - `created_at`: has a named downstream consumer — surfacing a fact's
      age lets a later context layer help the model weigh currency when
      two facts disagree ("learned yesterday" vs "learned 8 months ago").
    - `confidence`: how much the application trusts the fact, as opposed
      to how well it matched the query. A property OF the fact, not of the
      retrieval mechanism.
    - `memory_id`: kept for APPLICATION-level traceability only —
      correlating an answer back to the memories that were present, for
      debugging, logging, and later evaluation. It is not expected to be
      rendered into prompt text; a future formatting layer should omit it.
      It is also the key that lets an application look the full record back
      up in SemanticMemoryStore when it genuinely needs storage detail.

    Why other upstream fields are deliberately ABSENT:

    - `source_event_ids`: provenance into the EPISODIC layer — a coupling
      between two storage layers that an agent-context consumer cannot use
      and a model must never see. `memory_id` is sufficient for
      traceability: given it, the full record (and its provenance) is one
      store lookup away. Copying it here would duplicate storage data
      across the boundary for no consumer.
    - `active`: a storage lifecycle flag belonging to the supersession
      system. The retriever already excludes inactive records (16D), so
      every item reaching this contract is active by construction — the
      field would always be True and would leak lifecycle semantics.
    - `session_id`: lives on the CONTAINER, not on every item — see
      MemoryContext.
    - Vectors / embeddings / dimensions: never cross this boundary at all.
    """

    memory_id: str
    content: str
    similarity: float
    created_at: datetime
    confidence: float

    def __post_init__(self) -> None:
        _require_non_empty_str("memory_id", self.memory_id)
        _require_non_empty_str("content", self.content)

        if isinstance(self.similarity, bool) or not isinstance(self.similarity, (int, float)):
            raise ValueError("similarity must be a number.")
        if not (-1.0 <= float(self.similarity) <= 1.0):
            raise ValueError("similarity must be within [-1.0, 1.0].")
        object.__setattr__(self, "similarity", float(self.similarity))

        if not isinstance(self.created_at, datetime):
            raise ValueError("created_at must be a datetime.")
        if self.created_at.tzinfo is None or self.created_at.tzinfo.utcoffset(self.created_at) is None:
            raise ValueError("created_at must be timezone-aware.")

        if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)):
            raise ValueError("confidence must be a number.")
        if not (0.0 <= float(self.confidence) <= 1.0):
            raise ValueError("confidence must be between 0.0 and 1.0 inclusive.")
        object.__setattr__(self, "confidence", float(self.confidence))


@dataclass(frozen=True)
class MemoryContext:
    """The memory available to the agent for ONE session, as ordered items.

    Session safety — why `session_id` sits HERE rather than on each item,
    or nowhere at all:

    The tempting minimal design is `MemoryContext(items=...)` with no
    session anywhere, on the reasoning that items already came out of a
    session-scoped retrieval. That was rejected: if nothing records which
    session a context is FOR, then nothing can ever CHECK it. A future bug
    that builds a context from two retrievals, or caches one and reuses it
    for a different session, would be undetectable — at construction, at
    use, and in tests alike. The invariant "Session A memory must never
    become Session B context" would be merely hoped for rather than
    enforceable.

    Putting `session_id` on the container makes the context state what it
    is for, exactly once, and makes that claim verifiable: `build_memory_
    context` checks EVERY retrieved record's session against it and raises
    on any mismatch, turning a would-be cross-session leak into an
    immediate, loud failure. That mirrors the decisions already made
    downstream — the vector index partitions structurally (16C) and the
    retriever raises rather than silently filtering on a session
    disagreement (16D) — so this layer keeps the same stance rather than
    weakening it at the last hop.

    Per-item `session_id` was rejected as redundant: every item in a
    context shares one session by the container's own invariant, so
    repeating it on each item would create a second place for the same
    truth to be stated (and therefore a way for the two to disagree).

    `items` is a tuple, and a list passed in is copied into one, so a
    caller cannot mutate a context after construction. Item ORDER is
    preserved exactly as given — normally the retriever's similarity
    ordering — and is never re-sorted here; ranking is not this layer's
    job.

    An empty context (`items=()`) is valid and ordinary: it means
    retrieval found nothing relevant, which is a normal outcome, not an
    error.
    """

    session_id: str
    items: tuple[MemoryContextItem, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty_str("session_id", self.session_id)
        object.__setattr__(self, "session_id", self.session_id.strip())

        if not isinstance(self.items, (tuple, list)):
            raise ValueError("items must be a tuple or list of MemoryContextItem.")
        for index, item in enumerate(self.items):
            if not isinstance(item, MemoryContextItem):
                raise ValueError(f"items[{index}] must be a MemoryContextItem.")
        object.__setattr__(self, "items", tuple(self.items))


def build_memory_context(session_id: str, retrieved: Sequence[RetrievedMemory]) -> MemoryContext:
    """Project retrieval results into the agent-facing contract.

    This function is the ONE place where retrieval types cross into
    context types — which is why it, rather than the dataclasses above,
    is what imports `RetrievedMemory`. Consumers of the contract use
    `MemoryContext`/`MemoryContextItem` and never need to know
    `RetrievedMemory` or `SemanticMemoryRecord` exist.

    It is a pure translation: no retrieval, no re-ranking, no filtering by
    similarity, no deduplication. Input order is preserved exactly, and
    every input becomes exactly one output item.

    Session safety is enforced here because this is the one moment the
    information is available: each `RetrievedMemory` still carries its
    full record, including that record's own `session_id`. Any record not
    belonging to `session_id` raises rather than being silently dropped —
    a record from the wrong session at this point means an upstream
    integrity bug, and quietly filtering it would hide the bug while
    leaving the underlying leak path in place.
    """
    _require_non_empty_str("session_id", session_id)
    normalized_session_id = session_id.strip()

    if not isinstance(retrieved, (tuple, list)):
        raise ValueError("retrieved must be a tuple or list of RetrievedMemory.")

    items: list[MemoryContextItem] = []
    for index, entry in enumerate(retrieved):
        if not isinstance(entry, RetrievedMemory):
            raise ValueError(f"retrieved[{index}] must be a RetrievedMemory.")

        record = entry.memory
        if record.session_id.strip() != normalized_session_id:
            raise ValueError(
                f"session isolation violation: retrieved memory {record.memory_id!r} belongs to session "
                f"{record.session_id!r}, but the context is being built for session "
                f"{normalized_session_id!r}."
            )

        items.append(
            MemoryContextItem(
                memory_id=record.memory_id,
                content=record.content,
                similarity=entry.similarity,
                created_at=record.created_at,
                confidence=record.confidence,
            )
        )

    return MemoryContext(session_id=normalized_session_id, items=tuple(items))
