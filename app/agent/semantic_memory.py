"""Semantic memory — the fourth memory layer (Step 16A), and the last one
before real retrieval/embeddings work begins (see the Step 16 architecture
review — "Semantic Memory Blueprint" — for the full design this module
implements a small, deliberately inert slice of).

    AgentState          WORKING memory: current execution.      1 request.
    ConversationMemory   RAW turns, verbatim.                    1 session.
    EpisodicMemory        "Something happened."                  session(+).
    SemanticMemory (here) "This is a durable fact."               durable.

An episodic record is provenance: "User said during conversation: I prefer
Python." A semantic memory is a distilled, durable claim derived FROM one or
more episodic events: "User prefers Python." Episodic memory does not
disappear once a semantic memory exists — semantic memory is a DERIVED
read-model over episodic memory (Step 16, decision 1), never a replacement
for it. This module does not build that derivation pipeline; it only
defines what a derived fact looks like once one exists, and how it is
stored.

--------------------------------------------------------------------------
Step 16A scope — deliberately inert
--------------------------------------------------------------------------
This module is a data model and a storage abstraction ONLY. It contains:
- ONE immutable record type (SemanticMemoryRecord).
- ONE minimal Protocol (SemanticMemoryStore): add / get / list_recent /
  clear.
- ONE in-process implementation (InMemorySemanticMemory).

It deliberately contains NONE of the following (all later, separately
scoped phases — see the Step 16 blueprint's phased roadmap):
- Embeddings, an embedding model, or any vector representation.
- A vector store, similarity search, or ranking of any kind.
- Automatic LLM-based extraction of facts from episodic records — for now,
  callers (today: only tests) construct SemanticMemoryRecord directly.
- Conflict resolution / supersession LOGIC. The `active` field exists
  (see the class docstring) so the record shape does not need to be
  redesigned when supersession is built, but nothing in this module ever
  flips it — that is future work.
- Wiring into ChatService, AgentOrchestrator, AgentState, or the API.
  Nothing outside this file and its tests knows this module exists yet.

There is no module-level mutable store anywhere in this file: every
`InMemorySemanticMemory()` instance owns its own dicts, explicitly
constructed and injected by a caller — never a hidden singleton (same rule
as every other memory layer in this codebase — see app/agent/memory.py and
app/agent/episodic_memory.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, runtime_checkable


def _require_non_empty_str(field_name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")


@dataclass(frozen=True)
class SemanticMemoryRecord:
    """One durable, immutable fact, derived from one or more episodic
    events.

    Field-by-field justification (Step 16A, Part 2 — the minimum this
    architecture actually needs, not the full candidate list):

    - `memory_id`: a stable identity for this specific fact-slot. Unlike
      EpisodicMemoryRecord's `event_id` (intentionally NOT a lookup key —
      see app/agent/episodic_memory.py's "Duplicate event_id" section),
      `memory_id` IS a real lookup key here: `SemanticMemoryStore.get()`
      addresses records by it (Part 5 explicitly asks for `get()`, which
      episodic memory's Protocol does not have). That one extra operation
      is why semantic memory needs id uniqueness and episodic memory does
      not — see InMemorySemanticMemory.add.

    - `session_id`: scope. Session-scoped ONLY, for now (Part 3) — the
      same explicit, non-authentication scope key used by
      SessionMemoryStore and EpisodicMemory. Kept as a plain field (not a
      richer "scope" value object) for the same reason Step 14 didn't
      pre-build one for episodic memory: there is exactly one scope tier
      today, and widening to a user-scope tier later is an additive
      change, not a redesign — see the Step 16 blueprint, §7.

    - `content`: the durable fact itself, in natural language ("User
      prefers Python."). This is the ONLY field a future embedding step
      would embed (Step 16 blueprint, §5) — kept deliberately separate
      from provenance and metadata so it stays a clean, single assertion.

    - `created_at`: timezone-aware UTC creation timestamp. There is
      deliberately NO `updated_at`: this record is frozen, and "updating"
      a fact is a future supersession operation that creates a NEW record
      rather than mutating an old one (Step 16 blueprint, §13) — an
      `updated_at` field would imply a mutation path that does not exist
      yet, so it is left out rather than added and left meaningless.

    - `confidence`: a float in [0.0, 1.0]. This is NOT a statistically
      calibrated probability — nothing in this codebase runs a model that
      could produce one. It is an opaque, application-defined signal:
      today (direct construction only) it simply means "how sure is the
      caller that created this record." It exists now because the Step 16
      blueprint's future ranking formula (§9) consumes it directly, and
      retrofitting a confidence field onto every existing record later
      would be a breaking migration. Defaults to 1.0 (an explicitly
      asserted fact, taken at face value) since Step 16A has no extraction
      pipeline that would produce a lower, hedged value.

    - `source_event_ids`: a non-empty tuple of EpisodicMemoryRecord
      `event_id` values this fact was derived from (Part 7). Provenance is
      stored STRUCTURALLY, never folded into `content` as prose — so a
      future step can trace, debug, or invalidate a fact by following its
      source episodes without parsing natural language. At least one
      source event is REQUIRED: every semantic memory — however it is
      eventually created, including a future explicit "remember X"
      feature — originates from some completed interaction, and that
      interaction is itself an episodic event (Step 14). A fact with no
      traceable origin is exactly the kind of unverifiable claim the
      provenance requirement exists to rule out.

    - `active`: bool, default True. This is the ONE minimal state field
      Part 9 asks for — added now so a future supersession mechanism (a
      NEW record superseding an old one) does not require widening this
      dataclass's shape. Nothing in Step 16A ever sets it to False; there
      is no `supersede()` method and no conflict-resolution logic here.
      Its only job in this step is to exist so later work doesn't need to
      redesign every caller of this class.

    Immutability: the dataclass is frozen, matching EpisodicMemoryRecord.
    `source_event_ids` is copied into a real `tuple` at construction time
    (not merely trusted to already be one), so a caller's own list can't
    be mutated afterward to alter a record that's already been stored, and
    the stored value itself can never be appended to (tuples have no
    mutating methods) — the same guarantee EpisodicMemoryRecord gives its
    `metadata` via `MappingProxyType`, achieved here with the type that
    naturally fits an ordered, small collection of id strings.
    """

    memory_id: str
    session_id: str
    content: str
    created_at: datetime
    source_event_ids: tuple[str, ...] = field(default_factory=tuple)
    confidence: float = 1.0
    active: bool = True

    def __post_init__(self) -> None:
        _require_non_empty_str("memory_id", self.memory_id)
        _require_non_empty_str("session_id", self.session_id)
        _require_non_empty_str("content", self.content)

        if not isinstance(self.created_at, datetime):
            raise ValueError("created_at must be a datetime.")
        if self.created_at.tzinfo is None or self.created_at.tzinfo.utcoffset(self.created_at) is None:
            raise ValueError("created_at must be timezone-aware.")

        if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)):
            raise ValueError("confidence must be a number.")
        if not (0.0 <= float(self.confidence) <= 1.0):
            raise ValueError("confidence must be between 0.0 and 1.0 inclusive.")
        object.__setattr__(self, "confidence", float(self.confidence))

        if not isinstance(self.source_event_ids, (tuple, list)):
            raise ValueError("source_event_ids must be a tuple or list of strings.")
        if len(self.source_event_ids) == 0:
            raise ValueError("source_event_ids must contain at least one event id.")
        for event_id in self.source_event_ids:
            _require_non_empty_str("source_event_ids item", event_id)
        object.__setattr__(self, "source_event_ids", tuple(self.source_event_ids))

        if not isinstance(self.active, bool):
            raise ValueError("active must be a bool.")


@runtime_checkable
class SemanticMemoryStore(Protocol):
    """Session-partitioned, id-addressable storage for
    SemanticMemoryRecord objects.

    Deliberately minimal (Part 5) — no search/embedding/ranking methods.
    Those belong to a later, explicitly separate milestone (the Step 16
    blueprint's MemoryRetriever Protocol, not yet implemented).

    Step 16F added `replace` and `delete`, the two smallest CRUD
    completions that memory LIFECYCLE requires and that no workaround can
    substitute for. Records are frozen, so making one inactive
    (supersession) means swapping in a new version of it — impossible with
    only add/get/list/clear. `delete` is what lets the single component
    that writes to BOTH this store and the vector index keep the two
    consistent when a record leaves. Neither method is a search,
    embedding, or ranking concern, so the Protocol's original boundary is
    unchanged.
    """

    def add(self, record: SemanticMemoryRecord) -> None:
        ...

    def get(self, memory_id: str) -> SemanticMemoryRecord | None:
        ...

    def list_recent(self, session_id: str, limit: int = 10) -> list[SemanticMemoryRecord]:
        ...

    def replace(self, record: SemanticMemoryRecord) -> None:
        ...

    def delete(self, memory_id: str) -> None:
        ...

    def clear(self, session_id: str) -> None:
        ...


class InMemorySemanticMemory:
    """The only implementation for now: two plain in-process indexes —
    session_id -> list[SemanticMemoryRecord] (for `list_recent`/`clear`)
    and memory_id -> SemanticMemoryRecord (for `get`) — kept in sync.

    - Process-local, non-persistent, not thread-safe: same deliberate
      non-goal as every other in-memory store in this codebase.

    - `get(memory_id)` is why `memory_id` uniqueness IS enforced here,
      unlike EpisodicMemoryRecord's `event_id` (see the record docstring):
      `add()` raises ValueError if the id is already present, so `get()`
      always has an unambiguous single answer. This is a deliberate,
      documented difference from episodic memory's decision — not an
      inconsistency.

    - `list_recent(session_id, limit)` returns the `limit` most recent
      records for that session, NEWEST FIRST (index 0 is the latest) —
      the same ordering convention as EpisodicMemory.get_recent, for the
      same reason (Step 15's retention contract) and for consistency
      across this codebase's memory layers.

    - Retention: UNBOUNDED by default (`max_records_per_session=None`).
      This is a deliberate departure from EpisodicMemory's default cap
      (Part 10) — episodic memory is an ever-growing raw event log where
      old entries are expected to lose value and get evicted; semantic
      memory is supposed to stay small on its own (each fact is meant to
      be deduplicated/superseded down the line, not accumulated forever —
      Step 16 blueprint, §13), so imposing a default eviction policy on
      "durable facts" here would risk silently discarding exactly the
      long-lived knowledge this layer exists to keep. An OPTIONAL cap is
      still supported (same oldest-evicted-first mechanics as
      EpisodicMemory) as a defensive safety valve for a caller who wants
      one — it is just not the default.

    - `clear(session_id)` removes all of that session's records from BOTH
      indexes; a record's `memory_id` becomes reusable afterward (`get()`
      on the old id then returns None).

    - Duplicates (Part 11): two records with equal `content` but different
      `memory_id`s are explicitly ALLOWED and BOTH are stored — e.g. the
      same fact re-affirmed across two separate episodes. Recognizing that
      two pieces of text mean the same thing requires semantic comparison
      (embeddings), which is explicitly out of scope for Step 16A;
      content-level deduplication is deferred to a later retrieval-aware
      phase. What IS rejected is a literal `memory_id` collision (see
      above) — a different, narrower, purely-mechanical guarantee.

    - session_id is validated/normalized the same way as every other
      session-scoped store in this codebase: a non-empty string once
      stripped; the stripped form is the dict key.
    """

    def __init__(self, max_records_per_session: int | None = None):
        if max_records_per_session is not None and max_records_per_session < 1:
            raise ValueError("max_records_per_session must be >= 1 if provided.")
        self._max_records_per_session = max_records_per_session
        self._by_session: dict[str, list[SemanticMemoryRecord]] = {}
        self._by_id: dict[str, SemanticMemoryRecord] = {}

    def add(self, record: SemanticMemoryRecord) -> None:
        if not isinstance(record, SemanticMemoryRecord):
            raise ValueError("record must be a SemanticMemoryRecord.")
        if record.memory_id in self._by_id:
            raise ValueError(f"memory_id {record.memory_id!r} already exists.")

        session_key = self._normalize(record.session_id)
        session_records = self._by_session.setdefault(session_key, [])
        session_records.append(record)
        self._by_id[record.memory_id] = record

        if self._max_records_per_session is not None:
            overflow = len(session_records) - self._max_records_per_session
            if overflow > 0:
                evicted = session_records[:overflow]
                del session_records[:overflow]
                for evicted_record in evicted:
                    del self._by_id[evicted_record.memory_id]

    def get(self, memory_id: str) -> SemanticMemoryRecord | None:
        _require_non_empty_str("memory_id", memory_id)
        return self._by_id.get(memory_id)

    def list_recent(self, session_id: str, limit: int = 10) -> list[SemanticMemoryRecord]:
        session_key = self._normalize(session_id)
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("limit must be an integer >= 1.")
        records = self._by_session.get(session_key, [])
        most_recent = records[-limit:]
        return list(reversed(most_recent))

    def replace(self, record: SemanticMemoryRecord) -> None:
        """Swap an existing record for a new version of itself, IN PLACE —
        same memory_id, same position in the session's ordering.

        Position is preserved deliberately: `list_recent` is
        newest-first, and a record being superseded or having its
        provenance merged has not become "newer", so reordering it would
        silently change retrieval and retention behavior.

        The replacement must carry the same memory_id (that is what makes
        it a replacement rather than a new memory) and the same session_id
        (moving a record between sessions through a back door would
        breach the isolation every other layer enforces). Replacing a
        memory_id that does not exist raises, rather than quietly
        inserting — a replace that finds nothing to replace means the
        caller's view of the store is wrong.
        """
        if not isinstance(record, SemanticMemoryRecord):
            raise ValueError("record must be a SemanticMemoryRecord.")

        existing = self._by_id.get(record.memory_id)
        if existing is None:
            raise ValueError(f"memory_id {record.memory_id!r} does not exist; nothing to replace.")
        if self._normalize(existing.session_id) != self._normalize(record.session_id):
            raise ValueError(
                f"cannot move memory_id {record.memory_id!r} from session "
                f"{existing.session_id!r} to {record.session_id!r}."
            )

        session_key = self._normalize(record.session_id)
        session_records = self._by_session[session_key]
        session_records[session_records.index(existing)] = record
        self._by_id[record.memory_id] = record

    def delete(self, memory_id: str) -> None:
        """Remove one record entirely. A no-op if it was never stored,
        matching `clear`'s established "removing something that was never
        there is not an error" convention."""
        _require_non_empty_str("memory_id", memory_id)

        record = self._by_id.pop(memory_id, None)
        if record is None:
            return
        session_records = self._by_session.get(self._normalize(record.session_id))
        if session_records is not None and record in session_records:
            session_records.remove(record)

    def clear(self, session_id: str) -> None:
        session_key = self._normalize(session_id)
        removed = self._by_session.pop(session_key, [])
        for record in removed:
            del self._by_id[record.memory_id]

    def _normalize(self, session_id: str) -> str:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string.")
        return session_id.strip()
