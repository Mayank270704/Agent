"""Episodic memory — the third, distinct memory layer (Step 14).

    AgentState (app/agent/state.py):
        WORKING execution state for ONE agent execution/request.
        Lifetime: one request.

    ConversationMemory (app/agent/memory.py):
        Raw user/assistant conversation TURNS, verbatim.
        Lifetime: one session (see SessionMemoryStore).
        Answers: "What exactly was said?"

    EpisodicMemory (this module):
        Structured records of MEANINGFUL PAST EVENTS — not raw turns.
        Lifetime: potentially longer than one conversation session (the
        storage model here still partitions by session_id — see below —
        but the concept itself is not permanently tied to session
        semantics; that is an explicit non-goal of this step).
        Answers: "What meaningful event happened?"

These are deliberately kept as separate abstractions (Part 11) — merging
them would conflate "the exact words exchanged" with "a durable, compact
record of what happened," which have different retention, size, and (in a
future milestone) retrieval needs.

This module is intentionally a foundation, not a full memory system:
- ONE immutable record type (EpisodicMemoryRecord).
- ONE minimal Protocol (EpisodicMemory): add / get_recent / clear.
- ONE in-process implementation (InMemoryEpisodicMemory).

Explicitly NOT part of this step (future milestones):
- LLM-based memory extraction/classification/importance-scoring.
- search() / semantic_search() / similarity_search() / embedding() /
  retrieve() / rank() / summarize().
- Automatic retrieval into the agent's decision prompt.
- Persistence (database, files), a vector store, or RAG.

Records are created ONLY by explicit application code (Part 8) — see
app/agent/orchestrator.py's `_maybe_record_episode`, which is the sole
current caller. The LLM is never asked "should I remember this?".

There is no module-level mutable store anywhere in this file: every
`InMemoryEpisodicMemory()` instance owns its own dict, explicitly
constructed and injected by a caller — never a hidden singleton (same
rule as ConversationMemory/SessionMemoryStore, app/agent/memory.py).

--------------------------------------------------------------------------
Step 15 — retention/lifecycle contract (made explicit, behavior unchanged)
--------------------------------------------------------------------------

Step 14 already implemented the retention and ordering behavior below;
Step 15 does not change any of it — it documents the contract precisely
and adds the boundary/regression tests that pin it down:

1. Records are stored per session, in insertion order internally.
2. `get_recent(session_id, limit)` returns records NEWEST FIRST.
3. Once a session holds more than `max_records_per_session` records, the
   OLDEST records are evicted first — one eviction per record over the
   limit, so adding N records past the limit evicts exactly N records,
   deterministically, every time (see `InMemoryEpisodicMemory.add`).
4. Eviction is scoped to the OVERFLOWING session only; every other
   session's records are entirely unaffected by another session's
   eviction (there is no cross-session or global retention policy, and
   none is planned — Part 2).
5. `clear(session_id)` removes ALL of that session's records and nothing
   else; a subsequent `add()` for that same session starts from a clean,
   empty list — it is not merely "hidden," it is gone.

Duplicate event_id (Step 15, Part 6 — architectural decision):

`InMemoryEpisodicMemory` does NOT reject duplicate `event_id` values,
even within the same session, and this is intentional, not an oversight.
Reasoning:
- The store's only operations are `add` / `get_recent` / `clear` — there
  is no `get_by_id` or any other identity-based lookup anywhere in this
  Protocol, so `event_id` is never used as a storage/dedup key; it exists
  purely as an opaque, caller-supplied label on the record itself.
  Session_id + insertion order are the only things that give records
  meaning inside the store.
- The one real caller (`app/agent/orchestrator.py`'s
  `_maybe_record_episode`) generates `event_id` with `uuid.uuid4()` per
  record, which already makes collisions practically impossible without
  needing the store to enforce it.
- Enforcing uniqueness would require either a second index (an
  id -> record map, i.e. a small database) or an O(n) scan per `add()` —
  real infrastructure for a foundation step explicitly told to stay
  minimal (Part 6: "do not introduce a database or UUID service").
If a future milestone needs id-addressable lookup (e.g. "fetch episode
X"), that is the point to reconsider this — not before.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable


def _require_non_empty_str(field_name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")


@dataclass(frozen=True)
class EpisodicMemoryRecord:
    """One structured, immutable record of a meaningful past event.

    `event_type` describes WHAT kind of meaningful event occurred (e.g.
    "conversation_completed"). It stays a validated free-form string on
    purpose (Part 3) — not an enum — so future event types don't require
    touching this module; only application code that creates records needs
    to agree on the strings it uses.

    `metadata` is intentionally small and generic (`dict[str, Any]`), but
    is NOT a place to dump internal execution state (Part 17) — no
    prompts, chain-of-thought, raw model output, tool payloads, or
    secrets. See app/agent/orchestrator.py's `_maybe_record_episode` for
    the concrete, bounded metadata this codebase actually writes.

    Immutability: the dataclass itself is frozen, and `metadata` is copied
    and wrapped in a read-only `MappingProxyType` at construction time —
    both so a caller's own dict can't be mutated after the fact to alter a
    record that's already been stored, and so nothing can mutate the
    record's metadata after retrieval either.
    """

    event_id: str
    session_id: str
    event_type: str
    summary: str
    timestamp: datetime
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty_str("event_id", self.event_id)
        _require_non_empty_str("session_id", self.session_id)
        _require_non_empty_str("event_type", self.event_type)
        _require_non_empty_str("summary", self.summary)

        if not isinstance(self.timestamp, datetime):
            raise ValueError("timestamp must be a datetime.")
        if self.timestamp.tzinfo is None or self.timestamp.tzinfo.utcoffset(self.timestamp) is None:
            raise ValueError("timestamp must be timezone-aware.")

        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping.")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@runtime_checkable
class EpisodicMemory(Protocol):
    """Session-partitioned storage for EpisodicMemoryRecord objects.

    Deliberately minimal (Part 4) — no search/ranking/retrieval methods.
    Those belong to a later, explicitly separate milestone.
    """

    def add(self, record: EpisodicMemoryRecord) -> None:
        ...

    def get_recent(self, session_id: str, limit: int = 10) -> list[EpisodicMemoryRecord]:
        ...

    def clear(self, session_id: str) -> None:
        ...


class InMemoryEpisodicMemory:
    """The only implementation for now: a plain in-process dict of
    session_id -> list[EpisodicMemoryRecord].

    See this module's docstring ("Step 15 — retention/lifecycle contract")
    for the full, precise retention/ordering/isolation guarantees this
    class makes — summarized again briefly below.

    - Process-local, non-persistent: lost when the process restarts. Not
      thread-safe (same deliberate non-goal as ConversationMemory /
      SessionMemoryStore — see app/agent/memory.py).
    - Records for a session are kept in insertion order internally; oldest
      records are evicted first once `max_records_per_session` is
      exceeded, so the newest records always survive.
    - `get_recent(session_id, limit)` returns the `limit` most recent
      records, NEWEST FIRST (index 0 is the latest event) — chosen because
      "recent" reads most naturally as "what just happened," and this
      matches the eventual retrieval use case of showing/using the latest
      events first. This is the one documented ordering choice Part 5
      asks for.
    - Returned records are the actual stored `EpisodicMemoryRecord`
      objects (not defensive copies) — safe to hand out directly because
      the records themselves are frozen/immutable (see
      EpisodicMemoryRecord). The returned *list*, however, is always a new
      list, so appending/removing from it never affects internal storage.
    - `clear(session_id)` removes only that session's records; every other
      session is untouched.
    - session_id is validated the same way as SessionMemoryStore's (Part
      10 consistency): a non-empty string once stripped; the stripped form
      is the dict key, so `"A"` and `" A "` refer to the same session.
    """

    def __init__(self, max_records_per_session: int = 100):
        if max_records_per_session < 1:
            raise ValueError("max_records_per_session must be >= 1.")
        self._max_records_per_session = max_records_per_session
        self._records: dict[str, list[EpisodicMemoryRecord]] = {}

    def add(self, record: EpisodicMemoryRecord) -> None:
        if not isinstance(record, EpisodicMemoryRecord):
            raise ValueError("record must be an EpisodicMemoryRecord.")
        session_key = self._normalize(record.session_id)
        session_records = self._records.setdefault(session_key, [])
        session_records.append(record)
        overflow = len(session_records) - self._max_records_per_session
        if overflow > 0:
            del session_records[:overflow]

    def get_recent(self, session_id: str, limit: int = 10) -> list[EpisodicMemoryRecord]:
        session_key = self._normalize(session_id)
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("limit must be an integer >= 1.")
        records = self._records.get(session_key, [])
        most_recent = records[-limit:]
        return list(reversed(most_recent))

    def clear(self, session_id: str) -> None:
        session_key = self._normalize(session_id)
        self._records.pop(session_key, None)

    def _normalize(self, session_id: str) -> str:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string.")
        return session_id.strip()
