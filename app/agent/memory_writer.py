"""Semantic memory write pipeline (Step 16E-D) — the mirror image of
SemanticMemoryRetriever, and the component that finally closes the loop:

    MemoryCandidate[]  (app/agent/memory_extraction.py)
          |
      [1] safety validation      <- the trust boundary
          |
      [2] SemanticMemoryRecord   <- provenance attached here, not upstream
          |
      [3] EmbeddingProvider.embed(content)
          |
      [4] SemanticMemoryStore.add(record)
          |
      [5] VectorIndex.add(memory_id, session_id, vector)
          |
    retrievable by SemanticMemoryRetriever (16D)

--------------------------------------------------------------------------
Why validation lives HERE and not in the extractor
--------------------------------------------------------------------------
An extractor proposes; this class decides. That split is deliberate: the
safety policy must sit outside the component whose output it judges, so
that an LLM-backed extractor, a future rule-based one, a buggy one, or one
whose prompt was successfully manipulated are all subject to the identical
gate. `LLMMemoryExtractor`'s prompt asks the model to record facts rather
than instructions — that improves quality and is explicitly NOT relied on
for safety (see its docstring).

Rejected candidates are DROPPED, never rewritten. This class never edits,
truncates, or scrubs content to make it acceptable: a candidate either
passes as written or is discarded with a warning. Silently altering a fact
would store something the user never said.

Honest limitation: the instruction/secret checks below are pattern-based
and therefore imperfect in both directions. They will miss creatively
phrased directives and can reject an innocent sentence that happens to
match. They are a defense-in-depth gate that keeps the obvious cases out
of durable storage — NOT a guarantee, and not a substitute for the
untrusted-data framing applied at read time (see
app/agent/decision_maker.py's `_format_memory`).

--------------------------------------------------------------------------
Ordering, and what happens when a step fails
--------------------------------------------------------------------------
The required invariant is that a memory must never be RETRIEVABLE without
its record having been stored. Retrievability means "present in the vector
index", since the retriever searches the index and then resolves each hit
through the store.

Order is embed -> store -> index, chosen over store -> embed -> index
because embedding is both the most failure-prone step (dimension
mismatch, provider error) and the only one with no side effects, so doing
it first means a failure leaves nothing at all behind:

    embed fails  -> nothing stored, nothing indexed        (clean)
    store fails  -> nothing indexed (not reached yet)      (clean)
    index fails  -> record stored but not searchable       (benign residue)

Only the last case leaves residue, and it is the safe direction: the
record exists and is correct, it simply cannot be found yet. The reverse
ordering would instead leave an indexed vector with no record behind it,
which the retriever would encounter as a stale entry.

This is NOT transactional, and is not presented as such. There is no
rollback: `SemanticMemoryStore` exposes no per-record delete (only
`clear(session_id)`, which would destroy the whole session), so a failed
index step cannot be compensated without changing a stable contract —
which this milestone deliberately does not do. The limitation is
documented rather than papered over.
"""
from __future__ import annotations

import dataclasses
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Protocol, Sequence, runtime_checkable

from app.agent.embeddings import EmbeddingProvider
from app.agent.memory_extraction import MemoryCandidate
from app.agent.semantic_memory import SemanticMemoryRecord, SemanticMemoryStore
from app.agent.vector_index import VectorIndex

logger = logging.getLogger(__name__)

# Cosine similarity at or above which a new candidate is treated as
# restating a memory this session already holds (Step 16F-A). Set high on
# purpose: the cost of a false positive (two genuinely different facts
# merged into one, losing the second) is much worse than a false negative
# (one redundant record), and near-1.0 is the only region where sameness
# is safe to assume without semantic reasoning. Configurable per writer,
# and it MUST be recalibrated for whatever embedding model is in use.
DEFAULT_DUPLICATE_THRESHOLD = 0.95

# Candidates below this are dropped rather than stored (Step 16F-D). A low
# bar, not a quality filter: it discards only candidates an extractor has
# itself flagged as barely believable.
DEFAULT_MIN_CONFIDENCE = 0.3

# Phrasing that directs the ASSISTANT rather than describing the USER.
# Memory is durable data about a person, never an instruction channel, so a
# candidate that still reads as a command is refused even if a user really
# did phrase a genuine preference that way — the extractor is expected to
# rewrite such preferences as third-person facts ("User prefers metric
# units"), which pass cleanly.
_INSTRUCTION_PATTERNS = (
    r"\bignore\s+(all\s+|any\s+|previous\s+|prior\s+|the\s+)*(instruction|prompt|rule|direction)",
    r"\bdisregard\s+(all\s+|any\s+|previous\s+|prior\s+|the\s+)*(instruction|prompt|rule)",
    r"\b(you|assistant)\s+(must|should|will|shall|need to|have to)\b",
    r"\balways\s+(call|use|run|invoke|respond|reply|answer|say|tell|search|execute)\b",
    r"\bnever\s+(call|use|run|invoke|respond|reply|answer|say|tell|search|execute|mention)\b",
    r"\bfrom now on\b",
    r"\b(call|invoke|execute|run)\s+(the\s+)?\w*(tool|function|web_search|search)\b",
    r"\byour\s+(instruction|system prompt|rule|directive)",
    r"\breveal\s+(the\s+)?(system\s+)?prompt",
    r"^\s*(please\s+)?(do not|don't|always|never|ignore|disregard|reveal|forget)\b",
)

# Content that looks like a credential. Deliberately blunt: the cost of
# refusing an innocent sentence is one lost memory, while the cost of
# durably storing a real key is far higher.
_SECRET_PATTERNS = (
    r"\b(api[\s_-]?key|secret[\s_-]?key|access[\s_-]?token|auth[\s_-]?token|bearer\s+token)\b",
    r"\b(password|passwd|passphrase|credential)s?\b",
    r"\bsk-[A-Za-z0-9_-]{8,}",
    r"\bghp_[A-Za-z0-9]{8,}",
    r"\bAKIA[0-9A-Z]{8,}",
    r"\b[A-Fa-f0-9]{32,}\b",
    r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}",
)

_INSTRUCTION_REGEXES = tuple(re.compile(pattern, re.IGNORECASE) for pattern in _INSTRUCTION_PATTERNS)
_SECRET_REGEXES = tuple(re.compile(pattern, re.IGNORECASE) for pattern in _SECRET_PATTERNS)


def _require_non_empty(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")


def _require_event_ids(source_event_ids: object) -> None:
    if not isinstance(source_event_ids, (list, tuple)) or not source_event_ids:
        raise ValueError("source_event_ids must be a non-empty list or tuple.")
    for event_id in source_event_ids:
        if not isinstance(event_id, str) or not event_id.strip():
            raise ValueError("each source event id must be a non-empty string.")


def rejection_reason(content: str) -> str | None:
    """Returns why `content` must not be stored, or None if it may be.

    Exposed as a module-level function, not a private method, so the
    policy can be tested and reasoned about in isolation from the storage
    pipeline that applies it.
    """
    if not isinstance(content, str) or not content.strip():
        return "content is empty"

    for regex in _SECRET_REGEXES:
        if regex.search(content):
            return "content looks like it contains a credential"

    for regex in _INSTRUCTION_REGEXES:
        if regex.search(content):
            return "content reads as an instruction to the assistant rather than a fact about the user"

    return None


@runtime_checkable
class MemoryWriter(Protocol):
    """Turns proposed candidates into stored, retrievable semantic memory.

    Returns the records it actually persisted, which is not necessarily
    one per candidate: rejected candidates are dropped.
    """

    def write(
        self,
        session_id: str,
        candidates: Sequence[MemoryCandidate],
        source_event_ids: Sequence[str],
    ) -> list[SemanticMemoryRecord]:
        ...


class SemanticMemoryWriter:
    """The only implementation for now: composes a SemanticMemoryStore, an
    EmbeddingProvider, and a VectorIndex.

    Named for what it writes rather than where things live — like
    SemanticMemoryRetriever it owns no storage itself, only the three
    injected collaborators, so an `InMemory...` prefix would be false.

    All three are typed against their Protocols, never against
    InMemorySemanticMemory / DeterministicEmbeddingProvider /
    InMemoryVectorIndex, so a real embedding model and a database-backed
    index drop in with no change here. Provider/index dimension
    compatibility is checked EAGERLY at construction, for the same reason
    SemanticMemoryRetriever does it: a mismatch breaks every write without
    exception, so it is a wiring bug and belongs at wiring time rather
    than surfacing on an arbitrary later request.

    Provenance is attached HERE, from `source_event_ids` supplied by the
    caller (the episodic event that produced the interaction) — never by
    the extractor, which could otherwise forge it. `SemanticMemoryRecord`
    itself requires at least one source event id, so an unsourced memory
    cannot be constructed at all.

    Duplicate CONTENT is allowed through, matching the store's existing
    documented decision (16A): recognizing that two differently-worded
    facts mean the same thing needs semantic comparison, and dedup,
    merging, supersession and contradiction resolution are all explicitly
    later milestones. Every record gets a fresh uuid4 `memory_id`, so
    identity collisions cannot occur.
    """

    def __init__(
        self,
        semantic_memory: SemanticMemoryStore,
        embedding_provider: EmbeddingProvider,
        vector_index: VectorIndex,
        duplicate_threshold: float = DEFAULT_DUPLICATE_THRESHOLD,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        max_records_per_session: int | None = None,
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

        if isinstance(duplicate_threshold, bool) or not isinstance(duplicate_threshold, (int, float)):
            raise ValueError("duplicate_threshold must be a number.")
        if not (0.0 < float(duplicate_threshold) <= 1.0):
            raise ValueError("duplicate_threshold must be within (0.0, 1.0].")
        if isinstance(min_confidence, bool) or not isinstance(min_confidence, (int, float)):
            raise ValueError("min_confidence must be a number.")
        if not (0.0 <= float(min_confidence) <= 1.0):
            raise ValueError("min_confidence must be within [0.0, 1.0].")
        if max_records_per_session is not None and (
            isinstance(max_records_per_session, bool)
            or not isinstance(max_records_per_session, int)
            or max_records_per_session < 1
        ):
            raise ValueError("max_records_per_session must be an integer >= 1 if provided.")

        self.semantic_memory = semantic_memory
        self.embedding_provider = embedding_provider
        self.vector_index = vector_index
        self.duplicate_threshold = float(duplicate_threshold)
        self.min_confidence = float(min_confidence)
        self.max_records_per_session = max_records_per_session

    def write(
        self,
        session_id: str,
        candidates: Sequence[MemoryCandidate],
        source_event_ids: Sequence[str],
    ) -> list[SemanticMemoryRecord]:
        normalized_session_id = self._normalize(session_id)

        if not isinstance(candidates, (list, tuple)):
            raise ValueError("candidates must be a list or tuple of MemoryCandidate.")
        _require_event_ids(source_event_ids)

        written: list[SemanticMemoryRecord] = []
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, MemoryCandidate):
                raise ValueError(f"candidates[{index}] must be a MemoryCandidate.")

            reason = rejection_reason(candidate.content)
            if reason is not None:
                # Dropped, never rewritten -- see the module docstring.
                logger.warning("Rejected semantic memory candidate (%s): %r", reason, candidate.content)
                continue

            if candidate.confidence < self.min_confidence:
                logger.info(
                    "Dropping low-confidence memory candidate (%.2f < %.2f): %r",
                    candidate.confidence,
                    self.min_confidence,
                    candidate.content,
                )
                continue

            written.append(self._persist(normalized_session_id, candidate, tuple(source_event_ids)))

        return written

    def supersede(
        self,
        session_id: str,
        superseded_memory_id: str,
        candidate: MemoryCandidate,
        source_event_ids: Sequence[str],
    ) -> SemanticMemoryRecord:
        """Retire an existing memory and write its replacement.

        This is the MECHANISM for supersession (Step 16F-C). It is
        deliberately explicit rather than inferred: nothing in this
        codebase decides on its own that one memory supersedes another,
        because embedding similarity cannot distinguish "this contradicts
        that" from "this refines that" — see the module docstring's
        conflict note. A caller invokes this only when something outside
        the vector layer actually knows (a future resolver, or an explicit
        user action).

        The old record is NOT mutated — records are frozen. A new version
        of it is built with `active=False` and swapped in via
        `store.replace`, preserving its memory_id, content, provenance,
        and position. Nothing is deleted: the superseded fact stays
        auditable, and "what did the user used to prefer?" remains
        answerable.

        Its vector IS removed from the index, maintaining this class's
        invariant that the index holds exactly the ACTIVE records. That
        keeps an inactive memory from consuming a top-K slot in every
        future search only to be discarded when the retriever resolves it.
        """
        normalized_session_id = self._normalize(session_id)
        _require_non_empty(superseded_memory_id, "superseded_memory_id")
        if not isinstance(candidate, MemoryCandidate):
            raise ValueError("candidate must be a MemoryCandidate.")
        _require_event_ids(source_event_ids)

        existing = self.semantic_memory.get(superseded_memory_id)
        if existing is None:
            raise ValueError(f"memory_id {superseded_memory_id!r} does not exist; nothing to supersede.")
        if existing.session_id.strip() != normalized_session_id:
            raise ValueError(
                f"session isolation violation: memory {superseded_memory_id!r} belongs to session "
                f"{existing.session_id!r}, not {normalized_session_id!r}."
            )

        reason = rejection_reason(candidate.content)
        if reason is not None:
            # The security gate applies identically here: superseding is
            # not a route around it.
            raise ValueError(f"replacement candidate rejected ({reason}).")

        replacement = self._persist(normalized_session_id, candidate, tuple(source_event_ids))

        self.semantic_memory.replace(dataclasses.replace(existing, active=False))
        self.vector_index.remove(existing.memory_id, normalized_session_id)

        return replacement

    def _persist(
        self,
        session_id: str,
        candidate: MemoryCandidate,
        source_event_ids: tuple[str, ...],
    ) -> SemanticMemoryRecord:
        """embed -> [dedup] -> [retention] -> store -> index. See the
        module docstring for why this order, and for the non-atomicity it
        leaves."""
        # Embedded first so the vector is available for the duplicate
        # check -- no second embedding call is needed, and a failure here
        # still leaves nothing behind.
        vector = self.embedding_provider.embed(candidate.content)

        duplicate = self._find_duplicate(session_id, vector)
        if duplicate is not None:
            return self._merge_into(duplicate, candidate, source_event_ids)

        self._enforce_retention(session_id)

        record = SemanticMemoryRecord(
            memory_id=str(uuid.uuid4()),
            session_id=session_id,
            content=candidate.content,
            created_at=datetime.now(timezone.utc),
            source_event_ids=source_event_ids,
            confidence=candidate.confidence,
        )

        # Store: after this the fact is durable and correct.
        self.semantic_memory.add(record)

        # Index: only now does it become retrievable.
        self.vector_index.add(record.memory_id, record.session_id, vector)

        return record

    def _find_duplicate(self, session_id: str, vector) -> SemanticMemoryRecord | None:
        """Is this session already holding essentially the same fact?

        Reuses the vector index that already exists rather than adding any
        new comparison machinery: one top-1 search in the session's own
        partition, and a hit at or above `duplicate_threshold` counts as a
        restatement. Because the index holds exactly the ACTIVE records
        (see `supersede`), a superseded memory can never be matched here —
        so reaffirming a retired fact correctly creates a new one rather
        than quietly reviving the old.

        Exact string repetition is caught by this too, since identical
        text embeds to an identical vector (similarity 1.0) — hence no
        separate string-equality path, which would only cover the easiest
        case while implying the harder one was handled.

        A stale hit (indexed but no longer in the store) is treated as NOT
        a duplicate: merging provenance into a record that no longer
        exists would raise, and the conservative response to an
        inconsistency is to write the new memory rather than lose it.
        """
        hits = self.vector_index.search(session_id, vector, top_k=1)
        if not hits:
            return None
        if hits[0].similarity < self.duplicate_threshold:
            return None
        return self.semantic_memory.get(hits[0].memory_id)

    def _merge_into(
        self,
        existing: SemanticMemoryRecord,
        candidate: MemoryCandidate,
        source_event_ids: tuple[str, ...],
    ) -> SemanticMemoryRecord:
        """Fold a restatement into the memory it restates.

        Provenance is UNIONED, never replaced: the new interaction really
        did also support this fact, and dropping its event id would lose
        the evidence that the user said it twice — the explicit "do not
        silently discard provenance" requirement. Order is preserved and
        duplicates removed, so the merge is deterministic.

        Kept from the original: `memory_id` (stable identity),
        `created_at` (when the fact was FIRST learned, which is what makes
        it durable), and `content`. The new phrasing is discarded because
        nothing here can judge one wording better than another, and
        keeping the original also means the stored vector stays valid so
        no re-indexing is needed.

        Confidence becomes the MAXIMUM of the two. Independent
        reaffirmation should never make a fact less believed, and
        averaging or summing would invent a calibration this scale
        explicitly does not have.
        """
        merged_event_ids = list(existing.source_event_ids)
        for event_id in source_event_ids:
            if event_id not in merged_event_ids:
                merged_event_ids.append(event_id)

        merged = dataclasses.replace(
            existing,
            source_event_ids=tuple(merged_event_ids),
            confidence=max(existing.confidence, candidate.confidence),
        )
        self.semantic_memory.replace(merged)
        logger.info(
            "Merged duplicate memory candidate into existing memory %s: %r",
            existing.memory_id,
            candidate.content,
        )
        return merged

    def _enforce_retention(self, session_id: str) -> None:
        """Keep a session under its record cap, removing from BOTH stores.

        This class enforces retention itself rather than leaving it to the
        store's own `max_records_per_session` eviction, because the store
        cannot see the vector index: an eviction there would silently
        leave a vector whose record no longer exists. Evicting here — from
        the store and the index together — is what keeps the two
        consistent, and is why a writer-owned cap is the supported way to
        bound semantic memory.

        Oldest-first, matching the eviction direction every other memory
        layer in this codebase uses.
        """
        if self.max_records_per_session is None:
            return

        existing = self.semantic_memory.list_recent(session_id, limit=self.max_records_per_session + 1)
        overflow = len(existing) - self.max_records_per_session + 1
        if overflow <= 0:
            return

        # list_recent is newest-first, so the oldest are at the end.
        for record in list(reversed(existing))[:overflow]:
            self.semantic_memory.delete(record.memory_id)
            self.vector_index.remove(record.memory_id, session_id)
            logger.info("Evicted semantic memory %s to stay within retention cap.", record.memory_id)

    def _normalize(self, session_id: str) -> str:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string.")
        return session_id.strip()
