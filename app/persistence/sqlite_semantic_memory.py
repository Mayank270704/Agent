"""SQLite-backed SemanticMemoryStore and VectorIndex — Step 38.

`SQLiteSemanticMemory` and `SQLiteVectorIndex` implement the SAME
`SemanticMemoryStore` and `VectorIndex` Protocols already defined in
app/agent/semantic_memory.py and app/agent/vector_index.py — no change to
either module, no new Protocol. They are drop-in alternatives to
`InMemorySemanticMemory`/`InMemoryVectorIndex`, selected at the composition
root (app/semantic_memory_wiring.py / app/main.py) via the SAME
`MEMORY_BACKEND` setting Steps 36/37 already introduced.

Kept in one file, like app/persistence/sqlite_memory.py's conversation/
session pair (Step 36): these two are always wired together by
`build_semantic_memory()` and share this module's `_row_to_vector`-style
small helpers, even though — see below — they remain architecturally
INDEPENDENT of each other, exactly like their in-memory counterparts.

--------------------------------------------------------------------------
Deliberately NO foreign key between the two tables
--------------------------------------------------------------------------
`app/agent/vector_index.py`'s own module docstring states this as a
load-bearing design decision, repeatedly: VectorIndex and
SemanticMemoryStore are independent Protocols, neither depends on the
other's guarantees, and "this index must not assume SemanticMemoryStore's
guarantee anyway." A cross-table FK (`vectors.memory_id REFERENCES
records(memory_id)`) would silently couple them at the database level —
exactly the redesign Step 38 is told not to make — and would also reject
the Protocol-legal (if practically unused outside direct VectorIndex
tests) case of indexing a vector with no backing record, breaking
behavioral parity with `InMemoryVectorIndex` and the existing
`tests/test_vector_index.py` suite's own standalone usage.

What Step 38 DOES guarantee, "as far as the current architecture permits"
(its own phrasing): each store's OWN `add()` is atomic (wrapped in one
SQLite transaction), so a failure can never leave a half-written row in
EITHER table; the existing, UNCHANGED `SemanticMemoryWriter._persist()`
write order (embed -> store.add() -> index.add()) is exactly what already
keeps "a vector never outlives its record" true in practice, and that
order is not touched here.

--------------------------------------------------------------------------
Embedding-model identity (Step 38 requirement I)
--------------------------------------------------------------------------
Neither the `VectorIndex` Protocol's `add()`/`search()` methods nor
`EmbeddingProvider` itself carries a model-identity parameter — by design,
VectorIndex is meant to be "model independent" (see vector_index.py's
"Dependency direction" note). Persistence is what makes a NEW failure mode
reachable that never mattered when the index was wiped on every restart: a
persisted index built by one embedding model could silently be reopened
and searched by a DIFFERENT model later, producing meaningless similarity
scores with no error anywhere — the exact failure class
`local_embeddings.py` already refuses to allow for a live provider/index
dimension mismatch.

`SQLiteVectorIndex` closes this gap WITHOUT touching the Protocol: it
takes `embedding_model_name` as a required CONSTRUCTOR argument (a fact
about the index instance, exactly like `dimension` already is), tags every
vector it writes with that name, and — at construction time, the same
"wiring bug surfaces at wiring time" discipline `SemanticMemoryWriter`/
`SemanticMemoryRetriever` already apply to dimension mismatches — refuses
to open a database whose persisted vectors were tagged with a DIFFERENT
model name.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime

# `_validate_numeric_vector` is module-private to vector_index.py but is
# imported here deliberately, matching local_embeddings.py's identical
# precedent for `_require_valid_text`: it IS the one definition of "a
# valid vector" InMemoryVectorIndex itself uses, and re-implementing it
# here would create a second definition free to drift apart. Both modules
# live in the same package, so this is an internal detail, not a public
# coupling.
from app.agent.semantic_memory import SemanticMemoryRecord
from app.agent.vector_index import Vector, VectorSearchResult, _validate_numeric_vector, cosine_similarity
from app.persistence import db


def _require_non_empty_str(field_name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")


class SQLiteSemanticMemory:
    """SQLite-backed `SemanticMemoryStore`.

    Mirrors `InMemorySemanticMemory`'s documented behavior exactly:
    `memory_id` uniqueness is enforced GLOBALLY (unlike episodic memory's
    `event_id`), ordering for `list_recent` is insertion order (newest
    first), `replace()` preserves position (via SQLite's implicit
    `rowid`, which an UPDATE never changes), and retention is OPTIONAL and
    OFF by default — `build_semantic_memory()` always constructs this
    store uncapped; `SemanticMemoryWriter` remains the sole retention
    owner (see app/semantic_memory_wiring.py's own "Retention" section,
    unchanged by Step 38).
    """

    def __init__(self, db_path: str, max_records_per_session: int | None = None):
        if max_records_per_session is not None and max_records_per_session < 1:
            raise ValueError("max_records_per_session must be >= 1 if provided.")
        self._max_records_per_session = max_records_per_session
        self._conn = db.connect(db_path)
        self._lock = threading.Lock()

    def add(self, record: SemanticMemoryRecord) -> None:
        if not isinstance(record, SemanticMemoryRecord):
            raise ValueError("record must be a SemanticMemoryRecord.")
        session_key = self._normalize(record.session_id)
        with self._lock:
            with self._conn:
                existing = self._conn.execute(
                    "SELECT 1 FROM semantic_memory_records WHERE memory_id = ?", (record.memory_id,)
                ).fetchone()
                if existing is not None:
                    raise ValueError(f"memory_id {record.memory_id!r} already exists.")

                self._conn.execute(
                    "INSERT INTO semantic_memory_records "
                    "(memory_id, session_id, content, created_at, confidence, active, source_event_ids_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        record.memory_id,
                        session_key,
                        record.content,
                        record.created_at.isoformat(),
                        record.confidence,
                        int(record.active),
                        json.dumps(list(record.source_event_ids)),
                    ),
                )

                if self._max_records_per_session is not None:
                    count = self._conn.execute(
                        "SELECT COUNT(*) AS c FROM semantic_memory_records WHERE session_id = ?",
                        (session_key,),
                    ).fetchone()["c"]
                    overflow = count - self._max_records_per_session
                    if overflow > 0:
                        self._conn.execute(
                            "DELETE FROM semantic_memory_records WHERE rowid IN ("
                            "SELECT rowid FROM semantic_memory_records WHERE session_id = ? "
                            "ORDER BY rowid ASC LIMIT ?)",
                            (session_key, overflow),
                        )

    def get(self, memory_id: str) -> SemanticMemoryRecord | None:
        _require_non_empty_str("memory_id", memory_id)
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM semantic_memory_records WHERE memory_id = ?", (memory_id,)
            ).fetchone()
        return self._row_to_record(row) if row is not None else None

    def list_recent(self, session_id: str, limit: int = 10) -> list[SemanticMemoryRecord]:
        session_key = self._normalize(session_id)
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("limit must be an integer >= 1.")
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM semantic_memory_records WHERE session_id = ? ORDER BY rowid DESC LIMIT ?",
                (session_key, limit),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def replace(self, record: SemanticMemoryRecord) -> None:
        if not isinstance(record, SemanticMemoryRecord):
            raise ValueError("record must be a SemanticMemoryRecord.")
        with self._lock:
            with self._conn:
                row = self._conn.execute(
                    "SELECT session_id FROM semantic_memory_records WHERE memory_id = ?",
                    (record.memory_id,),
                ).fetchone()
                if row is None:
                    raise ValueError(f"memory_id {record.memory_id!r} does not exist; nothing to replace.")
                if self._normalize(row["session_id"]) != self._normalize(record.session_id):
                    raise ValueError(
                        f"cannot move memory_id {record.memory_id!r} from session "
                        f"{row['session_id']!r} to {record.session_id!r}."
                    )
                # A plain UPDATE never touches `rowid`, which is exactly
                # what preserves this record's position for list_recent's
                # ordering -- matching InMemorySemanticMemory.replace's own
                # explicit "same position in the session's ordering" rule.
                self._conn.execute(
                    "UPDATE semantic_memory_records SET content = ?, created_at = ?, confidence = ?, "
                    "active = ?, source_event_ids_json = ? WHERE memory_id = ?",
                    (
                        record.content,
                        record.created_at.isoformat(),
                        record.confidence,
                        int(record.active),
                        json.dumps(list(record.source_event_ids)),
                        record.memory_id,
                    ),
                )

    def delete(self, memory_id: str) -> None:
        _require_non_empty_str("memory_id", memory_id)
        with self._lock:
            with self._conn:
                self._conn.execute("DELETE FROM semantic_memory_records WHERE memory_id = ?", (memory_id,))

    def clear(self, session_id: str) -> None:
        session_key = self._normalize(session_id)
        with self._lock:
            with self._conn:
                self._conn.execute("DELETE FROM semantic_memory_records WHERE session_id = ?", (session_key,))

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _row_to_record(self, row: sqlite3.Row) -> SemanticMemoryRecord:
        return SemanticMemoryRecord(
            memory_id=row["memory_id"],
            session_id=row["session_id"],
            content=row["content"],
            created_at=datetime.fromisoformat(row["created_at"]),
            source_event_ids=tuple(json.loads(row["source_event_ids_json"])),
            confidence=row["confidence"],
            active=bool(row["active"]),
        )

    def _normalize(self, session_id: str) -> str:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string.")
        return session_id.strip()


class SQLiteVectorIndex:
    """SQLite-backed `VectorIndex`.

    Mirrors `InMemoryVectorIndex`'s documented behavior exactly: identity
    key is the PAIR (session_id, memory_id) — same memory_id under a
    different session is a fully independent row; search loads the
    requested session's vectors and runs the SAME `cosine_similarity`
    function with the SAME (similarity DESC, memory_id ASC) tie-break, so
    ordering/determinism is identical to the in-memory index, just sourced
    from SQL rows instead of a dict.
    """

    def __init__(self, db_path: str, dimension: int, embedding_model_name: str):
        if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 1:
            raise ValueError("dimension must be an integer >= 1.")
        _require_non_empty_str("embedding_model_name", embedding_model_name)
        self._dimension = dimension
        self._embedding_model_name = embedding_model_name.strip()
        self._conn = db.connect(db_path)
        self._lock = threading.Lock()
        self._reject_if_incompatible_model_already_persisted()

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def embedding_model_name(self) -> str:
        return self._embedding_model_name

    def add(self, memory_id: str, session_id: str, vector: Vector) -> None:
        _require_non_empty_str("memory_id", memory_id)
        validated_vector = self._validate_vector(vector)
        session_key = self._normalize(session_id)
        with self._lock:
            with self._conn:
                existing = self._conn.execute(
                    "SELECT 1 FROM semantic_memory_vectors WHERE memory_id = ? AND session_id = ?",
                    (memory_id, session_key),
                ).fetchone()
                if existing is not None:
                    raise ValueError(
                        f"memory_id {memory_id!r} already exists in session {session_id!r}; "
                        "call remove() first to replace it."
                    )
                self._conn.execute(
                    "INSERT INTO semantic_memory_vectors "
                    "(memory_id, session_id, dimension, vector_json, embedding_model_name) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        memory_id,
                        session_key,
                        self._dimension,
                        json.dumps(list(validated_vector)),
                        self._embedding_model_name,
                    ),
                )

    def remove(self, memory_id: str, session_id: str) -> None:
        _require_non_empty_str("memory_id", memory_id)
        session_key = self._normalize(session_id)
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "DELETE FROM semantic_memory_vectors WHERE memory_id = ? AND session_id = ?",
                    (memory_id, session_key),
                )

    def search(self, session_id: str, query_vector: Vector, top_k: int) -> list[VectorSearchResult]:
        session_key = self._normalize(session_id)
        validated_query = self._validate_vector(query_vector)
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be an integer >= 1.")

        with self._lock:
            rows = self._conn.execute(
                "SELECT memory_id, vector_json FROM semantic_memory_vectors WHERE session_id = ?",
                (session_key,),
            ).fetchall()

        scored = [
            (row["memory_id"], cosine_similarity(validated_query, tuple(json.loads(row["vector_json"]))))
            for row in rows
        ]
        scored.sort(key=lambda item: (-item[1], item[0]))  # similarity DESC, memory_id ASC
        return [VectorSearchResult(memory_id=mid, similarity=sim) for mid, sim in scored[:top_k]]

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _reject_if_incompatible_model_already_persisted(self) -> None:
        """Construction-time guard (Step 38 requirement I): refuse to open
        a database whose persisted vectors were tagged with a DIFFERENT
        embedding model than the one this instance declares. Mirrors the
        EAGER, at-construction dimension check SemanticMemoryWriter/
        SemanticMemoryRetriever already apply -- a mismatch here breaks
        every future search without exception, so it is a wiring bug and
        belongs at wiring time."""
        rows = self._conn.execute("SELECT DISTINCT embedding_model_name FROM semantic_memory_vectors").fetchall()
        existing_names = {row["embedding_model_name"] for row in rows}
        if existing_names and existing_names != {self._embedding_model_name}:
            self._conn.close()
            raise ValueError(
                f"persisted vectors in this database were created with embedding model(s) "
                f"{sorted(existing_names)!r}, but this index was constructed with "
                f"{self._embedding_model_name!r}; refusing to open an index that would compare "
                "incompatible embeddings."
            )

    def _validate_vector(self, vector: object) -> Vector:
        values = _validate_numeric_vector(vector, label="vector")
        if len(values) != self._dimension:
            raise ValueError(f"vector dimension {len(values)} does not match index dimension {self._dimension}.")
        return values

    def _normalize(self, session_id: str) -> str:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string.")
        return session_id.strip()
