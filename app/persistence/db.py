"""SQLite connection setup and schema/migration handling for persistent
memory backends (Step 36, extended by Step 37 and Step 38).

This module owns exactly two things, deliberately kept together in one
file rather than split further (unnecessary abstractions were explicitly
ruled out for this project; connection setup / schema application are a
single cohesive concern — a connection is never handed to a caller before
its schema has been verified or brought up to date):

    1. Opening a SQLite connection with the pragmas this architecture
       requires (foreign keys enforced, WAL journaling) and creating its
       parent directory if needed.
    2. Deterministic, explicit schema versioning via NUMBERED, INCREMENTAL
       migrations: a fresh database (version 0) has every migration up to
       CURRENT_SCHEMA_VERSION applied, in order; an existing database at
       some earlier recorded version has only the REMAINING migrations
       applied, preserving its existing data; an existing database at a
       version newer than this code supports fails loudly rather than
       guessing how to read data it was not built to handle; a version gap
       this code has no migration for (an internal consistency problem,
       not a normal deployment scenario) also fails loudly rather than
       silently skipping a step.

Step 36 shipped with only one schema version and necessarily no migration
path yet (there was nothing to migrate FROM). Step 37 is the first actual
schema change, so migrations became real: `_MIGRATIONS[2]` adds the
episodic-memory tables without touching `sessions`/`conversation_messages`
or any data already in them. Step 38 adds `_MIGRATIONS[3]`: semantic-memory
records and their vectors, again without touching any earlier table.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

CURRENT_SCHEMA_VERSION = 3

# Each entry is the set of DDL statements that take the schema from
# (version - 1) to `version`. IF NOT EXISTS / IF NOT EXISTS everywhere:
# these are also re-run, harmlessly, whenever a database already at that
# version is reopened (see _initialize_schema below).
_MIGRATIONS: dict[int, tuple[str, ...]] = {
    1: (
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            last_accessed_at TEXT NOT NULL,
            last_accessed_seq INTEGER NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_sessions_last_accessed_seq ON sessions(last_accessed_seq)",
        """
        CREATE TABLE IF NOT EXISTS conversation_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
            role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
            content TEXT NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_conversation_messages_session_id ON conversation_messages(session_id, id)",
    ),
    2: (
        # EPISODIC memory gets its OWN session-tracking table, deliberately
        # separate from `sessions` above (Step 37 design decision): in the
        # in-memory implementations, InMemoryEpisodicMemory and
        # InMemorySessionMemoryStore are two entirely independent
        # OrderedDicts with their own, unrelated max_sessions/LRU
        # bookkeeping -- one has no knowledge of the other. An FK against
        # the shared `sessions` table would silently invent a dependency
        # that does not exist today (episodic writes would start requiring
        # a session_store touch first) and would make the two stores'
        # independent eviction policies interfere with each other. A
        # dedicated `episodic_sessions` table preserves that independence
        # exactly while still giving episodic records a real FK + CASCADE
        # target, consistent with how Step 36 uses FK + CASCADE.
        """
        CREATE TABLE IF NOT EXISTS episodic_sessions (
            session_id TEXT PRIMARY KEY,
            last_accessed_seq INTEGER NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_episodic_sessions_last_accessed_seq ON episodic_sessions(last_accessed_seq)",
        # `event_id` is deliberately NOT the primary key: the existing
        # EpisodicMemory Protocol / InMemoryEpisodicMemory implementation
        # intentionally does not reject duplicate event_ids (see
        # app/agent/episodic_memory.py's module docstring, "Step 15, Part
        # 6") -- a PRIMARY KEY here would silently change that behavior.
        # `id` (AUTOINCREMENT) is the real identity/ordering key, exactly
        # mirroring conversation_messages' own pattern from Step 36.
        """
        CREATE TABLE IF NOT EXISTS episodic_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT NOT NULL,
            session_id TEXT NOT NULL REFERENCES episodic_sessions(session_id) ON DELETE CASCADE,
            event_type TEXT NOT NULL,
            summary TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            metadata_json TEXT NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_episodic_records_session_id ON episodic_records(session_id, id)",
    ),
    3: (
        # SEMANTIC memory records. Deliberately NO foreign key to `sessions`
        # (same reasoning as episodic_sessions above): InMemorySemanticMemory
        # has zero dependency on InMemorySessionMemoryStore, and in fact has
        # no session-COUNT cap/LRU concept at all (unlike episodic memory) --
        # SemanticMemoryWriter is the sole retention owner (see
        # app/semantic_memory_wiring.py), so there is no session-bookkeeping
        # table needed here at all, just a plain indexed session_id column.
        """
        CREATE TABLE IF NOT EXISTS semantic_memory_records (
            memory_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            confidence REAL NOT NULL,
            active INTEGER NOT NULL,
            source_event_ids_json TEXT NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_semantic_records_session ON semantic_memory_records(session_id)",
        # SEMANTIC memory VECTORS. Deliberately NO foreign key to
        # semantic_memory_records either, even though every production
        # vector does in practice belong to exactly one record: VectorIndex
        # and SemanticMemoryStore are independent Protocols BY DESIGN (see
        # app/agent/vector_index.py's "Dependency direction" section --
        # "this index must not assume SemanticMemoryStore's guarantee
        # anyway"), and the in-memory pair enforces nothing across them
        # either. A cross-table FK here would silently couple two
        # components the architecture explicitly keeps independent, and
        # would reject the Protocol-legal (if practically unused) case of
        # adding a vector with no backing record. Composite PRIMARY KEY
        # (memory_id, session_id) matches InMemoryVectorIndex's own identity
        # key exactly (Part 9 of its docstring): same memory_id + same
        # session -> rejected by the store (see SQLiteVectorIndex.add);
        # same memory_id + DIFFERENT session -> fully independent rows.
        """
        CREATE TABLE IF NOT EXISTS semantic_memory_vectors (
            memory_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            dimension INTEGER NOT NULL,
            vector_json TEXT NOT NULL,
            embedding_model_name TEXT NOT NULL,
            PRIMARY KEY (memory_id, session_id)
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_semantic_vectors_session ON semantic_memory_vectors(session_id)",
    ),
}


class SchemaVersionError(RuntimeError):
    """Raised when an existing database's recorded schema_version cannot be
    safely opened by this code: a version newer than this code supports, or
    a version gap with no migration defined for it. Never silently
    reinterpreted; see the module docstring."""


def read_schema_version(conn: sqlite3.Connection) -> int:
    """Cheap liveness/health probe (Step 39): read the recorded
    schema_version using an ALREADY-OPEN connection — no new connection is
    opened, no data table is scanned, just one indexed read against a
    single-row metadata table. Raises `sqlite3.Error` if the connection or
    underlying file is no longer usable (the caller's job to classify as
    "unhealthy"), and `SchemaVersionError` in the (should-be-impossible,
    since `connect()` always writes this row) case of a missing row.

    Deliberately does NOT re-run `_initialize_schema` or touch
    `semantic_memory_records`/`semantic_memory_vectors` — a health check
    must stay O(1) regardless of how much data a deployment has
    accumulated, never scan or validate the full dataset.
    """
    row = conn.execute("SELECT value FROM schema_meta WHERE key = 'schema_version'").fetchone()
    if row is None:
        raise SchemaVersionError("schema_meta has no recorded schema_version row.")
    return int(row["value"])


def connect(db_path: str) -> sqlite3.Connection:
    """Open (creating if necessary) a SQLite database at `db_path`,
    configured for this architecture's requirements, with its schema
    already verified, created, or migrated as needed.

    `db_path` must be a non-empty string — an empty/blank value is a
    configuration mistake, not a usable path, and is rejected immediately
    rather than attempting to open an unintended location (e.g. the
    process's current directory).
    """
    if not isinstance(db_path, str) or not db_path.strip():
        raise ValueError("db_path must be a non-empty string.")

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    _initialize_schema(conn)
    return conn


def _initialize_schema(conn: sqlite3.Connection) -> None:
    with conn:
        conn.execute("CREATE TABLE IF NOT EXISTS schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        row = conn.execute("SELECT value FROM schema_meta WHERE key = 'schema_version'").fetchone()
        existing_version = int(row["value"]) if row is not None else 0

        if existing_version > CURRENT_SCHEMA_VERSION:
            raise SchemaVersionError(
                f"database schema_version {existing_version} is newer than this code supports "
                f"({CURRENT_SCHEMA_VERSION}); refusing to open it."
            )

        for version in range(existing_version + 1, CURRENT_SCHEMA_VERSION + 1):
            statements = _MIGRATIONS.get(version)
            if statements is None:
                raise SchemaVersionError(
                    f"no migration is defined to reach schema version {version}; refusing to "
                    "open this database."
                )
            for statement in statements:
                conn.execute(statement)

        if existing_version != CURRENT_SCHEMA_VERSION:
            conn.execute(
                "INSERT INTO schema_meta (key, value) VALUES ('schema_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (str(CURRENT_SCHEMA_VERSION),),
            )
