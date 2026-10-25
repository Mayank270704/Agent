"""SQLite-backed EpisodicMemory — Step 37.

`SQLiteEpisodicMemory` implements the SAME `EpisodicMemory` Protocol already
defined in app/agent/episodic_memory.py — no change to that module, no new
Protocol. It is a drop-in alternative to `InMemoryEpisodicMemory`, selected
at the composition root (app/main.py) via the SAME `MEMORY_BACKEND` setting
Step 36 introduced (app/config.py) — one flag selects the backend for
conversation/session AND episodic memory together; there is no separate
flag, matching the "do not create unnecessary abstractions" instruction.

Design notes, matching app/agent/episodic_memory.py's documented behavior
(the "Step 15 — retention/lifecycle contract" section) as closely as a
SQL-backed store reasonably can:

    - Ordering: `get_recent(session_id, limit)` returns the `limit` most
      recent records NEWEST FIRST — identical to the in-memory
      implementation's `list(reversed(records[-limit:]))`. `id` (an
      AUTOINCREMENT primary key) gives a strictly increasing, unambiguous
      insertion order per session; `ORDER BY id DESC LIMIT ?` produces
      exactly the same sequence.

    - `event_id` is deliberately NOT a uniqueness constraint. The in-memory
      implementation's own module docstring documents, as an intentional
      architectural decision (not an oversight), that duplicate event_ids
      are accepted silently — the Protocol has no id-addressable lookup,
      so nothing depends on uniqueness. `id` (the surrogate AUTOINCREMENT
      key) is the only identity/ordering key this table actually needs;
      `event_id` is stored as plain, unconstrained data, preserving that
      exact behavior.

    - Per-session record cap (`max_records_per_session`, default 100):
      identical FIFO eviction to the in-memory version — the oldest
      records for that session are dropped once its count would exceed
      the limit, scoped to that session only.

    - Session-count cap (`max_sessions`, default 1000): identical LRU
      policy to the in-memory version. Episodic memory tracks its OWN
      session recency, independent of conversation/session memory's
      `sessions` table — see `episodic_sessions` in app/persistence/db.py
      for why these are deliberately separate tables. Both `add()` and
      `get_recent()` refresh an existing session's recency (matching the
      in-memory version's `move_to_end` calls in both methods); only
      inserting a brand-new session can trigger eviction, and the evicted
      session is always the least-recently-used one, never the one just
      touched.

    - Concurrency: one `threading.Lock` per store instance guards every
      access to the shared connection, the same role every other
      persistence-layer lock plays in this codebase (Step 36). Each
      logically atomic operation runs inside one `with connection:` block.

    - What is stored is exactly what `EpisodicMemoryRecord` represents:
      event_id, session_id, event_type, summary, timestamp, and the small
      structured metadata dict the ONE production caller
      (AgentOrchestrator._maybe_record_episode) populates. No tool call,
      observation, plan, raw model output, or telemetry field is ever
      written here — the `episodic_records` table has no column for any
      of them, and `metadata_json` round-trips only whatever the caller
      already put in `record.metadata`.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime

from app.agent.episodic_memory import EpisodicMemoryRecord
from app.persistence import db


def _row_to_record(row: sqlite3.Row) -> EpisodicMemoryRecord:
    return EpisodicMemoryRecord(
        event_id=row["event_id"],
        session_id=row["session_id"],
        event_type=row["event_type"],
        summary=row["summary"],
        timestamp=datetime.fromisoformat(row["timestamp"]),
        metadata=json.loads(row["metadata_json"]),
    )


class SQLiteEpisodicMemory:
    """SQLite-backed `EpisodicMemory`: session-partitioned, persistent
    storage for `EpisodicMemoryRecord` objects."""

    def __init__(self, db_path: str, max_records_per_session: int = 100, max_sessions: int = 1000):
        if max_records_per_session < 1:
            raise ValueError("max_records_per_session must be >= 1.")
        if max_sessions < 1:
            raise ValueError("max_sessions must be >= 1.")
        self._max_records_per_session = max_records_per_session
        self._max_sessions = max_sessions
        # db.connect() fails loudly if the path is invalid or the database
        # cannot be opened/created -- no fallback to an in-memory store
        # here, matching SQLiteSessionMemoryStore's identical rule (Step 36).
        self._conn = db.connect(db_path)
        self._lock = threading.Lock()

    def add(self, record: EpisodicMemoryRecord) -> None:
        if not isinstance(record, EpisodicMemoryRecord):
            raise ValueError("record must be an EpisodicMemoryRecord.")
        session_key = self._normalize(record.session_id)
        with self._lock:
            with self._conn:
                self._touch_session(session_key)

                self._conn.execute(
                    "INSERT INTO episodic_records "
                    "(event_id, session_id, event_type, summary, timestamp, metadata_json) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        record.event_id,
                        session_key,
                        record.event_type,
                        record.summary,
                        record.timestamp.isoformat(),
                        json.dumps(dict(record.metadata)),
                    ),
                )
                count = self._conn.execute(
                    "SELECT COUNT(*) AS c FROM episodic_records WHERE session_id = ?", (session_key,)
                ).fetchone()["c"]
                overflow = count - self._max_records_per_session
                if overflow > 0:
                    self._conn.execute(
                        "DELETE FROM episodic_records WHERE id IN ("
                        "SELECT id FROM episodic_records WHERE session_id = ? "
                        "ORDER BY id ASC LIMIT ?)",
                        (session_key, overflow),
                    )

    def get_recent(self, session_id: str, limit: int = 10) -> list[EpisodicMemoryRecord]:
        session_key = self._normalize(session_id)
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("limit must be an integer >= 1.")
        with self._lock:
            with self._conn:
                # Matches InMemoryEpisodicMemory.get_recent: only refresh
                # recency for a session that ALREADY exists -- never create
                # a new (empty) session entry merely by reading it.
                self._touch_session(session_key, create=False)

                rows = self._conn.execute(
                    "SELECT event_id, session_id, event_type, summary, timestamp, metadata_json "
                    "FROM episodic_records WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                    (session_key, limit),
                ).fetchall()
        return [_row_to_record(row) for row in rows]

    def clear(self, session_id: str) -> None:
        session_key = self._normalize(session_id)
        with self._lock:
            with self._conn:
                # ON DELETE CASCADE removes every episodic_records row for
                # this session too; a no-op if it was never created.
                self._conn.execute("DELETE FROM episodic_sessions WHERE session_id = ?", (session_key,))

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _touch_session(self, session_key: str, *, create: bool = True) -> None:
        """Marks `session_key` as most-recently-used, creating it if it
        doesn't exist yet (unless `create=False`) and evicting the
        least-recently-used session if a NEW insertion pushes the session
        count over `max_sessions`. Must be called while holding `self._lock`
        and inside an open `self._conn` transaction."""
        existing = self._conn.execute(
            "SELECT 1 FROM episodic_sessions WHERE session_id = ?", (session_key,)
        ).fetchone()
        next_seq = self._conn.execute(
            "SELECT COALESCE(MAX(last_accessed_seq), 0) + 1 AS n FROM episodic_sessions"
        ).fetchone()["n"]

        if existing is not None:
            self._conn.execute(
                "UPDATE episodic_sessions SET last_accessed_seq = ? WHERE session_id = ?",
                (next_seq, session_key),
            )
            return

        if not create:
            return

        self._conn.execute(
            "INSERT INTO episodic_sessions (session_id, last_accessed_seq) VALUES (?, ?)",
            (session_key, next_seq),
        )
        session_count = self._conn.execute("SELECT COUNT(*) AS c FROM episodic_sessions").fetchone()["c"]
        if session_count > self._max_sessions:
            # The least-recently-used session, never the one just inserted
            # (which always holds the highest seq value). ON DELETE CASCADE
            # removes its records too.
            self._conn.execute(
                "DELETE FROM episodic_sessions WHERE session_id = ("
                "SELECT session_id FROM episodic_sessions ORDER BY last_accessed_seq ASC LIMIT 1)"
            )

    def _normalize(self, session_id: str) -> str:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string.")
        return session_id.strip()
