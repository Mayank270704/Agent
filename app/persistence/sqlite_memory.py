"""SQLite-backed ConversationMemory / SessionMemoryStore — Step 36.

These two classes implement the SAME Protocols already defined in
app/agent/memory.py (`ConversationMemory`, `SessionMemoryStore`) — no new
Protocol, no change to that module. They are a drop-in alternative to
`InMemoryConversationMemory`/`InMemorySessionMemoryStore`, selected at the
composition root (app/main.py) via the `MEMORY_BACKEND` setting
(app/config.py). `InMemory*` remain the default and the implementation
every existing test exercises; nothing above the Protocol boundary changes
when this backend is selected instead.

Design notes, matching app/agent/memory.py's own documented behavior as
closely as a SQL-backed store reasonably can:

    - FIFO message eviction (`max_messages_per_session`, default 20):
      identical semantics to InMemoryConversationMemory — the OLDEST
      messages are dropped once a session's count would exceed the limit.
      `id` (an AUTOINCREMENT primary key) gives a strictly increasing,
      unambiguous insertion order per session, used both for `get_messages()`
      ordering and for selecting which rows to evict.

    - LRU session-count eviction (`max_sessions`, default 1000): identical
      policy to InMemorySessionMemoryStore — only a newly INSERTED session
      can trigger eviction (never a mere touch of an existing one), and the
      victim is always the least-recently-accessed session, never the one
      just created. `last_accessed_seq` (a monotonically increasing integer,
      recomputed under the same lock on every touch) is the ordering key,
      not `last_accessed_at` (a wall-clock timestamp) alone — two sessions
      touched within the same clock tick must still have an unambiguous
      recency order, exactly as OrderedDict's exact sequencing guarantees
      in the in-memory implementation. `last_accessed_at` is kept alongside
      it purely as human-readable/debuggable metadata; no eviction decision
      ever reads it.

    - Concurrency: one `threading.Lock` per store instance guards every
      access to the shared `sqlite3.Connection` (opened with
      `check_same_thread=False` so it can be used from FastAPI's
      threadpool-executed request handlers) — the same role
      `InMemorySessionMemoryStore`'s own lock plays, and for the identical
      reason: SQLite connections are not safe for concurrent use from
      multiple threads without external synchronization. Within that lock,
      each logically-atomic operation (a touch-or-create-plus-evict, or an
      append-plus-evict) runs inside one `with connection:` block, so a
      failure partway through rolls back the whole operation rather than
      leaving a partial mutation.

    - `SQLiteConversationMemory` holds no in-process message list of its
      own — it is a thin, stateless view over rows scoped by `session_id`.
      `SQLiteSessionMemoryStore.get_memory()` therefore returns a FRESH view
      object on every call (unlike `InMemorySessionMemoryStore`, which
      returns the SAME cached instance for a given session_id, because that
      instance IS the storage). This is a deliberate, harmless deviation:
      two different view instances for the same session_id always observe
      the same underlying rows, so nothing observable depends on object
      identity — and caching view objects here would just reintroduce, in
      Python, an eviction problem the database already solves.

    - What is stored is exactly what `ConversationMemory` represents: a
      role and a message. No tool call, observation, plan, raw model
      output, or telemetry field is ever written here — the
      `conversation_messages` table has no column for any of them.
"""
from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone

from app.persistence import db


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteConversationMemory:
    """SQLite-backed `ConversationMemory` for exactly one session.

    Always constructed by `SQLiteSessionMemoryStore.get_memory()` in
    practice — the session row it depends on (a foreign-key target for
    every message it writes) is created there, not here.
    """

    def __init__(
        self,
        connection: sqlite3.Connection,
        lock: threading.Lock,
        session_id: str,
        max_messages: int = 20,
    ):
        if max_messages < 1:
            raise ValueError("max_messages must be >= 1.")
        self._conn = connection
        self._lock = lock
        self._session_id = session_id
        self._max_messages = max_messages

    def add_user_message(self, message: str) -> None:
        self._append("user", message)

    def add_assistant_message(self, message: str) -> None:
        self._append("assistant", message)

    def get_messages(self) -> list[dict[str, str]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content FROM conversation_messages WHERE session_id = ? ORDER BY id ASC",
                (self._session_id,),
            ).fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in rows]

    def clear(self) -> None:
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "DELETE FROM conversation_messages WHERE session_id = ?", (self._session_id,)
                )

    def _append(self, role: str, message: str) -> None:
        if not isinstance(message, str) or not message.strip():
            raise ValueError(f"{role} message cannot be empty.")
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "INSERT INTO conversation_messages (session_id, role, content) VALUES (?, ?, ?)",
                    (self._session_id, role, message),
                )
                count = self._conn.execute(
                    "SELECT COUNT(*) AS c FROM conversation_messages WHERE session_id = ?",
                    (self._session_id,),
                ).fetchone()["c"]
                overflow = count - self._max_messages
                if overflow > 0:
                    self._conn.execute(
                        "DELETE FROM conversation_messages WHERE id IN ("
                        "SELECT id FROM conversation_messages WHERE session_id = ? "
                        "ORDER BY id ASC LIMIT ?)",
                        (self._session_id, overflow),
                    )


class SQLiteSessionMemoryStore:
    """SQLite-backed `SessionMemoryStore`: maps a session_id to its own
    `SQLiteConversationMemory`, all persisted in one SQLite database file.
    """

    def __init__(self, db_path: str, max_messages_per_session: int = 20, max_sessions: int = 1000):
        if max_messages_per_session < 1:
            raise ValueError("max_messages_per_session must be >= 1.")
        if max_sessions < 1:
            raise ValueError("max_sessions must be >= 1.")
        self._max_messages_per_session = max_messages_per_session
        self._max_sessions = max_sessions
        # db.connect() fails loudly (raises) if the path is invalid or the
        # database cannot be opened/created — there is no fallback to an
        # in-memory store here; a configured-but-unavailable SQLite backend
        # must fail application startup, not silently degrade (Step 35/36).
        self._conn = db.connect(db_path)
        self._lock = threading.Lock()

    def get_memory(self, session_id: str) -> SQLiteConversationMemory:
        normalized = self._normalize(session_id)
        with self._lock:
            with self._conn:
                existing = self._conn.execute(
                    "SELECT 1 FROM sessions WHERE session_id = ?", (normalized,)
                ).fetchone()
                now = _utc_now_iso()
                next_seq = self._conn.execute(
                    "SELECT COALESCE(MAX(last_accessed_seq), 0) + 1 AS n FROM sessions"
                ).fetchone()["n"]

                if existing is not None:
                    # Touch only: matches InMemorySessionMemoryStore's
                    # `move_to_end` on an existing session — eviction is
                    # never checked on a mere touch, only on insertion of a
                    # brand-new session below.
                    self._conn.execute(
                        "UPDATE sessions SET last_accessed_at = ?, last_accessed_seq = ? WHERE session_id = ?",
                        (now, next_seq, normalized),
                    )
                else:
                    self._conn.execute(
                        "INSERT INTO sessions (session_id, created_at, last_accessed_at, last_accessed_seq) "
                        "VALUES (?, ?, ?, ?)",
                        (normalized, now, now, next_seq),
                    )
                    count = self._conn.execute("SELECT COUNT(*) AS c FROM sessions").fetchone()["c"]
                    if count > self._max_sessions:
                        # The least-recently-used session, never the one
                        # just inserted (which always holds the highest
                        # seq value and therefore can never be the MIN).
                        # ON DELETE CASCADE removes its messages too.
                        self._conn.execute(
                            "DELETE FROM sessions WHERE session_id = ("
                            "SELECT session_id FROM sessions ORDER BY last_accessed_seq ASC LIMIT 1)"
                        )

        return SQLiteConversationMemory(
            self._conn, self._lock, normalized, max_messages=self._max_messages_per_session
        )

    def clear_session(self, session_id: str) -> None:
        normalized = self._normalize(session_id)
        with self._lock:
            with self._conn:
                # ON DELETE CASCADE removes every conversation_messages row
                # for this session too; a no-op if it was never created.
                self._conn.execute("DELETE FROM sessions WHERE session_id = ?", (normalized,))

    def close(self) -> None:
        """Release the underlying SQLite connection. Not required for
        normal operation (SQLite persists each committed transaction
        immediately; there is no buffered state an unclosed connection
        would lose) — provided for clean shutdown and so a second store
        instance can reliably reopen the same database file afterward
        (used by the persistence-across-restart tests)."""
        with self._lock:
            self._conn.close()

    def health_check(self) -> int:
        """Cheap liveness probe (Step 39): reuses this store's ALREADY-OPEN
        connection — opens nothing new, scans nothing — to confirm the
        underlying SQLite file is still reachable and reports its recorded
        schema version. Raises `sqlite3.Error`/`db.SchemaVersionError` if
        not. app/main.py's `/health` endpoint uses this ONE store as a
        representative probe for the whole shared database file (all five
        persistent stores point at the same file in production), rather
        than separately pinging all five -- a health check must stay O(1)
        and cheap on every request, and a shared file's reachability
        failure would surface identically through any one of them.
        """
        with self._lock:
            return db.read_schema_version(self._conn)

    def _normalize(self, session_id: str) -> str:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string.")
        return session_id.strip()
