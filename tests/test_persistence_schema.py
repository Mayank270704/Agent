"""Step 36/37/38, category H: schema/migration handling
(app/persistence/db.py).

Fully offline, pytest tmp_path databases only.
"""
from __future__ import annotations

import sqlite3

import pytest

from app.persistence import db


def test_fresh_database_initializes_schema_and_records_version(tmp_path) -> None:
    conn = db.connect(str(tmp_path / "fresh.db"))

    row = conn.execute("SELECT value FROM schema_meta WHERE key = 'schema_version'").fetchone()
    assert row["value"] == str(db.CURRENT_SCHEMA_VERSION)

    # Tables exist and are queryable (would raise sqlite3.OperationalError
    # if schema initialization silently failed to create them).
    conn.execute("SELECT * FROM sessions")
    conn.execute("SELECT * FROM conversation_messages")
    conn.execute("SELECT * FROM episodic_sessions")
    conn.execute("SELECT * FROM episodic_records")
    conn.execute("SELECT * FROM semantic_memory_records")
    conn.execute("SELECT * FROM semantic_memory_vectors")
    conn.close()


def test_reopening_an_up_to_date_database_succeeds(tmp_path) -> None:
    db_path = str(tmp_path / "reopen.db")
    conn1 = db.connect(db_path)
    conn1.close()

    conn2 = db.connect(db_path)  # must not raise -- matching version
    conn2.close()


def test_reopening_preserves_existing_data(tmp_path) -> None:
    db_path = str(tmp_path / "reopen_data.db")
    conn1 = db.connect(db_path)
    with conn1:
        conn1.execute(
            "INSERT INTO sessions (session_id, created_at, last_accessed_at, last_accessed_seq) "
            "VALUES ('a', 't', 't', 1)"
        )
    conn1.close()

    conn2 = db.connect(db_path)
    row = conn2.execute("SELECT session_id FROM sessions WHERE session_id = 'a'").fetchone()
    conn2.close()

    assert row["session_id"] == "a"


def test_future_schema_version_fails_loudly(tmp_path) -> None:
    db_path = str(tmp_path / "future.db")
    raw = sqlite3.connect(db_path)
    raw.execute("CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    raw.execute("INSERT INTO schema_meta (key, value) VALUES ('schema_version', '999')")
    raw.commit()
    raw.close()

    with pytest.raises(db.SchemaVersionError):
        db.connect(db_path)


def test_a_version_0_database_migrates_forward_successfully(tmp_path) -> None:
    """Step 36 version: this asserted version 0 failed loudly, because no
    migration path existed yet at all -- CURRENT_SCHEMA_VERSION was 1 with
    no predecessor to migrate FROM. Step 37 introduced a real, numbered
    migration path (see app/persistence/db.py's _MIGRATIONS), so version 0
    is now a legitimate starting point: every migration up to
    CURRENT_SCHEMA_VERSION is applied in order, and the database ends up
    fully current. This test's assertion is intentionally the OPPOSITE of
    its Step 36 version -- the premise it tested ("no migration exists")
    is exactly what Step 37 made false."""
    db_path = str(tmp_path / "version_zero.db")
    raw = sqlite3.connect(db_path)
    raw.execute("CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    raw.execute("INSERT INTO schema_meta (key, value) VALUES ('schema_version', '0')")
    raw.commit()
    raw.close()

    conn = db.connect(db_path)  # must not raise
    row = conn.execute("SELECT value FROM schema_meta WHERE key = 'schema_version'").fetchone()
    conn.execute("SELECT * FROM sessions")
    conn.execute("SELECT * FROM conversation_messages")
    conn.execute("SELECT * FROM episodic_records")
    conn.close()

    assert row["value"] == str(db.CURRENT_SCHEMA_VERSION)


def test_a_version_gap_with_no_defined_migration_fails_loudly(tmp_path, monkeypatch) -> None:
    """Defensive/internal-consistency test (Step 37): if a future schema
    bump ever forgets to add its migration to _MIGRATIONS, connect() must
    fail loudly rather than silently skip the missing step. Simulated by
    monkeypatching _MIGRATIONS to have a gap -- there is no way to
    construct this from real on-disk data today, since _MIGRATIONS is
    always complete in shipped code."""
    db_path = str(tmp_path / "gap.db")
    monkeypatch.setattr(db, "_MIGRATIONS", {1: db._MIGRATIONS[1]})  # pretend version 2 was never added

    with pytest.raises(db.SchemaVersionError):
        db.connect(db_path)


def test_version_mismatch_does_not_create_or_alter_data_tables(tmp_path) -> None:
    """A refused database must be left exactly as it was -- not partially
    upgraded, not left with half-created tables."""
    db_path = str(tmp_path / "refused.db")
    raw = sqlite3.connect(db_path)
    raw.execute("CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    raw.execute("INSERT INTO schema_meta (key, value) VALUES ('schema_version', '999')")
    raw.commit()

    with pytest.raises(db.SchemaVersionError):
        db.connect(db_path)

    tables = {
        row[0]
        for row in raw.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    }
    raw.close()
    assert "sessions" not in tables
    assert "conversation_messages" not in tables
    assert "episodic_sessions" not in tables
    assert "episodic_records" not in tables
    assert "semantic_memory_records" not in tables
    assert "semantic_memory_vectors" not in tables


def test_foreign_keys_pragma_is_enabled(tmp_path) -> None:
    conn = db.connect(str(tmp_path / "fk.db"))
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    conn.close()


def test_wal_journal_mode_is_enabled(tmp_path) -> None:
    conn = db.connect(str(tmp_path / "wal.db"))
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    conn.close()
    assert mode.lower() == "wal"


def test_deleting_a_session_cascades_to_its_messages() -> None:
    """Confirms the ON DELETE CASCADE foreign key actually behaves as
    declared, at the schema level (not just through the higher-level
    SQLiteSessionMemoryStore.clear_session, which is tested separately)."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    db._initialize_schema(conn)

    with conn:
        conn.execute(
            "INSERT INTO sessions (session_id, created_at, last_accessed_at, last_accessed_seq) "
            "VALUES ('a', 't', 't', 1)"
        )
        conn.execute(
            "INSERT INTO conversation_messages (session_id, role, content) VALUES ('a', 'user', 'hi')"
        )
    with conn:
        conn.execute("DELETE FROM sessions WHERE session_id = 'a'")

    row = conn.execute("SELECT COUNT(*) AS c FROM conversation_messages WHERE session_id = 'a'").fetchone()
    conn.close()
    assert row["c"] == 0


def test_deleting_an_episodic_session_cascades_to_its_records() -> None:
    """Same proof as above, for episodic_sessions -> episodic_records
    (Step 37) -- a deliberately SEPARATE FK target from `sessions`, since
    episodic memory's session bookkeeping is independent of conversation/
    session memory's (see app/persistence/db.py's migration-2 comment)."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    db._initialize_schema(conn)

    with conn:
        conn.execute("INSERT INTO episodic_sessions (session_id, last_accessed_seq) VALUES ('a', 1)")
        conn.execute(
            "INSERT INTO episodic_records (event_id, session_id, event_type, summary, timestamp, metadata_json) "
            "VALUES ('e1', 'a', 'conversation_completed', 'hi', 't', '{}')"
        )
    with conn:
        conn.execute("DELETE FROM episodic_sessions WHERE session_id = 'a'")

    row = conn.execute("SELECT COUNT(*) AS c FROM episodic_records WHERE session_id = 'a'").fetchone()
    conn.close()
    assert row["c"] == 0


def test_connect_rejects_a_blank_db_path() -> None:
    with pytest.raises(ValueError):
        db.connect("")
    with pytest.raises(ValueError):
        db.connect("   ")


def test_connect_creates_missing_parent_directories(tmp_path) -> None:
    nested_path = tmp_path / "nested" / "dirs" / "chatbot.db"
    conn = db.connect(str(nested_path))
    conn.close()

    assert nested_path.exists()


# ===========================================================================
# Step 38 — v2 -> v3 migration (adds semantic_memory_records/vectors)
# ===========================================================================

def test_a_version_2_database_migrates_to_v3_preserving_existing_data(tmp_path) -> None:
    """A database left at v2 by a Step-37-era process (before Step 38
    existed) must gain the semantic-memory tables on next open, without
    losing or altering anything already in its v1/v2 tables."""
    db_path = str(tmp_path / "v2_to_v3.db")
    conn1 = db.connect(db_path)
    with conn1:
        conn1.execute(
            "INSERT INTO sessions (session_id, created_at, last_accessed_at, last_accessed_seq) "
            "VALUES ('a', 't', 't', 1)"
        )
        conn1.execute(
            "INSERT INTO conversation_messages (session_id, role, content) VALUES ('a', 'user', 'hi')"
        )
    conn1.close()

    # Simulate a pre-Step-38 database by rewriting its recorded version
    # back down to 2 (what Step 37 would have left behind).
    raw = sqlite3.connect(db_path)
    raw.execute("UPDATE schema_meta SET value = '2' WHERE key = 'schema_version'")
    raw.commit()
    raw.close()

    conn2 = db.connect(db_path)  # must not raise; must add the v3 tables
    version_row = conn2.execute("SELECT value FROM schema_meta WHERE key = 'schema_version'").fetchone()
    session_row = conn2.execute("SELECT session_id FROM sessions WHERE session_id = 'a'").fetchone()
    message_row = conn2.execute("SELECT content FROM conversation_messages WHERE session_id = 'a'").fetchone()
    conn2.execute("SELECT * FROM semantic_memory_records")
    conn2.execute("SELECT * FROM semantic_memory_vectors")
    conn2.close()

    assert version_row["value"] == str(db.CURRENT_SCHEMA_VERSION)
    assert session_row["session_id"] == "a"
    assert message_row["content"] == "hi"


def test_semantic_memory_vectors_table_has_no_foreign_key_to_records(tmp_path) -> None:
    """Deliberate design choice (Step 38, see sqlite_semantic_memory.py's
    module docstring): VectorIndex and SemanticMemoryStore stay
    independent, exactly like their in-memory counterparts -- a vector can
    be added with no backing record, matching InMemoryVectorIndex's own
    documented independence from InMemorySemanticMemory."""
    conn = db.connect(str(tmp_path / "no_fk.db"))

    with conn:
        conn.execute(
            "INSERT INTO semantic_memory_vectors "
            "(memory_id, session_id, dimension, vector_json, embedding_model_name) "
            "VALUES ('orphan', 's', 4, '[1.0, 0.0, 0.0, 0.0]', 'model-x')"
        )  # must not raise -- no FK constraint ties this to semantic_memory_records

    row = conn.execute("SELECT memory_id FROM semantic_memory_vectors WHERE memory_id = 'orphan'").fetchone()
    conn.close()
    assert row["memory_id"] == "orphan"
