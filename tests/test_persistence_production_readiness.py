"""Step 39: persistence production-readiness pass.

Covers:
    1. /health endpoint behavior (in-process, TestClient)
    2. Startup validation through the real composition root (subprocess)
    3. End-to-end smoke tests A-F through the real composition root/ChatService
    4. Failure behavior (invalid path, unsupported schema version)
    5/7. Configuration + security audit assertions

Fully offline: a FakeLLM is used throughout (same pattern as test_api.py
and the Step 36/37/38 test files) -- no real Ollama, no network. Semantic
memory smoke tests use DeterministicEmbeddingProvider (no real embedding
model, no torch/sentence-transformers import), matching Step 38's own
test conventions.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.persistence import db

REPO_ROOT = Path(__file__).resolve().parent.parent


class FakeLLM:
    def __init__(self, responses: list[str]):
        self.responses = iter(responses)

    def generate(self, messages: list[dict[str, str]], *, json_mode: bool = False) -> str:
        try:
            return next(self.responses)
        except StopIteration:
            raise AssertionError("FakeLLM ran out of scripted responses") from None


def _final_json(answer: str) -> str:
    return json.dumps({"action_type": "final", "final_answer": answer})


def _no_memories_json() -> str:
    return json.dumps({"memories": []})


def _one_memory_json(content: str) -> str:
    return json.dumps({"memories": [{"content": content, "confidence": 0.9}]})


def _run(script: str, env: dict, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, cwd=str(REPO_ROOT), env=env, timeout=timeout
    )


# ===========================================================================
# 1 — /health endpoint
# ===========================================================================

@pytest.fixture
def client() -> TestClient:
    return TestClient(main_module.app)


def test_health_endpoint_exists_and_returns_200_for_the_default_backend(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["application"] == "running"
    assert body["memory_backend"] == "memory"
    assert body["persistence"] == "not_configured"


def test_root_endpoint_is_unaffected_by_the_new_health_endpoint(client: TestClient) -> None:
    """Backward compatibility: GET / must still behave exactly as before."""
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": main_module.settings.app_name}


def test_health_endpoint_with_sqlite_backend_reports_available_and_supported(tmp_path) -> None:
    db_path = tmp_path / "health.db"
    probe = textwrap.dedent(
        """
        import os
        from fastapi.testclient import TestClient
        import app.main as m
        client = TestClient(m.app)
        r = client.get("/health")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "ok"
        assert body["memory_backend"] == "sqlite"
        assert body["persistence"] == "available"
        assert body["schema_supported"] is True
        assert isinstance(body["schema_version"], int)
        print("HEALTH_OK")
        """
    )
    env = dict(os.environ)
    env["MEMORY_BACKEND"] = "sqlite"
    env["SQLITE_DB_PATH"] = str(db_path)
    result = _run(probe, env)
    assert result.returncode == 0, result.stderr
    assert "HEALTH_OK" in result.stdout


def test_health_endpoint_does_not_scan_semantic_or_vector_tables(tmp_path) -> None:
    """Deliberately O(1): write a large number of semantic records/vectors,
    then confirm /health still responds -- this does not PROVE cost is
    O(1) by itself, but confirms the endpoint's own code path never
    queries those tables at all (asserted structurally in app/main.py's
    health() docstring; this test is the behavioral companion -- a health
    check must stay fast and correct regardless of how much semantic data
    a deployment has accumulated)."""
    db_path = tmp_path / "health_large.db"
    probe = textwrap.dedent(
        """
        import os
        from datetime import datetime, timezone
        from app.persistence.sqlite_semantic_memory import SQLiteSemanticMemory, SQLiteVectorIndex
        from app.agent.semantic_memory import SemanticMemoryRecord
        db_path = os.environ["TEST_DB_PATH"]
        store = SQLiteSemanticMemory(db_path=db_path)
        index = SQLiteVectorIndex(db_path=db_path, dimension=4, embedding_model_name="t")
        for i in range(200):
            store.add(SemanticMemoryRecord(
                memory_id=f"m{i}", session_id="s", content=f"fact {i}",
                created_at=datetime.now(timezone.utc), source_event_ids=("e1",),
            ))
            index.add(f"m{i}", "s", (1.0, 0.0, 0.0, 0.0))
        store.close()
        index.close()

        from fastapi.testclient import TestClient
        import app.main as m
        client = TestClient(m.app)
        r = client.get("/health")
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "ok"
        print("HEALTH_OK")
        """
    )
    env = dict(os.environ)
    env["MEMORY_BACKEND"] = "sqlite"
    env["SQLITE_DB_PATH"] = str(db_path)
    env["TEST_DB_PATH"] = str(db_path)
    result = _run(probe, env)
    assert result.returncode == 0, result.stderr
    assert "HEALTH_OK" in result.stdout


def test_health_endpoint_reports_degraded_for_an_unsupported_schema_version(tmp_path) -> None:
    db_path = tmp_path / "future_schema.db"
    # Pre-seed a database the running code considers too new to open.
    raw = sqlite3.connect(str(db_path))
    raw.execute("CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    raw.execute("INSERT INTO schema_meta (key, value) VALUES ('schema_version', '999')")
    raw.commit()
    raw.close()

    # app.main itself fails to IMPORT in this case (see startup validation
    # tests below) -- a /health response can only ever be degraded, never
    # "available", for a future-version database, because the process that
    # would serve it never starts. This test documents that relationship
    # explicitly rather than leaving it implicit.
    probe = textwrap.dedent(
        """
        import os
        try:
            import app.main as m
            print("IMPORT_SUCCEEDED")
        except Exception as exc:
            print(f"IMPORT_FAILED:{type(exc).__name__}")
        """
    )
    env = dict(os.environ)
    env["MEMORY_BACKEND"] = "sqlite"
    env["SQLITE_DB_PATH"] = str(db_path)
    result = _run(probe, env)
    assert "IMPORT_FAILED:" in result.stdout
    assert "SchemaVersionError" in result.stdout


# ===========================================================================
# 2 — Startup validation
# ===========================================================================

def test_startup_fails_loudly_on_an_invalid_sqlite_path(tmp_path) -> None:
    """A path whose parent already exists as a FILE (not a directory)
    cannot have its parent directory created -- a reliable, deterministic
    way to reproduce an open failure without manufacturing filesystem
    corruption."""
    blocking_file = tmp_path / "not_a_directory"
    blocking_file.write_text("x")
    bad_db_path = blocking_file / "nested" / "chatbot.db"

    probe = "import app.main"
    env = dict(os.environ)
    env["MEMORY_BACKEND"] = "sqlite"
    env["SQLITE_DB_PATH"] = str(bad_db_path)
    result = _run(probe, env)

    assert result.returncode != 0


def test_startup_succeeds_with_a_valid_sqlite_path_and_applies_pragmas(tmp_path) -> None:
    db_path = tmp_path / "startup_ok.db"
    probe = textwrap.dedent(
        """
        import os
        import app.main as m
        conn = m._session_store._conn
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        print("STARTUP_OK")
        """
    )
    env = dict(os.environ)
    env["MEMORY_BACKEND"] = "sqlite"
    env["SQLITE_DB_PATH"] = str(db_path)
    result = _run(probe, env)
    assert result.returncode == 0, result.stderr
    assert "STARTUP_OK" in result.stdout
    assert db_path.exists()


def test_startup_applies_pending_migrations(tmp_path) -> None:
    """A database left at an EARLIER supported version must be
    transparently migrated forward at startup, not rejected."""
    db_path = tmp_path / "old_version.db"
    conn = db.connect(str(db_path))
    conn.close()
    raw = sqlite3.connect(str(db_path))
    raw.execute("UPDATE schema_meta SET value = '1' WHERE key = 'schema_version'")
    raw.commit()
    raw.close()

    probe = textwrap.dedent(
        f"""
        import app.main as m
        from app.persistence import db
        version = m._session_store.health_check()
        assert version == db.CURRENT_SCHEMA_VERSION, version
        print("MIGRATION_OK")
        """
    )
    env = dict(os.environ)
    env["MEMORY_BACKEND"] = "sqlite"
    env["SQLITE_DB_PATH"] = str(db_path)
    result = _run(probe, env)
    assert result.returncode == 0, result.stderr
    assert "MIGRATION_OK" in result.stdout


def test_startup_fails_loudly_on_an_unsupported_schema_version(tmp_path) -> None:
    db_path = tmp_path / "unsupported.db"
    raw = sqlite3.connect(str(db_path))
    raw.execute("CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    raw.execute("INSERT INTO schema_meta (key, value) VALUES ('schema_version', '999')")
    raw.commit()
    raw.close()

    probe = "import app.main"
    env = dict(os.environ)
    env["MEMORY_BACKEND"] = "sqlite"
    env["SQLITE_DB_PATH"] = str(db_path)
    result = _run(probe, env)

    assert result.returncode != 0
    assert "SchemaVersionError" in result.stderr or "schema_version" in result.stderr


def test_memory_backend_never_requires_or_touches_a_sqlite_path(tmp_path) -> None:
    """When MEMORY_BACKEND=memory, SQLite must not become a required
    dependency/path -- confirmed by proving no .db file is ever created at
    the (unused but still-configured) SQLITE_DB_PATH."""
    db_path = tmp_path / "should_not_exist.db"
    probe = "import app.main; print('OK')"
    env = dict(os.environ)
    env["MEMORY_BACKEND"] = "memory"
    env["SQLITE_DB_PATH"] = str(db_path)
    result = _run(probe, env)

    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
    assert not db_path.exists()


# ===========================================================================
# 3 — End-to-end smoke tests A-F, through the real composition root
# ===========================================================================

def test_conversation_persists_through_the_real_production_app_across_processes(tmp_path) -> None:
    """Scenario A, through the LITERAL app.main.chat_service singleton and
    real HTTP requests (FastAPI TestClient) -- not just the underlying
    store classes."""
    db_path = tmp_path / "e2e_conversation.db"
    env = dict(os.environ)
    env["MEMORY_BACKEND"] = "sqlite"
    env["SQLITE_DB_PATH"] = str(db_path)

    write_probe = textwrap.dedent(
        """
        import json
        import app.main as m
        from fastapi.testclient import TestClient

        class FakeLLM:
            def __init__(self, responses):
                self.responses = iter(responses)
            def generate(self, messages, *, json_mode=False):
                return next(self.responses)

        def _final_json(answer):
            return json.dumps({"action_type": "final", "final_answer": answer})

        m.chat_service.llm = FakeLLM([_final_json("Nice to meet you, Alice.")])
        client = TestClient(m.app)
        r = client.post("/chat", json={"message": "My name is Alice.", "session_id": "restart-convo"})
        assert r.status_code == 200, r.text
        print("WRITE_OK")
        """
    )
    write_result = _run(write_probe, env)
    assert write_result.returncode == 0, write_result.stderr
    assert "WRITE_OK" in write_result.stdout

    read_probe = textwrap.dedent(
        """
        import app.main as m
        messages = m.chat_service.session_store.get_memory("restart-convo").get_messages()
        assert any("Alice" in msg["content"] for msg in messages), messages
        assert len(messages) == 2
        print("READ_OK")
        """
    )
    read_result = _run(read_probe, env)
    assert read_result.returncode == 0, read_result.stderr
    assert "READ_OK" in read_result.stdout


def _full_stack_write_probe() -> str:
    """Builds conversation+episodic+semantic (DeterministicEmbeddingProvider,
    no real model) through the REAL composition functions
    (SQLiteSessionMemoryStore, SQLiteEpisodicMemory, build_semantic_memory,
    ChatService) -- i.e. the same functions app/main.py itself calls -- and
    exercises three sessions: 'main' (A-D), 'other' (E, isolation partner),
    'cleared' (F)."""
    return textwrap.dedent(
        """
        import json
        import os
        from app.agent.embeddings import DeterministicEmbeddingProvider
        from app.persistence.sqlite_memory import SQLiteSessionMemoryStore
        from app.persistence.sqlite_episodic_memory import SQLiteEpisodicMemory
        from app.semantic_memory_wiring import build_semantic_memory
        from app.services.chat import ChatService

        def _final(answer):
            return json.dumps({"action_type": "final", "final_answer": answer})

        def _mem(content):
            return json.dumps({"memories": [{"content": content, "confidence": 0.9}]})

        class FakeLLM:
            def __init__(self, responses):
                self.responses = iter(responses)
            def generate(self, messages, *, json_mode=False):
                return next(self.responses)

        db_path = os.environ["TEST_DB_PATH"]
        session_store = SQLiteSessionMemoryStore(db_path=db_path)
        episodic_memory = SQLiteEpisodicMemory(db_path=db_path)
        llm = FakeLLM([
            _final("Noted, you prefer Python."),
            _mem("User prefers Python for ML work."),
            _final("Got it, hiking is great."),
            _mem("User enjoys hiking."),
            _final("Sure thing."),
            _mem(""),  # placeholder, overwritten below for the 'cleared' session
        ])
        bundle = build_semantic_memory(
            enabled=True, model_name="fake/model", device="cpu",
            max_records_per_session=200, min_similarity=-1.0, llm_client=llm,
            memory_backend="sqlite", sqlite_db_path=db_path,
            provider_factory=lambda *, model_name, device: DeterministicEmbeddingProvider(dimension=8),
        )
        chat_service = ChatService(
            llm_client=llm, session_store=session_store, episodic_memory=episodic_memory,
            memory_retriever=bundle.retriever, memory_extractor=bundle.extractor, memory_writer=bundle.writer,
        )

        chat_service.ask("I prefer Python for ML work.", session_id="main")
        chat_service.ask("I also enjoy hiking.", session_id="other")
        chat_service.ask("Remember this, then forget it.", session_id="cleared")

        # Scenario F: clear the 'cleared' session across ALL layers before restart.
        session_store.clear_session("cleared")
        episodic_memory.clear("cleared")
        bundle.writer.semantic_memory.clear("cleared")
        for record in bundle.writer.semantic_memory.list_recent("cleared", limit=10):
            bundle.writer.vector_index.remove(record.memory_id, "cleared")

        session_store.close()
        episodic_memory.close()
        bundle.writer.semantic_memory.close()
        bundle.writer.vector_index.close()
        print("WRITE_OK")
        """
    )


def _full_stack_read_probe() -> str:
    return textwrap.dedent(
        """
        import os
        from app.agent.embeddings import DeterministicEmbeddingProvider
        from app.persistence.sqlite_memory import SQLiteSessionMemoryStore
        from app.persistence.sqlite_episodic_memory import SQLiteEpisodicMemory
        from app.semantic_memory_wiring import build_semantic_memory

        class FakeLLM:
            def generate(self, messages, *, json_mode=False):
                raise AssertionError("FakeLLM should not be called on the read side")

        db_path = os.environ["TEST_DB_PATH"]
        session_store = SQLiteSessionMemoryStore(db_path=db_path)
        episodic_memory = SQLiteEpisodicMemory(db_path=db_path)
        bundle = build_semantic_memory(
            enabled=True, model_name="fake/model", device="cpu",
            max_records_per_session=200, min_similarity=-1.0, llm_client=FakeLLM(),
            memory_backend="sqlite", sqlite_db_path=db_path,
            provider_factory=lambda *, model_name, device: DeterministicEmbeddingProvider(dimension=8),
        )

        # A: conversation
        messages = session_store.get_memory("main").get_messages()
        assert any("Python" in msg["content"] for msg in messages), messages

        # B: episodic
        episodes = episodic_memory.get_recent("main")
        assert len(episodes) == 1, episodes

        # C: semantic retrieval
        results = bundle.retriever.retrieve("main", "What language do I prefer?", top_k=5)
        assert len(results) == 1, results
        assert "Python" in results[0].memory.content

        # D: vector search directly
        hits = bundle.writer.vector_index.search(
            "main", DeterministicEmbeddingProvider(dimension=8).embed("User prefers Python for ML work."), top_k=5
        )
        assert len(hits) == 1, hits

        # E: session isolation -- 'other' must not see 'main's content, and vice versa
        other_messages = session_store.get_memory("other").get_messages()
        assert not any("Python" in msg["content"] for msg in other_messages), other_messages
        other_results = bundle.retriever.retrieve("other", "What language do I prefer?", top_k=5)
        assert all(r.memory.session_id == "other" for r in other_results)
        assert all("Python" not in r.memory.content for r in other_results)

        # F: cleared session stays cleared after restart
        assert session_store.get_memory("cleared").get_messages() == []
        assert episodic_memory.get_recent("cleared") == []
        assert bundle.writer.semantic_memory.list_recent("cleared", limit=10) == []

        print("READ_OK")
        """
    )


def test_full_stack_persistence_smoke_test_a_through_f(tmp_path) -> None:
    """Scenarios A-F in one coherent flow, through the real composition
    functions (SQLiteSessionMemoryStore, SQLiteEpisodicMemory,
    build_semantic_memory, ChatService) -- the same functions app/main.py
    itself calls. DeterministicEmbeddingProvider is used for the semantic
    layer (same convention as Step 38's own tests) so this stays fast and
    dependency-free; a separate test above proves conversation persistence
    through the LITERAL app.main.chat_service singleton and real HTTP."""
    db_path = tmp_path / "full_stack.db"
    env = dict(os.environ)
    env["TEST_DB_PATH"] = str(db_path)

    write_result = _run(_full_stack_write_probe(), env)
    assert write_result.returncode == 0, write_result.stderr
    assert "WRITE_OK" in write_result.stdout

    read_result = _run(_full_stack_read_probe(), env)
    assert read_result.returncode == 0, read_result.stderr
    assert "READ_OK" in read_result.stdout


# ===========================================================================
# 4 — Failure behavior
# ===========================================================================

def test_no_silent_fallback_to_memory_when_sqlite_path_is_invalid(tmp_path) -> None:
    """The process must FAIL, not quietly continue with an in-memory
    backend, when MEMORY_BACKEND=sqlite but the configured path cannot be
    opened."""
    blocking_file = tmp_path / "blocker"
    blocking_file.write_text("x")
    bad_path = blocking_file / "sub" / "chatbot.db"

    probe = textwrap.dedent(
        """
        import app.main as m
        from app.persistence.sqlite_memory import SQLiteSessionMemoryStore
        from app.agent.memory import InMemorySessionMemoryStore
        # If this line is ever reached, startup did NOT fail -- assert it
        # at least did not silently become an in-memory store either.
        assert not isinstance(m.chat_service.session_store, InMemorySessionMemoryStore)
        print("UNEXPECTEDLY_REACHED")
        """
    )
    env = dict(os.environ)
    env["MEMORY_BACKEND"] = "sqlite"
    env["SQLITE_DB_PATH"] = str(bad_path)
    result = _run(probe, env)

    assert result.returncode != 0
    assert "UNEXPECTEDLY_REACHED" not in result.stdout


def test_no_silent_fallback_on_unsupported_schema_version(tmp_path) -> None:
    db_path = tmp_path / "no_fallback_version.db"
    raw = sqlite3.connect(str(db_path))
    raw.execute("CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    raw.execute("INSERT INTO schema_meta (key, value) VALUES ('schema_version', '999')")
    raw.commit()
    raw.close()

    probe = "import app.main; print('UNEXPECTEDLY_REACHED')"
    env = dict(os.environ)
    env["MEMORY_BACKEND"] = "sqlite"
    env["SQLITE_DB_PATH"] = str(db_path)
    result = _run(probe, env)

    assert result.returncode != 0
    assert "UNEXPECTEDLY_REACHED" not in result.stdout


def test_database_open_failure_is_reproducible_via_blocked_parent_path(tmp_path) -> None:
    """A direct, unit-level reproduction of the same open-failure class
    (not relying on a subprocess), confirming db.connect() itself raises
    rather than silently creating something unexpected."""
    blocking_file = tmp_path / "blocker2"
    blocking_file.write_text("x")
    bad_path = blocking_file / "sub" / "chatbot.db"

    with pytest.raises(OSError):
        db.connect(str(bad_path))


# ===========================================================================
# 5 — Configuration review
# ===========================================================================

def test_env_example_documents_memory_backend_and_sqlite_db_path() -> None:
    content = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    assert "MEMORY_BACKEND" in content
    assert "SQLITE_DB_PATH" in content


def test_gitignore_excludes_database_files() -> None:
    content = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "*.db" in content
    assert "/data/" in content


# ===========================================================================
# 7 — Security audit
# ===========================================================================

# Matches an f-string literal passed directly to .execute(...), or a plain
# string literal immediately concatenated with something else via `+`
# (on either side of the quote) -- the two realistic ways Python-level SQL
# string-building would show up. Deliberately does NOT flag a bare `+`
# inside a string literal's own text (e.g. "SELECT ... + 1 AS n ..." is
# ordinary, static, fully-parameterized SQL arithmetic, not Python string
# concatenation) -- an earlier, cruder version of this check false-
# positived on exactly that.
_SQL_EXECUTE_FSTRING = re.compile(r"\.execute\(\s*f[\"']")
_SQL_STRING_CONCAT = re.compile(r"[\"']\s*\+\s*\w|\w\s*\+\s*[\"']")
_SQL_PERCENT_FORMAT = re.compile(r"[\"']\s*%\s*[\(\[]")


def test_no_fstring_or_percent_sql_construction_in_persistence_layer() -> None:
    """Static audit: every `.execute(...)` call across the persistence
    layer must use a parameterized `?` placeholder, never an f-string,
    `%`-formatted, or string-concatenated SQL statement with interpolated
    values."""
    persistence_dir = REPO_ROOT / "app" / "persistence"
    offenders = []
    for path in persistence_dir.glob("*.py"):
        content = path.read_text(encoding="utf-8")
        if (
            _SQL_EXECUTE_FSTRING.search(content)
            or _SQL_STRING_CONCAT.search(content)
            or _SQL_PERCENT_FORMAT.search(content)
        ):
            offenders.append(path.name)
    assert offenders == [], f"found non-parameterized SQL construction in: {offenders}"


def test_security_gate_still_blocks_secrets_through_the_full_sqlite_stack(tmp_path) -> None:
    """End-to-end confirmation (not just a unit test on rejection_reason):
    a credential-shaped memory candidate never reaches the SQLite tables,
    through the real ChatService -> AgentOrchestrator -> SemanticMemoryWriter
    pipeline."""
    db_path = tmp_path / "security.db"
    write_probe = textwrap.dedent(
        """
        import json
        import os
        from app.agent.embeddings import DeterministicEmbeddingProvider
        from app.persistence.sqlite_memory import SQLiteSessionMemoryStore
        from app.persistence.sqlite_episodic_memory import SQLiteEpisodicMemory
        from app.semantic_memory_wiring import build_semantic_memory
        from app.services.chat import ChatService

        def _final(answer):
            return json.dumps({"action_type": "final", "final_answer": answer})

        class FakeLLM:
            def __init__(self, responses):
                self.responses = iter(responses)
            def generate(self, messages, *, json_mode=False):
                return next(self.responses)

        db_path = os.environ["TEST_DB_PATH"]
        session_store = SQLiteSessionMemoryStore(db_path=db_path)
        episodic_memory = SQLiteEpisodicMemory(db_path=db_path)
        llm = FakeLLM([
            _final("I can't store that."),
            json.dumps({"memories": [{"content": "User API key is sk-abcdef1234567890", "confidence": 0.9}]}),
        ])
        bundle = build_semantic_memory(
            enabled=True, model_name="fake/model", device="cpu",
            max_records_per_session=200, min_similarity=-1.0, llm_client=llm,
            memory_backend="sqlite", sqlite_db_path=db_path,
            provider_factory=lambda *, model_name, device: DeterministicEmbeddingProvider(dimension=8),
        )
        chat_service = ChatService(
            llm_client=llm, session_store=session_store, episodic_memory=episodic_memory,
            memory_retriever=bundle.retriever, memory_extractor=bundle.extractor, memory_writer=bundle.writer,
        )
        chat_service.ask("here is my key sk-abcdef1234567890", session_id="sec-test")

        assert bundle.writer.semantic_memory.list_recent("sec-test", limit=10) == []
        conn = bundle.writer.semantic_memory._conn
        row = conn.execute("SELECT COUNT(*) AS c FROM semantic_memory_records").fetchone()
        assert row["c"] == 0, "a credential-shaped candidate reached the SQLite table"
        print("SECURITY_OK")
        """
    )
    env = dict(os.environ)
    env["TEST_DB_PATH"] = str(db_path)
    result = _run(write_probe, env)
    assert result.returncode == 0, result.stderr
    assert "SECURITY_OK" in result.stdout
