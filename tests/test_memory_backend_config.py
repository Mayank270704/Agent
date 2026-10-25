"""Step 36, category I: MEMORY_BACKEND / SQLITE_DB_PATH configuration
(app/config.py) and the composition-root wiring that selects a backend
(app/main.py).

The subprocess-based probes mirror the established pattern in
tests/test_semantic_memory_config.py::test_default_process_startup_enables_nothing
exactly: a settings-driven composition choice is proven end-to-end by
importing app.main fresh, in a clean interpreter, under controlled
environment variables -- not just by unit-testing app/config.py in
isolation.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from app.config import Settings, _parse_memory_backend

REPO_ROOT = Path(__file__).resolve().parent.parent


# ===========================================================================
# MEMORY_BACKEND parsing
# ===========================================================================

def test_memory_backend_defaults_to_memory() -> None:
    assert Settings().memory_backend == "memory"


def test_memory_backend_explicit_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMORY_BACKEND", "memory")
    assert Settings().memory_backend == "memory"


def test_memory_backend_explicit_sqlite(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMORY_BACKEND", "sqlite")
    assert Settings().memory_backend == "sqlite"


def test_memory_backend_parsing_is_case_insensitive_and_strips_whitespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEMORY_BACKEND", "  SQLite  ")
    assert Settings().memory_backend == "sqlite"


def test_invalid_memory_backend_raises_at_settings_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMORY_BACKEND", "redis")
    with pytest.raises(ValueError, match="invalid MEMORY_BACKEND"):
        Settings()


def test_unrecognized_memory_backend_never_silently_falls_back_to_sqlite_or_memory() -> None:
    """The failure mode of a typo must be a loud error, never a silent
    choice of either backend."""
    with pytest.raises(ValueError):
        _parse_memory_backend("postgres", default="memory")


def test_memory_backend_is_read_at_construction_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without default_factory, this would fail -- see the identical test
    for semantic_memory_enabled in test_semantic_memory_config.py."""
    assert Settings().memory_backend == "memory"

    monkeypatch.setenv("MEMORY_BACKEND", "sqlite")

    assert Settings().memory_backend == "sqlite"


def test_memory_backend_parsing_helper_uses_default_when_unset() -> None:
    assert _parse_memory_backend(None, default="memory") == "memory"
    assert _parse_memory_backend("", default="memory") == "memory"
    assert _parse_memory_backend("   ", default="memory") == "memory"


# ===========================================================================
# SQLITE_DB_PATH
# ===========================================================================

def test_sqlite_db_path_has_a_non_empty_default() -> None:
    path = Settings().sqlite_db_path
    assert path
    assert path.endswith("conversations.db")


def test_sqlite_db_path_is_read_at_construction_time(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    custom = str(tmp_path / "custom.db")
    monkeypatch.setenv("SQLITE_DB_PATH", custom)

    assert Settings().sqlite_db_path == custom


# ===========================================================================
# Composition-root wiring (app/main.py), proven end-to-end via subprocess,
# matching test_semantic_memory_config.py's established precedent.
# ===========================================================================

def test_default_process_startup_uses_the_in_memory_backend() -> None:
    probe = (
        "import app.main; "
        "from app.agent.memory import InMemorySessionMemoryStore; "
        "assert isinstance(app.main.chat_service.session_store, InMemorySessionMemoryStore); "
        "print('MEMORY_BACKEND_DEFAULT_OK')"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, cwd=str(REPO_ROOT)
    )

    assert result.returncode == 0, result.stderr
    assert "MEMORY_BACKEND_DEFAULT_OK" in result.stdout


def test_sqlite_backend_env_selects_the_sqlite_session_store(tmp_path) -> None:
    db_path = tmp_path / "probe.db"
    probe = (
        "import app.main; "
        "from app.persistence.sqlite_memory import SQLiteSessionMemoryStore; "
        "assert isinstance(app.main.chat_service.session_store, SQLiteSessionMemoryStore); "
        "print('MEMORY_BACKEND_SQLITE_OK')"
    )
    env = dict(os.environ)
    env["MEMORY_BACKEND"] = "sqlite"
    env["SQLITE_DB_PATH"] = str(db_path)
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, cwd=str(REPO_ROOT), env=env
    )

    assert result.returncode == 0, result.stderr
    assert "MEMORY_BACKEND_SQLITE_OK" in result.stdout
    assert db_path.exists()  # the database file was actually created at startup


def test_sqlite_backend_actually_persists_a_conversation_through_the_real_app(tmp_path) -> None:
    """An end-to-end proof that selecting the sqlite backend at the
    composition root produces a session_store that genuinely persists --
    not just an instance of the right class."""
    db_path = tmp_path / "probe_e2e.db"
    probe = (
        "import app.main; "
        "app.main.chat_service.session_store.get_memory('e2e-session').add_user_message('hello'); "
        "app.main.chat_service.session_store.close(); "
        "print('WRITE_OK')"
    )
    env = dict(os.environ)
    env["MEMORY_BACKEND"] = "sqlite"
    env["SQLITE_DB_PATH"] = str(db_path)
    write_result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, cwd=str(REPO_ROOT), env=env
    )
    assert write_result.returncode == 0, write_result.stderr
    assert "WRITE_OK" in write_result.stdout

    read_probe = (
        "import app.main; "
        "messages = app.main.chat_service.session_store.get_memory('e2e-session').get_messages(); "
        "assert messages == [{'role': 'user', 'content': 'hello'}], messages; "
        "print('READ_OK')"
    )
    read_result = subprocess.run(
        [sys.executable, "-c", read_probe], capture_output=True, text=True, cwd=str(REPO_ROOT), env=env
    )
    assert read_result.returncode == 0, read_result.stderr
    assert "READ_OK" in read_result.stdout


def test_invalid_memory_backend_env_fails_application_startup_loudly() -> None:
    probe = "import app.main"
    env = dict(os.environ)
    env["MEMORY_BACKEND"] = "redis"
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, cwd=str(REPO_ROOT), env=env
    )

    assert result.returncode != 0
    assert "invalid MEMORY_BACKEND" in result.stderr
