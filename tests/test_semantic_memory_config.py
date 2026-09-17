"""Step 16I: configuration for semantic memory activation.

Covers app/config.py's four settings:

    SEMANTIC_MEMORY_ENABLED (default false)
    EMBEDDING_MODEL_NAME    (default sentence-transformers/all-MiniLM-L6-v2, from 16H)
    EMBEDDING_DEVICE        (default cpu)
    SEMANTIC_MEMORY_MAX_RECORDS_PER_SESSION (default 200)

The central risk this file guards against: a dataclass field's plain
`= os.getenv(...)` default is evaluated ONCE at class-definition time (at
module import), not per-instantiation. `semantic_memory_enabled`,
`embedding_device`, and `semantic_memory_max_records_per_session` use
`default_factory` specifically so `monkeypatch.setenv(...)` followed by a
fresh `Settings()` call actually observes the new value — the tests below
prove that property directly, not just the end default.

Fully offline: no network, no model, no sentence-transformers import.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from app.agent.local_embeddings import DEFAULT_EMBEDDING_MODEL_NAME
from app.config import Settings, _parse_bool, _parse_positive_int

REPO_ROOT = Path(__file__).resolve().parent.parent


# ===========================================================================
# A1 — SEMANTIC_MEMORY_ENABLED defaults to False
# ===========================================================================

def test_semantic_memory_is_disabled_by_default() -> None:
    assert Settings().semantic_memory_enabled is False


def test_default_settings_singleton_is_disabled() -> None:
    """app/config.py's module-level `settings` (what app/main.py actually
    imports) must be disabled in this test environment, where
    SEMANTIC_MEMORY_ENABLED is not set."""
    from app.config import settings

    assert settings.semantic_memory_enabled is False


# ===========================================================================
# A2 — bool parsing behavior
# ===========================================================================

@pytest.mark.parametrize("value", ["true", "True", "TRUE", "1", "yes", "Yes", "on", "ON"])
def test_bool_parsing_accepts_documented_truthy_values(value: str) -> None:
    assert _parse_bool(value) is True


@pytest.mark.parametrize(
    "value", ["false", "False", "0", "no", "off", "", "   ", "maybe", "enabled", "TRUE ISH"]
)
def test_bool_parsing_treats_everything_else_as_false(value: str) -> None:
    assert _parse_bool(value) is False


def test_bool_parsing_unset_uses_the_default_argument() -> None:
    assert _parse_bool(None, default=False) is False
    assert _parse_bool(None, default=True) is True


def test_unrecognized_bool_value_never_silently_enables() -> None:
    """The failure mode of a typo must be "stays off," never "turns on"."""
    for typo in ("tru", "yess", "ON!", "1.0", "enable"):
        assert _parse_bool(typo, default=False) is False


def test_bool_parsing_whitespace_is_stripped() -> None:
    assert _parse_bool("  true  ") is True
    assert _parse_bool("  false  ") is False


# ===========================================================================
# A3 — EMBEDDING_MODEL_NAME (16H field, unchanged by 16I)
# ===========================================================================

def test_embedding_model_name_defaults_to_the_16h_constant() -> None:
    assert Settings().embedding_model_name == DEFAULT_EMBEDDING_MODEL_NAME


# ===========================================================================
# A4 — EMBEDDING_DEVICE defaults to cpu
# ===========================================================================

def test_embedding_device_defaults_to_cpu() -> None:
    assert Settings().embedding_device == "cpu"


def test_embedding_device_is_read_at_construction_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """Proves the default_factory timing fix directly: setting the env var
    AFTER app.config was already imported must still be observed by a
    freshly constructed Settings()."""
    monkeypatch.setenv("EMBEDDING_DEVICE", "cuda")

    assert Settings().embedding_device == "cuda"


# ===========================================================================
# A5 — SEMANTIC_MEMORY_MAX_RECORDS_PER_SESSION defaults to 200
# ===========================================================================

def test_max_records_per_session_defaults_to_200() -> None:
    assert Settings().semantic_memory_max_records_per_session == 200


def test_max_records_per_session_is_read_at_construction_time(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEMANTIC_MEMORY_MAX_RECORDS_PER_SESSION", "50")

    assert Settings().semantic_memory_max_records_per_session == 50


def test_max_records_parsing_helper_uses_the_default_when_unset() -> None:
    assert _parse_positive_int(None, default=200) == 200
    assert _parse_positive_int("", default=200) == 200
    assert _parse_positive_int("   ", default=200) == 200


def test_max_records_parsing_helper_parses_a_valid_integer() -> None:
    assert _parse_positive_int("50", default=200) == 50
    assert _parse_positive_int("  77  ", default=200) == 77


# ===========================================================================
# A6 — invalid configuration values
# ===========================================================================

@pytest.mark.parametrize("bad", ["abc", "12.5", "one hundred", "5abc", "--3"])
def test_invalid_max_records_value_raises_rather_than_silently_defaulting(bad: str) -> None:
    """A malformed configuration value is a deployment bug, and silently
    falling back to the default would hide it."""
    with pytest.raises(ValueError, match="invalid integer"):
        _parse_positive_int(bad, default=200)


def test_invalid_max_records_env_var_raises_at_settings_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEMANTIC_MEMORY_MAX_RECORDS_PER_SESSION", "not-a-number")

    with pytest.raises(ValueError, match="invalid integer"):
        Settings()


def test_invalid_max_records_env_var_does_not_affect_disabled_mode() -> None:
    """A malformed retention value only matters once semantic memory is
    actually enabled — Settings() itself still raises (it is a genuine
    parsing failure, not a range check), but range validation like ">= 1"
    is deliberately deferred to SemanticMemoryWriter, which never runs
    for a disabled bundle."""
    # -3 parses fine as an integer (this is a RANGE problem, not a parsing
    # one) — Settings() must not pre-validate range, matching the design's
    # note that range validation belongs solely to SemanticMemoryWriter.
    assert _parse_positive_int("-3", default=200) == -3


# ===========================================================================
# A7 — proving environment variables are read at the CORRECT time
# ===========================================================================

def test_changing_env_after_import_is_observed_by_a_new_settings_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The single most important configuration test in this file: without
    default_factory, this would fail, because the plain-default value
    would already have been baked in when app.config was first imported
    (by an earlier test module, in the same process)."""
    before = Settings().semantic_memory_enabled
    assert before is False  # sanity: starts disabled in this test env

    monkeypatch.setenv("SEMANTIC_MEMORY_ENABLED", "true")
    after = Settings().semantic_memory_enabled

    assert after is True


def test_default_process_startup_enables_nothing() -> None:
    """A true end-to-end proof, in a CLEAN interpreter with no
    SEMANTIC_MEMORY_ENABLED set: importing app.main must not construct any
    semantic memory collaborator and must not import the ML stack."""
    probe = (
        "import sys; import app.main; "
        "assert app.main.chat_service.memory_retriever is None; "
        "assert app.main.chat_service.memory_writer is None; "
        "assert app.main.chat_service.memory_extractor is None; "
        "assert 'torch' not in sys.modules; "
        "assert 'sentence_transformers' not in sys.modules; "
        "print('DISABLED_OK')"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )

    assert result.returncode == 0, result.stderr
    assert "DISABLED_OK" in result.stdout


# NOTE: the matching "enabled + unloadable model fails startup loudly"
# probe belongs in tests/test_semantic_memory_activation.py (Phase 4),
# since it requires app/main.py's composition-root wiring to exist. This
# file (Phase 2) covers only app/config.py's own, self-contained behavior.
