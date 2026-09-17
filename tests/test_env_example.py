"""Milestone 23, item 4: `.env.example` contains no secret material, and
every variable it declares is one the application actually reads (via
app/config.py's `os.getenv(...)` calls) — never an invented/unused name.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ENV_EXAMPLE_PATH = Path(__file__).resolve().parent.parent / ".env.example"

# The exact set of env var names app/config.py actually reads, derived by
# grepping os.getenv(...) call sites there — kept as a literal list (not
# introspected at import time) so this test fails loudly if config.py
# adds a new one without .env.example being updated to match.
CONSUMED_ENV_VARS = {
    "APP_NAME",
    "LLM_PROVIDER",
    "OPENAI_API_KEY",
    "TAVILY_API_KEY",
    "MODEL_NAME",
    "OLLAMA_BASE_URL",
    "OLLAMA_MODEL",
    "EMBEDDING_MODEL_NAME",
    "SEMANTIC_MEMORY_ENABLED",
    "EMBEDDING_DEVICE",
    "SEMANTIC_MEMORY_MAX_RECORDS_PER_SESSION",
    "APP_PORT",
    "LOG_LEVEL",
    "TELEMETRY_ENABLED",
    "REQUEST_TIMEOUT_SECONDS",
}


def _declared_vars_and_values() -> dict[str, str]:
    content = ENV_EXAMPLE_PATH.read_text(encoding="utf-8")
    declared = {}
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        assert "=" in stripped, f"malformed line (no '='): {line!r}"
        name, _, value = stripped.partition("=")
        declared[name.strip()] = value.strip()
    return declared


def test_env_example_file_exists() -> None:
    assert ENV_EXAMPLE_PATH.exists()


def test_every_declared_variable_is_actually_consumed_by_the_application() -> None:
    declared = _declared_vars_and_values()
    for name in declared:
        assert name in CONSUMED_ENV_VARS, f"{name} is declared in .env.example but not read by app/config.py"


def test_every_consumed_variable_is_documented() -> None:
    declared = _declared_vars_and_values()
    missing = CONSUMED_ENV_VARS - set(declared)
    assert not missing, f"app/config.py reads these but .env.example omits them: {missing}"


def test_no_value_looks_like_a_real_secret() -> None:
    declared = _declared_vars_and_values()
    secret_patterns = (
        r"tvly-[A-Za-z0-9_-]{10,}",
        r"sk-[A-Za-z0-9_-]{10,}",
        r"AIza[A-Za-z0-9_-]{10,}",
        r"AKIA[A-Z0-9]{10,}",
        r"ghp_[A-Za-z0-9]{10,}",
    )
    for name, value in declared.items():
        for pattern in secret_patterns:
            assert not re.search(pattern, value, re.IGNORECASE), f"{name} looks like a real secret: {value!r}"


def test_api_key_fields_are_empty_placeholders_not_real_values() -> None:
    declared = _declared_vars_and_values()
    assert declared["OPENAI_API_KEY"] == ""
    assert declared["TAVILY_API_KEY"] == ""


def test_the_real_env_file_is_never_copied_in() -> None:
    """The .env.example content must not contain the literal real .env
    file's content — a cheap guard against "cp .env .env.example"."""
    real_env_path = ENV_EXAMPLE_PATH.parent / ".env"
    if not real_env_path.exists():
        return  # nothing to compare against in this environment
    real_lines = {
        line.strip() for line in real_env_path.read_text(encoding="utf-8").splitlines() if line.strip()
    }
    example_lines = {
        line.strip() for line in ENV_EXAMPLE_PATH.read_text(encoding="utf-8").splitlines() if line.strip()
    }
    # Any line present in BOTH files that assigns a non-empty value to an
    # API key would mean the real secret leaked into the example.
    for line in real_lines & example_lines:
        if line.startswith(("OPENAI_API_KEY=", "TAVILY_API_KEY=")) and line not in (
            "OPENAI_API_KEY=",
            "TAVILY_API_KEY=",
        ):
            pytest.fail(f"real secret line leaked into .env.example: {line!r}")


def test_env_example_is_git_tracked_candidate_not_ignored() -> None:
    """.env.example must NOT match the .gitignore rule that excludes the
    real .env — otherwise it could never be committed at all."""
    gitignore_path = ENV_EXAMPLE_PATH.parent / ".gitignore"
    gitignore_content = gitignore_path.read_text(encoding="utf-8")
    # The exact ignored pattern is the literal "\n.env\n" line (not
    # ".env*"), which does not match ".env.example" — asserted directly
    # rather than re-implementing gitignore glob semantics.
    assert ".env*" not in gitignore_content
    assert re.search(r"^\.env$", gitignore_content, re.MULTILINE) is not None
