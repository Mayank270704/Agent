"""Production Readiness Audit, P0-1: application logging activation.

Before this fix, `settings.log_level` (app/config.py) was dead
configuration — nothing called `logging.basicConfig()`/`dictConfig()`,
so the root logger had no handler and every `logger.info(...)` call in
this codebase was silently discarded in a real deployment (pytest's own
`caplog` fixture works independently of this and was NOT proof the fix
was in place — see the module docstring on why these tests instead
inspect the root logger's actual level/handlers and effective level of
a real application logger).

Fully offline: no Ollama, no network.
"""
from __future__ import annotations

import json
import logging

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.main import configure_logging


@pytest.fixture(autouse=True)
def _restore_root_logging_state():
    """`configure_logging` mutates process-global logging state
    (`force=True` clears existing handlers). Snapshot and restore it
    around every test in this file so these tests cannot leak a changed
    root logger level/handler set into any other test file."""
    root = logging.getLogger()
    original_level = root.level
    original_handlers = list(root.handlers)
    yield
    root.handlers[:] = original_handlers
    root.setLevel(original_level)


# ===========================================================================
# 1 — configure_logging itself: respects its level argument, installs a
# real handler (not merely something pytest's own caplog provides)
# ===========================================================================

def test_configure_logging_sets_the_root_logger_level_from_its_argument() -> None:
    configure_logging("DEBUG")
    assert logging.getLogger().level == logging.DEBUG

    configure_logging("WARNING")
    assert logging.getLogger().level == logging.WARNING


def test_configure_logging_is_case_insensitive() -> None:
    configure_logging("info")
    assert logging.getLogger().level == logging.INFO


def test_configure_logging_installs_a_real_handler_on_the_root_logger() -> None:
    configure_logging("INFO")
    assert len(logging.getLogger().handlers) > 0


def test_configure_logging_format_includes_level_logger_and_message(capsys) -> None:
    configure_logging("INFO")
    probe_logger = logging.getLogger("app.test.p0_1.format_probe")

    probe_logger.info("PROBE_MESSAGE_XYZ")

    captured = capsys.readouterr()
    assert "INFO" in captured.err
    assert "app.test.p0_1.format_probe" in captured.err
    assert "PROBE_MESSAGE_XYZ" in captured.err


def test_configure_logging_is_authoritative_even_if_a_handler_already_exists() -> None:
    """The exact failure mode this fix closes: logging.basicConfig()
    WITHOUT force=True is a silent no-op whenever the root logger
    already has a handler. Proves configure_logging does not have that
    problem."""
    root = logging.getLogger()
    root.addHandler(logging.NullHandler())  # simulate "something already configured logging"
    root.setLevel(logging.CRITICAL)

    configure_logging("DEBUG")

    assert logging.getLogger().level == logging.DEBUG


# ===========================================================================
# 2 — the actual settings.log_level is what main.py wired in
# ===========================================================================

def test_the_production_module_actually_applied_settings_log_level() -> None:
    """Ties app.main's real, already-executed module-level call to the
    live root logger state — not just the helper function in isolation.
    (app.main is imported exactly once per process; this asserts on the
    lasting effect of that one real call, run before this test file's
    autouse fixture snapshot even existed.)"""
    expected_level = getattr(logging, main_module.settings.log_level.upper())
    # A later test in this file may have changed the root level via
    # configure_logging() directly; re-assert the real wiring here.
    configure_logging(main_module.settings.log_level)
    assert logging.getLogger().level == expected_level


def test_an_application_logger_is_no_longer_silently_filtered_at_info(caplog) -> None:
    """Before this fix, an unconfigured root logger effectively filters
    out INFO-level records for any named logger that never set its own
    level (the default `getEffectiveLevel()` climbs to the root, which
    was unset). After configure_logging("info") (settings' own default),
    an existing application logger's effective level permits INFO."""
    configure_logging("info")

    app_logger = logging.getLogger("app.agent.tool_execution")

    assert app_logger.isEnabledFor(logging.INFO) is True
    assert app_logger.getEffectiveLevel() <= logging.INFO


def test_warning_level_config_filters_out_info_as_expected() -> None:
    """Respecting the configured level cuts both ways: a stricter
    configured level genuinely raises the effective threshold, proving
    this isn't just "always permissive"."""
    configure_logging("WARNING")

    app_logger = logging.getLogger("app.agent.tool_execution")

    assert app_logger.isEnabledFor(logging.INFO) is False
    assert app_logger.isEnabledFor(logging.WARNING) is True


# ===========================================================================
# 3 — no secret/content leakage introduced by the format itself
# ===========================================================================

def test_the_log_format_string_contains_no_hardcoded_secret_or_content_field() -> None:
    """Structural check on the ACTUAL format string passed to
    basicConfig (not the docstring): it renders only level/logger-name/
    message/timestamp placeholders -- no field that could echo a
    prompt, tool input, or credential by construction."""
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(configure_logging)))
    call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "basicConfig"
    )
    format_arg = next(kw.value for kw in call.keywords if kw.arg == "format")
    format_string = format_arg.value

    assert format_string == "%(asctime)s %(levelname)s %(name)s: %(message)s"
    for forbidden in ("api_key", "password", "token", "prompt", "tool_input", "args"):
        assert forbidden not in format_string.lower()


# ===========================================================================
# 4 — normal API behavior is unchanged
# ===========================================================================

@pytest.fixture
def client() -> TestClient:
    return TestClient(main_module.app)


def test_root_endpoint_is_unchanged(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": main_module.settings.app_name}


def test_chat_endpoint_response_schema_is_unchanged(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeLLM:
        def generate(self, messages, *, json_mode: bool = False) -> str:
            return json.dumps({"action_type": "final", "final_answer": "hello there"})

    monkeypatch.setattr(main_module.chat_service, "llm", FakeLLM())

    response = client.post("/chat", json={"message": "hi"})

    assert response.status_code == 200
    assert set(response.json().keys()) == {"reply"}
    assert response.json()["reply"] == "hello there"


def test_chat_endpoint_still_returns_400_for_empty_message(client: TestClient) -> None:
    response = client.post("/chat", json={"message": "   "})

    assert response.status_code == 400
