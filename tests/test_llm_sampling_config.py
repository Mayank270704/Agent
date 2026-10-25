"""Milestone 44.1 — optional Ollama sampling configuration
(OLLAMA_TEMPERATURE / OLLAMA_SEED, app/config.py) and its wiring into the
request payload LLMClient sends (app/models/llm.py).

Both default to unset/None: the central claim this file proves is that an
existing deployment's behavior is byte-identical unless an operator
explicitly opts in — see tests/test_llm_client.py for the pre-existing
payload-shape tests this file is additive to, not a replacement for.
"""
from __future__ import annotations

import json

import pytest

from app.config import Settings, _parse_optional_float, _parse_optional_int
from app.models.llm import LLMClient


# ===========================================================================
# Settings parsing
# ===========================================================================

def test_ollama_temperature_defaults_to_none() -> None:
    assert Settings().ollama_temperature is None


def test_ollama_seed_defaults_to_none() -> None:
    assert Settings().ollama_seed is None


def test_ollama_temperature_explicit_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_TEMPERATURE", "0.2")
    assert Settings().ollama_temperature == 0.2


def test_ollama_seed_explicit_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_SEED", "42")
    assert Settings().ollama_seed == 42


def test_ollama_temperature_blank_string_is_treated_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_TEMPERATURE", "   ")
    assert Settings().ollama_temperature is None


def test_invalid_ollama_temperature_raises_at_settings_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_TEMPERATURE", "not-a-number")
    with pytest.raises(ValueError, match="invalid float configuration value"):
        Settings()


def test_invalid_ollama_seed_raises_at_settings_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_SEED", "not-an-int")
    with pytest.raises(ValueError, match="invalid integer configuration value"):
        Settings()


def test_ollama_temperature_is_read_at_construction_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without default_factory, this would fail -- matches the established
    pattern in tests/test_memory_backend_config.py/test_semantic_memory_
    config.py for every other default_factory field."""
    assert Settings().ollama_temperature is None

    monkeypatch.setenv("OLLAMA_TEMPERATURE", "0.5")
    assert Settings().ollama_temperature == 0.5


def test_parse_optional_float_unset_returns_none() -> None:
    assert _parse_optional_float(None) is None
    assert _parse_optional_float("") is None


def test_parse_optional_int_unset_returns_none() -> None:
    assert _parse_optional_int(None) is None
    assert _parse_optional_int("") is None


def test_parse_optional_int_accepts_zero_and_negative() -> None:
    """Unlike _parse_positive_int, a seed is not required to be positive."""
    assert _parse_optional_int("0") == 0
    assert _parse_optional_int("-1") == -1


# ===========================================================================
# LLMClient payload wiring
# ===========================================================================

class _FakeHTTPResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False


def _capture_payload(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout=None):  # noqa: ANN001
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        body = json.dumps({"message": {"content": "ok"}}).encode("utf-8")
        return _FakeHTTPResponse(body)

    monkeypatch.setattr("app.models.llm.urllib.request.urlopen", fake_urlopen)
    return captured


def test_default_client_sends_no_options_field(monkeypatch: pytest.MonkeyPatch) -> None:
    """The central claim: an LLMClient constructed with no temperature/seed
    (every pre-44.1 call site, and every 44.1 call site where the operator
    left the new settings unset) sends a payload identical to before this
    milestone -- no "options" key at all."""
    captured = _capture_payload(monkeypatch)
    client = LLMClient(provider="ollama", model_name="llama3.2:3b", base_url="http://localhost:11434")

    client.generate([{"role": "user", "content": "hi"}])

    assert "options" not in captured["payload"]


def test_temperature_only_adds_options_with_temperature(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_payload(monkeypatch)
    client = LLMClient(
        provider="ollama", model_name="llama3.2:3b", base_url="http://localhost:11434", temperature=0.2
    )

    client.generate([{"role": "user", "content": "hi"}])

    assert captured["payload"]["options"] == {"temperature": 0.2}


def test_seed_only_adds_options_with_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_payload(monkeypatch)
    client = LLMClient(provider="ollama", model_name="llama3.2:3b", base_url="http://localhost:11434", seed=42)

    client.generate([{"role": "user", "content": "hi"}])

    assert captured["payload"]["options"] == {"seed": 42}


def test_temperature_and_seed_together(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_payload(monkeypatch)
    client = LLMClient(
        provider="ollama",
        model_name="llama3.2:3b",
        base_url="http://localhost:11434",
        temperature=0.0,
        seed=7,
    )

    client.generate([{"role": "user", "content": "hi"}])

    assert captured["payload"]["options"] == {"temperature": 0.0, "seed": 7}


def test_sampling_options_coexist_with_json_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sampling config must not interfere with the existing json_mode flag
    (decision_maker's own use of generate(..., json_mode=True))."""
    captured = _capture_payload(monkeypatch)
    client = LLMClient(
        provider="ollama", model_name="llama3.2:3b", base_url="http://localhost:11434", temperature=0.3
    )

    client.generate([{"role": "user", "content": "hi"}], json_mode=True)

    assert captured["payload"]["format"] == "json"
    assert captured["payload"]["options"] == {"temperature": 0.3}
