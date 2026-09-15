from __future__ import annotations

import json

import pytest

from app.models.llm import LLMClient


class _FakeHTTPResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False


def _make_client() -> LLMClient:
    return LLMClient(provider="ollama", model_name="llama3.2:3b", base_url="http://localhost:11434")


def _capture_payload(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout=None):  # noqa: ANN001
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        body = json.dumps({"message": {"content": "ok"}}).encode("utf-8")
        return _FakeHTTPResponse(body)

    monkeypatch.setattr("app.models.llm.urllib.request.urlopen", fake_urlopen)
    return captured


def test_generate_without_json_mode_preserves_existing_payload_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """Backward compatibility: existing callers (Router, Orchestrator) call
    generate(messages) with no json_mode argument at all — the payload sent
    to Ollama must be identical to before this option existed."""
    captured = _capture_payload(monkeypatch)
    client = _make_client()

    reply = client.generate([{"role": "user", "content": "hi"}])

    assert reply == "ok"
    assert "format" not in captured["payload"]


def test_generate_with_json_mode_false_also_omits_format(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_payload(monkeypatch)
    client = _make_client()

    client.generate([{"role": "user", "content": "hi"}], json_mode=False)

    assert "format" not in captured["payload"]


def test_generate_with_json_mode_true_adds_format_json(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_payload(monkeypatch)
    client = _make_client()

    client.generate([{"role": "user", "content": "hi"}], json_mode=True)

    assert captured["payload"]["format"] == "json"
