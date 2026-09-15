from __future__ import annotations

import io
import json
import urllib.error
from datetime import datetime

import pytest

from app.tools.web_search import WebSearchTool
from app.tools.base import ToolResult


class _FakeHTTPResponse:
    """Stands in for the object returned by `urllib.request.urlopen`."""

    def __init__(self, body: bytes):
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False


def _make_tool(api_key: str = "test-key") -> WebSearchTool:
    tool = WebSearchTool()
    tool.api_key = api_key
    return tool


def _forbid_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("network should not be called")

    monkeypatch.setattr("app.tools.web_search.urllib.request.urlopen", fail)


# ---------------------------------------------------------------------------
# Query validation — must fail before any network call is attempted.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_query", ["", "   ", None])
def test_execute_validates_query(monkeypatch: pytest.MonkeyPatch, bad_query: str | None) -> None:
    _forbid_network(monkeypatch)
    tool = _make_tool()

    with pytest.raises(ValueError):
        tool.execute(bad_query)  # type: ignore[arg-type]


def test_execute_missing_api_key_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _forbid_network(monkeypatch)
    tool = _make_tool(api_key="")

    with pytest.raises(ValueError, match="TAVILY_API_KEY"):
        tool.execute("some query")


# ---------------------------------------------------------------------------
# Successful response parsing + metadata preservation.
# ---------------------------------------------------------------------------

def test_execute_parses_and_normalizes_results(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout=None):  # noqa: ANN001
        captured["request"] = request
        payload = {
            "results": [
                {
                    "title": "A", "url": "https://a.com", "content": "content a",
                    "score": 0.9, "published_date": "2026-01-01",
                },
                {
                    "title": "B", "url": "https://b.com", "snippet": "content b",
                    "publishedDate": "2026-01-02", "domain": "b.com",
                },
                {
                    "title": "C", "url": "https://c.com", "content": "content c",
                    "date": "2026-01-03", "site": "c.com",
                },
                "not-a-dict-should-be-skipped",
            ]
        }
        return _FakeHTTPResponse(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr("app.tools.web_search.urllib.request.urlopen", fake_urlopen)

    tool = _make_tool()
    result = tool.execute("current gold price")

    assert isinstance(result, ToolResult)
    assert result.success is True
    assert result.error is None

    results = result.data
    assert len(results) == 3  # non-dict item skipped

    assert results[0]["title"] == "A"
    assert results[0]["published_date"] == "2026-01-01"
    assert "source" not in results[0]

    assert results[1]["content"] == "content b"  # falls back to "snippet"
    assert results[1]["published_date"] == "2026-01-02"  # from "publishedDate"
    assert results[1]["source"] == "b.com"  # from "domain"

    assert results[2]["content"] == "content c"
    assert results[2]["published_date"] == "2026-01-03"  # from "date"
    assert results[2]["source"] == "c.com"  # from "site"

    for item in results:
        assert "retrieved_at" in item
        datetime.fromisoformat(str(item["retrieved_at"]))  # must be valid ISO

    sent_payload = json.loads(captured["request"].data.decode("utf-8"))
    assert sent_payload["query"] == "current gold price"
    assert sent_payload["api_key"] == tool.api_key
    assert sent_payload["time_range"] == "day"  # "current" -> day recency


# ---------------------------------------------------------------------------
# Recency / time-range selection — pure functions, no network involved.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "query,expected",
    [
        ("What is a neural network?", False),
        ("What is the current gold price?", True),
        ("Latest AI news", True),
        ("What happened this week in tech?", True),
    ],
)
def test_is_time_sensitive_query(query: str, expected: bool) -> None:
    tool = _make_tool()
    assert tool._is_time_sensitive_query(query) is expected


@pytest.mark.parametrize(
    "query,expected_time_range",
    [
        ("What is a neural network?", None),
        ("current gold price", "day"),
        ("gold price today", "day"),
        ("latest AI news", "week"),
        ("recent AI breakthroughs", "week"),
        ("news this week", "week"),
        ("news this month", "month"),
    ],
)
def test_get_search_options_time_range_selection(query: str, expected_time_range: str | None) -> None:
    tool = _make_tool()
    recency_detected, time_range = tool._get_search_options(query)
    assert time_range == expected_time_range
    assert recency_detected is (expected_time_range is not None)


# ---------------------------------------------------------------------------
# HTTP / network / malformed-response failure handling.
# ---------------------------------------------------------------------------

def test_execute_raises_runtime_error_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request, timeout=None):  # noqa: ANN001
        raise urllib.error.HTTPError(
            url="https://api.tavily.com/search",
            code=401,
            msg="Unauthorized",
            hdrs=None,  # type: ignore[arg-type]
            fp=io.BytesIO(b'{"error":"invalid api key"}'),
        )

    monkeypatch.setattr("app.tools.web_search.urllib.request.urlopen", fake_urlopen)

    tool = _make_tool()
    result = tool.execute("some query")

    assert isinstance(result, ToolResult)
    assert result.success is False
    assert result.data is None
    assert "HTTP 401" in result.error


def test_execute_raises_runtime_error_on_network_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request, timeout=None):  # noqa: ANN001
        raise urllib.error.URLError("timed out")

    monkeypatch.setattr("app.tools.web_search.urllib.request.urlopen", fake_urlopen)

    tool = _make_tool()
    result = tool.execute("some query")

    assert isinstance(result, ToolResult)
    assert result.success is False
    assert result.data is None
    assert "network or timeout" in result.error


def test_execute_raises_runtime_error_on_malformed_json(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request, timeout=None):  # noqa: ANN001
        return _FakeHTTPResponse(b"not valid json")

    monkeypatch.setattr("app.tools.web_search.urllib.request.urlopen", fake_urlopen)

    tool = _make_tool()
    result = tool.execute("some query")

    assert isinstance(result, ToolResult)
    assert result.success is False
    assert result.data is None
    assert "not valid JSON" in result.error


# ---------------------------------------------------------------------------
# Live integration smoke test — real network call to the real Tavily API.
# Excluded from the default suite via `-m "not integration"`.
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_web_search_tool_live_tavily_smoke() -> None:
    tool = WebSearchTool()
    if not tool.api_key or not tool.api_key.strip():
        pytest.skip("TAVILY_API_KEY not configured; skipping live Tavily integration test")

    general_query = "What is a neural network?"
    general_result = tool.execute(general_query)
    assert isinstance(general_result, ToolResult)
    assert general_result.success is True
    assert isinstance(general_result.data, list)
    assert tool._is_time_sensitive_query(general_query) is False

    current_query = "What is the current gold price in India today?"
    current_result = tool.execute(current_query)
    assert current_result.success is True
    assert isinstance(current_result.data, list)
    recency_detected, time_range = tool._get_search_options(current_query)
    assert recency_detected is True
    assert time_range == "day"
