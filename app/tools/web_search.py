from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
import urllib.error
import urllib.request

from app.config import settings
from app.tools.base import RiskLevel, ToolCapability, ToolResult

logger = logging.getLogger(__name__)


class WebSearchTool:
    name = "web_search"
    description = "Search the web for current information using the Tavily Search API."
    input_schema: dict[str, str] = {"query": "string"}
    output_description = "A list of web search results, each with title, url, content, and source metadata."
    # Milestone 18: this tool's defining characteristic is the external
    # HTTP call to a third-party API (Tavily) — untrusted response
    # content, the query leaving this process, and independent network
    # failure modes — not a local read/write distinction, hence
    # EXTERNAL_NETWORK rather than READ. Classified MEDIUM (not LOW) for
    # that reason, though it remains non-destructive and needs no
    # confirmation.
    capability = ToolCapability.EXTERNAL_NETWORK
    risk_level = RiskLevel.MEDIUM
    requires_confirmation = False

    def __init__(self):
        self.api_key = settings.tavily_api_key

    def _is_time_sensitive_query(self, query: str) -> bool:
        lowered = query.lower()
        time_sensitive_keywords = (
            "today",
            "current",
            "latest",
            "recent",
            "now",
            "live",
            "this week",
            "this month",
            "latest news",
            "current price",
            "current rate",
            "current status",
        )
        return any(keyword in lowered for keyword in time_sensitive_keywords)

    def _get_search_options(self, query: str) -> tuple[bool, str | None]:
        if not self._is_time_sensitive_query(query):
            return False, None

        lowered = query.lower()
        if any(keyword in lowered for keyword in ("today", "now", "live", "current")):
            return True, "day"
        if any(keyword in lowered for keyword in ("latest", "recent")):
            return True, "week"
        if "this week" in lowered:
            return True, "week"
        if "this month" in lowered:
            return True, "month"
        return True, "week"

    def execute(self, input: str | None = None) -> ToolResult:
        """Run a Tavily web search for `input` (the search query).

        Raises ValueError for invalid input (empty query, missing API key
        configuration) — these are precondition failures, not operational
        ones. Returns ToolResult(success=False, error=...) for network/API
        failures, which are expected, recoverable failure modes (see
        app.tools.base for the full error contract).
        """
        query = input
        if query is None or not str(query).strip():
            raise ValueError("Search query cannot be empty.")

        if not self.api_key or not self.api_key.strip():
            raise ValueError("Missing TAVILY_API_KEY configuration. Set it in the .env file.")

        try:
            cleaned = self._search(str(query).strip())
        except RuntimeError as exc:
            return ToolResult.fail(str(exc))

        return ToolResult.ok(cleaned)

    def _search(self, original_query: str) -> list[dict[str, object]]:
        """Existing Tavily REST call + result cleanup, unchanged from before
        this abstraction — only the caller-facing `execute()` contract above
        changed, not search semantics."""
        recency_detected, time_range = self._get_search_options(original_query)

        payload: dict[str, object] = {
            "api_key": self.api_key,
            "query": original_query,
            "max_results": 5,
        }
        if recency_detected and time_range:
            payload["time_range"] = time_range

        logger.info(
            "Tavily search request: query=%s recency_detected=%s time_range=%s",
            original_query,
            recency_detected,
            time_range,
        )

        request = urllib.request.Request(
            "https://api.tavily.com/search",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                raw_data = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Tavily API request failed with HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Tavily request failed due to network or timeout: {exc}") from exc

        try:
            data = json.loads(raw_data)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Tavily response was not valid JSON.") from exc

        results = data.get("results", [])
        retrieval_timestamp = datetime.now(timezone.utc).isoformat()
        cleaned = []
        for item in results:
            if not isinstance(item, dict):
                continue

            published_date = (
                item.get("published_date")
                or item.get("publishedDate")
                or item.get("published")
                or item.get("date")
                or None
            )
            source = item.get("source") or item.get("domain") or item.get("site") or None

            cleaned_item = {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content") or item.get("snippet") or "",
                "score": item.get("score"),
                "retrieved_at": retrieval_timestamp,
            }

            if published_date is not None:
                cleaned_item["published_date"] = published_date
            if source is not None:
                cleaned_item["source"] = source

            cleaned.append(cleaned_item)

        logger.info(
            "Tavily search completed: original_query=%s recency_detected=%s time_range=%s result_count=%d",
            original_query,
            recency_detected,
            time_range,
            len(cleaned),
        )
        return cleaned
