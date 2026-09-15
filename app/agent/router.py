from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Literal

from app.config import settings
from app.models.llm import LLMClient

logger = logging.getLogger(__name__)

Route = Literal["llm", "web", "time", "date"]


@dataclass(frozen=True)
class RouterDecision:
    needs_web: bool
    reason: str
    search_query: str
    route: Route = "llm"


class Router:
    """Decides whether a user request needs external/current web information."""

    def __init__(self, llm_client: LLMClient | None = None):
        self.llm = llm_client or LLMClient(
            provider=settings.llm_provider,
            model_name=settings.model_name,
            api_key=settings.openai_api_key,
            base_url=settings.ollama_base_url,
        )

    def decide(self, user_message: str) -> RouterDecision:
        if user_message is None or not str(user_message).strip():
            raise ValueError("User message cannot be empty.")

        normalized = str(user_message).strip()

        deterministic = self._deterministic_temporal_check(normalized)
        if deterministic is not None:
            return deterministic

        prompt = self._build_prompt(normalized)

        try:
            raw_response = self.llm.generate([
                {"role": "user", "content": prompt},
            ])
        except RuntimeError as exc:
            logger.warning("Router LLM call failed: %s", exc)
            return RouterDecision(
                needs_web=False,
                reason="The router could not reliably decide whether external web information was needed.",
                search_query="",
                route="llm",
            )

        return self._parse_decision(raw_response)

    def _deterministic_temporal_check(self, user_message: str) -> RouterDecision | None:
        normalized = re.sub(r"\s+", " ", user_message.strip().lower())
        explicit_date_pattern = (
            r"\b(?:"
            r"\d{1,2}\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
            r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
            r"nov(?:ember)?|dec(?:ember)?)\s+\d{4}|"
            r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
            r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
            r"nov(?:ember)?|dec(?:ember)?)\s+\d{1,2},?\s+\d{4}|"
            r"\d{4}[-/]\d{1,2}[-/]\d{1,2}"
            r")\b"
        )
        explicit_date_match = re.search(explicit_date_pattern, normalized)
        if explicit_date_match:
            weekday_question = re.search(
                r"\b(?:what|which)\s+(?:day|weekday)(?:\s+of\s+the\s+week)?\s+(?:was|is|will\s+be)\b",
                normalized,
            )
            if weekday_question:
                return RouterDecision(
                    needs_web=False,
                    reason="This is a deterministic weekday calculation for a specified date.",
                    search_query=explicit_date_match.group(0).strip(),
                    route="date",
                )
            return RouterDecision(
                needs_web=True,
                reason="This question asks about information associated with a specified date.",
                search_query=normalized,
                route="web",
            )

        temporal_patterns = (
            "which year is this",
            "which year is it",
            "what year is it",
            "what is the current year",
            "what date is today",
            "what is today\'s date",
            "what day is it today",
            "what is the current date",
            "what time is it",
            "what is the date right now",
            "what day is today",
            "what's today's date",
            "what year is it now",
            "what is the current time",
            "what day is it",
            "what date is it",
        )

        if any(pattern in normalized for pattern in temporal_patterns):
            search_query = self._generate_temporal_search_query(normalized)
            return RouterDecision(
                needs_web=False,
                reason="This question asks for the current date, time, or year and depends on present temporal context.",
                search_query="",
                route="time",
            )

        return None

    def _generate_temporal_search_query(self, normalized_message: str) -> str:
        if "time" in normalized_message or "what time" in normalized_message:
            return "current time today"

        if "date" in normalized_message or "day" in normalized_message:
            if "today" in normalized_message:
                return "today's date"
            return "current date today"

        if "year" in normalized_message:
            if "current" in normalized_message or "today" in normalized_message:
                return "current year today"
            return "current year today"

        return "current information today"

    def _build_prompt(self, user_message: str) -> str:
        return f"""
You are a router for a chatbot. Decide which path should answer the user's request.

Return STRICT JSON only with this shape:
{{
    "route": "web",
    "needs_web": true,
    "reason": "The user is asking for current information from the web.",
  "search_query": "latest AI news"
}}

Rules:
- Set route="time" only for direct questions asking for the current local date, year, day, or time, such as "What year is it?" or "What is today's date?".
- Set route="date" for deterministic weekday questions about a specified date, such as "What day was 27 July 2026?". These questions do not need web search or an LLM calculation.
- Set route="web" when the request needs current, recent, live, or present-world facts that are not simply local date/time information.
- Questions about a specific numeric or named date, including historical or future dates, must use route="web". Examples include "What day was 12 September 2026?" and "What happened on 12 September 2026?".
- Set route="llm" for stable knowledge, explanations, writing, and casual conversation.
- For route="web", set needs_web=true and provide a useful search_query.
- For route="llm", route="time", or route="date", set needs_web=false. For route="llm" and route="time", set search_query to an empty string.
- Current information that should use the web includes:
  - latest/current/recent information
  - live prices
  - current weather
  - current sports scores/results
  - current news
  - current events
  - current product/service availability
  - information that explicitly depends on the present state of the world
  - requests where factual freshness is important
- Do not classify every query containing "current" as route="time". Current prices, population, software versions, news, weather, and similar facts use route="web".
- Use route="llm" for:
  - general concepts
  - programming explanations
  - mathematics
  - algorithms
  - machine learning concepts
  - explanations of historical/stable knowledge
  - writing/rewriting
  - casual conversation
  - reasoning that can be completed from the model's existing knowledge

Important: do not rely only on keywords. Use semantic judgment.

User request:
{user_message}
""".strip()

    def _parse_decision(self, raw_response: str) -> RouterDecision:
        cleaned = (raw_response or "").strip()

        if not cleaned:
            return RouterDecision(
                needs_web=False,
                reason="The router decision could not be parsed because the model returned no content.",
                search_query="",
            )

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            return RouterDecision(
                needs_web=False,
                reason="The router decision could not be parsed because the model output was not valid JSON.",
                search_query="",
            )

        if not isinstance(parsed, dict):
            return RouterDecision(
                needs_web=False,
                reason="The router decision could not be parsed because the model output was not a JSON object.",
                search_query="",
            )

        needs_web = parsed.get("needs_web")
        reason = parsed.get("reason")
        search_query = parsed.get("search_query")
        route = parsed.get("route")

        if route is None:
            if not isinstance(needs_web, bool):
                return RouterDecision(
                    needs_web=False,
                    reason="The router decision could not be parsed because neither route nor needs_web was valid.",
                    search_query="",
                    route="llm",
                )
            route = "web" if needs_web else "llm"

        if route not in ("llm", "web", "time", "date"):
            return RouterDecision(
                needs_web=False,
                reason="The router decision could not be parsed because route was invalid.",
                search_query="",
                route="llm",
            )

        if not isinstance(reason, str):
            return RouterDecision(
                needs_web=False,
                reason="The router decision could not be parsed because reason was not a string.",
                search_query="",
                route="llm",
            )

        if not isinstance(search_query, str):
            search_query = ""

        if route == "web":
            needs_web = True
        else:
            needs_web = False
            search_query = ""

        if route == "web" and not search_query.strip():
            search_query = "current information today"

        return RouterDecision(
            needs_web=needs_web,
            reason=reason,
            search_query=search_query.strip() if search_query else "",
            route=route,
        )
