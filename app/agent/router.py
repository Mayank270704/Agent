"""Single-shot classifier: user message -> one of a fixed set of routes.

As of Step 7, AgentOrchestrator (app/agent/orchestrator.py) no longer uses
this module — it delegates execution to AgentLoop + LLMDecisionMaker
(app/agent/decision_maker.py) instead. Router is kept in the codebase,
unmodified and independently tested (tests/test_router.py), rather than
deleted, because it isn't obviously dead weight — see the relationship below.

Router vs. AgentDecision/LLMDecisionMaker — same underlying question ("what
should happen with this request?"), answered very differently:

- Router is a closed, one-shot classifier over a fixed `Route` enum (llm /
  web / time / date), with a cheap deterministic regex fast-path in front of
  a single LLM call, and no visibility into tool results — it decides once,
  before anything has run.
- LLMDecisionMaker produces an open-ended AgentDecision (any registered tool,
  not a fixed set) and is called *repeatedly* by AgentLoop, each time with
  the growing execution history (prior tool calls/observations) in its
  prompt — it can change its mind after seeing what a tool returned, which
  Router structurally cannot do.

Router is not currently used, but it is NOT obviously redundant: its
deterministic fast-path is cheap, fast, and has zero LLM-reliability risk
for the patterns it covers, which the current pure-LLM-decision loop lacks
entirely (every iteration now costs a real Ollama call, even for something
as simple as "what time is it?").

UPDATE (Step 8): `decide()` (the full classifier, including its own LLM
fallback call) is still unused by AgentOrchestrator, and stays that way —
wiring it in would create a second competing decision-maker, which is
exactly what Step 8 was told to avoid. Instead, Router's existing
LLM-free deterministic regex layer (`_deterministic_temporal_check`) is now
also exposed through `classify_hint()`, a tiny, three-bucket, read-only
classification with NO LLM call and NO side effects. LLMDecisionMaker
(app/agent/decision_maker.py) uses it purely as an *advisory* line in its
own prompt — never as a bypass. LLMDecisionMaker still makes every actual
decision; the hint can be, and sometimes is, ignored by the model.

UPDATE (Step 8C): `classify_hint()` gained one small, explainable addition —
a keyword/phrase check for explicit current/recency intent ("latest",
"recent", "current") or an explicit search/look-up request ("search the
web", "search for", "look up", "look for"), used only when the existing
`_deterministic_temporal_check` found no match. This closes a measured gap
(see the Step 8B baseline): phrases like "What is the latest AI news?" or
"Search the web for information about RAG." previously fell through to
GENERAL even though they clearly signal a need for external information.
`_deterministic_temporal_check` and `decide()` themselves are UNCHANGED —
only `classify_hint()`'s own web-intent layer is new. This is deliberately
narrow (no bare "search", so "How does Google Search work?" still reads as
a static question) rather than a general named-entity or topic classifier —
see the module-level rationale for why Router still isn't wired into
AgentOrchestrator even with this addition (this remains a controlled
experiment, not a decision to keep or remove Router).
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from app.config import settings
from app.models.llm import LLMClient

logger = logging.getLogger(__name__)

Route = Literal["llm", "web", "time", "date"]


class RoutingHint(Enum):
    """A coarse, advisory-only classification of a user message, produced
    with zero LLM calls. Deliberately just three buckets — this is NOT a
    replacement for LLMDecisionMaker's own judgment, only a cheap nudge for
    the small set of patterns that are genuinely unambiguous.

    - GENERAL: no strong routing signal either way.
    - TOOL_LIKELY: the request strongly appears to need an external/current-
      information tool (e.g. a web search), but not deterministically so.
    - DETERMINISTIC_TOOL: the request unambiguously needs a deterministic
      tool such as time/date — the strongest of the three signals.
    """

    GENERAL = "general"
    TOOL_LIKELY = "tool_likely"
    DETERMINISTIC_TOOL = "deterministic_tool"


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

    # Small, explainable web-intent signal (Step 8C) — deliberately NOT a
    # general topic/entity classifier. Single-word recency signals plus a
    # handful of explicit search/look-up phrases; "search" alone is excluded
    # on purpose so a static question like "How does Google Search work?"
    # doesn't false-positive.
    _WEB_INTENT_KEYWORDS = ("latest", "recent", "current")
    _WEB_INTENT_PHRASES = ("search the web", "search for", "look up", "look for")

    def classify_hint(self, user_message: str) -> RoutingHint:
        """Cheap, deterministic, LLM-free pre-filter producing a coarse,
        advisory routing hint. Reuses the exact same regex patterns as the
        deterministic fast-path in `decide()` above — it never calls the LLM
        fallback and never itself decides anything; it only recognizes the
        small set of patterns `decide()` already treats as unambiguous, plus
        one additional LLM-free web-intent check (see `_has_web_intent`).
        """
        if user_message is None or not str(user_message).strip():
            return RoutingHint.GENERAL

        normalized = str(user_message).strip()

        deterministic = self._deterministic_temporal_check(normalized)
        if deterministic is not None:
            if deterministic.route in ("time", "date"):
                return RoutingHint.DETERMINISTIC_TOOL
            if deterministic.route == "web":
                return RoutingHint.TOOL_LIKELY
            return RoutingHint.GENERAL

        if self._has_web_intent(normalized):
            return RoutingHint.TOOL_LIKELY

        return RoutingHint.GENERAL

    def _has_web_intent(self, user_message: str) -> bool:
        """True for explicit current/recency wording or an explicit
        search/look-up request. Only called after the deterministic temporal
        check already found no match, so it never overrides a time/date
        classification."""
        normalized = re.sub(r"\s+", " ", user_message.strip().lower())
        if any(keyword in normalized for keyword in self._WEB_INTENT_KEYWORDS):
            return True
        return any(phrase in normalized for phrase in self._WEB_INTENT_PHRASES)

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
