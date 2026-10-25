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
"recent", "current") or an explicit search/look-up request (originally
"search the web", "search for", "look up", "look for" — see the Step 27
update below for why three of those four were later removed), used only
when the existing `_deterministic_temporal_check` found no match. This
closes a measured gap (see the Step 8B baseline): phrases like "What is
the latest AI news?" or "Search the web for information about RAG."
previously fell through to GENERAL even though they clearly signal a need
for external information. `_deterministic_temporal_check` and `decide()`
themselves are UNCHANGED — only `classify_hint()`'s own web-intent layer is
new. This is deliberately narrow (no bare "search", so "How does Google
Search work?" still reads as a static question) rather than a general
named-entity or topic classifier — see the module-level rationale for why
Router still isn't wired into AgentOrchestrator even with this addition
(this remains a controlled experiment, not a decision to keep or remove
Router).

UPDATE (Step 26): `classify_hint()`'s web-intent layer above is advisory
only — it adds a prompt hint, but the capability assessment measured that
llama3.2:3b still frequently ignores it (the same class of under-calling
Step 20's `deterministic_tool_route` fixed for time/date). `deterministic_
web_search_route()` is the AUTHORITATIVE counterpart, added for the exact
same reason and with the exact same discipline: it recognizes a
deliberately NARROW subset of `_has_web_intent`'s territory — explicit,
imperative search COMMANDS that ALSO name "web"/"internet"/"online"
("search the web for X", "search online for X", "look up X online", "find
X online"), never the broader recency-keyword signal ("latest", "recent",
"current") that `_has_web_intent` also matches, and never a bare topical
question. A recency keyword alone stays advisory-only (too many non-search
readings — "current mood", "recent changes to my code" — to select a tool
with certainty); an explicit command is a much stronger signal, but even
"search for X"/"look up X" ALONE (no web/online/internet reference) was
found, during this milestone's own adversarial testing, to false-positive
on ordinary non-web requests ("search for the bug in this function", "look
up to your role model") — so an explicit locality qualifier is required on
every pattern, not merely an imperative verb. See `deterministic_web_
search_route`'s own docstring for the exact pattern set, the negation
guard, and the adversarial cases that shaped it.

UPDATE (Step 27): Step 26 narrowed the AUTHORITATIVE route to exclude bare
"search for X"/"look up X" but left `_has_web_intent`'s ADVISORY phrase set
containing exactly those same bare phrases — an inconsistency: the same
words Step 26 proved too ambiguous to select a tool WITH certainty were
still nudging the model TOWARD that same wrong tool via the ROUTING HINT
line `LLMDecisionMaker._format_hint` renders for `RoutingHint.TOOL_LIKELY`.
A real-model diagnostic (with-hint vs. hint-disabled, 3 trials each, same
9 ambiguous/local-context cases, same model, same temperature — the only
variable changed was whether the hint fired) measured a 33% (9/27)
unnecessary-`web_search` rate with the hint active on those cases, dropping
to 0% (0/27) with it disabled. That isolates the phrase set itself, not
model nondeterminism alone, as the dominant contributor. `_WEB_INTENT_
PHRASES` below was narrowed to drop "search for"/"look up"/"look for",
keeping only "search the web" — bringing the advisory hint's bare-verb
handling into the same, already-proven-safer posture the deterministic
route has used since Step 26. The bare recency KEYWORDS ("latest",
"recent", "current") are unchanged and untouched by this update; they were
not implicated by the diagnostic and remain a materially different signal
(a topic being described as current, not an imperative verb).
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
    #
    # UPDATE (Step 27): "search for"/"look up"/"look for" were REMOVED from
    # this list — they used to sit here as bare phrases, same as they still
    # do NOT sit in `_WEB_SEARCH_COMMAND_PATTERNS` below (Step 26 already
    # excluded them there for an identical reason). Leaving them here was an
    # inconsistency: Step 26 proved bare "search for X"/"look up X" is too
    # ambiguous to select a tool WITH certainty, but the identical bare
    # phrases were still nudging the model TOWARD that same wrong tool via
    # this advisory hint. A real-model diagnostic run for Step 27
    # (with-hint vs. hint-disabled, 3 trials each, 9 cases: "look up to your
    # role model," "search for the bug in this function," "search for a
    # job," etc.) measured a 33% (9/27) unnecessary-web_search rate WITH
    # this phrase set active on those exact cases, dropping to 0% (0/27)
    # with the hint disabled — i.e. the SAME cases, same model, same
    # temperature; only the hint changed. That isolates this phrase set,
    # not model nondeterminism alone, as the dominant cause. "search the
    # web" is kept: it is already covered by the (much narrower, qualifier-
    # requiring) deterministic route below, so it is largely redundant here
    # in production, but it remains a correct, unambiguous signal for any
    # caller that uses `classify_hint()` on its own.
    _WEB_INTENT_KEYWORDS = ("latest", "recent", "current")
    _WEB_INTENT_PHRASES = ("search the web",)

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

    def deterministic_tool_route(self, user_message: str) -> tuple[str, str | None] | None:
        """Public, LLM-free accessor over the EXISTING deterministic
        temporal matcher, narrowed to the cases it already classifies with
        certainty as a LOCAL time/date tool operation.

        Returns `(tool_name, tool_input)` — either `("time", None)` (the
        time tool takes no input) or `("date", "<the matched date text>")`
        — or `None` for everything else.

        This method adds NO new matching of its own: it delegates verbatim
        to `_deterministic_temporal_check` (unchanged since Step 8) and only
        translates that existing verdict into the tool-name/tool-input pair
        a caller would need. In particular it deliberately returns `None`
        for:
        - `route="web"` (e.g. "What happened on 12 September 2026?") — a
          specified date in a question is not the same thing as a local
          weekday calculation, and which external tool (if any) should
          answer it is exactly the open-ended judgment that must stay with
          LLMDecisionMaker.
        - anything the matcher does not recognize at all, including
          ambiguous temporal phrasing ("Explain today's date conceptually",
          "What is the current price of Bitcoin?"). Ambiguity is left to
          the normal LLM decision path, by design.

        See app/agent/decision_maker.py for the one caller, and why acting
        on this verdict is opt-in there rather than automatic here — this
        method decides nothing and has no side effects; it only reports
        what the existing matcher already knows.
        """
        if user_message is None or not str(user_message).strip():
            return None

        decision = self._deterministic_temporal_check(str(user_message).strip())
        if decision is None:
            return None
        if decision.route == "time":
            return ("time", None)
        if decision.route == "date":
            return ("date", decision.search_query or None)
        return None

    # Step 26 — authoritative web-search-COMMAND routing. Deliberately a
    # much narrower set than `_WEB_INTENT_KEYWORDS`/`_WEB_INTENT_PHRASES`
    # above (which stay advisory-only, unchanged): every pattern here
    # requires BOTH an imperative search/look-up/find verb AND an explicit
    # "web"/"internet"/"online" qualifier — never one without the other.
    # Each pattern has exactly one capture group — the text to search for.
    #
    # This is narrower than the Step 26 design brief's own Category A list
    # (which includes bare "search for ..."), and that narrowing is
    # evidence-driven, not a judgment call made in the abstract: adversarial
    # testing during this milestone (see tests/test_deterministic_web_
    # search_routing.py) found that a bare, unqualified "search for X" /
    # "look up X" — with no web/online/internet reference at all — false-
    # positives on ordinary, non-web requests that merely happen to share
    # the verb: "search for the bug in this function", "look up to your
    # role model" (a different phrasal verb entirely), "search for a job".
    # None of those should ever bypass the model's own judgment. Requiring
    # an explicit web/online/internet qualifier keeps every remaining
    # pattern true to its literal words: if the user says "the web" or
    # "online", routing to `web_search` is never a misreading, regardless
    # of how unusual the search subject is. Bare "search for"/"look up"
    # remain covered by the EXISTING advisory hint (`_has_web_intent`,
    # unchanged) — nudged, never auto-selected.
    #
    # Step 28: `\bfind\s+(.+?)\s+on\s+the\s+(?:web|internet)\b` was added
    # below to close a coverage gap found during Step 27's evaluation —
    # "find X on the internet"/"find X on the web" was falling through to
    # the LLM even though "look up X on the web/internet" (the otherwise
    # identical pattern, two lines up) was already covered. It is the exact
    # same qualifier-required shape as every other pattern here, just for
    # the "find" verb's "on the web/internet" phrasing instead of "online"
    # — not a new class of match, only a missing symmetric case of an
    # already-present one.
    _WEB_SEARCH_COMMAND_PATTERNS = (
        re.compile(r"\bsearch\s+(?:the\s+)?(?:web|internet)\s+for\s+(.+)"),
        re.compile(r"\bsearch\s+online\s+for\s+(.+)"),
        re.compile(r"\blook\s+up\s+(.+?)\s+online\b"),
        re.compile(r"\blook\s+up\s+(.+?)\s+on\s+the\s+(?:web|internet)\b"),
        re.compile(r"\bfind\s+online\s+(.+)"),
        re.compile(r"\bfind\s+(.+?)\s+online\b"),
        re.compile(r"\bfind\s+(.+?)\s+on\s+the\s+(?:web|internet)\b"),
    )

    # A message containing any of these anywhere is never routed
    # deterministically — negated/hypothetical phrasing ("don't search the
    # web for X", "no need to look up X") is exactly the ambiguity this
    # narrow route must not paper over. It falls through to the ordinary
    # LLM decision path instead, unchanged.
    _WEB_SEARCH_NEGATION_MARKERS = (
        "don't",
        "do not",
        "doesn't",
        "won't",
        "never",
        "shouldn't",
        "no need to",
        "without searching",
        "stop searching",
    )

    def deterministic_web_search_route(self, user_message: str) -> tuple[str, str] | None:
        """Public, LLM-free accessor recognizing an explicit web-search
        COMMAND with certainty — the `web_search` counterpart to
        `deterministic_tool_route`'s time/date coverage above.

        Returns `("web_search", "<extracted query text>")`, or `None` for
        everything else: no command phrase matched (including a bare,
        unqualified "search for X"/"look up X" with no "web"/"online"/
        "internet" reference — see `_WEB_SEARCH_COMMAND_PATTERNS`'s own
        comment for why that qualifier is required on every pattern), the
        phrase matched but cleanup left no usable query text, or a negation
        marker was present anywhere in the message. This method does not
        call `_deterministic_temporal_check` itself —
        precedence between the two deterministic routes is decided by the
        one caller (`LLMDecisionMaker._deterministic_web_decision`, called
        only after the temporal check already returned `None` — see that
        module for why), not duplicated here.

        Deliberately excluded, matching Category D of the Step 26 design:
        a bare recency keyword ("What's the latest on X?"), a bare
        current-information question ("What is the current price of
        Bitcoin?"), and a historical/static question ("When was the Eiffel
        Tower built?") all return `None` here — none of them are an
        explicit instruction to search, so which tool (if any) should
        answer them stays the LLM's judgment call, exactly as it already
        is for the equivalent non-command cases in `deterministic_tool_
        route`.
        """
        if user_message is None or not str(user_message).strip():
            return None

        normalized = re.sub(r"\s+", " ", str(user_message).strip().lower())

        if any(marker in normalized for marker in self._WEB_SEARCH_NEGATION_MARKERS):
            return None

        for pattern in self._WEB_SEARCH_COMMAND_PATTERNS:
            match = pattern.search(normalized)
            if not match:
                continue
            query = self._clean_web_search_query(match.group(1))
            if query:
                return ("web_search", query)

        return None

    def _clean_web_search_query(self, raw_query: str) -> str | None:
        """Strip a redundant trailing "online"/"on the web"/"on the
        internet" qualifier (the command phrase already established that —
        repeating it inside the search query itself would only dilute a
        Tavily search) and surrounding punctuation/whitespace. Returns
        `None` for a query that is empty after cleanup, e.g. "search for"
        with no object at all."""
        cleaned = re.sub(r"\s+(?:online|on the web|on the internet)\s*$", "", raw_query.strip())
        cleaned = cleaned.strip(" ?!.,")
        return cleaned or None

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

        # Step 44.2: "what is the current date" was removed from this
        # substring-anywhere tuple -- it false-positived on "what is the
        # current date format?" (substring-anywhere matching cannot tell a
        # trailing qualifier like "format?" apart from nothing at all). It
        # now lives in `unambiguous_current_time_phrases` below instead,
        # which requires the WHOLE message to match (after stripping
        # trailing punctuation) -- "what is the current date?" still
        # routes; "what is the current date format?" correctly does not.
        # No other entry here was found to have this problem against this
        # milestone's evidence (see tests/test_router.py), so none of the
        # others were moved.
        temporal_patterns = (
            "which year is this",
            "which year is it",
            "what year is it",
            "what is the current year",
            "what date is today",
            "what is today\'s date",
            "what day is it today",
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

        # Step 30 — a small, evidence-bounded set of additional unambiguous
        # CURRENT-TIME phrasings, found missing by the Step 29 end-to-end
        # evaluation (Category B: "Can you tell me the current time?" and
        # "What's the time?" both fell through to LLM judgment and were
        # missed in the majority of trials, despite being exactly as
        # unambiguous as the already-covered "What time is it?").
        #
        # Deliberately matched as the ENTIRE message (after stripping
        # trailing punctuation only), NOT as a substring-anywhere check
        # like `temporal_patterns` above: "what is the time" and "can you
        # tell me the current time" are short, generic-sounding fragments
        # that DO occur inside unrelated, realistic questions this exact
        # deployment already fields as an assistant used for technical
        # help — e.g. "What is the time complexity of this algorithm?" or
        # "Can you tell me the current time complexity of this sort?" (see
        # the Step 29 evaluation's own Category G — "search for"/"look
        # up"/local-technical-context phrasing is a proven-real class of
        # request here). A substring-anywhere match would silently
        # misroute those to the local clock instead of answering the real
        # question — exactly the "broad regex that routes every query
        # containing 'time'" this milestone was told not to build.
        # Requiring the phrase to be the WHOLE message (trailing "?"/"!"/
        # "." aside) closes the Step 29 gap without that risk: a longer
        # sentence built AROUND one of these phrases never matches.
        #
        # Step 44.2 — a read-only diagnosis of a live "today date" failure
        # found this exact same gap on the DATE side: short, fragment-style
        # date requests ("today date", "today's date", "current date",
        # "date today") matched none of `temporal_patterns` above (that
        # tuple requires a full question shape — "what is today's date",
        # "what date is today", etc. — and is checked substring-anywhere,
        # which these shorter fragments don't satisfy either way) and fell
        # through to the ordinary LLM decision call, where the model
        # sometimes answered from unrelated prior conversation context
        # instead of the date (reproduced directly, including in a FRESH
        # session with no prior context — a model-reliability gap once
        # deterministic certainty is absent, not a memory-contamination
        # bug). These resolve to the SAME `route="time"` bucket as
        # "what is today's date?" already does (TimeTool, not DateTool —
        # `date`/DateTool stays reserved for a weekday calculation against
        # an explicitly specified date, matching every other phrasing in
        # this bucket). "time now" is added for the same reason, alongside
        # the existing current-time fragments it already sits next to.
        #
        # Using the SAME whole-message discipline as the block above (not
        # added to the substring-anywhere `temporal_patterns` tuple) is
        # deliberately load-bearing here, not incidental: "current date" as
        # a substring-anywhere match would wrongly fire on "what is the
        # current date format?" or "what's the current date for the
        # meeting?" — exactly the "broad keyword matching" this fix must
        # not become. As a whole-message match, a longer sentence built
        # around one of these fragments (including "i asked today date",
        # which contains "today date" but is not equal to it) never
        # matches, by construction.
        unambiguous_current_time_phrases = (
            "what's the time",
            "what is the time",
            "can you tell me the current time",
            "can you tell me what time it is",
            "time now",
            "today date",
            "today's date",
            "current date",
            "date today",
            "what is the current date",
        )
        if normalized.rstrip("?!.").strip() in unambiguous_current_time_phrases:
            return RouterDecision(
                needs_web=False,
                reason="This question asks for the current time and depends on present temporal context.",
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
