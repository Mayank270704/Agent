from __future__ import annotations

import json

import pytest

from app.agent.router import Router, RoutingHint


class FakeLLM:
    """Replays a fixed sequence of responses; records every prompt it receives."""

    def __init__(self, responses: list[str]):
        self.responses = iter(responses)
        self.calls: list[list[dict[str, str]]] = []

    def generate(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        return next(self.responses)


class RaisingLLM:
    """Fails the test if the router ever calls the LLM (used for deterministic paths)."""

    def generate(self, messages: list[dict[str, str]]) -> str:
        raise AssertionError("LLM should not be called for deterministic routing")


class ErrorLLM:
    """Simulates an LLM/connection failure (e.g. Ollama unreachable)."""

    def generate(self, messages: list[dict[str, str]]) -> str:
        raise RuntimeError("simulated LLM/connection failure")


# ---------------------------------------------------------------------------
# Deterministic (regex-based) routing — must NOT call the LLM at all.
# ---------------------------------------------------------------------------

DETERMINISTIC_EXAMPLES = (
    ("What time is it?", "time"),
    ("What is today's date?", "time"),
    ("What year is it?", "time"),
    ("What day is today?", "time"),
    ("What day was 27 July 2026?", "date"),
    ("What day was 12 September 2026?", "date"),
    ("What day is 25 December 2026?", "date"),
    ("What happened on 12 September 2026?", "web"),
)


@pytest.mark.parametrize("text,expected_route", DETERMINISTIC_EXAMPLES)
def test_deterministic_temporal_routing(text: str, expected_route: str) -> None:
    router = Router(RaisingLLM())
    decision = router.decide(text)
    assert decision.route == expected_route
    assert decision.needs_web is (expected_route == "web")
    if expected_route == "time":
        assert decision.search_query == ""


def test_deterministic_date_route_search_query_is_the_matched_date() -> None:
    router = Router(RaisingLLM())
    decision = router.decide("What day was 27 July 2026?")
    assert decision.route == "date"
    assert "27 july 2026" in decision.search_query


# ---------------------------------------------------------------------------
# LLM-JSON routing — only reached when no deterministic pattern matches.
# ---------------------------------------------------------------------------

def test_llm_json_routing_normal_llm_route() -> None:
    llm = FakeLLM([json.dumps({
        "route": "llm", "needs_web": False, "reason": "stable", "search_query": "",
    })])
    router = Router(llm)

    decision = router.decide("What is a neural network?")

    assert decision.route == "llm"
    assert decision.needs_web is False
    assert len(llm.calls) == 1


def test_llm_json_routing_web_route() -> None:
    llm = FakeLLM([json.dumps({
        "route": "web", "needs_web": True, "reason": "current price",
        "search_query": "today's gold price",
    })])
    router = Router(llm)

    decision = router.decide("What is today's gold price?")

    assert decision.route == "web"
    assert decision.needs_web is True
    assert decision.search_query == "today's gold price"


def test_llm_json_routing_web_route_current_gold_price() -> None:
    llm = FakeLLM([json.dumps({
        "route": "web", "needs_web": True, "reason": "current price",
        "search_query": "current gold price",
    })])
    router = Router(llm)

    decision = router.decide("What is the current gold price?")

    assert decision.route == "web"
    assert decision.needs_web is True


# ---------------------------------------------------------------------------
# Fallback / error-handling paths.
# ---------------------------------------------------------------------------

def test_malformed_json_falls_back_to_llm_route() -> None:
    llm = FakeLLM(["not valid json {"])
    router = Router(llm)

    decision = router.decide("Some ambiguous question")

    assert decision.route == "llm"
    assert decision.needs_web is False
    assert decision.search_query == ""


def test_invalid_route_value_falls_back_to_llm_route() -> None:
    llm = FakeLLM([json.dumps({
        "route": "not_a_real_route", "needs_web": False, "reason": "x", "search_query": "",
    })])
    router = Router(llm)

    decision = router.decide("Some ambiguous question")

    assert decision.route == "llm"
    assert decision.needs_web is False


def test_non_dict_json_falls_back_to_llm_route() -> None:
    llm = FakeLLM([json.dumps(["route", "web"])])
    router = Router(llm)

    decision = router.decide("Some ambiguous question")

    assert decision.route == "llm"
    assert decision.needs_web is False


def test_llm_runtime_error_falls_back_to_llm_route() -> None:
    router = Router(ErrorLLM())

    decision = router.decide("Some ambiguous question")

    assert decision.route == "llm"
    assert decision.needs_web is False
    assert "could not reliably decide" in decision.reason


def test_empty_message_raises_value_error() -> None:
    router = Router(RaisingLLM())

    with pytest.raises(ValueError):
        router.decide("   ")


# ---------------------------------------------------------------------------
# classify_hint() (Step 8, extended Step 8C) — LLM-free, advisory-only,
# three buckets: GENERAL / TOOL_LIKELY / DETERMINISTIC_TOOL.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected_hint",
    [
        # Part 8, items 1-3: deterministic date/time signals.
        ("What time is it?", RoutingHint.DETERMINISTIC_TOOL),
        ("What is today's date?", RoutingHint.DETERMINISTIC_TOOL),
        ("What day is 25 December 2026?", RoutingHint.DETERMINISTIC_TOOL),
        ("What day was 27 July 2026?", RoutingHint.DETERMINISTIC_TOOL),
        ("What happened on 12 September 2026?", RoutingHint.TOOL_LIKELY),
        # Part 8, items 4-6: explicit current/recency or search-intent wording.
        ("What is the latest AI news?", RoutingHint.TOOL_LIKELY),
        ("Search for current NVIDIA news.", RoutingHint.TOOL_LIKELY),
        ("Search the web for recent RAG research.", RoutingHint.TOOL_LIKELY),
        # Part 8, items 7-9: named entities / topics WITHOUT recency wording
        # must stay GENERAL — a mention of NVIDIA/OpenAI/a technical term is
        # not itself a routing signal (Part 5's over-routing guard).
        ("Explain what NVIDIA is.", RoutingHint.GENERAL),
        ("What is a transformer?", RoutingHint.GENERAL),
        ("What is OpenAI?", RoutingHint.GENERAL),
        ("What is 2 + 2?", RoutingHint.GENERAL),
        ("Explain what machine learning is.", RoutingHint.GENERAL),
        ("How does Google Search work?", RoutingHint.GENERAL),  # "search" alone must not trigger
    ],
)
def test_classify_hint_buckets(text: str, expected_hint: RoutingHint) -> None:
    router = Router(RaisingLLM())  # must never call the LLM

    assert router.classify_hint(text) is expected_hint


def test_classify_hint_does_not_call_the_llm() -> None:
    router = Router(RaisingLLM())

    router.classify_hint("What time is it?")  # would raise if it touched the LLM
    router.classify_hint("What is the latest AI news?")  # would raise if it touched the LLM


@pytest.mark.parametrize("blank", ["", "   ", None])
def test_classify_hint_blank_input_is_general(blank: str | None) -> None:
    router = Router(RaisingLLM())

    assert router.classify_hint(blank) is RoutingHint.GENERAL


def test_classify_hint_never_executes_a_tool() -> None:
    """Part 8, item 10. classify_hint() takes only a string and returns an
    enum member — it has no tool/registry reference to execute anything
    through, by construction. This test documents and locks in that fact."""
    import inspect

    signature = inspect.signature(Router.classify_hint)
    assert list(signature.parameters) == ["self", "user_message"]

    router = Router(RaisingLLM())
    result = router.classify_hint("Search the web for recent RAG research.")
    assert isinstance(result, RoutingHint)


# ---------------------------------------------------------------------------
# Step 27 — the advisory hint's bare "search for"/"look up"/"look for" was
# REMOVED from `_WEB_INTENT_PHRASES` (kept in Step 8C's original list, and
# left untouched by Step 26, which only narrowed the AUTHORITATIVE route).
# A real-model diagnostic measured a 33% (9/27) unnecessary-web_search rate
# with the hint active on exactly these kinds of local-context/ambiguous
# cases, dropping to 0% (0/27) with the hint disabled on the same cases —
# see app/agent/router.py's Step 27 module-docstring update for the full
# methodology. These tests lock in the resulting, narrower classify_hint()
# behavior and guard against the phrase set silently creeping back.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        # The Step 27 design brief's own ambiguity examples.
        "look up to your role model",
        "search for the bug in this function",
        "search for the value in this array",
        "look up the meaning of this variable",
        "search for the answer in the text above",
        "search for a job",
        "look for my keys",
        "I am looking for a new job.",
        # Technical/local-context searches (Phase 6 category G).
        "Search for the error in this code: def foo(): retun 1",
        "Look up the definition of this variable in my code above.",
        "Find the bug in this function.",
    ],
)
def test_classify_hint_no_longer_flags_bare_search_or_look_up_verbs(text: str) -> None:
    """Step 27: with no recency keyword and no "search the web" phrase, a
    bare search/look-up/find verb is GENERAL now, not TOOL_LIKELY — the
    advisory hint no longer nudges the model toward `web_search` for
    ordinary local-context or ambiguous requests."""
    router = Router(RaisingLLM())  # must never call the LLM

    assert router.classify_hint(text) is RoutingHint.GENERAL


@pytest.mark.parametrize(
    "text",
    [
        # "search the web" itself is explicitly KEPT in `_WEB_INTENT_PHRASES`.
        "Search the web for information about RAG.",
        "search the web for anything",
        # A recency KEYWORD alone is untouched by the Step 27 change —
        # still TOOL_LIKELY even though the message also happens to use a
        # now-removed bare verb.
        "Search for current AI news.",
        "Look up the latest research on LLM agents.",
        "What is the latest AI news?",
        "What are the recent developments in AI?",
        "What is the current price of Bitcoin?",
    ],
)
def test_classify_hint_still_flags_explicit_web_phrase_and_recency_keywords(text: str) -> None:
    """Regression guard: the Step 27 narrowing must not silence the
    remaining, still-justified signals — "search the web" and the three
    recency keywords ("latest", "recent", "current") are unchanged."""
    router = Router(RaisingLLM())

    assert router.classify_hint(text) is RoutingHint.TOOL_LIKELY


@pytest.mark.parametrize(
    "text",
    [
        # Historical/static questions that merely look temporal (Phase 6
        # category E) must stay GENERAL — untouched by Step 27, confirmed
        # here as a regression guard.
        "What year was Python created?",
        "When was the Eiffel Tower built?",
        "Who wrote Romeo and Juliet?",
    ],
)
def test_classify_hint_historical_questions_remain_general(text: str) -> None:
    router = Router(RaisingLLM())

    assert router.classify_hint(text) is RoutingHint.GENERAL


@pytest.mark.parametrize(
    "text",
    [
        # Negated search commands: unaffected by Step 27 (this is the
        # AUTHORITATIVE deterministic route's own, pre-existing negation
        # guard — see app/agent/router.py's `_WEB_SEARCH_NEGATION_MARKERS`,
        # Step 26 — not the advisory hint touched by this milestone).
        # Regression-guarded here per Phase 6's explicit "negation" category.
        "Don't search the web for anything.",
        "No need to look up the weather online.",
    ],
)
def test_deterministic_web_search_route_still_declines_negated_commands(text: str) -> None:
    assert Router.__new__(Router).deterministic_web_search_route(text) is None


def test_deterministic_temporal_route_still_takes_precedence_unaffected_by_step_27() -> None:
    """Regression guard per Phase 6's "time/date precedence" category:
    Step 27 touched only the ADVISORY hint's phrase set, never
    `deterministic_tool_route`/`_deterministic_temporal_check`."""
    router = Router.__new__(Router)

    assert router.deterministic_tool_route("What time is it?") == ("time", None)
    assert router.deterministic_tool_route("What day is 25 December 2026?") == ("date", "25 december 2026")


# ---------------------------------------------------------------------------
# Step 44.2 — short, fragment-style date/time requests now route
# deterministically too (a read-only diagnosis found "today date" matched
# NEITHER the substring-anywhere `temporal_patterns` tuple NOR the
# whole-message `unambiguous_current_time_phrases` tuple, and fell all the
# way through to an unreliable LLM decision call, even in a fresh session
# with no prior conversation to "contaminate" it). All of these resolve to
# the SAME `route="time"` bucket "what is today's date?" already used
# (TimeTool, not DateTool — DateTool stays reserved for a weekday
# calculation against an explicitly specified date).
# ---------------------------------------------------------------------------

NEW_FRAGMENT_DATE_TIME_EXAMPLES = (
    "today date",
    "today's date",
    "current date",
    "date today",
    "time now",
)


@pytest.mark.parametrize("text", NEW_FRAGMENT_DATE_TIME_EXAMPLES)
def test_new_fragment_date_time_phrases_route_deterministically(text: str) -> None:
    router = Router.__new__(Router)
    assert router.deterministic_tool_route(text) == ("time", None)


@pytest.mark.parametrize("text", NEW_FRAGMENT_DATE_TIME_EXAMPLES)
def test_new_fragment_date_time_phrases_never_call_the_llm(text: str) -> None:
    """Same discipline as every other deterministic case: RaisingLLM proves
    no LLM call happens on the path that reaches this decision."""
    router = Router(RaisingLLM())
    decision = router.decide(text)
    assert decision.route == "time"
    assert decision.needs_web is False


# Pre-existing phrasings that must keep working exactly as before this
# milestone (not touched, or in "what is the current date"'s case, moved to
# a different matching mechanism that still matches this exact text).
EXISTING_POSITIVE_EXAMPLES = (
    "what is today's date?",
    "what is the current time?",
    "what date is today",
    "what day is it",
    "what's the time",
    "what is the time",
    "what is the current date",
    "what is the current date?",
)


@pytest.mark.parametrize("text", EXISTING_POSITIVE_EXAMPLES)
def test_existing_date_time_phrasings_still_route_deterministically(text: str) -> None:
    router = Router.__new__(Router)
    assert router.deterministic_tool_route(text) == ("time", None)


# Longer, realistic sentences that merely CONTAIN a date/time-ish fragment
# must NOT deterministically route -- the exact "broad keyword matching"
# failure mode this milestone was explicitly told not to introduce.
# "what is the current date format?" is a regression guard for a
# PRE-EXISTING false positive this milestone found and fixed (unrelated to
# the new fragment phrases above): the old `temporal_patterns` tuple
# matched "what is the current date" as a substring-anywhere, which also
# matched this longer sentence. It was moved to the whole-message-only
# tuple below instead -- see app/agent/router.py.
NEGATIVE_EXAMPLES = (
    "what happened today?",
    "today's weather",
    "today's cricket score",
    "tell me today's news",
    "date format in Python",
    "calculate the date difference",
    "what date was the match?",
    "set a date for my meeting",
    "what is the current date format?",
    "what's the current date for the meeting?",
)


@pytest.mark.parametrize("text", NEGATIVE_EXAMPLES)
def test_sentences_merely_containing_a_date_time_fragment_do_not_route_deterministically(
    text: str,
) -> None:
    router = Router.__new__(Router)
    assert router.deterministic_tool_route(text) is None


def test_conversational_reference_to_today_date_does_not_route_deterministically() -> None:
    """Explicit regression guard: "i asked today date" CONTAINS the new
    "today date" fragment but is not EQUAL to it -- whole-message matching
    must reject it, exactly as it already rejects longer sentences built
    around the pre-existing time phrases above. This is conversational
    context/clarification, not a clean current-date request, and must be
    left to the ordinary LLM decision path like any other ambiguous input."""
    router = Router.__new__(Router)
    assert router.deterministic_tool_route("i asked today date") is None
    assert router.deterministic_tool_route("I asked today date") is None
    assert router.deterministic_tool_route("i asked today date again") is None


def test_new_fragment_phrases_are_case_insensitive_and_tolerate_punctuation() -> None:
    router = Router.__new__(Router)
    assert router.deterministic_tool_route("Today Date") == ("time", None)
    assert router.deterministic_tool_route("TODAY DATE?") == ("time", None)
    assert router.deterministic_tool_route("  today   date  ") == ("time", None)


def test_date_tool_route_for_an_explicit_date_is_unaffected() -> None:
    """The DateTool-selecting route (an explicit calendar date) is a
    completely separate branch of `_deterministic_temporal_check` from the
    fragment phrases this milestone added -- unaffected by this change."""
    router = Router.__new__(Router)
    assert router.deterministic_tool_route("What day was 27 July 2026?") == ("date", "27 july 2026")
