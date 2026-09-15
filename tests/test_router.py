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
