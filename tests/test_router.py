from __future__ import annotations

import json

import pytest

from app.agent.router import Router


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
