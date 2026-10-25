"""Step 26 — real-model before/after evaluation for deterministic
web-search-command routing.

Reuses the EXISTING routing-evaluation harness (app/agent/evaluation.py:
EvaluationCase, evaluate_all, summarize, compare, format_comparison_report —
the same machinery tests/test_routing_evaluation.py already exercises
against real Ollama) rather than inventing a second evaluation framework —
see the Step 26 investigation notes on reuse. Only the DATASET is new: a
small, explicit set targeting exactly the classes Step 26's design brief
asked for (explicit web-search commands, latest/current questions without a
command, static questions, historical/static questions that merely look
temporal, time/date, adversarial/ambiguous bare verbs, and one multi-part
query), sized deliberately small (15 cases) rather than a large brittle
matrix.

Two informational, integration-marked tests:
1. `test_real_ollama_web_search_command_routing_evaluation` — the new
   dataset run once against the CURRENT production decision-maker
   configuration (temporal + web-search deterministic routing both on).
2. `test_real_ollama_web_search_routing_ab_experiment` — the real A/B
   measurement Phase 6 asks for: BASELINE (temporal routing on, web-search
   routing off — i.e. this codebase's configuration immediately before
   this milestone) vs. EXPERIMENT (both on — this milestone's change),
   against the real local model, reporting accuracy deltas, LLM call
   counts, and latency for each arm. Never asserts a minimum accuracy or
   that the experiment "wins" — llama3.2:3b output is nondeterministic,
   and the deterministic route's own effect is already proven exactly, at
   zero LLM calls, by tests/test_deterministic_web_search_routing.py; this
   evaluation exists to observe real end-to-end behavior, not to gate the
   suite on it.
"""
from __future__ import annotations

import time
import urllib.error
import urllib.request

import pytest

from app.agent.decision_maker import LLMDecisionMaker
from app.agent.evaluation import (
    EvaluationCase,
    compare,
    evaluate_all,
    format_comparison_report,
    format_report,
    summarize,
)
from app.agent.loop import ActionType
from app.agent.tool_registry import ToolRegistry
from app.config import settings
from app.models.llm import LLMClient
from app.tools.date import DateTool
from app.tools.time import TimeTool
from app.tools.web_search import WebSearchTool


def _standard_registry() -> ToolRegistry:
    """The real, production tool set, used read-only for metadata — exactly
    like test_routing_evaluation.py's own helper. evaluate_case() only ever
    calls decision_maker.decide(), never .execute(), so this never touches
    Tavily or the network regardless of what the model decides."""
    registry = ToolRegistry()
    registry.register(WebSearchTool())
    registry.register(TimeTool())
    registry.register(DateTool())
    return registry


class CountingLLMClient:
    """Wraps a real LLMClient, counting calls and accumulating wall-clock
    time spent inside `generate()` — Phase 6 asks for both "number of LLM
    calls" and "latency" per arm. A deterministically-routed case never
    reaches the wrapped client at all (see LLMDecisionMaker._deterministic_
    web_decision), so a lower count here directly measures how many real
    model calls this milestone's change avoids, not merely a proxy for it.
    """

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner
        self.call_count = 0
        self.total_seconds = 0.0

    def generate(self, messages: list[dict[str, str]], *, json_mode: bool = False) -> str:
        self.call_count += 1
        start = time.monotonic()
        try:
            return self._inner.generate(messages, json_mode=json_mode)
        finally:
            self.total_seconds += time.monotonic() - start


# ---------------------------------------------------------------------------
# The Step 26 evaluation dataset — 15 cases across 7 categories, matching
# the design brief's own category list (A explicit commands, B current-info
# questions, D static-but-temporal-looking negatives, E time/date, plus
# adversarial and multi-part additions this milestone's own testing added).
# ---------------------------------------------------------------------------

WEB_SEARCH_COMMAND_EVALUATION_CASES: list[EvaluationCase] = [
    # Category W — explicit web-search COMMANDS naming web/online/internet:
    # this milestone's actual deterministic target.
    EvaluationCase("w1", "Search the web for the latest AI news.", ActionType.TOOL, "W", "web_search"),
    EvaluationCase("w2", "Look up the current price of Bitcoin online.", ActionType.TOOL, "W", "web_search"),
    EvaluationCase("w3", "Find the best pizza recipe online.", ActionType.TOOL, "W", "web_search"),
    EvaluationCase("w4", "Search the internet for python async tutorials.", ActionType.TOOL, "W", "web_search"),
    EvaluationCase("w5", "Please search online for today's top tech headlines.", ActionType.TOOL, "W", "web_search"),
    # Category B — current-information QUESTIONS with no command phrasing:
    # deliberately OUTSIDE this milestone's deterministic route (stays
    # advisory-hint + LLM judgment, unchanged).
    EvaluationCase("b1", "What is the latest AI news?", ActionType.TOOL, "B", "web_search"),
    EvaluationCase("b2", "What is the current price of Bitcoin?", ActionType.TOOL, "B", "web_search"),
    # Category A — static questions -> FINAL, unaffected by either route.
    EvaluationCase("a1", "What is Python?", ActionType.FINAL, "A"),
    EvaluationCase("a2", "Explain machine learning.", ActionType.FINAL, "A"),
    # Category D — historical/static questions that merely LOOK temporal ->
    # FINAL. The Step 26 brief's explicit negative-example category.
    EvaluationCase("d1", "What year was Python created?", ActionType.FINAL, "D"),
    EvaluationCase("d2", "When was the Eiffel Tower built?", ActionType.FINAL, "D"),
    # Category E — existing deterministic time/date coverage (Step 20),
    # included to confirm this milestone leaves it completely unaffected.
    EvaluationCase("e1", "What time is it?", ActionType.TOOL, "E", "time"),
    EvaluationCase("e2", "What day is 25 December 2026?", ActionType.TOOL, "E", "date"),
    # Category X — adversarial/ambiguous bare verbs (no web/online/internet
    # qualifier). Best-guess expectation only (genuinely ambiguous, like the
    # existing dataset's own "trick" category) — graded for visibility, not
    # as a pass/fail requirement; see the module docstring.
    EvaluationCase("x1", "Search for the mistake in my code.", ActionType.FINAL, "X"),
    EvaluationCase("x2", "Look up to your role model.", ActionType.FINAL, "X"),
    # Category M — one multi-part query mixing a deterministic-temporal
    # phrase with an explicit web-search command, to confirm existing
    # temporal precedence (Step 20) still wins even now that a second
    # deterministic route exists (see app/agent/decision_maker.py's
    # `decide()` ordering).
    EvaluationCase(
        "m1", "What time is it, and also search the web for the latest AI news?", ActionType.TOOL, "M", "time"
    ),
]


def _ollama_is_available() -> bool:
    try:
        urllib.request.urlopen(f"{settings.ollama_base_url}/api/tags", timeout=2)
        return True
    except (urllib.error.URLError, OSError):
        return False


@pytest.mark.integration
def test_real_ollama_web_search_command_routing_evaluation() -> None:
    if not _ollama_is_available():
        pytest.skip(f"Ollama is not reachable at {settings.ollama_base_url}; skipping live web-search evaluation.")

    counting_client = CountingLLMClient(
        LLMClient(provider="ollama", model_name=settings.model_name, base_url=settings.ollama_base_url)
    )
    decision_maker = LLMDecisionMaker(
        llm_client=counting_client,
        tool_registry=_standard_registry(),
        deterministic_temporal_routing=True,
        deterministic_web_search_routing=True,
    )

    start = time.monotonic()
    results = evaluate_all(decision_maker, WEB_SEARCH_COMMAND_EVALUATION_CASES)
    wall_seconds = time.monotonic() - start
    summary = summarize(results)

    print("\n" + "=" * 70)
    print("STEP 26 REAL llama3.2:3b WEB-SEARCH-COMMAND EVALUATION")
    print("(informational — not a pass/fail gate)")
    print("=" * 70)
    print(format_report(summary, results))
    print(
        f"llm_calls={counting_client.call_count}/{len(WEB_SEARCH_COMMAND_EVALUATION_CASES)} "
        f"llm_seconds={counting_client.total_seconds:.2f} wall_seconds={wall_seconds:.2f}"
    )
    print("=" * 70)

    # Structural sanity only — never a quality gate on real, nondeterministic
    # model output.
    assert summary.total == len(WEB_SEARCH_COMMAND_EVALUATION_CASES)
    # The 5 Category W cases are matched by the deterministic route with
    # certainty (proven exactly, offline, by test_deterministic_web_search_
    # routing.py) — this IS a structural guarantee, not an observation of
    # model behavior, so it is safe to assert here even though the rest of
    # this test is informational only.
    assert counting_client.call_count <= len(WEB_SEARCH_COMMAND_EVALUATION_CASES) - 5


@pytest.mark.integration
def test_real_ollama_web_search_routing_ab_experiment() -> None:
    """BASELINE = this codebase's configuration immediately before this
    milestone (temporal deterministic routing on, web-search deterministic
    routing off — i.e. `deterministic_web_search_routing=False`, still
    reachable via the advisory hint only). EXPERIMENT = this milestone's
    actual change (both on) — the exact configuration app/main.py now
    ships. Same dataset, same real model, run twice."""
    if not _ollama_is_available():
        pytest.skip(f"Ollama is not reachable at {settings.ollama_base_url}; skipping live A/B experiment.")

    baseline_client = CountingLLMClient(
        LLMClient(provider="ollama", model_name=settings.model_name, base_url=settings.ollama_base_url)
    )
    baseline_decision_maker = LLMDecisionMaker(
        llm_client=baseline_client,
        tool_registry=_standard_registry(),
        deterministic_temporal_routing=True,
        deterministic_web_search_routing=False,
    )
    baseline_start = time.monotonic()
    baseline_results = evaluate_all(baseline_decision_maker, WEB_SEARCH_COMMAND_EVALUATION_CASES)
    baseline_wall_seconds = time.monotonic() - baseline_start
    baseline_summary = summarize(baseline_results)

    experiment_client = CountingLLMClient(
        LLMClient(provider="ollama", model_name=settings.model_name, base_url=settings.ollama_base_url)
    )
    experiment_decision_maker = LLMDecisionMaker(
        llm_client=experiment_client,
        tool_registry=_standard_registry(),
        deterministic_temporal_routing=True,
        deterministic_web_search_routing=True,
    )
    experiment_start = time.monotonic()
    experiment_results = evaluate_all(experiment_decision_maker, WEB_SEARCH_COMMAND_EVALUATION_CASES)
    experiment_wall_seconds = time.monotonic() - experiment_start
    experiment_summary = summarize(experiment_results)

    comparison = compare(baseline_summary, experiment_summary)

    print("\n" + "=" * 70)
    print("STEP 26 REAL llama3.2:3b A/B EXPERIMENT: web-search-command routing OFF vs. ON")
    print("(informational only — do not assume the experiment wins on every metric)")
    print("=" * 70)
    print("--- BASELINE (deterministic_web_search_routing=False) ---")
    print(format_report(baseline_summary, baseline_results))
    print(
        f"llm_calls={baseline_client.call_count}/{len(WEB_SEARCH_COMMAND_EVALUATION_CASES)} "
        f"llm_seconds={baseline_client.total_seconds:.2f} wall_seconds={baseline_wall_seconds:.2f}"
    )
    print("--- EXPERIMENT (deterministic_web_search_routing=True) ---")
    print(format_report(experiment_summary, experiment_results))
    print(
        f"llm_calls={experiment_client.call_count}/{len(WEB_SEARCH_COMMAND_EVALUATION_CASES)} "
        f"llm_seconds={experiment_client.total_seconds:.2f} wall_seconds={experiment_wall_seconds:.2f}"
    )
    print("--- COMPARISON ---")
    print(format_comparison_report(comparison))
    print("=" * 70)

    # Structural sanity only.
    assert baseline_summary.total == len(WEB_SEARCH_COMMAND_EVALUATION_CASES)
    assert experiment_summary.total == len(WEB_SEARCH_COMMAND_EVALUATION_CASES)
    # The experiment arm calls the LLM strictly less often than the
    # baseline arm — the 5 Category W cases never reach the model in the
    # experiment (deterministic, proven offline), while the baseline arm
    # always consults it. This is a structural guarantee of the dispatch
    # logic, not an observation of nondeterministic model behavior.
    assert experiment_client.call_count < baseline_client.call_count
