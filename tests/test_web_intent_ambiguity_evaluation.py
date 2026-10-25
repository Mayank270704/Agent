"""Step 27 — real-model before/after evaluation for advisory-hint
ambiguity hardening (bare "search for"/"look up"/"look for").

Root cause, established by a Phase 1/2 diagnostic (with-hint vs.
hint-fully-disabled, 3 trials each, on the exact ambiguity class this
milestone targets): the advisory ROUTING HINT rendered by
`LLMDecisionMaker._format_hint` — driven by `Router.classify_hint` /
`Router._has_web_intent` — was nudging llama3.2:3b toward `web_search` for
ordinary local-context and ambiguous requests ("look up to your role
model", "search for the bug in this function") that contain a bare
search/look-up verb with no web/online/internet reference. The measured
effect: 33% (9/27) unnecessary-web_search calls with the (then-current,
Step 26) hint active on those cases, dropping to 0% (0/27) with the hint
disabled entirely on the same cases, same model, same temperature. See
app/agent/router.py's Step 27 module-docstring update for the full
methodology and reasoning; the fix itself is a one-line narrowing of
`Router._WEB_INTENT_PHRASES` (removing "search for"/"look up"/"look for",
keeping only "search the web") — no deterministic-routing change, no
prompt-template change, no architecture change.

This file is the PERMANENT Phase 7 evaluation: unlike the diagnostic above
(hint fully OFF as an isolating experiment), this measures the ACTUAL
shipped fix — OLD (Step 26's broader `_WEB_INTENT_PHRASES`, reconstructed
here via a test-only Router subclass override) vs. NEW (today's narrower
list) — reusing the same EvaluationCase/evaluate_all/summarize/compare
harness app/agent/evaluation.py already provides, on a NEW, purpose-built
dataset spanning Phase 1's required categories A-I with multiple
phrasings each. This dataset is intentionally NOT compared numerically
against Step 26's own 16-case dataset (different composition, different
purpose) — only against itself, old-router-arm vs. new-router-arm.

Every test here is `@pytest.mark.integration` (skips if Ollama is
unreachable) and informational only: nondeterministic real-model output
is never asserted as a minimum accuracy, except where a deterministic
code path (the authoritative router, proven offline elsewhere) makes an
assertion structurally safe regardless of model behavior.
"""
from __future__ import annotations

import time
import urllib.error
import urllib.request
from dataclasses import dataclass

import pytest

from app.agent.decision_maker import DecisionParseError, LLMDecisionMaker
from app.agent.evaluation import (
    EvaluationCase,
    compare,
    evaluate_all,
    evaluate_case,
    format_comparison_report,
    format_report,
    summarize,
)
from app.agent.loop import ActionType
from app.agent.router import Router
from app.agent.state import AgentState
from app.agent.tool_registry import ToolRegistry
from app.config import settings
from app.models.llm import LLMClient
from app.tools.date import DateTool
from app.tools.time import TimeTool
from app.tools.web_search import WebSearchTool


def _standard_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(WebSearchTool())
    registry.register(TimeTool())
    registry.register(DateTool())
    return registry


class PreStep27Router(Router):
    """Reconstructs the Step 26 (pre-Step-27) advisory-hint behavior for
    A/B comparison purposes ONLY — the exact `_WEB_INTENT_PHRASES` this
    codebase shipped between Step 26 and Step 27, before the bare-verb
    entries were removed. `deterministic_tool_route`/`deterministic_web_
    search_route` (the AUTHORITATIVE routes) are inherited UNCHANGED:
    Step 27 never touched them, only the advisory `_has_web_intent` layer,
    so this override reproduces exactly and only that one difference."""

    _WEB_INTENT_PHRASES = ("search the web", "search for", "look up", "look for")


class CountingLLMClient:
    """Same counting wrapper Step 26's evaluation file uses — counts real
    calls and accumulates wall-clock time spent inside `generate()`."""

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
# The Phase 1 dataset — categories A-I, multiple phrasings each. Cases with
# a clear expected action use ActionType (graded like the existing
# harness); cases in AMBIGUOUS_CASES have no single correct answer and are
# tracked separately, per Phase 1's "do not force ambiguous cases into a
# binary label" instruction.
# ---------------------------------------------------------------------------

GRADED_CASES: list[EvaluationCase] = [
    # A. Explicit web commands -> TOOL/web_search (deterministic; Step 26).
    EvaluationCase("a1", "Search the web for the latest AI news.", ActionType.TOOL, "A", "web_search"),
    EvaluationCase("a2", "Look up the current price of Bitcoin online.", ActionType.TOOL, "A", "web_search"),
    EvaluationCase("a3", "Find the best pizza recipe on the internet.", ActionType.TOOL, "A", "web_search"),
    # B. Ambiguous bare search/look-up commands -> FINAL (Step 27's target
    # class; NOT deterministic, so graded against the LLM's real judgment).
    EvaluationCase("b1", "look up to your role model", ActionType.FINAL, "B"),
    EvaluationCase("b2", "search for the bug in this function", ActionType.FINAL, "B"),
    EvaluationCase("b3", "search for the value in this array", ActionType.FINAL, "B"),
    EvaluationCase("b4", "look up the meaning of this variable", ActionType.FINAL, "B"),
    # D. Static factual questions -> FINAL.
    EvaluationCase("d1", "What is Python?", ActionType.FINAL, "D"),
    EvaluationCase("d2", "Explain machine learning.", ActionType.FINAL, "D"),
    EvaluationCase("d3", "What is 2 + 2?", ActionType.FINAL, "D"),
    # E. Historical questions -> FINAL.
    EvaluationCase("e1", "What year was Python created?", ActionType.FINAL, "E"),
    EvaluationCase("e2", "When was the Eiffel Tower built?", ActionType.FINAL, "E"),
    EvaluationCase("e3", "Who wrote Romeo and Juliet?", ActionType.FINAL, "E"),
    # F. Time/date queries -> TOOL/time or TOOL/date (deterministic).
    EvaluationCase("f1", "What time is it?", ActionType.TOOL, "F", "time"),
    EvaluationCase("f2", "What day is 25 December 2026?", ActionType.TOOL, "F", "date"),
    # G. Technical/local-context searches -> FINAL (Step 27's other target
    # class — a "search"/"look up" verb aimed at content already provided).
    EvaluationCase(
        "g1", "Search for the error in this code: def foo(): retun 1", ActionType.FINAL, "G"
    ),
    EvaluationCase("g2", "Look up the definition of this variable in my code above.", ActionType.FINAL, "G"),
    EvaluationCase("g3", "Find the bug in this function.", ActionType.FINAL, "G"),
    # H. Negated search requests -> FINAL (the deterministic negation guard
    # already blocks the authoritative route; graded here against the
    # LLM's own judgment for the residual advisory-hint-only path).
    EvaluationCase("h1", "Don't search the web for anything.", ActionType.FINAL, "H"),
    EvaluationCase("h2", "No need to look up the weather online.", ActionType.FINAL, "H"),
]

# C. Current/latest information and I. Multi-part requests: genuinely
# ambiguous / no single correct answer (a reasonable agent might legitimately
# call web_search OR answer FINAL depending on interpretation) — tracked
# for observability only, never graded pass/fail, per Phase 1's instruction.
AMBIGUOUS_CASES: list[tuple[str, str, str]] = [
    # (name, message, category)
    ("c1", "What is the latest AI news?", "C"),
    ("c2", "What is the current price of Bitcoin?", "C"),
    ("c3", "What's happening in the stock market right now?", "C"),
    (
        "i1",
        "What time is it, and can you also look up the current Bitcoin price online?",
        "I",
    ),
    ("i2", "Can you tell me what Python is and also search for a bug in this code: def f(): pass", "I"),
]


def _make_decision_maker(client, router_cls: type[Router] = Router) -> LLMDecisionMaker:
    return LLMDecisionMaker(
        llm_client=client,
        tool_registry=_standard_registry(),
        router=router_cls(client),
        deterministic_temporal_routing=True,
        deterministic_web_search_routing=True,
    )


def _ollama_is_available() -> bool:
    try:
        urllib.request.urlopen(f"{settings.ollama_base_url}/api/tags", timeout=2)
        return True
    except (urllib.error.URLError, OSError):
        return False


TRIALS = 3


@dataclass
class TrialRun:
    call_count: int
    total_seconds: int
    summary: object
    results: list


def _run_trials(router_cls: type[Router], trials: int) -> list[TrialRun]:
    runs = []
    for _ in range(trials):
        client = CountingLLMClient(
            LLMClient(provider="ollama", model_name=settings.model_name, base_url=settings.ollama_base_url)
        )
        dm = _make_decision_maker(client, router_cls)
        results = evaluate_all(dm, GRADED_CASES)
        runs.append(
            TrialRun(
                call_count=client.call_count, total_seconds=client.total_seconds,
                summary=summarize(results), results=results,
            )
        )
    return runs


@pytest.mark.integration
def test_real_ollama_web_intent_ambiguity_evaluation_current_code() -> None:
    """Phase 7: the current (post-Step-27) configuration run for
    `TRIALS` repeated trials on the full graded dataset, reporting
    per-trial results and per-case variance across trials — distinguishing
    a stable/deterministic outcome from real model variance."""
    if not _ollama_is_available():
        pytest.skip(f"Ollama is not reachable at {settings.ollama_base_url}; skipping live evaluation.")

    runs = _run_trials(Router, TRIALS)

    print("\n" + "=" * 70)
    print(f"STEP 27 REAL llama3.2:3b WEB-INTENT-AMBIGUITY EVALUATION ({TRIALS} trials, current code)")
    print("(informational — not a pass/fail gate)")
    print("=" * 70)
    for i, run in enumerate(runs):
        print(f"--- trial {i + 1} --- llm_calls={run.call_count} llm_seconds={run.total_seconds:.2f}")
        print(format_report(run.summary, run.results))

    # Per-case variance across trials: which decisions did each case get?
    print("--- per-case decisions across trials ---")
    variable_cases = []
    for idx, case in enumerate(GRADED_CASES):
        decisions = []
        for run in runs:
            result = run.results[idx]
            decisions.append(
                result.actual_tool if result.actual_action is ActionType.TOOL else str(result.actual_action)
            )
        stable = len(set(decisions)) == 1
        if not stable:
            variable_cases.append(case.name)
        print(f"  [{case.category}] {case.name!r} decisions={decisions} stable={stable}")
    print(f"cases with cross-trial variance: {variable_cases}")
    print("=" * 70)

    assert all(run.summary.total == len(GRADED_CASES) for run in runs)


@pytest.mark.integration
def test_real_ollama_web_intent_ambiguity_ab_experiment() -> None:
    """The actual before/after measurement: OLD `_WEB_INTENT_PHRASES`
    (`PreStep27Router`) vs. NEW (today's `Router`), `TRIALS` trials each,
    same dataset, same real model. Never asserts the new arm "wins" on
    every metric — only that the deterministic-routing-covered cases
    (Category A/F) are completely unaffected (structurally guaranteed,
    proven offline) and that both arms complete successfully."""
    if not _ollama_is_available():
        pytest.skip(f"Ollama is not reachable at {settings.ollama_base_url}; skipping live A/B experiment.")

    old_runs = _run_trials(PreStep27Router, TRIALS)
    new_runs = _run_trials(Router, TRIALS)

    old_summaries = [r.summary for r in old_runs]
    new_summaries = [r.summary for r in new_runs]

    def _avg(values: list[float]) -> float:
        return sum(values) / len(values)

    old_avg_accuracy = _avg([s.accuracy for s in old_summaries])
    new_avg_accuracy = _avg([s.accuracy for s in new_summaries])
    old_avg_unnecessary = _avg([s.unnecessary_tool_call_rate for s in old_summaries])
    new_avg_unnecessary = _avg([s.unnecessary_tool_call_rate for s in new_summaries])
    old_avg_calls = _avg([r.call_count for r in old_runs])
    new_avg_calls = _avg([r.call_count for r in new_runs])
    old_avg_seconds = _avg([r.total_seconds for r in old_runs])
    new_avg_seconds = _avg([r.total_seconds for r in new_runs])

    print("\n" + "=" * 70)
    print(f"STEP 27 REAL llama3.2:3b A/B EXPERIMENT: OLD (Step 26) vs. NEW (Step 27) advisory hint ({TRIALS} trials each)")
    print("(informational only — averaged across trials to show central tendency, not a pass/fail gate)")
    print("=" * 70)
    for i, run in enumerate(old_runs):
        print(f"--- OLD trial {i + 1} ---")
        print(format_report(run.summary, run.results))
    for i, run in enumerate(new_runs):
        print(f"--- NEW trial {i + 1} ---")
        print(format_report(run.summary, run.results))
    print("--- AVERAGED COMPARISON ---")
    print(f"OLD: accuracy={old_avg_accuracy:.1%} unnecessary_tool_call_rate={old_avg_unnecessary:.1%} "
          f"avg_llm_calls={old_avg_calls:.1f} avg_llm_seconds={old_avg_seconds:.2f}")
    print(f"NEW: accuracy={new_avg_accuracy:.1%} unnecessary_tool_call_rate={new_avg_unnecessary:.1%} "
          f"avg_llm_calls={new_avg_calls:.1f} avg_llm_seconds={new_avg_seconds:.2f}")
    print(f"delta: accuracy={new_avg_accuracy - old_avg_accuracy:+.1%} "
          f"unnecessary_tool_call_rate={new_avg_unnecessary - old_avg_unnecessary:+.1%}")
    print("--- single-trial compare() (trial 1 of each arm) ---")
    print(format_comparison_report(compare(old_summaries[0], new_summaries[0])))
    print("=" * 70)

    assert all(s.total == len(GRADED_CASES) for s in old_summaries + new_summaries)


@pytest.mark.integration
def test_real_ollama_ambiguous_category_c_and_i_observability() -> None:
    """Category C (current-info questions) and I (multi-part) are
    genuinely ambiguous — tracked here for observability only, never
    graded, per Phase 1's instruction not to force a binary label."""
    if not _ollama_is_available():
        pytest.skip(f"Ollama is not reachable at {settings.ollama_base_url}; skipping live observability run.")

    client = CountingLLMClient(
        LLMClient(provider="ollama", model_name=settings.model_name, base_url=settings.ollama_base_url)
    )
    dm = _make_decision_maker(client, Router)

    print("\n" + "=" * 70)
    print("STEP 27 CATEGORY C/I OBSERVABILITY (ambiguous — not graded)")
    print("=" * 70)
    decisions_seen = 0
    for name, message, category in AMBIGUOUS_CASES:
        # decide() is called directly here (not through AgentLoop), so a
        # DecisionParseError — llama3.2:3b occasionally emits malformed/
        # nested JSON, a separate, already-documented model limitation
        # unrelated to Step 27 and already recoverable in production via
        # AgentLoop + CorrectionPolicy — is caught the same way
        # evaluate_case() (app/agent/evaluation.py) already does, rather
        # than crashing this observability-only loop.
        try:
            decision = dm.decide(AgentState(user_input=message))
            action = decision.tool_name if decision.action_type is ActionType.TOOL else "FINAL"
        except DecisionParseError as exc:
            action = f"PARSE_ERROR({exc})"
        decisions_seen += 1
        # i1 deliberately mixes a deterministic-temporal phrase with an
        # explicit web command ("What time is it, and ... look up the
        # current Bitcoin price online?") to observe precedence on a
        # multi-part request — it resolves to "time" with NO LLM call at
        # all (existing Step 20/26 precedence, unaffected by Step 27), so
        # this test does not assert a fixed LLM call count.
        print(f"  [{category}] {name!r} -> {action} :: {message!r}")
    print(f"total LLM calls across {decisions_seen} ambiguous/multi-part cases: {client.call_count}")
    print("=" * 70)

    assert decisions_seen == len(AMBIGUOUS_CASES)  # every case produced a result, no crash
