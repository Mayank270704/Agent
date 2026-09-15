"""Tests for the routing-evaluation harness (app/agent/evaluation.py).

Three groups:

1. Deterministic tests (Part 3/5/8, Step 8B) using FakeLLM — these verify
   the EVALUATION FRAMEWORK's classification logic is correct, not whether
   a fake LLM "can think." No network, no Ollama, no Tavily.
2. Deterministic A/B mechanism tests (Part 7, Step 8C) — verify the
   comparison math and the NoHintRouter baseline-construction mechanism.
3. Two informational, integration-marked real-Ollama evaluations (Part 6
   and Part 7/11) — never required for the normal suite, never assert a
   minimum accuracy (llama3.2:3b output is nondeterministic), never execute
   web_search (so they can never call Tavily — only the DECISION is graded).
"""
from __future__ import annotations

import json

import pytest

from app.agent.decision_maker import LLMDecisionMaker
from app.agent.evaluation import (
    CaseResult,
    CaseVerdict,
    ComparisonSummary,
    EvaluationCase,
    EvaluationSummary,
    NoHintRouter,
    ROUTING_EVALUATION_CASES,
    compare,
    evaluate_all,
    evaluate_case,
    format_comparison_report,
    format_report,
    summarize,
)
from app.agent.loop import ActionType
from app.agent.router import RoutingHint
from app.agent.tool_registry import ToolRegistry
from app.config import settings
from app.models.llm import LLMClient
from app.tools.date import DateTool
from app.tools.time import TimeTool
from app.tools.web_search import WebSearchTool


class FakeLLM:
    """Replays a fixed sequence of raw responses, one per generate() call."""

    def __init__(self, responses: list[str]):
        self.responses = iter(responses)

    def generate(self, messages: list[dict[str, str]], *, json_mode: bool = False) -> str:
        return next(self.responses)


def _standard_registry() -> ToolRegistry:
    """The real, production tool set, used read-only for metadata (name,
    description, input_schema, output_description). evaluate_case() only
    ever calls decision_maker.decide() — never .execute() — so this never
    touches Tavily or the network, regardless of what the model decides."""
    registry = ToolRegistry()
    registry.register(WebSearchTool())
    registry.register(TimeTool())
    registry.register(DateTool())
    return registry


def _final_json(answer: str = "an answer") -> str:
    return json.dumps({"action_type": "final", "final_answer": answer})


def _tool_json(name: str, tool_input: str | None = "some input") -> str:
    return json.dumps({"action_type": "tool", "tool_name": name, "tool_input": tool_input})


def _decision_maker(response: str) -> LLMDecisionMaker:
    return LLMDecisionMaker(llm_client=FakeLLM([response]), tool_registry=_standard_registry())


CASE_FINAL = EvaluationCase("case_final", "What is Python?", ActionType.FINAL, "A")
CASE_WEB = EvaluationCase("case_web", "What is the latest AI news?", ActionType.TOOL, "B", "web_search")


# ---------------------------------------------------------------------------
# EvaluationCase invariants.
# ---------------------------------------------------------------------------

def test_evaluation_case_rejects_tool_expectation_without_tool_name() -> None:
    with pytest.raises(ValueError):
        EvaluationCase("bad", "x", ActionType.TOOL, "X")


def test_evaluation_case_rejects_final_expectation_with_a_tool_name() -> None:
    with pytest.raises(ValueError):
        EvaluationCase("bad", "x", ActionType.FINAL, "X", expected_tool="web_search")


# ---------------------------------------------------------------------------
# Part 3 / Part 5: verify the evaluation FRAMEWORK's classification logic.
# Each test feeds a specific, sometimes deliberately wrong, scripted
# response and checks the verdict — never treating a merely-valid TOOL
# decision as automatically correct.
# ---------------------------------------------------------------------------

def test_correct_final_decision_is_classified_correct() -> None:
    result = evaluate_case(_decision_maker(_final_json()), CASE_FINAL)

    assert result.verdict is CaseVerdict.CORRECT
    assert result.is_correct is True
    assert result.actual_action is ActionType.FINAL


def test_correct_tool_decision_is_classified_correct() -> None:
    result = evaluate_case(_decision_maker(_tool_json("web_search", "latest AI news")), CASE_WEB)

    assert result.verdict is CaseVerdict.CORRECT
    assert result.actual_action is ActionType.TOOL
    assert result.actual_tool == "web_search"


def test_expected_final_but_got_tool_is_unnecessary_tool_call() -> None:
    """A syntactically valid TOOL decision is NOT automatically correct when
    FINAL was actually expected."""
    result = evaluate_case(_decision_maker(_tool_json("web_search", "python")), CASE_FINAL)

    assert result.verdict is CaseVerdict.UNNECESSARY_TOOL_CALL
    assert result.is_correct is False


def test_expected_tool_but_got_final_is_missed_tool_call() -> None:
    result = evaluate_case(_decision_maker(_final_json("I don't know.")), CASE_WEB)

    assert result.verdict is CaseVerdict.MISSED_TOOL_CALL


def test_expected_tool_but_got_a_different_registered_tool_is_wrong_tool() -> None:
    """Expected web_search, predicted date -> wrong, not "close enough"."""
    result = evaluate_case(_decision_maker(_tool_json("date", "some date")), CASE_WEB)

    assert result.verdict is CaseVerdict.WRONG_TOOL
    assert result.actual_tool == "date"


def test_expected_tool_but_got_an_unregistered_tool_is_invalid_tool() -> None:
    """Expected web_search, predicted "calculator" -> both wrong AND
    invalid; must be classified as invalid_tool specifically, distinct from
    a generic parse error or a wrong-but-registered tool."""
    result = evaluate_case(_decision_maker(_tool_json("calculator", "2 + 2")), CASE_WEB)

    assert result.verdict is CaseVerdict.INVALID_TOOL
    assert result.actual_action is None  # no AgentDecision was ever produced


def test_malformed_json_is_classified_as_parse_error_not_invalid_tool() -> None:
    result = evaluate_case(_decision_maker("not valid json {"), CASE_FINAL)

    assert result.verdict is CaseVerdict.PARSE_ERROR


# ---------------------------------------------------------------------------
# Part 4: the five summary metrics, checked against hand-built CaseResults
# so each formula is verified independently of decide()/parsing.
# ---------------------------------------------------------------------------

def _result(case: EvaluationCase, verdict: CaseVerdict, actual_action=None, actual_tool=None) -> CaseResult:
    return CaseResult(case=case, verdict=verdict, actual_action=actual_action, actual_tool=actual_tool)


def test_summarize_computes_overall_accuracy() -> None:
    results = [
        _result(CASE_FINAL, CaseVerdict.CORRECT, ActionType.FINAL),
        _result(CASE_WEB, CaseVerdict.WRONG_TOOL, ActionType.TOOL, "date"),
    ]

    summary = summarize(results)

    assert summary.total == 2
    assert summary.correct == 1
    assert summary.accuracy == 0.5


def test_summarize_computes_tool_selection_accuracy() -> None:
    web1 = EvaluationCase("w1", "x", ActionType.TOOL, "B", "web_search")
    web2 = EvaluationCase("w2", "y", ActionType.TOOL, "B", "web_search")
    results = [
        _result(web1, CaseVerdict.CORRECT, ActionType.TOOL, "web_search"),
        _result(web2, CaseVerdict.WRONG_TOOL, ActionType.TOOL, "date"),
    ]

    summary = summarize(results)

    assert summary.tool_selection_accuracy == 0.5


def test_summarize_computes_unnecessary_tool_call_rate() -> None:
    final1 = EvaluationCase("f1", "x", ActionType.FINAL, "A")
    final2 = EvaluationCase("f2", "y", ActionType.FINAL, "A")
    results = [
        _result(final1, CaseVerdict.CORRECT, ActionType.FINAL),
        _result(final2, CaseVerdict.UNNECESSARY_TOOL_CALL, ActionType.TOOL, "web_search"),
    ]

    summary = summarize(results)

    assert summary.unnecessary_tool_call_rate == 0.5


def test_summarize_computes_missed_tool_call_rate() -> None:
    web1 = EvaluationCase("w1", "x", ActionType.TOOL, "B", "web_search")
    web2 = EvaluationCase("w2", "y", ActionType.TOOL, "B", "web_search")
    results = [
        _result(web1, CaseVerdict.CORRECT, ActionType.TOOL, "web_search"),
        _result(web2, CaseVerdict.MISSED_TOOL_CALL, ActionType.FINAL),
    ]

    summary = summarize(results)

    assert summary.missed_tool_call_rate == 0.5


def test_summarize_computes_invalid_tool_rate() -> None:
    web1 = EvaluationCase("w1", "x", ActionType.TOOL, "B", "web_search")
    results = [
        _result(web1, CaseVerdict.INVALID_TOOL),
        _result(CASE_FINAL, CaseVerdict.CORRECT, ActionType.FINAL),
        _result(CASE_FINAL, CaseVerdict.CORRECT, ActionType.FINAL),
        _result(CASE_FINAL, CaseVerdict.CORRECT, ActionType.FINAL),
    ]

    summary = summarize(results)

    assert summary.invalid_tool_rate == 0.25


def test_summarize_on_empty_results_has_zero_rates_not_errors() -> None:
    summary = summarize([])

    assert summary.total == 0
    assert summary.accuracy == 0.0
    assert summary.tool_selection_accuracy == 0.0
    assert summary.unnecessary_tool_call_rate == 0.0
    assert summary.missed_tool_call_rate == 0.0
    assert summary.invalid_tool_rate == 0.0


def test_format_report_produces_readable_text_without_crashing() -> None:
    results = evaluate_all(_decision_maker(_final_json()), [CASE_FINAL])
    summary = summarize(results)

    report = format_report(summary, results)

    assert isinstance(report, str)
    assert "accuracy=" in report


# ---------------------------------------------------------------------------
# Part 1 dataset sanity.
# ---------------------------------------------------------------------------

def test_dataset_has_expected_size_and_categories() -> None:
    assert 25 <= len(ROUTING_EVALUATION_CASES) <= 40
    assert {c.category for c in ROUTING_EVALUATION_CASES} == {"A", "B", "C", "D", "E", "F", "G"}


def test_dataset_case_names_are_unique() -> None:
    names = [c.name for c in ROUTING_EVALUATION_CASES]
    assert len(names) == len(set(names))


def _expected_response_json(case: EvaluationCase) -> str:
    if case.expected_action is ActionType.FINAL:
        return _final_json("a correct answer")
    return _tool_json(case.expected_tool, "some appropriate input")


def test_full_dataset_with_perfectly_scripted_llm_scores_100_percent() -> None:
    """End-to-end wiring check: if the model always returns exactly what's
    expected, accuracy must be 100%. This also catches a malformed dataset
    entry (e.g. an expected_tool that doesn't match a registered tool)."""
    llm = FakeLLM([_expected_response_json(case) for case in ROUTING_EVALUATION_CASES])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=_standard_registry())

    results = evaluate_all(decision_maker, ROUTING_EVALUATION_CASES)
    summary = summarize(results)

    assert summary.total == len(ROUTING_EVALUATION_CASES)
    assert summary.accuracy == 1.0
    assert summary.invalid_tool_rate == 0.0
    assert summary.unnecessary_tool_call_rate == 0.0
    assert summary.missed_tool_call_rate == 0.0


# ---------------------------------------------------------------------------
# Part 8: explicit coverage of the named failure modes.
# ---------------------------------------------------------------------------

def test_part8_1_two_plus_two_does_not_need_a_tool() -> None:
    case = next(c for c in ROUTING_EVALUATION_CASES if c.user_input == "What is 2 + 2?")

    result = evaluate_case(_decision_maker(_final_json("4")), case)

    assert result.verdict is CaseVerdict.CORRECT
    assert result.actual_action is ActionType.FINAL


def test_part8_2_latest_ai_news_uses_web_search() -> None:
    case = next(c for c in ROUTING_EVALUATION_CASES if c.user_input == "What is the latest AI news?")

    result = evaluate_case(_decision_maker(_tool_json("web_search", "latest AI news")), case)

    assert result.verdict is CaseVerdict.CORRECT
    assert result.actual_tool == "web_search"


def test_part8_3_christmas_2026_uses_date_tool() -> None:
    case = next(c for c in ROUTING_EVALUATION_CASES if c.user_input == "What day is 25 December 2026?")

    result = evaluate_case(_decision_maker(_tool_json("date", "25 December 2026")), case)

    assert result.verdict is CaseVerdict.CORRECT
    assert result.actual_tool == "date"


def test_part8_4_what_time_is_it_uses_time_tool() -> None:
    case = next(c for c in ROUTING_EVALUATION_CASES if c.user_input == "What time is it?")

    result = evaluate_case(_decision_maker(_tool_json("time", None)), case)

    assert result.verdict is CaseVerdict.CORRECT
    assert result.actual_tool == "time"


def test_part8_5_explain_transformer_is_final() -> None:
    case = EvaluationCase("transformer_explain", "Explain what a transformer is.", ActionType.FINAL, "A")

    result = evaluate_case(_decision_maker(_final_json("A transformer is a neural network architecture.")), case)

    assert result.verdict is CaseVerdict.CORRECT


def test_part8_6_search_rag_developments_uses_web_search() -> None:
    case = EvaluationCase(
        "rag_search", "Search the web for the latest developments in RAG.", ActionType.TOOL, "F", "web_search"
    )

    result = evaluate_case(_decision_maker(_tool_json("web_search", "latest developments in RAG")), case)

    assert result.verdict is CaseVerdict.CORRECT


def test_part8_7_calculator_selection_is_invalid_tool() -> None:
    case = next(c for c in ROUTING_EVALUATION_CASES if c.user_input == "What is 2 + 2?")

    result = evaluate_case(_decision_maker(_tool_json("calculator", "2 + 2")), case)

    assert result.verdict is CaseVerdict.INVALID_TOOL


# ---------------------------------------------------------------------------
# Part 7 (Step 8C): A/B comparison mechanism. NoHintRouter/compare() are
# tested deterministically here; the real behavioral effect of the hint can
# only be measured against the real model (see the integration test below).
# ---------------------------------------------------------------------------

def test_no_hint_router_always_returns_general() -> None:
    router = NoHintRouter()

    assert router.classify_hint("What time is it?") is RoutingHint.GENERAL
    assert router.classify_hint("What is the latest AI news?") is RoutingHint.GENERAL
    assert router.classify_hint("") is RoutingHint.GENERAL


def _summary(**overrides: object) -> EvaluationSummary:
    defaults: dict[str, object] = dict(
        total=27,
        correct=17,
        accuracy=0.63,
        tool_selection_accuracy=0.412,
        unnecessary_tool_call_rate=0.0,
        missed_tool_call_rate=0.529,
        invalid_tool_rate=0.0,
        by_verdict={},
    )
    defaults.update(overrides)
    return EvaluationSummary(**defaults)  # type: ignore[arg-type]


def test_compare_computes_accuracy_delta() -> None:
    comparison = compare(_summary(accuracy=0.5), _summary(accuracy=0.7))

    assert comparison.accuracy_delta == pytest.approx(0.2)


def test_compare_computes_all_five_deltas() -> None:
    baseline = _summary(
        accuracy=0.6, tool_selection_accuracy=0.4, unnecessary_tool_call_rate=0.0,
        missed_tool_call_rate=0.5, invalid_tool_rate=0.0,
    )
    experiment = _summary(
        accuracy=0.7, tool_selection_accuracy=0.6, unnecessary_tool_call_rate=0.1,
        missed_tool_call_rate=0.2, invalid_tool_rate=0.05,
    )

    comparison = compare(baseline, experiment)

    assert comparison.accuracy_delta == pytest.approx(0.1)
    assert comparison.tool_selection_accuracy_delta == pytest.approx(0.2)
    assert comparison.unnecessary_tool_call_rate_delta == pytest.approx(0.1)
    assert comparison.missed_tool_call_rate_delta == pytest.approx(-0.3)
    assert comparison.invalid_tool_rate_delta == pytest.approx(0.05)


def test_format_comparison_report_produces_readable_text() -> None:
    comparison = compare(_summary(), _summary(accuracy=0.7))

    report = format_comparison_report(comparison)

    assert isinstance(report, str)
    assert "baseline:" in report
    assert "experiment:" in report
    assert "delta:" in report


def test_ab_mechanism_baseline_and_experiment_run_independently() -> None:
    """Deterministic sanity check of the A/B WIRING itself — NOT of real
    model behavior. A FakeLLM doesn't read the routing hint to decide
    anything, so both arms necessarily score identically here; only the real
    Ollama test below can measure an actual behavioral difference."""
    responses = [_expected_response_json(case) for case in ROUTING_EVALUATION_CASES]

    baseline_llm = FakeLLM(list(responses))
    baseline_decision_maker = LLMDecisionMaker(
        llm_client=baseline_llm, tool_registry=_standard_registry(), router=NoHintRouter()
    )
    baseline_summary = summarize(evaluate_all(baseline_decision_maker, ROUTING_EVALUATION_CASES))

    experiment_llm = FakeLLM(list(responses))
    experiment_decision_maker = LLMDecisionMaker(llm_client=experiment_llm, tool_registry=_standard_registry())
    experiment_summary = summarize(evaluate_all(experiment_decision_maker, ROUTING_EVALUATION_CASES))

    comparison = compare(baseline_summary, experiment_summary)

    assert isinstance(comparison, ComparisonSummary)
    assert baseline_summary.accuracy == 1.0
    assert experiment_summary.accuracy == 1.0
    assert comparison.accuracy_delta == 0.0


# ---------------------------------------------------------------------------
# Part 6: real llama3.2:3b evaluation. Informational only — never asserts a
# minimum accuracy (a single run of a nondeterministic 3B model proves
# nothing about long-run reliability), never required for the normal suite,
# and never executes web_search (so it can never call Tavily).
# ---------------------------------------------------------------------------

def _ollama_is_available() -> bool:
    import urllib.error
    import urllib.request

    try:
        urllib.request.urlopen(f"{settings.ollama_base_url}/api/tags", timeout=2)
        return True
    except (urllib.error.URLError, OSError):
        return False


@pytest.mark.integration
def test_real_ollama_routing_evaluation() -> None:
    if not _ollama_is_available():
        pytest.skip(f"Ollama is not reachable at {settings.ollama_base_url}; skipping live routing evaluation.")

    llm_client = LLMClient(provider="ollama", model_name=settings.model_name, base_url=settings.ollama_base_url)
    decision_maker = LLMDecisionMaker(llm_client=llm_client, tool_registry=_standard_registry())

    results = evaluate_all(decision_maker, ROUTING_EVALUATION_CASES)
    summary = summarize(results)

    print("\n" + "=" * 70)
    print("REAL llama3.2:3b ROUTING EVALUATION (informational — not a pass/fail gate)")
    print("=" * 70)
    print(format_report(summary, results))
    print("=" * 70)

    # Structural sanity only. Deliberately NOT asserting a minimum accuracy:
    # this test must never fail the suite just because the local model made
    # a poor decision on a nondeterministic run.
    assert summary.total == len(ROUTING_EVALUATION_CASES)


# ---------------------------------------------------------------------------
# Part 7/11: the real A/B experiment. Runs the SAME 27-case dataset twice —
# once with NoHintRouter (baseline, no routing hint at all) and once with
# the real Router (experiment, LLMDecisionMaker's current default wiring) —
# against the real model, and reports the comparison. Informational only:
# never asserts a minimum accuracy, and never assumes the experiment wins.
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_real_ollama_routing_hint_ab_experiment() -> None:
    if not _ollama_is_available():
        pytest.skip(f"Ollama is not reachable at {settings.ollama_base_url}; skipping live A/B experiment.")

    llm_client = LLMClient(provider="ollama", model_name=settings.model_name, base_url=settings.ollama_base_url)

    baseline_decision_maker = LLMDecisionMaker(
        llm_client=llm_client, tool_registry=_standard_registry(), router=NoHintRouter()
    )
    baseline_results = evaluate_all(baseline_decision_maker, ROUTING_EVALUATION_CASES)
    baseline_summary = summarize(baseline_results)

    experiment_decision_maker = LLMDecisionMaker(llm_client=llm_client, tool_registry=_standard_registry())
    experiment_results = evaluate_all(experiment_decision_maker, ROUTING_EVALUATION_CASES)
    experiment_summary = summarize(experiment_results)

    comparison = compare(baseline_summary, experiment_summary)

    print("\n" + "=" * 70)
    print("REAL llama3.2:3b A/B EXPERIMENT: no-hint baseline vs. Router-hint experiment")
    print("(informational only — do not assume the hint is beneficial)")
    print("=" * 70)
    print("--- BASELINE (no hint) ---")
    print(format_report(baseline_summary, baseline_results))
    print("--- EXPERIMENT (Router hint) ---")
    print(format_report(experiment_summary, experiment_results))
    print("--- COMPARISON ---")
    print(format_comparison_report(comparison))
    print("=" * 70)

    # Structural sanity only — never a quality gate on real model output,
    # and never a pass/fail judgment on whether the hint helped.
    assert baseline_summary.total == len(ROUTING_EVALUATION_CASES)
    assert experiment_summary.total == len(ROUTING_EVALUATION_CASES)
