"""Routing-decision evaluation harness: measures how reliably a DecisionMaker
(today: LLMDecisionMaker backed by local Ollama + llama3.2:3b) chooses FINAL
vs. TOOL, and which tool, against a small hand-picked dataset.

This module is measurement-only. It does not decide, execute, route, or
change anything about the live agent architecture — it exists to establish a
baseline BEFORE deciding whether Router needs to be reintroduced as a fast
routing-hint layer (see app/agent/router.py's module docstring for that
still-open question). Nothing here assumes the answer.

It never executes a tool: `evaluate_case` calls `decision_maker.decide(...)`
only, exactly like AgentLoop would for one iteration, and stops there. A
"tool" decision is graded by name alone, never run — so evaluating against
`web_search` costs nothing and never touches Tavily.

Step 8C adds an A/B comparison: `NoHintRouter` is a Router stand-in that
always reports RoutingHint.GENERAL, used as the "baseline" (no routing hint)
arm via LLMDecisionMaker's existing `router=` injection point — no
production code changes were needed to support this experiment. `compare()`
diffs two EvaluationSummarys (baseline vs. experiment) into a
ComparisonSummary of per-metric deltas.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from app.agent.decision_maker import DecisionParseError
from app.agent.loop import ActionType, AgentDecision, DecisionMaker
from app.agent.router import RoutingHint
from app.agent.state import AgentState


@dataclass(frozen=True)
class EvaluationCase:
    """One routing-decision test case: a user input and what a well-behaved
    decision maker should do with it, given the currently registered tools."""

    name: str
    user_input: str
    expected_action: ActionType
    category: str
    expected_tool: str | None = None  # only meaningful when expected_action is TOOL

    def __post_init__(self) -> None:
        if self.expected_action is ActionType.TOOL and not self.expected_tool:
            raise ValueError(f"Case {self.name!r}: TOOL cases must specify expected_tool.")
        if self.expected_action is ActionType.FINAL and self.expected_tool:
            raise ValueError(f"Case {self.name!r}: FINAL cases must not specify expected_tool.")


class CaseVerdict(Enum):
    """The specific way a case did (or didn't) match its expectation.

    Deliberately NOT a bare "pass"/"fail": a TOOL decision that produces
    valid JSON is not automatically graded correct (see the module's Part 5
    rationale) — the verdict distinguishes *why* it was wrong.
    """

    CORRECT = "correct"
    UNNECESSARY_TOOL_CALL = "unnecessary_tool_call"  # expected FINAL, got TOOL
    MISSED_TOOL_CALL = "missed_tool_call"  # expected TOOL, got FINAL
    WRONG_TOOL = "wrong_tool"  # got TOOL, but not the expected tool name
    INVALID_TOOL = "invalid_tool"  # model selected a tool that isn't registered
    PARSE_ERROR = "parse_error"  # any other malformed/invalid model output


@dataclass(frozen=True)
class CaseResult:
    """The outcome of running one EvaluationCase through a real decide() call."""

    case: EvaluationCase
    verdict: CaseVerdict
    actual_action: ActionType | None  # None only for PARSE_ERROR/INVALID_TOOL (no decision was produced)
    actual_tool: str | None = None
    detail: str = ""  # the final answer / tool_input / exception message, for debugging output

    @property
    def is_correct(self) -> bool:
        return self.verdict is CaseVerdict.CORRECT


def evaluate_case(decision_maker: DecisionMaker, case: EvaluationCase) -> CaseResult:
    """Run one case through `decision_maker.decide()` and grade the outcome.

    Never executes a tool — grading is by decision content only. A fresh
    AgentState is used per case (this evaluates single-turn routing, not
    multi-step behavior).
    """
    state = AgentState(user_input=case.user_input)

    try:
        decision: AgentDecision = decision_maker.decide(state)
    except DecisionParseError as exc:
        verdict = CaseVerdict.INVALID_TOOL if "unregistered tool" in str(exc).lower() else CaseVerdict.PARSE_ERROR
        return CaseResult(case=case, verdict=verdict, actual_action=None, actual_tool=None, detail=str(exc))

    if decision.action_type is ActionType.FINAL:
        verdict = CaseVerdict.CORRECT if case.expected_action is ActionType.FINAL else CaseVerdict.MISSED_TOOL_CALL
        return CaseResult(
            case=case, verdict=verdict, actual_action=ActionType.FINAL, detail=decision.final_answer or ""
        )

    # decision.action_type is ActionType.TOOL
    if case.expected_action is ActionType.FINAL:
        verdict = CaseVerdict.UNNECESSARY_TOOL_CALL
    elif decision.tool_name != case.expected_tool:
        verdict = CaseVerdict.WRONG_TOOL
    else:
        verdict = CaseVerdict.CORRECT

    return CaseResult(
        case=case,
        verdict=verdict,
        actual_action=ActionType.TOOL,
        actual_tool=decision.tool_name,
        detail=decision.tool_input or "",
    )


def evaluate_all(decision_maker: DecisionMaker, cases: list[EvaluationCase]) -> list[CaseResult]:
    return [evaluate_case(decision_maker, case) for case in cases]


@dataclass(frozen=True)
class EvaluationSummary:
    """The five metrics from Part 4, plus a raw verdict breakdown for
    debugging. Rates are 0.0 when their denominator is empty, never NaN."""

    total: int
    correct: int
    accuracy: float
    tool_selection_accuracy: float  # correct-tool-name rate, among cases that expected a tool
    unnecessary_tool_call_rate: float  # among cases that expected FINAL
    missed_tool_call_rate: float  # among cases that expected a tool
    invalid_tool_rate: float  # among all cases
    by_verdict: dict[CaseVerdict, int] = field(default_factory=dict)


def summarize(results: list[CaseResult]) -> EvaluationSummary:
    total = len(results)
    by_verdict: dict[CaseVerdict, int] = {}
    for result in results:
        by_verdict[result.verdict] = by_verdict.get(result.verdict, 0) + 1

    final_expected = [r for r in results if r.case.expected_action is ActionType.FINAL]
    tool_expected = [r for r in results if r.case.expected_action is ActionType.TOOL]

    correct = by_verdict.get(CaseVerdict.CORRECT, 0)
    tool_correct = sum(1 for r in tool_expected if r.verdict is CaseVerdict.CORRECT)
    unnecessary = sum(1 for r in final_expected if r.verdict is CaseVerdict.UNNECESSARY_TOOL_CALL)
    missed = sum(1 for r in tool_expected if r.verdict is CaseVerdict.MISSED_TOOL_CALL)
    invalid = by_verdict.get(CaseVerdict.INVALID_TOOL, 0)

    return EvaluationSummary(
        total=total,
        correct=correct,
        accuracy=(correct / total) if total else 0.0,
        tool_selection_accuracy=(tool_correct / len(tool_expected)) if tool_expected else 0.0,
        unnecessary_tool_call_rate=(unnecessary / len(final_expected)) if final_expected else 0.0,
        missed_tool_call_rate=(missed / len(tool_expected)) if tool_expected else 0.0,
        invalid_tool_rate=(invalid / total) if total else 0.0,
        by_verdict=by_verdict,
    )


def format_report(summary: EvaluationSummary, results: list[CaseResult] | None = None) -> str:
    """A short, human-readable report — used by the informational real-Ollama
    evaluation (Part 6), never asserted on for pass/fail."""
    lines = [
        f"total={summary.total} correct={summary.correct} accuracy={summary.accuracy:.1%}",
        f"tool_selection_accuracy={summary.tool_selection_accuracy:.1%}",
        f"unnecessary_tool_call_rate={summary.unnecessary_tool_call_rate:.1%}",
        f"missed_tool_call_rate={summary.missed_tool_call_rate:.1%}",
        f"invalid_tool_rate={summary.invalid_tool_rate:.1%}",
        "by_verdict: " + ", ".join(f"{v.value}={n}" for v, n in sorted(summary.by_verdict.items(), key=lambda kv: kv[0].value)),
    ]
    if results:
        lines.append("failures:")
        for result in results:
            if not result.is_correct:
                lines.append(
                    f"  [{result.case.category}] {result.case.name!r} "
                    f"expected={result.case.expected_action.value}/{result.case.expected_tool} "
                    f"got={result.verdict.value}"
                    + (f"/{result.actual_tool}" if result.actual_tool else "")
                )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Part 1: the evaluation dataset (~27 cases across 7 categories).
#
# Category E ("current date" phrasing) is pinned to expected_tool="time", not
# "date": TimeTool's output already includes the current date (see
# app/tools/time.py), and Router's own pre-existing deterministic fast-path
# (_deterministic_temporal_check) already classifies these exact phrases as
# "time", not "date" — this dataset matches that established, tested
# semantics rather than inventing a new expectation.
# ---------------------------------------------------------------------------

ROUTING_EVALUATION_CASES: list[EvaluationCase] = [
    # Category A — static / general questions -> FINAL
    EvaluationCase("a1", "What is Python?", ActionType.FINAL, "A"),
    EvaluationCase("a2", "Explain machine learning.", ActionType.FINAL, "A"),
    EvaluationCase("a3", "What is gradient descent?", ActionType.FINAL, "A"),
    EvaluationCase("a4", "What is a transformer?", ActionType.FINAL, "A"),
    EvaluationCase("a5", "How does a neural network work?", ActionType.FINAL, "A"),
    EvaluationCase("a6", "What is 2 + 2?", ActionType.FINAL, "A"),
    # Category B — current information -> TOOL web_search
    EvaluationCase("b1", "What is the latest AI news?", ActionType.TOOL, "B", "web_search"),
    EvaluationCase("b2", "What happened recently in artificial intelligence?", ActionType.TOOL, "B", "web_search"),
    EvaluationCase("b3", "What are the latest developments in NVIDIA?", ActionType.TOOL, "B", "web_search"),
    EvaluationCase("b4", "Search for current AI news.", ActionType.TOOL, "B", "web_search"),
    EvaluationCase("b5", "What is the current situation with OpenAI?", ActionType.TOOL, "B", "web_search"),
    # Category C — current time -> TOOL time
    EvaluationCase("c1", "What time is it?", ActionType.TOOL, "C", "time"),
    EvaluationCase("c2", "What is the current time?", ActionType.TOOL, "C", "time"),
    EvaluationCase("c3", "Tell me the time right now.", ActionType.TOOL, "C", "time"),
    # Category D — date / weekday for a specified date -> TOOL date
    EvaluationCase("d1", "What day is 25 December 2026?", ActionType.TOOL, "D", "date"),
    EvaluationCase("d2", "What weekday is 1 January 2027?", ActionType.TOOL, "D", "date"),
    EvaluationCase("d3", "Which day of the week is 15 August 2026?", ActionType.TOOL, "D", "date"),
    # Category E — current date -> TOOL time (see rationale above)
    EvaluationCase("e1", "What is today's date?", ActionType.TOOL, "E", "time"),
    EvaluationCase("e2", "What date is it today?", ActionType.TOOL, "E", "time"),
    EvaluationCase("e3", "Tell me today's date.", ActionType.TOOL, "E", "time"),
    # Category F — web search, not obviously "current" -> TOOL web_search
    EvaluationCase("f1", "Search the web for information about RAG.", ActionType.TOOL, "F", "web_search"),
    EvaluationCase("f2", "Find recent papers about transformers.", ActionType.TOOL, "F", "web_search"),
    EvaluationCase("f3", "Look up the latest research on LLM agents.", ActionType.TOOL, "F", "web_search"),
    # Category G — trick / negative cases -> FINAL (named entity != need for a tool)
    EvaluationCase("g1", "Explain today's date conceptually.", ActionType.FINAL, "G"),
    EvaluationCase("g2", "What is a neural network?", ActionType.FINAL, "G"),
    EvaluationCase("g3", "How does Google Search work?", ActionType.FINAL, "G"),
    EvaluationCase("g4", "Can you explain what NVIDIA is?", ActionType.FINAL, "G"),
]


# ---------------------------------------------------------------------------
# Part 7: A/B comparison — baseline (no routing hint) vs. experiment (Router
# hint + LLMDecisionMaker, i.e. LLMDecisionMaker's current default wiring).
# ---------------------------------------------------------------------------


class NoHintRouter:
    """A Router stand-in that always reports RoutingHint.GENERAL — used to
    construct the "baseline" (no routing hint) arm of the experiment via
    LLMDecisionMaker's existing `router=` constructor parameter. Zero
    production code changes were needed to support this: it's the same
    injection point tests already used to supply a FakeRouter."""

    def classify_hint(self, user_message: str) -> RoutingHint:
        return RoutingHint.GENERAL


@dataclass(frozen=True)
class ComparisonSummary:
    """Baseline vs. experiment, plus the delta for each of the five metrics.
    A positive `*_delta` means the experiment scored higher on that metric —
    which is an improvement for accuracy/tool_selection_accuracy, but a
    REGRESSION for the three failure-rate metrics."""

    baseline: EvaluationSummary
    experiment: EvaluationSummary
    accuracy_delta: float
    tool_selection_accuracy_delta: float
    unnecessary_tool_call_rate_delta: float
    missed_tool_call_rate_delta: float
    invalid_tool_rate_delta: float


def compare(baseline: EvaluationSummary, experiment: EvaluationSummary) -> ComparisonSummary:
    return ComparisonSummary(
        baseline=baseline,
        experiment=experiment,
        accuracy_delta=experiment.accuracy - baseline.accuracy,
        tool_selection_accuracy_delta=experiment.tool_selection_accuracy - baseline.tool_selection_accuracy,
        unnecessary_tool_call_rate_delta=experiment.unnecessary_tool_call_rate - baseline.unnecessary_tool_call_rate,
        missed_tool_call_rate_delta=experiment.missed_tool_call_rate - baseline.missed_tool_call_rate,
        invalid_tool_rate_delta=experiment.invalid_tool_rate - baseline.invalid_tool_rate,
    )


def format_comparison_report(comparison: ComparisonSummary) -> str:
    def _delta(value: float) -> str:
        sign = "+" if value >= 0 else ""
        return f"{sign}{value:.1%}"

    return "\n".join([
        f"baseline:   accuracy={comparison.baseline.accuracy:.1%} "
        f"tool_selection_accuracy={comparison.baseline.tool_selection_accuracy:.1%} "
        f"unnecessary_tool_call_rate={comparison.baseline.unnecessary_tool_call_rate:.1%} "
        f"missed_tool_call_rate={comparison.baseline.missed_tool_call_rate:.1%} "
        f"invalid_tool_rate={comparison.baseline.invalid_tool_rate:.1%}",
        f"experiment: accuracy={comparison.experiment.accuracy:.1%} "
        f"tool_selection_accuracy={comparison.experiment.tool_selection_accuracy:.1%} "
        f"unnecessary_tool_call_rate={comparison.experiment.unnecessary_tool_call_rate:.1%} "
        f"missed_tool_call_rate={comparison.experiment.missed_tool_call_rate:.1%} "
        f"invalid_tool_rate={comparison.experiment.invalid_tool_rate:.1%}",
        f"delta:      accuracy={_delta(comparison.accuracy_delta)} "
        f"tool_selection_accuracy={_delta(comparison.tool_selection_accuracy_delta)} "
        f"unnecessary_tool_call_rate={_delta(comparison.unnecessary_tool_call_rate_delta)} "
        f"missed_tool_call_rate={_delta(comparison.missed_tool_call_rate_delta)} "
        f"invalid_tool_rate={_delta(comparison.invalid_tool_rate_delta)}",
    ])
