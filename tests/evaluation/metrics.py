"""Step 29 — metric computation over `CaseExecutionResult`s (categories
A-H) and over the bespoke pass/fail check lists categories I-N produce.

Every metric is its own separate number (Phase 4: "Do NOT collapse all of
these into one arbitrary score"). `task_success_rate` IS an aggregate, but
its formula is stated explicitly in its own docstring, exactly as Phase 4
requires when producing any overall metric.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from tests.evaluation.runner import CaseExecutionResult


def _rate(numerator: int, denominator: int) -> float:
    return (numerator / denominator) if denominator else 0.0


def _percentile(values: list[float], pct: float) -> float:
    """Simple, dependency-free percentile (nearest-rank method) — no numpy
    in this project's core dependencies (see requirements.txt), and this
    dataset's sample sizes don't warrant adding one."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round(pct * (len(ordered) - 1)))))
    return ordered[index]


@dataclass
class CategoryMetrics:
    category: str
    total_trials: int
    cases: int
    task_success_rate: float
    final_answer_correctness_rate: float | None  # None if no case in this category has a checker
    tool_selection_accuracy: float | None  # None if no case in this category requires a tool
    missed_tool_call_rate: float | None
    unnecessary_tool_call_rate: float | None
    tool_execution_success_rate: float | None
    avg_llm_calls: float
    avg_tool_calls: float
    avg_latency_seconds: float
    min_latency_seconds: float
    max_latency_seconds: float
    p50_latency_seconds: float
    p95_latency_seconds: float


def task_success(result: CaseExecutionResult) -> bool:
    """Step 29 Phase 4, item 1 — the explicit formula:

    A trial counts as a TASK SUCCESS if and only if ALL of:
      1. The HTTP request itself succeeded (status_code == 200) and the
         production request-lifecycle telemetry recorded REQUEST_COMPLETED
         (never REQUEST_FAILED, never neither).
      2. Tool selection was correct: if the case requires a specific tool,
         that tool was proposed at least once; if the case requires no
         tool, none was proposed (no unnecessary call).
      3. Where a deterministic final-answer check exists for the case, it
         passed. Cases with NO checker (open-ended prose) are judged
         success on criteria 1-2 alone — see the dataset module's
         docstring for why forcing a checker onto open-ended prose would
         itself be dishonest grading, not stricter grading.

    This is a conjunction, not a weighted score — any single failure
    (wrong/missing tool, a failed request, or a failed deterministic
    check) makes the trial NOT a success, full stop.
    """
    if result.status_code != 200 or result.request_succeeded is not True:
        return False
    if result.case.requires_tool and result.missed_tool_call:
        return False
    if result.unnecessary_tool_call:
        return False
    answer_ok = result.final_answer_ok
    if answer_ok is False:
        return False
    return True


def compute_category_metrics(category: str, results: list[CaseExecutionResult]) -> CategoryMetrics:
    total = len(results)
    cases = len({r.case.case_id for r in results})

    successes = sum(1 for r in results if task_success(r))

    checked = [r for r in results if r.case.final_answer_check is not None]
    answer_correct = sum(1 for r in checked if r.final_answer_ok)

    tool_required = [r for r in results if r.case.requires_tool]
    tool_correct = sum(1 for r in tool_required if r.tool_selection_ok)
    missed = sum(1 for r in tool_required if r.missed_tool_call)

    tool_not_required = [r for r in results if not r.case.requires_tool]
    unnecessary = sum(1 for r in tool_not_required if r.unnecessary_tool_call)

    all_tool_executions = [success for r in results for (_, success) in r.executed_tools]

    latencies = [r.latency_seconds for r in results]

    return CategoryMetrics(
        category=category,
        total_trials=total,
        cases=cases,
        task_success_rate=_rate(successes, total),
        final_answer_correctness_rate=_rate(answer_correct, len(checked)) if checked else None,
        tool_selection_accuracy=_rate(tool_correct, len(tool_required)) if tool_required else None,
        missed_tool_call_rate=_rate(missed, len(tool_required)) if tool_required else None,
        unnecessary_tool_call_rate=_rate(unnecessary, len(tool_not_required)) if tool_not_required else None,
        tool_execution_success_rate=(
            _rate(sum(1 for s in all_tool_executions if s), len(all_tool_executions))
            if all_tool_executions else None
        ),
        avg_llm_calls=(sum(r.llm_call_count for r in results) / total) if total else 0.0,
        avg_tool_calls=(sum(len(r.executed_tools) for r in results) / total) if total else 0.0,
        avg_latency_seconds=(sum(latencies) / total) if total else 0.0,
        min_latency_seconds=min(latencies) if latencies else 0.0,
        max_latency_seconds=max(latencies) if latencies else 0.0,
        p50_latency_seconds=_percentile(latencies, 0.50),
        p95_latency_seconds=_percentile(latencies, 0.95),
    )


def format_category_report(m: CategoryMetrics) -> str:
    def _pct(v: float | None) -> str:
        return "n/a" if v is None else f"{v:.1%}"

    return (
        f"[{m.category}] trials={m.total_trials} cases={m.cases} "
        f"task_success_rate={_pct(m.task_success_rate)} "
        f"final_answer_correctness_rate={_pct(m.final_answer_correctness_rate)} "
        f"tool_selection_accuracy={_pct(m.tool_selection_accuracy)} "
        f"missed_tool_call_rate={_pct(m.missed_tool_call_rate)} "
        f"unnecessary_tool_call_rate={_pct(m.unnecessary_tool_call_rate)} "
        f"tool_execution_success_rate={_pct(m.tool_execution_success_rate)}\n"
        f"      avg_llm_calls={m.avg_llm_calls:.2f} avg_tool_calls={m.avg_tool_calls:.2f} "
        f"latency(avg/min/max/p50/p95)="
        f"{m.avg_latency_seconds:.2f}/{m.min_latency_seconds:.2f}/{m.max_latency_seconds:.2f}/"
        f"{m.p50_latency_seconds:.2f}/{m.p95_latency_seconds:.2f}s"
    )


@dataclass
class VarianceReport:
    """Per-case cross-trial stability (Step 29 Phase 6 — do not hide model
    variance)."""

    case_id: str
    category: str
    trial_outcomes: list[bool]  # task_success per trial, in order
    stable: bool


def compute_variance(results: list[CaseExecutionResult]) -> list[VarianceReport]:
    by_case: dict[str, list[CaseExecutionResult]] = {}
    for r in results:
        by_case.setdefault(r.case.case_id, []).append(r)

    reports = []
    for case_id, case_results in by_case.items():
        case_results.sort(key=lambda r: r.trial_index)
        outcomes = [task_success(r) for r in case_results]
        reports.append(
            VarianceReport(
                case_id=case_id,
                category=case_results[0].case.category,
                trial_outcomes=outcomes,
                stable=len(set(outcomes)) <= 1,
            )
        )
    return reports


@dataclass
class CheckResult:
    """A single named pass/fail check — used for categories I-N, whose
    ground truth is a mechanism property (memory correctness, security,
    timeout, telemetry), not a model-graded task outcome."""

    name: str
    passed: bool
    detail: str = ""


def summarize_checks(checks: list[CheckResult]) -> tuple[int, int, float]:
    """Returns (passed_count, total_count, pass_rate)."""
    total = len(checks)
    passed = sum(1 for c in checks if c.passed)
    return passed, total, _rate(passed, total)
