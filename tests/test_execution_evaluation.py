"""Milestone 19, Phase 5: app/agent/execution_evaluation.py.

Proves the full-execution evaluation harness: it runs a real
AgentOrchestrator end to end with a scripted LLM and fake tools, derives
every metric purely from the returned AgentResult + collected AgentEvent
stream, and never alters execution. Fully offline: no Ollama, no Tavily,
no LLM judge.
"""
from __future__ import annotations

import json

from app.agent.execution_evaluation import ExecutionCase, run_all, run_case, summarize
from app.agent.tool_registry import ToolRegistry
from app.tools.base import ToolResult


class _Tool:
    def __init__(self, name: str, *, result: ToolResult | None = None):
        self.name = name
        self.description = "x"
        self.input_schema: dict[str, str] = {}
        self._result = result or ToolResult.ok({"ok": True})
        self.execute_count = 0

    def execute(self, input: str | None = None) -> ToolResult:
        self.execute_count += 1
        return self._result


class _ValidatingTool(_Tool):
    def validate(self, input: str | None = None) -> None:
        if input is None or not str(input).strip():
            raise ValueError("value cannot be empty.")


def _final_json(answer: str) -> str:
    return json.dumps({"action_type": "final", "final_answer": answer})


def _tool_json(tool_name: str, tool_input: str | None) -> str:
    return json.dumps({"action_type": "tool", "tool_name": tool_name, "tool_input": tool_input})


def _registry(*tools) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


# ===========================================================================
# Completion / answer / tool selection
# ===========================================================================

def test_a_completed_case_with_the_expected_tool_and_answer_passes() -> None:
    tool = _Tool("time")
    case = ExecutionCase(
        name="c1",
        user_input="what time is it",
        llm_responses=[_tool_json("time", None), _final_json("It is noon.")],
        expected_completion=True,
        expected_tools=("time",),
        expected_answer_contains="noon",
    )

    result = run_case(case, _registry(tool))

    assert result.completed is True
    assert result.executed_tools == ("time",)
    assert result.answer_matches is True
    assert result.passed is True
    assert tool.execute_count == 1


def test_a_case_missing_its_answer_predicate_fails() -> None:
    case = ExecutionCase(
        name="c2",
        user_input="what time is it",
        llm_responses=[_final_json("I have no idea.")],
        expected_completion=True,
        expected_answer_contains="noon",
    )

    result = run_case(case, ToolRegistry())

    assert result.completed is True
    assert result.answer_matches is False
    assert result.passed is False


def test_an_unnecessary_tool_call_is_reported_and_fails_the_case() -> None:
    tool = _Tool("web_search")
    case = ExecutionCase(
        name="c3",
        user_input="what is 2+2",
        llm_responses=[_tool_json("web_search", "2+2"), _final_json("4")],
        expected_completion=True,
        expected_tools=(),  # no tool call was expected at all
        forbidden_tools=("web_search",),
    )

    result = run_case(case, _registry(tool))

    assert result.executed_tools == ("web_search",)
    assert result.unnecessary_tools == ("web_search",)
    assert result.passed is False  # forbidden tool was executed


def test_a_missed_expected_tool_is_reflected_in_expected_vs_executed() -> None:
    case = ExecutionCase(
        name="c4",
        user_input="what time is it",
        llm_responses=[_final_json("I don't know.")],  # model answered directly, no tool used
        expected_completion=True,
        expected_tools=("time",),
    )

    result = run_case(case, ToolRegistry())

    assert result.proposed_tools == ()
    assert result.executed_tools == ()
    assert result.passed is False  # expected tool was never executed


# ===========================================================================
# Denial / correction / rejection counts
# ===========================================================================

def test_a_denied_tool_is_counted_and_is_not_an_executed_tool() -> None:
    from app.agent.orchestrator import AgentOrchestrator
    from app.agent.permissions import AllowlistPermissionPolicy, ExecutionContext
    from app.agent.telemetry import EventEmitter, ListEventSink
    from app.agent.tool_execution import ToolExecutionGate
    from app.agent.execution_evaluation import _grade

    tool = _Tool("delete_file")
    registry = _registry(tool)
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy(frozenset()))

    class _FakeLLM:
        def generate(self, messages, *, json_mode: bool = False) -> str:
            return _tool_json("delete_file", "x")

    sink = ListEventSink()
    emitter = EventEmitter(sink, request_id="eval-c5")
    orchestrator = AgentOrchestrator(
        llm_client=_FakeLLM(), tool_registry=registry, tool_execution_gate=gate, event_emitter=emitter
    )
    result_obj = orchestrator.process("delete it")

    case = ExecutionCase(
        name="c5", user_input="delete it", llm_responses=[], expected_completion=False, forbidden_tools=("delete_file",)
    )
    result = _grade(case, result_obj.answer, result_obj.status, tuple(sink.events))

    assert result.denied_tools == ("delete_file",)
    assert result.executed_tools == ()
    assert result.completed is False
    assert result.passed is True  # expected_completion=False matched, no forbidden tool EXECUTED
    assert tool.execute_count == 0


def test_a_correction_is_counted() -> None:
    tool = _ValidatingTool("writer")
    case = ExecutionCase(
        name="c6",
        user_input="write it",
        llm_responses=[_tool_json("writer", "   "), _tool_json("writer", "ok"), _final_json("done")],
        expected_completion=True,
        expected_tools=("writer",),
    )

    # Requires BOTH a ToolExecutionGate (the tool's validate() hook is only
    # ever consulted by the gate — see app/agent/tool_execution.py; with
    # no gate, "   " would reach execute() unchecked) and a correction
    # policy to actually recover. Built manually rather than through
    # run_case's no-gate default, to prove correction_count reflects a
    # REAL validate() rejection followed by a real correction.
    from app.agent.orchestrator import AgentOrchestrator
    from app.agent.permissions import AllowlistPermissionPolicy
    from app.agent.reliability import BudgetedCorrectionPolicy
    from app.agent.telemetry import EventEmitter, ListEventSink
    from app.agent.tool_execution import ToolExecutionGate
    from app.agent.execution_evaluation import _grade, _ScriptedLLM

    registry = _registry(tool)
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy({"writer"}))
    sink = ListEventSink()
    emitter = EventEmitter(sink, request_id="eval-c6")
    orchestrator = AgentOrchestrator(
        llm_client=_ScriptedLLM(case.llm_responses),
        tool_registry=registry,
        tool_execution_gate=gate,
        correction_policy=BudgetedCorrectionPolicy(max_corrections=3),
        event_emitter=emitter,
    )
    result_obj = orchestrator.process(case.user_input)
    result = _grade(case, result_obj.answer, result_obj.status, tuple(sink.events))

    assert result.correction_count == 1
    assert result.completed is True
    assert tool.execute_count == 1


# ===========================================================================
# Iteration count / latency / plan step count
# ===========================================================================

def test_iteration_count_matches_the_number_of_decide_calls() -> None:
    tool = _Tool("time")
    case = ExecutionCase(
        name="c7",
        user_input="what time is it",
        llm_responses=[_tool_json("time", None), _final_json("noon")],
        expected_completion=True,
    )

    result = run_case(case, _registry(tool))

    assert result.iteration_count == 2  # one TOOL decide, one FINAL decide


def test_latency_ms_is_populated_and_non_negative() -> None:
    case = ExecutionCase(
        name="c8", user_input="hello", llm_responses=[_final_json("hi")], expected_completion=True
    )

    result = run_case(case, ToolRegistry())

    assert result.latency_ms is not None
    assert result.latency_ms >= 0.0


def test_plan_step_count_is_none_when_no_plan_generator_is_used() -> None:
    case = ExecutionCase(
        name="c9", user_input="hello", llm_responses=[_final_json("hi")], expected_completion=True
    )

    result = run_case(case, ToolRegistry())

    assert result.plan_step_count is None


# ===========================================================================
# Evaluation never alters execution
# ===========================================================================

def test_running_the_same_case_twice_produces_identical_tool_execution_counts() -> None:
    tool_a = _Tool("time")
    tool_b = _Tool("time")
    case = ExecutionCase(
        name="c10",
        user_input="what time is it",
        llm_responses=[_tool_json("time", None), _final_json("noon")],
        expected_completion=True,
    )

    result_a = run_case(case, _registry(tool_a))
    result_b = run_case(case, _registry(tool_b))

    assert tool_a.execute_count == tool_b.execute_count == 1
    assert result_a.completed == result_b.completed
    assert result_a.executed_tools == result_b.executed_tools


def test_run_all_grades_every_case_independently_with_fresh_correlation() -> None:
    cases = [
        ExecutionCase(name="a", user_input="hi", llm_responses=[_final_json("hello")], expected_completion=True),
        ExecutionCase(name="b", user_input="hi", llm_responses=[_final_json("hello")], expected_completion=True),
    ]

    results = run_all(cases, ToolRegistry())

    assert len(results) == 2
    assert all(r.completed for r in results)


# ===========================================================================
# Summary
# ===========================================================================

def test_summarize_computes_rates_over_a_batch() -> None:
    passing = ExecutionCase(
        name="pass", user_input="hi", llm_responses=[_final_json("hello")], expected_completion=True
    )
    failing = ExecutionCase(
        name="fail",
        user_input="hi",
        llm_responses=[_final_json("hello")],
        expected_completion=True,
        expected_answer_contains="goodbye",  # will not match -> fails
    )

    results = run_all([passing, failing], ToolRegistry())
    summary = summarize(results)

    assert summary.total == 2
    assert summary.passed == 1
    assert summary.pass_rate == 0.5
    assert summary.completion_rate == 1.0
    assert summary.avg_latency_ms is not None


def test_summarize_handles_an_empty_batch_without_nan() -> None:
    summary = summarize([])

    assert summary.total == 0
    assert summary.pass_rate == 0.0
    assert summary.avg_latency_ms is None


# ===========================================================================
# No Ollama / Tavily required — structural proof
# ===========================================================================

def test_module_never_imports_network_capable_libraries() -> None:
    import app.agent.execution_evaluation as module

    source = __import__("inspect").getsource(module)
    import_lines = [line for line in source.splitlines() if line.strip().startswith(("import ", "from "))]
    for forbidden in ("urllib", "requests", "httpx", "tavily", "openai"):
        assert not any(forbidden in line.lower() for line in import_lines)
