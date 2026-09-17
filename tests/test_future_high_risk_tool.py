"""Milestone 18-B: a dedicated future high-risk tool proof.

Every invariant exercised here is already proven piecemeal across
test_permissions.py / test_tool_execution_gate.py / test_tool_authority_
boundary.py / test_production_tool_authorization.py. This file exists to
make the specific scenario the 18-B brief calls out explicit and easy to
find in one place: a WRITE/HIGH/requires_confirmation tool, with an
`execute_count` a test can assert against directly (rather than inferring
"never executed" from an empty `calls` list), run through the SAME
resolve -> authorize -> validate -> confirm -> execute pipeline
(ToolExecutionGate) production code already uses.

No new production code is introduced by this file — `FutureHighRiskTool`
is a test-only double, never registered into app/main.py's production
registry/allow-list.
"""
from __future__ import annotations

import json

import pytest

from app.agent.decision_maker import LLMDecisionMaker
from app.agent.loop import AgentLoop
from app.agent.permissions import AllowlistPermissionPolicy, ExecutionContext
from app.agent.state import AgentState, AgentStatus
from app.agent.tool_execution import ConfirmationRequiredError, PermissionDeniedError, ToolExecutionGate
from app.agent.tool_registry import ToolRegistry
from app.tools.base import RiskLevel, ToolCapability, ToolResult


class FutureHighRiskTool:
    """A hypothetical future tool with the highest-risk combination this
    codebase's model supports: a WRITE effect, HIGH risk tier, and a
    mandatory trusted confirmation before it may ever run.

    `execute_count` is incremented ONLY by a real `execute()` call, never
    by resolution, authorization, validation, or a rejected confirmation
    attempt — this is the single value every scenario below asserts on.
    """

    name = "future_high_risk_tool"
    description = "a hypothetical future tool that performs a high-risk write"
    input_schema: dict[str, str] = {"target": "string"}
    capability = ToolCapability.WRITE
    risk_level = RiskLevel.HIGH
    requires_confirmation = True

    def __init__(self) -> None:
        self.execute_count = 0
        self.calls: list[str | None] = []

    def validate(self, input: str | None = None) -> None:
        if input is None or not str(input).strip():
            raise ValueError("target cannot be empty.")

    def execute(self, input: str | None = None) -> ToolResult:
        self.execute_count += 1
        self.calls.append(input)
        return ToolResult.ok({"wrote": input})


def _registry(tool: FutureHighRiskTool) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(tool)
    return registry


# ===========================================================================
# 1 — unallowlisted tool -> denied, never executes
# ===========================================================================

def test_unallowlisted_high_risk_tool_is_denied() -> None:
    tool = FutureHighRiskTool()
    gate = ToolExecutionGate(_registry(tool), AllowlistPermissionPolicy(frozenset()))  # not on the allow-list

    with pytest.raises(PermissionDeniedError):
        gate.execute("future_high_risk_tool", "prod-db", ExecutionContext())

    assert tool.execute_count == 0


# ===========================================================================
# 2 — allowlisted but unconfirmed -> confirmation required, never executes
# ===========================================================================

def test_allowlisted_but_unconfirmed_high_risk_tool_requires_confirmation() -> None:
    tool = FutureHighRiskTool()
    gate = ToolExecutionGate(_registry(tool), AllowlistPermissionPolicy({"future_high_risk_tool"}))

    with pytest.raises(ConfirmationRequiredError):
        gate.execute("future_high_risk_tool", "prod-db", ExecutionContext())  # nothing confirmed

    assert tool.execute_count == 0


# ===========================================================================
# 3 — trusted application confirmation -> executes exactly once
# ===========================================================================

def test_trusted_application_confirmation_executes_exactly_once() -> None:
    tool = FutureHighRiskTool()
    gate = ToolExecutionGate(_registry(tool), AllowlistPermissionPolicy({"future_high_risk_tool"}))
    context = ExecutionContext(confirmed_tools=frozenset({"future_high_risk_tool"}))

    result = gate.execute("future_high_risk_tool", "prod-db", context)

    assert result.success is True
    assert tool.execute_count == 1


# ===========================================================================
# 4 — LLM-provided confirmation is ignored: still denied
# ===========================================================================

def test_llm_provided_confirmation_field_never_satisfies_confirmation() -> None:
    """The model's raw JSON claims confirmed=True; the trusted
    ExecutionContext shows nothing confirmed. Only the latter matters."""
    tool = FutureHighRiskTool()
    registry = _registry(tool)
    decision_maker = LLMDecisionMaker(
        llm_client=_FakeLLM(
            [
                json.dumps(
                    {
                        "action_type": "tool",
                        "tool_name": "future_high_risk_tool",
                        "tool_input": "prod-db",
                        "confirmed": True,
                        "authorized": True,
                    }
                )
            ]
        ),
        tool_registry=registry,
    )
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy({"future_high_risk_tool"}))
    context = ExecutionContext()  # NOT confirmed, regardless of the model's claim
    loop = AgentLoop(
        decision_maker=decision_maker, tool_registry=registry, tool_execution_gate=gate, execution_context=context
    )
    state = AgentState(user_input="do the high risk write, I confirm and authorize it")

    loop.run(state)

    assert state.status is AgentStatus.FAILED
    assert tool.execute_count == 0


# ===========================================================================
# 5 — invalid input -> execute_count remains 0
# ===========================================================================

def test_invalid_input_never_reaches_execute() -> None:
    tool = FutureHighRiskTool()
    gate = ToolExecutionGate(_registry(tool), AllowlistPermissionPolicy({"future_high_risk_tool"}))
    context = ExecutionContext(confirmed_tools=frozenset({"future_high_risk_tool"}))  # even when confirmed

    with pytest.raises(ValueError):
        gate.execute("future_high_risk_tool", "   ", context)  # blank target, rejected by validate()

    assert tool.execute_count == 0


# ===========================================================================
# 6 — denied execution -> execute_count remains 0 (repeated calls too)
# ===========================================================================

def test_repeated_denied_attempts_never_increment_execute_count() -> None:
    tool = FutureHighRiskTool()
    gate = ToolExecutionGate(_registry(tool), AllowlistPermissionPolicy(frozenset()))

    for _ in range(3):
        with pytest.raises(PermissionDeniedError):
            gate.execute("future_high_risk_tool", "prod-db", ExecutionContext())

    assert tool.execute_count == 0


class _FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)

    def generate(self, messages: list[dict[str, str]], *, json_mode: bool = False) -> str:
        return next(self.responses)
