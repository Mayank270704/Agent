"""Milestone 18, Phase 7: hardening review.

Targeted checks for the specific risks Phase 7 calls out: model-controlled
fields, permission/confirmation escalation, error exposure, logging, and
duplicate execution paths. Most of these invariants are already proven in
test_permissions.py / test_tool_execution_gate.py / test_tool_authority_
boundary.py / test_agent_loop_permissions.py — this file covers the
handful of interactions between Milestone 18 and pre-existing Milestone
17 hardening (F8) that are not exercised elsewhere.
"""
from __future__ import annotations

import logging

import pytest

from app.agent.loop import AgentDecision, AgentLoop
from app.agent.orchestrator import AgentOrchestrator
from app.agent.permissions import AllowlistPermissionPolicy, ExecutionContext
from app.agent.state import AgentState, AgentStatus
from app.agent.tool_execution import ToolExecutionGate
from app.agent.tool_registry import ToolRegistry
from app.tools.base import ToolResult


class FakeLLM:
    def generate(self, messages, *, json_mode: bool = False) -> str:
        raise AssertionError("FakeLLM should never be called in these tests")


class RepeatToolDecisionMaker:
    def __init__(self, tool_name: str, tool_input: str | None = None):
        self.tool_name = tool_name
        self.tool_input = tool_input

    def decide(self, state: AgentState) -> AgentDecision:
        return AgentDecision.tool(self.tool_name, self.tool_input)


class RecordingTool:
    def __init__(self, name: str, *, requires_confirmation: bool = False):
        self.name = name
        self.description = f"tool '{name}'"
        self.input_schema: dict[str, str] = {}
        self.requires_confirmation = requires_confirmation
        self.calls: list[str | None] = []

    def execute(self, input: str | None = None) -> ToolResult:
        self.calls.append(input)
        return ToolResult.ok("done")


# ===========================================================================
# Error exposure: Milestone 17's F8 generic-answer hardening covers the
# two NEW Milestone 18 failure types too, with no code change needed to
# _failure_answer -- it treats every state.errors entry uniformly.
# ===========================================================================

def test_permission_denied_produces_the_generic_failure_answer_not_the_raw_message() -> None:
    tool = RecordingTool("delete_secret_project")
    registry = ToolRegistry()
    registry.register(tool)
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy(frozenset()))
    orchestrator = AgentOrchestrator(
        llm_client=FakeLLM(),
        tool_registry=registry,
        decision_maker=RepeatToolDecisionMaker("delete_secret_project"),
    )
    # AgentOrchestrator does not accept a gate directly (Milestone 18 is
    # wired at the AgentLoop level) -- exercise the same _failure_answer
    # hardening by driving AgentLoop directly and reusing the
    # orchestrator's exact _failure_answer logic.
    loop = AgentLoop(
        decision_maker=RepeatToolDecisionMaker("delete_secret_project"),
        tool_registry=registry,
        tool_execution_gate=gate,
    )
    state = AgentState(user_input="delete the secret project")
    loop.run(state)

    answer = orchestrator._failure_answer(state)

    assert "could not complete this request" in answer
    assert "delete_secret_project" not in answer
    assert "not authorized" not in answer
    assert "delete_secret_project" in state.errors[-1].message  # detail preserved internally


def test_confirmation_required_produces_the_generic_failure_answer_not_the_raw_message() -> None:
    tool = RecordingTool("wipe_database", requires_confirmation=True)
    registry = ToolRegistry()
    registry.register(tool)
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy({"wipe_database"}))
    orchestrator = AgentOrchestrator(
        llm_client=FakeLLM(), tool_registry=registry, decision_maker=RepeatToolDecisionMaker("wipe_database")
    )
    loop = AgentLoop(
        decision_maker=RepeatToolDecisionMaker("wipe_database"),
        tool_registry=registry,
        tool_execution_gate=gate,
        execution_context=ExecutionContext(),
    )
    state = AgentState(user_input="wipe it")
    loop.run(state)

    answer = orchestrator._failure_answer(state)

    assert "could not complete this request" in answer
    assert "wipe_database" not in answer
    assert "confirmation" not in answer.lower()
    assert "wipe_database" in state.errors[-1].message


# ===========================================================================
# Duplicate execution paths: once a gate is configured, IT (not
# AgentLoop.tools) is the authoritative registry for resolution.
# ===========================================================================

def test_gate_registry_is_authoritative_over_loop_tools_when_both_are_configured() -> None:
    """If a caller mistakenly points the loop and the gate at two
    DIFFERENT registries, the gate's own registry is what actually gets
    consulted for resolution/authorization -- there is no second,
    loop-level path that could resolve against the other one instead."""
    loop_registry = ToolRegistry()
    loop_registry.register(RecordingTool("only_in_loop_registry"))

    gate_registry = ToolRegistry()
    gate_tool = RecordingTool("only_in_gate_registry")
    gate_registry.register(gate_tool)
    gate = ToolExecutionGate(gate_registry, AllowlistPermissionPolicy({"only_in_gate_registry"}))

    loop = AgentLoop(
        decision_maker=RepeatToolDecisionMaker("only_in_gate_registry"),
        tool_registry=loop_registry,  # deliberately mismatched
        tool_execution_gate=gate,
        max_iterations=1,
    )
    state = AgentState(user_input="hello")

    loop.run(state)

    # Resolved and ran through the GATE's registry, proving there is no
    # separate, loop-owned resolution path active alongside the gate.
    assert len(gate_tool.calls) == 1


def test_loop_tools_registry_is_never_consulted_for_resolution_once_a_gate_is_present() -> None:
    loop_registry = ToolRegistry()
    loop_registry.register(RecordingTool("time"))  # a tool that WOULD resolve here

    gate_registry = ToolRegistry()  # empty -- "time" is NOT registered here
    gate = ToolExecutionGate(gate_registry, AllowlistPermissionPolicy({"time"}))

    loop = AgentLoop(
        decision_maker=RepeatToolDecisionMaker("time"), tool_registry=loop_registry, tool_execution_gate=gate
    )
    state = AgentState(user_input="hello")

    loop.run(state)

    # If loop.tools were ever consulted as a fallback, this would succeed.
    # It must instead fail exactly as an unregistered tool would.
    assert state.status is AgentStatus.FAILED


# ===========================================================================
# Logging: sensitive-looking tool_input never appears anywhere in a log,
# through the FULL AgentLoop pipeline (not just the gate in isolation).
# ===========================================================================

def test_no_log_line_anywhere_contains_the_raw_tool_input(caplog: pytest.LogCaptureFixture) -> None:
    tool = RecordingTool("delete_file")
    registry = ToolRegistry()
    registry.register(tool)
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy(frozenset()))
    secret_input = "super-secret-target-path/hunter2.db"
    loop = AgentLoop(
        decision_maker=RepeatToolDecisionMaker("delete_file", secret_input),
        tool_registry=registry,
        tool_execution_gate=gate,
    )
    state = AgentState(user_input="delete the sensitive file")

    with caplog.at_level(logging.DEBUG):
        loop.run(state)

    assert secret_input not in caplog.text
    assert "hunter2" not in caplog.text


def test_gate_execute_signature_has_no_channel_for_a_model_supplied_claim() -> None:
    """Structural proof at the call-signature level: ToolExecutionGate.
    execute() takes exactly (tool_name, tool_input, context) — there is no
    parameter through which an "authorized"/"confirmed"/"capability"
    claim could even be passed in, regardless of what a caller might try
    to forward from a model's raw output."""
    import inspect

    from app.agent.tool_execution import ToolExecutionGate

    parameters = list(inspect.signature(ToolExecutionGate.execute).parameters)

    assert parameters == ["self", "tool_name", "tool_input", "context"]
