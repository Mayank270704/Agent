"""Milestone 18, Phase 6: the LLM authority boundary, end to end.

Proves that a model's raw JSON output — even when it explicitly includes
`authorized`, `permission`, `capability`, `risk_level`, or `confirmed`
fields — has ZERO effect on what actually executes. The pipeline exercised
here is the REAL production one: raw JSON string -> LLMDecisionMaker ->
AgentDecision -> AgentLoop -> ToolExecutionGate -> PermissionPolicy ->
Tool.execute(). No mocking of any authorization step — a real
AllowlistPermissionPolicy makes the real decision.

Fully offline: FakeLLM replays scripted JSON strings; no Ollama, no
network.
"""
from __future__ import annotations

import json

import pytest

from app.agent.decision_maker import LLMDecisionMaker
from app.agent.loop import AgentLoop
from app.agent.permissions import AllowlistPermissionPolicy, ExecutionContext
from app.agent.state import AgentState, AgentStatus
from app.agent.tool_execution import ToolExecutionGate
from app.agent.tool_registry import ToolRegistry
from app.tools.base import ToolResult


class FakeLLM:
    def __init__(self, responses: list[str]):
        self.responses = iter(responses)

    def generate(self, messages: list[dict[str, str]], *, json_mode: bool = False) -> str:
        return next(self.responses)


class RecordingTool:
    def __init__(self, name: str, *, result: ToolResult | None = None):
        self.name = name
        self.description = f"tool '{name}'"
        self.input_schema: dict[str, str] = {"value": "string"}
        self._result = result or ToolResult.ok({"ok": True})
        self.calls: list[str | None] = []

    def execute(self, input: str | None = None) -> ToolResult:
        self.calls.append(input)
        return self._result


def _final_json(answer: str) -> str:
    return json.dumps({"action_type": "final", "final_answer": answer})


def _build(registry: ToolRegistry, *, allowed: frozenset[str], confirmed: frozenset[str] = frozenset()):
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy(allowed))
    context = ExecutionContext(confirmed_tools=confirmed)
    return gate, context


# ===========================================================================
# INVARIANT 4 — model-provided "authorized" claims are ignored
# ===========================================================================

def test_model_claiming_authorized_true_for_a_denied_tool_is_still_denied() -> None:
    tool = RecordingTool("delete_file")
    registry = ToolRegistry()
    registry.register(tool)
    decision_maker = LLMDecisionMaker(
        llm_client=FakeLLM(
            [
                json.dumps(
                    {
                        "action_type": "tool",
                        "tool_name": "delete_file",
                        "tool_input": "x",
                        "authorized": True,
                        "permission": "admin",
                    }
                )
            ]
        ),
        tool_registry=registry,
    )
    gate, context = _build(registry, allowed=frozenset())  # NOT allowed, regardless of the claim
    loop = AgentLoop(
        decision_maker=decision_maker, tool_registry=registry, tool_execution_gate=gate, execution_context=context
    )
    state = AgentState(user_input="delete something")

    loop.run(state)

    assert state.status is AgentStatus.FAILED
    assert tool.calls == []


def test_model_claiming_authorized_true_for_an_allowed_tool_is_irrelevant_it_would_run_anyway() -> None:
    """The claim carries no weight in EITHER direction — an allowed tool
    runs because it is on the allow-list, not because the model said so."""
    tool = RecordingTool("time")
    registry = ToolRegistry()
    registry.register(tool)
    decision_maker = LLMDecisionMaker(
        llm_client=FakeLLM(
            [
                json.dumps({"action_type": "tool", "tool_name": "time", "tool_input": None, "authorized": True}),
                _final_json("It is 12:00."),
            ]
        ),
        tool_registry=registry,
    )
    gate, context = _build(registry, allowed=frozenset({"time"}))
    loop = AgentLoop(
        decision_maker=decision_maker, tool_registry=registry, tool_execution_gate=gate, execution_context=context
    )
    state = AgentState(user_input="what time is it")

    loop.run(state)

    assert len(tool.calls) == 1


# ===========================================================================
# INVARIANT 5/6 — model-provided capability/risk claims are ignored
# ===========================================================================

def test_model_claiming_capability_read_for_a_destructive_tool_does_not_downgrade_it() -> None:
    """The example from the milestone brief: descriptor says DESTRUCTIVE
    (via requires_confirmation), model says capability=READ — the
    application must keep treating it as needing confirmation."""
    tool = RecordingTool("delete_file")
    tool.requires_confirmation = True  # type: ignore[attr-defined]
    tool.capability = "DESTRUCTIVE"  # type: ignore[attr-defined]  # the REAL, app-set value
    registry = ToolRegistry()
    registry.register(tool)
    decision_maker = LLMDecisionMaker(
        llm_client=FakeLLM(
            [
                json.dumps(
                    {
                        "action_type": "tool",
                        "tool_name": "delete_file",
                        "tool_input": "x",
                        "capability": "READ",  # the model's claim
                        "risk_level": "LOW",  # the model's claim
                    }
                )
            ]
        ),
        tool_registry=registry,
    )
    gate, context = _build(registry, allowed=frozenset({"delete_file"}))  # confirmed_tools empty -> not trusted
    loop = AgentLoop(
        decision_maker=decision_maker, tool_registry=registry, tool_execution_gate=gate, execution_context=context
    )
    state = AgentState(user_input="delete the file")

    loop.run(state)

    # Still blocked on confirmation -- the model's "capability"/"risk_level"
    # claims changed nothing about the REAL, application-set descriptor.
    assert state.status is AgentStatus.FAILED
    assert tool.calls == []


# ===========================================================================
# INVARIANT 7/12 — model-provided confirmation claims are ignored
# ===========================================================================

def test_model_claiming_confirmed_true_cannot_satisfy_confirmation() -> None:
    tool = RecordingTool("delete_file")
    tool.requires_confirmation = True  # type: ignore[attr-defined]
    registry = ToolRegistry()
    registry.register(tool)
    decision_maker = LLMDecisionMaker(
        llm_client=FakeLLM(
            [
                json.dumps(
                    {
                        "action_type": "tool",
                        "tool_name": "delete_file",
                        "tool_input": "x",
                        "confirmed": True,
                        "confirmation": "yes",
                    }
                )
            ]
        ),
        tool_registry=registry,
    )
    # ExecutionContext has NO trusted confirmation for "delete_file",
    # regardless of what the model claimed.
    gate, context = _build(registry, allowed=frozenset({"delete_file"}), confirmed=frozenset())
    loop = AgentLoop(
        decision_maker=decision_maker, tool_registry=registry, tool_execution_gate=gate, execution_context=context
    )
    state = AgentState(user_input="delete the file, I confirm it")

    loop.run(state)

    assert state.status is AgentStatus.FAILED
    assert tool.calls == []


def test_only_trusted_application_confirmation_permits_execution() -> None:
    """The positive case: identical model output, but THIS TIME the
    trusted ExecutionContext (never the model) shows confirmation."""
    tool = RecordingTool("delete_file")
    tool.requires_confirmation = True  # type: ignore[attr-defined]
    registry = ToolRegistry()
    registry.register(tool)
    decision_maker = LLMDecisionMaker(
        llm_client=FakeLLM(
            [
                json.dumps({"action_type": "tool", "tool_name": "delete_file", "tool_input": "x", "confirmed": True}),
                _final_json("Deleted."),
            ]
        ),
        tool_registry=registry,
    )
    gate, context = _build(registry, allowed=frozenset({"delete_file"}), confirmed=frozenset({"delete_file"}))
    loop = AgentLoop(
        decision_maker=decision_maker, tool_registry=registry, tool_execution_gate=gate, execution_context=context
    )
    state = AgentState(user_input="delete the file")

    loop.run(state)

    assert state.status is AgentStatus.COMPLETED
    assert len(tool.calls) == 1


# ===========================================================================
# The extra fields never even survive parsing
# ===========================================================================

def test_extra_model_fields_never_reach_the_agent_decision_object() -> None:
    registry = ToolRegistry()
    registry.register(RecordingTool("time"))
    decision_maker = LLMDecisionMaker(
        llm_client=FakeLLM(
            [
                json.dumps(
                    {
                        "action_type": "tool",
                        "tool_name": "time",
                        "tool_input": None,
                        "authorized": True,
                        "permission": "admin",
                        "capability": "READ",
                        "risk_level": "LOW",
                        "confirmed": True,
                    }
                )
            ]
        ),
        tool_registry=registry,
    )
    state = AgentState(user_input="what time is it")

    decision = decision_maker.decide(state)

    for forbidden in ("authorized", "permission", "capability", "risk_level", "confirmed"):
        assert not hasattr(decision, forbidden)


def test_agent_decision_has_a_fixed_closed_set_of_fields() -> None:
    """Structural proof, not just an absence-of-attribute check: dataclass
    fields are enumerable, so this proves no extra field could ever sneak
    in regardless of what any future parsing change might do."""
    from dataclasses import fields

    from app.agent.loop import AgentDecision

    field_names = {f.name for f in fields(AgentDecision)}

    assert field_names == {"action_type", "tool_name", "tool_input", "final_answer"}


# ===========================================================================
# INVARIANT 11 — permission denial cannot self-escalate via correction
# ===========================================================================

def test_denied_tool_cannot_be_recovered_through_correction_even_when_repeated() -> None:
    """The exact scenario the milestone brief forbids: permission denied
    -> correction -> same privileged request -> permission denied -> ...
    must terminate on the FIRST denial, not loop."""
    from app.agent.reliability import BudgetedCorrectionPolicy

    tool = RecordingTool("delete_file")
    registry = ToolRegistry()
    registry.register(tool)
    decision_maker = LLMDecisionMaker(
        llm_client=FakeLLM(
            [
                json.dumps({"action_type": "tool", "tool_name": "delete_file", "tool_input": "x"}),
                json.dumps({"action_type": "tool", "tool_name": "delete_file", "tool_input": "x"}),
                json.dumps({"action_type": "tool", "tool_name": "delete_file", "tool_input": "x"}),
            ]
        ),
        tool_registry=registry,
    )
    gate, context = _build(registry, allowed=frozenset())  # never allowed
    loop = AgentLoop(
        decision_maker=decision_maker,
        tool_registry=registry,
        tool_execution_gate=gate,
        execution_context=context,
        correction_policy=BudgetedCorrectionPolicy(max_corrections=5),  # plenty of budget -- must not matter
        max_iterations=5,
    )
    state = AgentState(user_input="please delete the file")

    loop.run(state)

    assert state.status is AgentStatus.FAILED
    assert state.step == 1  # terminated on the FIRST attempt, never retried
    assert state.corrections == []  # no correction was ever recorded for it
    assert tool.calls == []
