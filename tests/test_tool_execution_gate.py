"""Milestone 18, Phase 6: app/agent/tool_execution.py — ToolExecutionGate
in isolation, without AgentLoop.

Proves the gate itself: resolve -> authorize -> validate -> confirm ->
execute, in that order, using ONLY the existing ToolRegistry and a real
AllowlistPermissionPolicy. No LLM, no network, fully deterministic.
"""
from __future__ import annotations

import pytest

from app.agent.permissions import AllowlistPermissionPolicy, ExecutionContext
from app.agent.tool_execution import ConfirmationRequiredError, PermissionDeniedError, ToolExecutionGate
from app.agent.tool_registry import ToolNotFoundError, ToolRegistry
from app.tools.base import ToolResult


class RecordingTool:
    """A minimal Tool double that records every execute() call, so tests
    can assert it was called EXACTLY ONCE (or never)."""

    def __init__(self, name: str = "recorder", *, result: ToolResult | None = None, raises: Exception | None = None):
        self.name = name
        self.description = f"records calls to '{name}'"
        self.input_schema: dict[str, str] = {}
        self._result = result or ToolResult.ok("done")
        self._raises = raises
        self.calls: list[str | None] = []

    def execute(self, input: str | None = None) -> ToolResult:
        self.calls.append(input)
        if self._raises is not None:
            raise self._raises
        return self._result


class ValidatingTool(RecordingTool):
    """A Tool that ALSO defines the optional `validate()` hook, so tests
    can prove the gate calls it, and calls it in the right place relative
    to confirmation."""

    def __init__(self, *args: object, validation_error: str | None = None, **kwargs: object):
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._validation_error = validation_error
        self.validate_calls: list[str | None] = []

    def validate(self, input: str | None = None) -> None:
        self.validate_calls.append(input)
        if self._validation_error is not None:
            raise ValueError(self._validation_error)


def _registry(*tools) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


# ===========================================================================
# Construction validation
# ===========================================================================

def test_construction_requires_a_real_tool_registry() -> None:
    with pytest.raises(ValueError):
        ToolExecutionGate("not a registry", AllowlistPermissionPolicy({"x"}))  # type: ignore[arg-type]


def test_construction_requires_a_real_permission_policy() -> None:
    with pytest.raises(ValueError):
        ToolExecutionGate(ToolRegistry(), "not a policy")  # type: ignore[arg-type]


def test_gate_holds_no_separate_tool_table() -> None:
    """The gate must reuse the SAME ToolRegistry, never build its own."""
    registry = _registry(RecordingTool("time"))
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy({"time"}))

    assert gate.tools is registry


# ===========================================================================
# 1 — RESOLVE: unregistered tools never execute
# ===========================================================================

def test_unregistered_tool_raises_tool_not_found_error() -> None:
    gate = ToolExecutionGate(ToolRegistry(), AllowlistPermissionPolicy({"ghost"}))

    with pytest.raises(ToolNotFoundError):
        gate.execute("ghost", None, ExecutionContext())


def test_unregistered_tool_never_reaches_execute_even_if_the_name_is_allow_listed() -> None:
    """Being on the allow-list is irrelevant if the tool was never
    registered — ToolRegistry remains authoritative for existence."""
    tool = RecordingTool("ghost")
    gate = ToolExecutionGate(ToolRegistry(), AllowlistPermissionPolicy({"ghost"}))  # NOT registered

    with pytest.raises(ToolNotFoundError):
        gate.execute("ghost", None, ExecutionContext())

    assert tool.calls == []


# ===========================================================================
# 2 — AUTHORIZE: denied tools never execute
# ===========================================================================

def test_denied_tool_raises_permission_denied_error() -> None:
    tool = RecordingTool("delete_file")
    gate = ToolExecutionGate(_registry(tool), AllowlistPermissionPolicy(frozenset()))  # nothing allowed

    with pytest.raises(PermissionDeniedError):
        gate.execute("delete_file", None, ExecutionContext())

    assert tool.calls == []


def test_allowed_tool_executes() -> None:
    tool = RecordingTool("time", result=ToolResult.ok({"time": "12:00"}))
    gate = ToolExecutionGate(_registry(tool), AllowlistPermissionPolicy({"time"}))

    result = gate.execute("time", None, ExecutionContext())

    assert result.success is True
    assert result.data == {"time": "12:00"}
    assert tool.calls == [None]


def test_authorization_order_authorize_before_execute() -> None:
    """A denied tool must never even be CALLED — proving order, not just
    outcome."""
    tool = RecordingTool("delete_file")
    gate = ToolExecutionGate(_registry(tool), AllowlistPermissionPolicy({"time"}))  # delete_file not listed

    with pytest.raises(PermissionDeniedError):
        gate.execute("delete_file", "x", ExecutionContext())

    assert tool.calls == []


# ===========================================================================
# 3/4 — VALIDATE then CONFIRM: order matters
# ===========================================================================

def test_confirmation_required_blocks_execution_without_trust() -> None:
    tool = RecordingTool("delete_file")
    descriptor_tool = tool
    registry = _registry(tool)
    # requires_confirmation is read from the tool object, not the registry —
    # set it directly for this test double.
    tool.requires_confirmation = True  # type: ignore[attr-defined]
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy({"delete_file"}))

    with pytest.raises(ConfirmationRequiredError):
        gate.execute("delete_file", None, ExecutionContext())

    assert tool.calls == []


def test_trusted_confirmation_permits_execution() -> None:
    tool = RecordingTool("delete_file")
    tool.requires_confirmation = True  # type: ignore[attr-defined]
    gate = ToolExecutionGate(_registry(tool), AllowlistPermissionPolicy({"delete_file"}))
    context = ExecutionContext(confirmed_tools=frozenset({"delete_file"}))

    result = gate.execute("delete_file", None, context)

    assert result.success is True
    assert tool.calls == [None]


def test_validate_is_called_when_the_tool_defines_it() -> None:
    tool = ValidatingTool("date")
    gate = ToolExecutionGate(_registry(tool), AllowlistPermissionPolicy({"date"}))

    gate.execute("date", "2026-01-01", ExecutionContext())

    assert tool.validate_calls == ["2026-01-01"]


def test_validate_failure_raises_value_error_and_never_executes() -> None:
    tool = ValidatingTool("date", validation_error="bad date")
    gate = ToolExecutionGate(_registry(tool), AllowlistPermissionPolicy({"date"}))

    with pytest.raises(ValueError, match="bad date"):
        gate.execute("date", "not-a-date", ExecutionContext())

    assert tool.calls == []  # execute() itself was never reached


def test_validate_runs_before_the_confirmation_check() -> None:
    """A tool that BOTH needs confirmation AND has a separate validate()
    hook: an invalid input is reported before the confirmation gate is
    even consulted — the documented order (resolve -> authorize ->
    validate -> confirm -> execute)."""
    tool = ValidatingTool("delete_file", validation_error="bad target")
    tool.requires_confirmation = True  # type: ignore[attr-defined]
    gate = ToolExecutionGate(_registry(tool), AllowlistPermissionPolicy({"delete_file"}))

    with pytest.raises(ValueError, match="bad target"):
        gate.execute("delete_file", "garbage", ExecutionContext())  # not confirmed either

    assert tool.calls == []


def test_a_tool_with_no_validate_hook_is_unaffected() -> None:
    """Every tool in this codebase today has no separate validate() — the
    gate must be a complete no-op for that hook in that case."""
    tool = RecordingTool("time")
    gate = ToolExecutionGate(_registry(tool), AllowlistPermissionPolicy({"time"}))

    result = gate.execute("time", None, ExecutionContext())

    assert result.success is True
    assert not hasattr(tool, "validate_calls")


def test_invalid_input_from_execute_itself_still_propagates_as_value_error() -> None:
    """For a tool with no separate validate(), the existing ValueError
    contract inside execute() is unchanged."""
    tool = RecordingTool("date", raises=ValueError("Could not parse a supported date"))
    gate = ToolExecutionGate(_registry(tool), AllowlistPermissionPolicy({"date"}))

    with pytest.raises(ValueError, match="Could not parse"):
        gate.execute("date", "garbage", ExecutionContext())


# ===========================================================================
# 5 — EXECUTE: exactly once on the successful path
# ===========================================================================

def test_tool_is_executed_exactly_once_on_the_allow_path() -> None:
    tool = RecordingTool("time")
    gate = ToolExecutionGate(_registry(tool), AllowlistPermissionPolicy({"time"}))

    gate.execute("time", None, ExecutionContext())

    assert len(tool.calls) == 1


@pytest.mark.parametrize(
    "setup",
    [
        "unregistered",
        "denied",
        "confirmation_required",
        "validate_hook_rejects",
    ],
)
def test_tool_executes_zero_times_on_every_blocked_path(setup: str) -> None:
    """"Blocked" means the pipeline stops BEFORE ever calling execute() —
    which is only true for resolve/authorize/confirm failures, and for a
    separate `validate()` hook rejecting input. A tool with NO separate
    validate() hook that raises ValueError from INSIDE execute() itself
    has, by definition, already been called once — that case is covered
    separately by test_invalid_input_from_execute_itself_still_propagates_
    as_value_error, not here."""
    tool = ValidatingTool("target", validation_error="bad") if setup == "validate_hook_rejects" else RecordingTool(
        "target"
    )
    registry = ToolRegistry() if setup == "unregistered" else _registry(tool)
    allowed = frozenset() if setup == "denied" else frozenset({"target"})
    if setup == "confirmation_required":
        tool.requires_confirmation = True  # type: ignore[attr-defined]
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy(allowed))

    with pytest.raises((ToolNotFoundError, PermissionDeniedError, ConfirmationRequiredError, ValueError)):
        gate.execute("target", None, ExecutionContext())

    assert tool.calls == []


def test_the_gate_returns_the_tools_own_operational_failure_unmodified() -> None:
    """A tool's own ToolResult(success=False, ...) is NOT the gate's
    business — the gate never wraps, alters, or reinterprets it."""
    tool = RecordingTool("web_search", result=ToolResult.fail("network timeout"))
    gate = ToolExecutionGate(_registry(tool), AllowlistPermissionPolicy({"web_search"}))

    result = gate.execute("web_search", "q", ExecutionContext())

    assert result.success is False
    assert result.error == "network timeout"


# ===========================================================================
# Observability — safe structured events, never payloads
# ===========================================================================

def test_denial_is_logged_with_tool_name_only(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    tool = RecordingTool("delete_file")
    gate = ToolExecutionGate(_registry(tool), AllowlistPermissionPolicy(frozenset()))

    with caplog.at_level(logging.WARNING, logger="app.agent.tool_execution"):
        with pytest.raises(PermissionDeniedError):
            gate.execute("delete_file", "secret input text", ExecutionContext())

    assert "tool.authorization.denied" in caplog.text
    assert "tool=delete_file" in caplog.text
    assert "secret input text" not in caplog.text


def test_confirmation_required_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    tool = RecordingTool("delete_file")
    tool.requires_confirmation = True  # type: ignore[attr-defined]
    gate = ToolExecutionGate(_registry(tool), AllowlistPermissionPolicy({"delete_file"}))

    with caplog.at_level(logging.WARNING, logger="app.agent.tool_execution"):
        with pytest.raises(ConfirmationRequiredError):
            gate.execute("delete_file", None, ExecutionContext())

    assert "tool.confirmation.required" in caplog.text


def test_successful_execution_is_logged_without_raw_tool_output(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    tool = RecordingTool("time", result=ToolResult.ok({"secret_looking_data": "should not be logged"}))
    gate = ToolExecutionGate(_registry(tool), AllowlistPermissionPolicy({"time"}))

    with caplog.at_level(logging.INFO, logger="app.agent.tool_execution"):
        gate.execute("time", None, ExecutionContext())

    assert "tool.execution.completed" in caplog.text
    assert "should not be logged" not in caplog.text
