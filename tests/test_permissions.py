"""Milestone 18, Phase 6: app/agent/permissions.py in isolation.

No AgentLoop, no gate, no LLM, no network — this file proves
`AllowlistPermissionPolicy` and `ExecutionContext` are correct as PURE,
standalone components: deterministic, LLM-independent, non-self-
escalating, and never derived from anything but their own explicit
construction-time/call-time arguments.
"""
from __future__ import annotations

import pytest

from app.agent.permissions import (
    AllowlistPermissionPolicy,
    ExecutionContext,
    PermissionDecision,
    PermissionPolicy,
)
from app.tools.base import RiskLevel, ToolCapability, ToolDescriptor


def _descriptor(name: str = "time", **overrides: object) -> ToolDescriptor:
    fields: dict[str, object] = {
        "name": name,
        "description": f"the {name} tool",
        "input_schema": {},
    }
    fields.update(overrides)
    return ToolDescriptor(**fields)  # type: ignore[arg-type]


# ===========================================================================
# ExecutionContext
# ===========================================================================

def test_execution_context_defaults_to_no_confirmations_and_no_session() -> None:
    context = ExecutionContext()

    assert context.session_id is None
    assert context.confirmed_tools == frozenset()


def test_execution_context_is_confirmed_checks_the_trusted_set_only() -> None:
    context = ExecutionContext(confirmed_tools=frozenset({"send_email"}))

    assert context.is_confirmed("send_email") is True
    assert context.is_confirmed("delete_file") is False


def test_execution_context_is_frozen() -> None:
    context = ExecutionContext()

    with pytest.raises((AttributeError, TypeError)):
        context.session_id = "x"  # type: ignore[misc]


def test_two_execution_contexts_do_not_share_confirmed_tools() -> None:
    """Guards against a mutable-default-argument bug."""
    a = ExecutionContext()
    b = ExecutionContext(confirmed_tools=frozenset({"x"}))

    assert a.confirmed_tools == frozenset()
    assert b.confirmed_tools == frozenset({"x"})


# ===========================================================================
# AllowlistPermissionPolicy — resolution/construction
# ===========================================================================

def test_policy_conforms_to_the_permission_policy_protocol() -> None:
    policy = AllowlistPermissionPolicy({"time"})

    assert isinstance(policy, PermissionPolicy)


@pytest.mark.parametrize("bad", ["time", 123, None, object()])
def test_construction_rejects_a_non_collection_allowed_tools(bad: object) -> None:
    with pytest.raises(ValueError):
        AllowlistPermissionPolicy(bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_entries", [{"", "time"}, {"  ", "time"}, {123, "time"}])
def test_construction_rejects_invalid_entries(bad_entries: set) -> None:
    with pytest.raises(ValueError):
        AllowlistPermissionPolicy(bad_entries)  # type: ignore[arg-type]


def test_allowed_tools_accepts_list_tuple_set_or_frozenset() -> None:
    for container in (["time"], ("time",), {"time"}, frozenset({"time"})):
        policy = AllowlistPermissionPolicy(container)
        assert policy.evaluate(_descriptor("time"), ExecutionContext()) is PermissionDecision.ALLOW


def test_empty_allowlist_denies_everything() -> None:
    policy = AllowlistPermissionPolicy(frozenset())

    assert policy.evaluate(_descriptor("time"), ExecutionContext()) is PermissionDecision.DENY


# ===========================================================================
# ALLOW / DENY — the core decision
# ===========================================================================

def test_explicitly_allowed_tool_is_allowed() -> None:
    policy = AllowlistPermissionPolicy({"time", "date"})

    assert policy.evaluate(_descriptor("time"), ExecutionContext()) is PermissionDecision.ALLOW


def test_unlisted_tool_is_denied() -> None:
    policy = AllowlistPermissionPolicy({"time"})

    assert policy.evaluate(_descriptor("date"), ExecutionContext()) is PermissionDecision.DENY


def test_future_tool_is_not_automatically_allowed() -> None:
    """A tool the policy has never heard of — e.g. registered later — must
    default to DENY, never ALLOW."""
    policy = AllowlistPermissionPolicy({"time", "date", "web_search"})

    for future_tool in ("send_email", "delete_file", "make_payment", "deploy_application"):
        assert policy.evaluate(_descriptor(future_tool), ExecutionContext()) is PermissionDecision.DENY


def test_the_policy_never_reads_the_tool_registry() -> None:
    """Structural proof of 'do not create allow-all-registered-tools': the
    policy class has no ToolRegistry reference at all, so it CANNOT be
    driven by what happens to be registered."""
    policy = AllowlistPermissionPolicy({"time"})

    assert not hasattr(policy, "tools")
    assert not hasattr(policy, "tool_registry")
    assert not hasattr(policy, "registry")


# ===========================================================================
# CONFIRM — layered on top of ALLOW, never softens a DENY
# ===========================================================================

def test_allowed_tool_requiring_confirmation_without_trust_returns_confirm() -> None:
    policy = AllowlistPermissionPolicy({"delete_file"})
    descriptor = _descriptor("delete_file", requires_confirmation=True)

    assert policy.evaluate(descriptor, ExecutionContext()) is PermissionDecision.CONFIRM


def test_allowed_tool_requiring_confirmation_with_trust_returns_allow() -> None:
    policy = AllowlistPermissionPolicy({"delete_file"})
    descriptor = _descriptor("delete_file", requires_confirmation=True)
    context = ExecutionContext(confirmed_tools=frozenset({"delete_file"}))

    assert policy.evaluate(descriptor, context) is PermissionDecision.ALLOW


def test_confirmation_for_one_tool_does_not_confirm_another() -> None:
    policy = AllowlistPermissionPolicy({"delete_file", "make_payment"})
    context = ExecutionContext(confirmed_tools=frozenset({"delete_file"}))

    payment = _descriptor("make_payment", requires_confirmation=True)

    assert policy.evaluate(payment, context) is PermissionDecision.CONFIRM


def test_a_denied_tool_is_never_softened_to_confirm() -> None:
    """CONFIRM must never be used to rescue a tool that is not on the
    allow-list at all — DENY wins outright, regardless of
    requires_confirmation or trusted confirmation state."""
    policy = AllowlistPermissionPolicy({"time"})  # "delete_file" not listed
    descriptor = _descriptor("delete_file", requires_confirmation=True)
    context = ExecutionContext(confirmed_tools=frozenset({"delete_file"}))  # even if "confirmed"

    assert policy.evaluate(descriptor, context) is PermissionDecision.DENY


def test_tool_not_requiring_confirmation_is_allowed_even_with_empty_context() -> None:
    policy = AllowlistPermissionPolicy({"time"})
    descriptor = _descriptor("time", requires_confirmation=False)

    assert policy.evaluate(descriptor, ExecutionContext()) is PermissionDecision.ALLOW


# ===========================================================================
# Capability/risk are read but never drive the decision by themselves
# ===========================================================================

@pytest.mark.parametrize("capability", list(ToolCapability))
def test_capability_alone_does_not_change_the_allow_deny_outcome(capability: ToolCapability) -> None:
    """The allow-list, not capability, is authoritative for ALLOW/DENY —
    a DESTRUCTIVE tool that IS on the allow-list (with no confirmation
    requirement) is allowed; capability is descriptive metadata here."""
    policy = AllowlistPermissionPolicy({"x"})
    descriptor = _descriptor("x", capability=capability)

    assert policy.evaluate(descriptor, ExecutionContext()) is PermissionDecision.ALLOW


@pytest.mark.parametrize("risk_level", list(RiskLevel))
def test_risk_level_alone_does_not_change_the_allow_deny_outcome(risk_level: RiskLevel) -> None:
    policy = AllowlistPermissionPolicy({"x"})
    descriptor = _descriptor("x", risk_level=risk_level)

    assert policy.evaluate(descriptor, ExecutionContext()) is PermissionDecision.ALLOW


# ===========================================================================
# Determinism / purity / no LLM / no network
# ===========================================================================

def test_evaluate_is_deterministic() -> None:
    policy = AllowlistPermissionPolicy({"time"})
    descriptor = _descriptor("time")
    context = ExecutionContext()

    results = {policy.evaluate(descriptor, context) for _ in range(5)}

    assert results == {PermissionDecision.ALLOW}


def test_evaluate_does_not_mutate_the_descriptor_or_context() -> None:
    policy = AllowlistPermissionPolicy({"time"})
    descriptor = _descriptor("time")
    context = ExecutionContext()

    policy.evaluate(descriptor, context)

    assert descriptor == _descriptor("time")
    assert context == ExecutionContext()


def test_repeated_denial_never_becomes_allow_non_self_escalation() -> None:
    """Calling evaluate() repeatedly for a denied tool, with the SAME
    context, must never itself cause a later call to return something
    more permissive — the policy holds no mutable call history."""
    policy = AllowlistPermissionPolicy({"time"})
    descriptor = _descriptor("delete_file")
    context = ExecutionContext()

    results = [policy.evaluate(descriptor, context) for _ in range(10)]

    assert all(r is PermissionDecision.DENY for r in results)


def test_permission_policy_module_makes_no_network_calls() -> None:
    """Structural proof: neither module imports anything network-capable."""
    import app.agent.permissions as permissions_module

    source = __import__("inspect").getsource(permissions_module)
    for forbidden in ("urllib", "requests", "socket", "httpx"):
        assert forbidden not in source


def test_permission_policy_module_never_imports_an_llm_client() -> None:
    import app.agent.permissions as permissions_module

    source = __import__("inspect").getsource(permissions_module)
    assert "import" not in "\n".join(
        line for line in source.splitlines() if "llm" in line.lower()
    )


# ===========================================================================
# Type safety of the decision itself
# ===========================================================================

def test_permission_decision_is_a_closed_typed_enum_not_a_string() -> None:
    policy = AllowlistPermissionPolicy({"time"})

    decision = policy.evaluate(_descriptor("time"), ExecutionContext())

    assert isinstance(decision, PermissionDecision)
    assert not isinstance(decision, str)


def test_invalid_descriptor_type_is_rejected() -> None:
    policy = AllowlistPermissionPolicy({"time"})

    with pytest.raises(ValueError):
        policy.evaluate("time", ExecutionContext())  # type: ignore[arg-type]


def test_invalid_context_type_is_rejected() -> None:
    policy = AllowlistPermissionPolicy({"time"})

    with pytest.raises(ValueError):
        policy.evaluate(_descriptor("time"), "trust me")  # type: ignore[arg-type]
