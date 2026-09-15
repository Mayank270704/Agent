from __future__ import annotations

import pytest

from app.agent.tool_registry import ToolNotFoundError, ToolRegistrationError, ToolRegistry
from app.tools.base import Tool, ToolResult


class FakeTool:
    """A minimal duck-typed tool used only to test the registry. Deliberately
    does not inherit from anything, to prove the registry works with any
    object that structurally satisfies the Tool Protocol."""

    def __init__(self, name: str = "fake", description: str = "A fake tool for tests."):
        self.name = name
        self.description = description
        self.calls: list[str | None] = []

    def execute(self, input: str | None = None) -> ToolResult:
        self.calls.append(input)
        return ToolResult.ok(f"handled: {input}")


def test_register_and_get_tool() -> None:
    registry = ToolRegistry()
    tool = FakeTool(name="fake")

    registry.register(tool)

    assert registry.get("fake") is tool


def test_list_tools_returns_all_registered_tools() -> None:
    registry = ToolRegistry()
    tool_a = FakeTool(name="a")
    tool_b = FakeTool(name="b")

    registry.register(tool_a)
    registry.register(tool_b)

    listed = registry.list_tools()

    assert len(listed) == 2
    assert set(listed) == {tool_a, tool_b}


def test_has_reports_registration_status() -> None:
    registry = ToolRegistry()
    registry.register(FakeTool(name="fake"))

    assert registry.has("fake") is True
    assert registry.has("missing") is False


def test_duplicate_registration_raises_and_does_not_replace_existing_tool() -> None:
    registry = ToolRegistry()
    original = FakeTool(name="fake", description="original")
    registry.register(original)

    with pytest.raises(ToolRegistrationError):
        registry.register(FakeTool(name="fake", description="replacement"))

    # the original registration must survive untouched, not be silently overwritten
    assert registry.get("fake") is original
    assert registry.get("fake").description == "original"


def test_unknown_tool_lookup_raises_clear_exception() -> None:
    registry = ToolRegistry()

    with pytest.raises(ToolNotFoundError):
        registry.get("does-not-exist")


def test_registry_accepts_any_tool_protocol_conformant_object() -> None:
    tool = FakeTool(name="protocol-check")
    assert isinstance(tool, Tool)  # structural conformance, no inheritance required

    registry = ToolRegistry()
    registry.register(tool)

    result = registry.get("protocol-check").execute("ping")

    assert result.success is True
    assert result.data == "handled: ping"
    assert tool.calls == ["ping"]


def test_empty_registry_lists_no_tools() -> None:
    registry = ToolRegistry()

    assert registry.list_tools() == []
