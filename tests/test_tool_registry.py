from __future__ import annotations

import pytest

from app.agent.tool_registry import ToolNotFoundError, ToolRegistrationError, ToolRegistry
from app.tools.base import Tool, ToolDescriptor, ToolResult


class FakeTool:
    """A minimal duck-typed tool used only to test the registry. Deliberately
    does not inherit from anything, to prove the registry works with any
    object that structurally satisfies the Tool Protocol."""

    def __init__(
        self,
        name: str = "fake",
        description: str = "A fake tool for tests.",
        input_schema: dict[str, str] | None = None,
    ):
        self.name = name
        self.description = description
        self.input_schema = input_schema if input_schema is not None else {}
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


# ---------------------------------------------------------------------------
# Structured metadata: describe() / describe_all() (Step 9).
# ---------------------------------------------------------------------------

def test_describe_returns_structured_metadata_for_one_tool() -> None:
    registry = ToolRegistry()
    registry.register(FakeTool(name="fake", description="A fake tool.", input_schema={"query": "string"}))

    descriptor = registry.describe("fake")

    assert isinstance(descriptor, ToolDescriptor)
    assert descriptor.name == "fake"
    assert descriptor.description == "A fake tool."
    assert descriptor.input_schema == {"query": "string"}
    assert descriptor.output_description == ""  # optional, absent on FakeTool
    assert descriptor.permissions == ()  # optional, absent on FakeTool


def test_describe_unknown_tool_raises_the_same_error_as_get() -> None:
    registry = ToolRegistry()

    with pytest.raises(ToolNotFoundError):
        registry.describe("does-not-exist")


def test_describe_all_lists_structured_metadata_for_every_registered_tool() -> None:
    registry = ToolRegistry()
    registry.register(FakeTool(name="a", description="Tool A", input_schema={"x": "string"}))
    registry.register(FakeTool(name="b", description="Tool B"))

    descriptors = registry.describe_all()

    assert {d.name for d in descriptors} == {"a", "b"}
    by_name = {d.name: d for d in descriptors}
    assert by_name["a"].description == "Tool A"
    assert by_name["a"].input_schema == {"x": "string"}
    assert by_name["b"].input_schema == {}


def test_describe_all_on_empty_registry_returns_empty_list() -> None:
    registry = ToolRegistry()

    assert registry.describe_all() == []


def test_describe_reads_optional_output_description_and_permissions_when_present() -> None:
    class ToolWithOptionalMetadata:
        name = "rich_tool"
        description = "A tool with optional metadata."
        input_schema: dict[str, str] = {"input": "string"}
        output_description = "Returns a rich result."
        permissions = ("network",)

        def execute(self, input: str | None = None) -> ToolResult:
            return ToolResult.ok(input)

    registry = ToolRegistry()
    registry.register(ToolWithOptionalMetadata())

    descriptor = registry.describe("rich_tool")

    assert descriptor.output_description == "Returns a rich result."
    assert descriptor.permissions == ("network",)


def test_describe_does_not_execute_the_tool() -> None:
    class RaisingIfExecuted:
        name = "raising_tool"
        description = "Should never be executed by describe()."
        input_schema: dict[str, str] = {}

        def execute(self, input: str | None = None) -> ToolResult:
            raise AssertionError("describe() must never call execute()")

    registry = ToolRegistry()
    registry.register(RaisingIfExecuted())

    registry.describe("raising_tool")  # would raise if it executed the tool
    registry.describe_all()  # same, for the bulk path


# ---------------------------------------------------------------------------
# Compatibility with the real, existing tools.
# ---------------------------------------------------------------------------

def test_describe_all_works_with_the_real_existing_tools() -> None:
    from app.tools.date import DateTool
    from app.tools.time import TimeTool
    from app.tools.web_search import WebSearchTool

    registry = ToolRegistry()
    registry.register(WebSearchTool())
    registry.register(TimeTool())
    registry.register(DateTool())

    descriptors = {d.name: d for d in registry.describe_all()}

    assert set(descriptors) == {"web_search", "time", "date"}
    assert descriptors["web_search"].input_schema == {"query": "string"}
    assert descriptors["time"].input_schema == {}
    assert descriptors["date"].input_schema == {"date": "string"}
    for descriptor in descriptors.values():
        assert descriptor.description  # every real tool still has a description
        assert descriptor.output_description  # every real tool now defines one
