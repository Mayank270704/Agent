from __future__ import annotations

import pytest

from app.tools.base import Tool, ToolResult
from app.tools.date import DateTool
from app.tools.time import TimeTool
from app.tools.web_search import WebSearchTool


# ---------------------------------------------------------------------------
# ToolResult envelope itself.
# ---------------------------------------------------------------------------

def test_tool_result_ok_helper() -> None:
    result = ToolResult.ok({"x": 1})

    assert result.success is True
    assert result.data == {"x": 1}
    assert result.error is None


def test_tool_result_fail_helper() -> None:
    result = ToolResult.fail("boom")

    assert result.success is False
    assert result.data is None
    assert result.error == "boom"


# ---------------------------------------------------------------------------
# Structural conformance: every tool has name/description and matches the
# Tool Protocol (isinstance works because Tool is @runtime_checkable).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tool", [WebSearchTool(), TimeTool(), DateTool()])
def test_tool_conforms_to_tool_protocol(tool: Tool) -> None:
    assert isinstance(tool, Tool)
    assert isinstance(tool.name, str) and tool.name
    assert isinstance(tool.description, str) and tool.description
    assert isinstance(tool.input_schema, dict)
    for key, value in tool.input_schema.items():
        assert isinstance(key, str) and isinstance(value, str)


# ---------------------------------------------------------------------------
# Each tool's successful execute() returns a ToolResult carrying its expected data.
# ---------------------------------------------------------------------------

def test_time_tool_execute_returns_tool_result_with_expected_fields() -> None:
    tool = TimeTool()

    result = tool.execute()

    assert isinstance(result, ToolResult)
    assert result.success is True
    assert result.error is None
    for key in ("date", "time", "year", "day_of_week", "timezone", "iso"):
        assert key in result.data


def test_date_tool_execute_returns_tool_result_with_expected_fields() -> None:
    tool = DateTool()

    result = tool.execute("27 July 2026")

    assert isinstance(result, ToolResult)
    assert result.success is True
    assert result.error is None
    for key in ("date", "day", "month", "year", "day_of_week"):
        assert key in result.data
    assert result.data["day_of_week"] == "Monday"


def test_web_search_tool_execute_missing_key_still_raises_value_error() -> None:
    """Invalid input/config (missing API key) is a precondition failure, so it
    stays a raised ValueError rather than becoming ToolResult(success=False) —
    see app.tools.base for the error-design rationale. Full network-path
    coverage (success + operational-failure ToolResult cases) lives in
    test_web_search.py."""
    tool = WebSearchTool()
    tool.api_key = ""

    with pytest.raises(ValueError):
        tool.execute("some query")
