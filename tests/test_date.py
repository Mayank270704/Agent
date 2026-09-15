from __future__ import annotations

import pytest

from app.tools.base import ToolResult
from app.tools.date import DateTool


CASES = (
    ("27 July 2026", "Monday"),
    ("12 September 2026", "Saturday"),
    ("25 December 2026", "Friday"),
)


@pytest.mark.parametrize("date_input,expected_day", CASES)
def test_date_tool_computes_expected_weekday(date_input: str, expected_day: str) -> None:
    tool = DateTool()

    result = tool.execute(date_input)

    assert isinstance(result, ToolResult)
    assert result.success is True
    assert result.error is None

    data = result.data
    assert data["day_of_week"] == expected_day
    assert data["date"]
    assert isinstance(data["day"], int)
    assert isinstance(data["month"], int)
    assert isinstance(data["year"], int)


def test_date_tool_invalid_input_raises_value_error() -> None:
    tool = DateTool()

    with pytest.raises(ValueError):
        tool.execute("this is not a date")


def test_date_tool_empty_input_raises_value_error() -> None:
    tool = DateTool()

    with pytest.raises(ValueError):
        tool.execute("   ")
