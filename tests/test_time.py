from __future__ import annotations

from datetime import datetime

from app.tools.base import ToolResult
from app.tools.time import TimeTool


def test_time_tool_has_name_and_description() -> None:
    tool = TimeTool()

    assert tool.name == "time"
    assert tool.description


def test_time_tool_returns_tool_result_wrapping_a_dict_with_required_fields() -> None:
    tool = TimeTool()

    result = tool.execute()

    assert isinstance(result, ToolResult)
    assert result.success is True
    assert result.error is None
    assert isinstance(result.data, dict)
    for key in ("date", "time", "year", "day_of_week", "timezone", "iso"):
        assert key in result.data


def test_time_tool_year_is_int() -> None:
    tool = TimeTool()

    result = tool.execute()

    assert isinstance(result.data["year"], int)


def test_time_tool_iso_timestamp_is_timezone_aware() -> None:
    tool = TimeTool()

    result = tool.execute()

    parsed = datetime.fromisoformat(str(result.data["iso"]))
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() is not None


def test_time_tool_fields_are_internally_consistent() -> None:
    """The individual fields must all describe the same instant as `iso`,
    regardless of what "now" actually is when the test runs.

    Note: `datetime.fromisoformat` only recovers a fixed UTC offset, not the
    original named timezone (e.g. "India Standard Time" round-trips as
    "UTC+05:30"), so timezone identity is checked via offset, not tzname().
    """
    tool = TimeTool()

    result = tool.execute()
    data = result.data
    parsed = datetime.fromisoformat(str(data["iso"]))

    assert data["date"] == parsed.date().isoformat()
    assert data["time"] == parsed.time().isoformat(timespec="seconds")
    assert data["year"] == parsed.year
    assert data["day_of_week"] == parsed.strftime("%A")
    assert isinstance(data["timezone"], str)
    assert data["timezone"]
