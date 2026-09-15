from __future__ import annotations

from datetime import datetime

from app.tools.base import ToolResult


class TimeTool:
    name = "time"
    description = "Provides the current local/system date and time from the runtime clock."

    def execute(self, input: str | None = None) -> ToolResult:
        """Return the current local date and time in a predictable structure.

        Takes no meaningful input; `input` exists only to satisfy the shared
        Tool execution contract and is ignored.
        """
        current = datetime.now().astimezone()
        timezone_name = current.tzname()

        data = {
            "date": current.date().isoformat(),
            "time": current.time().isoformat(timespec="seconds"),
            "year": current.year,
            "day_of_week": current.strftime("%A"),
            "timezone": timezone_name,
            "iso": current.isoformat(),
        }
        return ToolResult.ok(data)
