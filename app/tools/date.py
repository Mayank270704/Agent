from __future__ import annotations

import re
from datetime import date, datetime

from app.tools.base import RiskLevel, ToolCapability, ToolResult


class DateTool:
    name = "date"
    description = "Determines the weekday and calendar details for a specified date."
    input_schema: dict[str, str] = {"date": "string"}
    output_description = "A dict with the parsed date, weekday, day, month, and year."
    # Milestone 18: pure local computation over caller-supplied text — no
    # network call, no mutation of any application or filesystem state.
    # Lowest available capability/risk tier; never needs confirmation.
    capability = ToolCapability.READ
    risk_level = RiskLevel.LOW
    requires_confirmation = False

    _DATE_FORMATS = (
        "%d %B %Y",
        "%d %b %Y",
        "%B %d %Y",
        "%b %d %Y",
        "%Y-%m-%d",
        "%Y/%m/%d",
    )

    def execute(self, input: str | None = None) -> ToolResult:
        """Parse `input` as a date and return its weekday/calendar details.

        Raises ValueError for missing/unparseable input: invalid input is a
        precondition failure, not an operational one, so it is not wrapped
        in a failed ToolResult (see app.tools.base for the error contract).
        """
        if input is None or not str(input).strip():
            raise ValueError("Date input cannot be empty.")

        parsed_date = self._parse_date(str(input).strip())
        data = {
            "date": parsed_date.isoformat(),
            "day_of_week": parsed_date.strftime("%A"),
            "day": parsed_date.day,
            "month": parsed_date.month,
            "year": parsed_date.year,
        }
        return ToolResult.ok(data)

    def _parse_date(self, date_input: str) -> date:
        normalized = re.sub(r"[,?]", "", date_input)
        normalized = re.sub(r"\s+", " ", normalized).strip()

        for date_format in self._DATE_FORMATS:
            try:
                return datetime.strptime(normalized, date_format).date()
            except ValueError:
                continue

        date_match = re.search(
            r"\b(\d{1,2}\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
            r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
            r"nov(?:ember)?|dec(?:ember)?)\s+\d{4})\b",
            normalized,
            re.IGNORECASE,
        )
        if date_match:
            for date_format in ("%d %B %Y", "%d %b %Y"):
                try:
                    return datetime.strptime(date_match.group(1), date_format).date()
                except ValueError:
                    continue

        raise ValueError(f"Could not parse a supported date from: {date_input}")
