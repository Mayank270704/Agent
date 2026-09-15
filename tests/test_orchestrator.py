from __future__ import annotations

import pytest

from app.agent.orchestrator import AgentOrchestrator
from app.agent.router import RouterDecision
from app.agent.tool_registry import ToolRegistry
from app.tools.base import ToolResult


class FakeRouter:
    def __init__(self, decision: RouterDecision):
        self.decision = decision

    def decide(self, user_message: str) -> RouterDecision:
        return self.decision


class FakeLLM:
    def __init__(self):
        self.prompts: list[str] = []

    def generate(self, messages: list[dict[str, str]]) -> str:
        self.prompts.append(messages[0]["content"])
        return "fake answer"


class FakeWebSearchTool:
    name = "web_search"
    description = "Fake web search tool for tests."

    def __init__(self, *, fail: bool = False, fail_error: str = "simulated search failure"):
        self.calls: list[str] = []
        self._fail = fail
        self._fail_error = fail_error

    def execute(self, input: str | None = None) -> ToolResult:
        self.calls.append(input)
        if self._fail:
            return ToolResult.fail(self._fail_error)
        return ToolResult.ok([{"title": "Fake source", "url": "https://example.com", "content": "Fake result"}])


class FakeTimeTool:
    name = "time"
    description = "Fake time tool for tests."

    def __init__(self):
        self.calls = 0

    def execute(self, input: str | None = None) -> ToolResult:
        self.calls += 1
        return ToolResult.ok({
            "date": "runtime-date",
            "time": "runtime-time",
            "year": 9999,
            "day_of_week": "runtime-day",
            "timezone": "runtime-zone",
            "iso": "runtime-iso",
        })


class FakeDateTool:
    name = "date"
    description = "Fake date tool for tests."

    def __init__(self):
        self.calls: list[str] = []

    def execute(self, input: str | None = None) -> ToolResult:
        self.calls.append(input)
        return ToolResult.ok({
            "date": "2026-07-27",
            "day_of_week": "Monday",
            "day": 27,
            "month": 7,
            "year": 2026,
        })


def _build_orchestrator(route: str):
    llm = FakeLLM()
    web_tool = FakeWebSearchTool()
    time_tool = FakeTimeTool()
    date_tool = FakeDateTool()
    orchestrator = AgentOrchestrator(
        llm_client=llm,
        router=FakeRouter(RouterDecision(
            needs_web=route == "web",
            reason=f"{route} route",
            search_query="test search" if route == "web" else "",
            route=route,  # type: ignore[arg-type]
        )),
        web_search_tool=web_tool,  # type: ignore[arg-type]
        time_tool=time_tool,  # type: ignore[arg-type]
        date_tool=date_tool,  # type: ignore[arg-type]
    )
    return orchestrator, llm, web_tool, time_tool, date_tool


# ---------------------------------------------------------------------------
# Comprehensive route-dispatch + tool-isolation matrix.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "route,expected_web,expected_time,expected_date",
    [
        ("llm", False, False, False),
        ("web", True, False, False),
        ("time", False, True, False),
        ("date", False, False, True),
    ],
)
def test_orchestrator_route_dispatch_and_tool_isolation(
    route: str, expected_web: bool, expected_time: bool, expected_date: bool
) -> None:
    orchestrator, llm, web_tool, time_tool, date_tool = _build_orchestrator(route)

    result = orchestrator.process("test question")

    assert result.used_web is expected_web
    assert result.used_time is expected_time
    assert result.used_date is expected_date
    assert bool(web_tool.calls) is expected_web
    assert time_tool.calls == (1 if expected_time else 0)
    assert len(date_tool.calls) == (1 if expected_date else 0)
    assert len(llm.prompts) == (0 if expected_date else 1)

    if expected_time:
        assert "runtime-date" in llm.prompts[0]
        assert "runtime-time" in llm.prompts[0]
    if expected_date:
        assert result.answer == "27 July 2026 was a Monday."


# ---------------------------------------------------------------------------
# Explicit per-route isolation checks (named per the required coverage list).
# ---------------------------------------------------------------------------

def test_llm_route_does_not_call_any_tool() -> None:
    orchestrator, llm, web_tool, time_tool, date_tool = _build_orchestrator("llm")

    orchestrator.process("test question")

    assert web_tool.calls == []
    assert time_tool.calls == 0
    assert date_tool.calls == []
    assert len(llm.prompts) == 1


def test_web_route_calls_only_web_search_tool() -> None:
    orchestrator, llm, web_tool, time_tool, date_tool = _build_orchestrator("web")

    orchestrator.process("test question")

    assert web_tool.calls == ["test search"]
    assert time_tool.calls == 0
    assert date_tool.calls == []


def test_time_route_calls_only_time_tool() -> None:
    orchestrator, llm, web_tool, time_tool, date_tool = _build_orchestrator("time")

    orchestrator.process("test question")

    assert time_tool.calls == 1
    assert web_tool.calls == []
    assert date_tool.calls == []


def test_date_route_calls_only_date_tool_and_skips_llm() -> None:
    orchestrator, llm, web_tool, time_tool, date_tool = _build_orchestrator("date")

    orchestrator.process("test question")

    assert date_tool.calls == ["test question"]
    assert web_tool.calls == []
    assert time_tool.calls == 0
    assert llm.prompts == []


def test_web_route_falls_back_gracefully_when_search_tool_fails() -> None:
    """Previously untested: WebSearchTool now reports operational failures via
    ToolResult(success=False) instead of raising RuntimeError, and the
    orchestrator must still degrade to the same fallback answer without
    calling the LLM."""
    llm = FakeLLM()
    web_tool = FakeWebSearchTool(fail=True, fail_error="Tavily request failed due to network or timeout")
    time_tool = FakeTimeTool()
    date_tool = FakeDateTool()
    orchestrator = AgentOrchestrator(
        llm_client=llm,
        router=FakeRouter(RouterDecision(
            needs_web=True,
            reason="web route",
            search_query="test search",
            route="web",  # type: ignore[arg-type]
        )),
        web_search_tool=web_tool,  # type: ignore[arg-type]
        time_tool=time_tool,  # type: ignore[arg-type]
        date_tool=date_tool,  # type: ignore[arg-type]
    )

    result = orchestrator.process("test question")

    assert result.used_web is True
    assert result.sources == []
    assert "could not complete the web search" in result.answer
    assert len(llm.prompts) == 0  # LLM must not be called when search fails
    assert web_tool.calls == ["test search"]


# ---------------------------------------------------------------------------
# ToolRegistry integration: DI flows through the registry, not per-tool
# attributes, and a prebuilt registry works exactly the same way.
# ---------------------------------------------------------------------------

def test_orchestrator_registers_injected_tools_in_a_tool_registry() -> None:
    orchestrator, _llm, web_tool, time_tool, date_tool = _build_orchestrator("llm")

    assert isinstance(orchestrator.tools, ToolRegistry)
    assert orchestrator.tools.get("web_search") is web_tool
    assert orchestrator.tools.get("time") is time_tool
    assert orchestrator.tools.get("date") is date_tool


def test_orchestrator_accepts_a_prebuilt_tool_registry() -> None:
    """DI also works by handing the orchestrator a fully-built registry
    directly, proving tool execution goes through whatever registry it is
    given rather than through hardcoded per-tool attributes."""
    llm = FakeLLM()
    web_tool = FakeWebSearchTool()
    time_tool = FakeTimeTool()
    date_tool = FakeDateTool()

    registry = ToolRegistry()
    registry.register(web_tool)
    registry.register(time_tool)
    registry.register(date_tool)

    orchestrator = AgentOrchestrator(
        llm_client=llm,
        router=FakeRouter(RouterDecision(
            needs_web=False, reason="time route", search_query="", route="time",  # type: ignore[arg-type]
        )),
        tool_registry=registry,
    )

    result = orchestrator.process("test question")

    assert orchestrator.tools is registry
    assert result.used_time is True
    assert time_tool.calls == 1
    assert web_tool.calls == []
    assert date_tool.calls == []


def test_empty_message_raises_value_error() -> None:
    orchestrator, _llm, _web_tool, _time_tool, _date_tool = _build_orchestrator("llm")

    with pytest.raises(ValueError):
        orchestrator.process("   ")
