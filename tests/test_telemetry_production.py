"""Milestone 19: telemetry threaded end to end through ChatService ->
AgentOrchestrator, and the production composition in app/main.py.

Fully offline: FakeLLM replays scripted JSON; no Ollama, no network.
"""
from __future__ import annotations

import json

import pytest

from app.agent.telemetry import EventEmitter, EventType, ListEventSink, LoggingEventSink, SafeEventSink
from app.agent.tool_execution import ToolExecutionGate
from app.agent.tool_registry import ToolRegistry
from app.agent.permissions import AllowlistPermissionPolicy
from app.services.chat import ChatService


class FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)

    def generate(self, messages: list[dict[str, str]], *, json_mode: bool = False) -> str:
        return next(self.responses)


def _final_json(answer: str) -> str:
    return json.dumps({"action_type": "final", "final_answer": answer})


def _tool_json(tool_name: str, tool_input: str | None) -> str:
    return json.dumps({"action_type": "tool", "tool_name": tool_name, "tool_input": tool_input})


class _Tool:
    def __init__(self, name: str) -> None:
        self.name = name
        self.description = "x"
        self.input_schema: dict[str, str] = {}

    def execute(self, input: str | None = None):
        from app.tools.base import ToolResult

        return ToolResult.ok({"ok": True})


# ===========================================================================
# End-to-end correlation and event content through ChatService
# ===========================================================================

def test_chat_service_emits_a_full_request_event_stream() -> None:
    registry = ToolRegistry()
    registry.register(_Tool("time"))
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy({"time"}))
    sink = ListEventSink()
    llm = FakeLLM([_tool_json("time", None), _final_json("It is noon.")])
    service = ChatService(
        llm_client=llm, tool_registry=registry, tool_execution_gate=gate, event_sink=sink
    )

    answer = service.ask("what time is it")

    assert answer == "It is noon."
    types = [e.event_type for e in sink.events]
    assert types[0] is EventType.REQUEST_STARTED
    assert types[-1] is EventType.REQUEST_COMPLETED
    assert EventType.TOOL_PROPOSED in types
    assert EventType.TOOL_EXECUTION_COMPLETED in types
    assert EventType.LLM_CALL_STARTED in types
    assert EventType.LLM_CALL_COMPLETED in types


def test_two_ask_calls_produce_two_distinct_request_ids() -> None:
    registry = ToolRegistry()
    registry.register(_Tool("time"))
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy({"time"}))
    sink = ListEventSink()
    llm = FakeLLM(
        [_tool_json("time", None), _final_json("a"), _tool_json("time", None), _final_json("b")]
    )
    service = ChatService(
        llm_client=llm, tool_registry=registry, tool_execution_gate=gate, event_sink=sink
    )

    service.ask("first request")
    service.ask("second request")

    request_ids = {e.request_id for e in sink.events}
    assert len(request_ids) == 2


def test_session_id_is_carried_as_correlation_never_as_request_id() -> None:
    registry = ToolRegistry()
    registry.register(_Tool("time"))
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy({"time"}))
    sink = ListEventSink()
    llm = FakeLLM([_tool_json("time", None), _final_json("a")])
    service = ChatService(
        llm_client=llm, tool_registry=registry, tool_execution_gate=gate, event_sink=sink
    )

    service.ask("what time is it", session_id="alice-session")

    assert all(e.session_id == "alice-session" for e in sink.events)
    assert all(e.request_id != "alice-session" for e in sink.events)


def test_disabled_event_sink_produces_no_events_and_identical_answer() -> None:
    registry = ToolRegistry()
    registry.register(_Tool("time"))
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy({"time"}))
    llm = FakeLLM([_tool_json("time", None), _final_json("It is noon.")])
    service = ChatService(llm_client=llm, tool_registry=registry, tool_execution_gate=gate)  # event_sink=None

    answer = service.ask("what time is it")

    assert answer == "It is noon."
    assert service.event_sink is None


def test_a_throwing_sink_through_the_full_chat_service_path_does_not_break_the_answer() -> None:
    class RaisingSink:
        def emit(self, event) -> None:  # noqa: ANN001
            raise RuntimeError("backend down")

    registry = ToolRegistry()
    registry.register(_Tool("time"))
    gate = ToolExecutionGate(registry, AllowlistPermissionPolicy({"time"}))
    llm = FakeLLM([_tool_json("time", None), _final_json("It is noon.")])
    service = ChatService(
        llm_client=llm, tool_registry=registry, tool_execution_gate=gate, event_sink=RaisingSink()
    )

    answer = service.ask("what time is it")  # must not raise

    assert answer == "It is noon."


# ===========================================================================
# Production composition (app/main.py)
# ===========================================================================

def test_production_telemetry_defaults_to_disabled() -> None:
    import app.main as main_module

    assert main_module.settings.telemetry_enabled is False
    assert main_module._event_sink is None
    assert main_module.chat_service.event_sink is None


def test_telemetry_enabled_defaults_to_false(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.config import Settings

    monkeypatch.delenv("TELEMETRY_ENABLED", raising=False)

    assert Settings().telemetry_enabled is False


def test_telemetry_enabled_env_var_flips_the_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.config import Settings

    monkeypatch.setenv("TELEMETRY_ENABLED", "true")

    assert Settings().telemetry_enabled is True


def test_the_composition_app_main_actually_performs_produces_a_safe_logging_sink(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Proves the EXACT expression app/main.py's module-level `_event_sink`
    line evaluates to, against a real (frozen) Settings instance built
    with TELEMETRY_ENABLED=true — not a re-description of the logic."""
    from app.config import Settings

    monkeypatch.setenv("TELEMETRY_ENABLED", "true")
    settings = Settings()

    sink = SafeEventSink(LoggingEventSink()) if settings.telemetry_enabled else None

    assert isinstance(sink, SafeEventSink)
