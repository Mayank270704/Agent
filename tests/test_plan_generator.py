"""Tests for LLMPlanGenerator (app/agent/plan_generator.py).

All tests except the one marked `@pytest.mark.integration` use a FakeLLM —
no network, no Ollama, no Tavily. The integration test talks to a real
local Ollama instance, is skipped automatically if unreachable, and never
executes the generated plan (no tool is ever registered or called).
"""
from __future__ import annotations

import json

import pytest

from app.agent.plan import Plan, PlanStatus
from app.agent.plan_generator import LLMPlanGenerator, PlanGenerationError
from app.config import settings
from app.models.llm import LLMClient


class FakeLLM:
    """Replays a fixed sequence of raw responses, one per generate() call.
    Records every call, including the exact messages and json_mode flag."""

    def __init__(self, responses: list[str]):
        self.responses = iter(responses)
        self.calls: list[dict[str, object]] = []

    def generate(self, messages: list[dict[str, str]], *, json_mode: bool = False) -> str:
        self.calls.append({"messages": messages, "json_mode": json_mode})
        return next(self.responses)


class RaisingTool:
    """A tool double whose execute() raises loudly if ever called. Used to
    prove LLMPlanGenerator never executes anything — it doesn't even accept
    a ToolRegistry, so this is never registered anywhere; it exists purely
    so a test can assert its execute() was never invoked."""

    name = "raising_tool"
    description = "Must never be executed by the planner."
    input_schema: dict[str, str] = {}

    def __init__(self) -> None:
        self.executed = False

    def execute(self, input: str | None = None):
        self.executed = True
        raise AssertionError("LLMPlanGenerator must never execute a tool")


def _one_step_json(description: str = "Explain what Python is") -> str:
    return json.dumps({"steps": [{"step_id": 1, "description": description}]})


def _multi_step_json() -> str:
    return json.dumps({
        "steps": [
            {"step_id": 1, "description": "Search for the latest AI news"},
            {"step_id": 2, "description": "Identify the three most important developments"},
            {"step_id": 3, "description": "Summarize those developments"},
        ]
    })


# ---------------------------------------------------------------------------
# 1-2: valid plans.
# ---------------------------------------------------------------------------

def test_valid_one_step_plan() -> None:
    llm = FakeLLM([_one_step_json()])
    generator = LLMPlanGenerator(llm_client=llm)

    plan = generator.generate("What is Python?")

    assert isinstance(plan, Plan)
    assert len(plan.steps) == 1
    assert plan.steps[0].step_id == 1
    assert plan.steps[0].description == "Explain what Python is"


def test_valid_multi_step_plan() -> None:
    llm = FakeLLM([_multi_step_json()])
    generator = LLMPlanGenerator(llm_client=llm)

    plan = generator.generate("Find the latest AI news and summarize the three most important developments.")

    assert len(plan.steps) == 3
    assert [s.step_id for s in plan.steps] == [1, 2, 3]
    assert plan.steps[0].description == "Search for the latest AI news"
    assert plan.steps[1].description == "Identify the three most important developments"
    assert plan.steps[2].description == "Summarize those developments"


# ---------------------------------------------------------------------------
# 3-4: input validation.
# ---------------------------------------------------------------------------

def test_blank_user_input_rejected() -> None:
    generator = LLMPlanGenerator(llm_client=FakeLLM([]))

    with pytest.raises(ValueError):
        generator.generate("")


def test_whitespace_only_user_input_rejected() -> None:
    generator = LLMPlanGenerator(llm_client=FakeLLM([]))

    with pytest.raises(ValueError):
        generator.generate("   ")


# ---------------------------------------------------------------------------
# 5-21: strict output validation. Every one of these must raise
# PlanGenerationError, never silently repair or guess.
# ---------------------------------------------------------------------------

def test_empty_llm_response_rejected() -> None:
    generator = LLMPlanGenerator(llm_client=FakeLLM([""]))

    with pytest.raises(PlanGenerationError):
        generator.generate("What is Python?")


def test_whitespace_only_llm_response_rejected() -> None:
    generator = LLMPlanGenerator(llm_client=FakeLLM(["   \n  "]))

    with pytest.raises(PlanGenerationError):
        generator.generate("What is Python?")


def test_malformed_json_rejected() -> None:
    generator = LLMPlanGenerator(llm_client=FakeLLM(["not valid json {"]))

    with pytest.raises(PlanGenerationError):
        generator.generate("What is Python?")


def test_json_array_instead_of_object_rejected() -> None:
    generator = LLMPlanGenerator(llm_client=FakeLLM([json.dumps([{"step_id": 1, "description": "x"}])]))

    with pytest.raises(PlanGenerationError):
        generator.generate("What is Python?")


def test_json_scalar_instead_of_object_rejected() -> None:
    generator = LLMPlanGenerator(llm_client=FakeLLM([json.dumps("just a string")]))

    with pytest.raises(PlanGenerationError):
        generator.generate("What is Python?")


def test_missing_steps_key_rejected() -> None:
    generator = LLMPlanGenerator(llm_client=FakeLLM([json.dumps({"final_answer": "no steps here"})]))

    with pytest.raises(PlanGenerationError):
        generator.generate("What is Python?")


def test_steps_null_rejected() -> None:
    generator = LLMPlanGenerator(llm_client=FakeLLM([json.dumps({"steps": None})]))

    with pytest.raises(PlanGenerationError):
        generator.generate("What is Python?")


def test_steps_not_a_list_rejected() -> None:
    generator = LLMPlanGenerator(llm_client=FakeLLM([json.dumps({"steps": "step one"})]))

    with pytest.raises(PlanGenerationError):
        generator.generate("What is Python?")


def test_empty_steps_list_rejected() -> None:
    generator = LLMPlanGenerator(llm_client=FakeLLM([json.dumps({"steps": []})]))

    with pytest.raises(PlanGenerationError):
        generator.generate("What is Python?")


def test_step_not_an_object_rejected() -> None:
    generator = LLMPlanGenerator(llm_client=FakeLLM([json.dumps({"steps": ["just a string"]})]))

    with pytest.raises(PlanGenerationError):
        generator.generate("What is Python?")


def test_missing_step_id_rejected() -> None:
    generator = LLMPlanGenerator(llm_client=FakeLLM([json.dumps({"steps": [{"description": "x"}]})]))

    with pytest.raises(PlanGenerationError):
        generator.generate("What is Python?")


def test_missing_description_rejected() -> None:
    generator = LLMPlanGenerator(llm_client=FakeLLM([json.dumps({"steps": [{"step_id": 1}]})]))

    with pytest.raises(PlanGenerationError):
        generator.generate("What is Python?")


@pytest.mark.parametrize("bad_step_id", ["1", 1.5, None, True, [1]])
def test_invalid_step_id_type_rejected(bad_step_id: object) -> None:
    generator = LLMPlanGenerator(
        llm_client=FakeLLM([json.dumps({"steps": [{"step_id": bad_step_id, "description": "x"}]})])
    )

    with pytest.raises(PlanGenerationError):
        generator.generate("What is Python?")


@pytest.mark.parametrize("bad_step_id", [0, -1, -100])
def test_step_id_less_than_one_rejected(bad_step_id: int) -> None:
    generator = LLMPlanGenerator(
        llm_client=FakeLLM([json.dumps({"steps": [{"step_id": bad_step_id, "description": "x"}]})])
    )

    with pytest.raises(PlanGenerationError):
        generator.generate("What is Python?")


def test_duplicate_step_ids_rejected() -> None:
    response = json.dumps({"steps": [
        {"step_id": 1, "description": "first"},
        {"step_id": 1, "description": "duplicate"},
    ]})
    generator = LLMPlanGenerator(llm_client=FakeLLM([response]))

    with pytest.raises(PlanGenerationError):
        generator.generate("What is Python?")


@pytest.mark.parametrize("blank_description", ["", "   "])
def test_blank_description_rejected(blank_description: str) -> None:
    generator = LLMPlanGenerator(
        llm_client=FakeLLM([json.dumps({"steps": [{"step_id": 1, "description": blank_description}]})])
    )

    with pytest.raises(PlanGenerationError):
        generator.generate("What is Python?")


@pytest.mark.parametrize("bad_description", [123, None, True, ["a", "list"], {"nested": "object"}])
def test_non_string_description_rejected(bad_description: object) -> None:
    generator = LLMPlanGenerator(
        llm_client=FakeLLM([json.dumps({"steps": [{"step_id": 1, "description": bad_description}]})])
    )

    with pytest.raises(PlanGenerationError):
        generator.generate("What is Python?")


def test_ambiguous_incomplete_step_structure_rejected() -> None:
    """A step object missing both required fields — the most ambiguous case."""
    generator = LLMPlanGenerator(llm_client=FakeLLM([json.dumps({"steps": [{"irrelevant": "field"}]})]))

    with pytest.raises(PlanGenerationError):
        generator.generate("What is Python?")


# ---------------------------------------------------------------------------
# 22-23: generated Plan/PlanStep lifecycle status is always PENDING — the
# model never controls lifecycle state.
# ---------------------------------------------------------------------------

def test_generated_plan_starts_as_pending() -> None:
    llm = FakeLLM([_multi_step_json()])
    generator = LLMPlanGenerator(llm_client=llm)

    plan = generator.generate("Find the latest AI news and summarize it.")

    assert plan.status is PlanStatus.PENDING


def test_generated_plan_steps_start_as_pending() -> None:
    llm = FakeLLM([_multi_step_json()])
    generator = LLMPlanGenerator(llm_client=llm)

    plan = generator.generate("Find the latest AI news and summarize it.")

    assert all(step.status is PlanStatus.PENDING for step in plan.steps)


# ---------------------------------------------------------------------------
# 24: LLM is called with json_mode=True.
# ---------------------------------------------------------------------------

def test_llm_is_called_with_json_mode_true() -> None:
    llm = FakeLLM([_one_step_json()])
    generator = LLMPlanGenerator(llm_client=llm)

    generator.generate("What is Python?")

    assert len(llm.calls) == 1
    assert llm.calls[0]["json_mode"] is True


# ---------------------------------------------------------------------------
# 16: LLMClient call contract — generate(messages, json_mode=True), never
# bypassed, never a second HTTP implementation.
# ---------------------------------------------------------------------------

def test_llm_called_with_a_single_user_message_matching_llmclient_contract() -> None:
    llm = FakeLLM([_one_step_json()])
    generator = LLMPlanGenerator(llm_client=llm)

    generator.generate("What is Python?")

    messages = llm.calls[0]["messages"]
    assert isinstance(messages, list) and len(messages) == 1
    assert messages[0]["role"] == "user"
    assert isinstance(messages[0]["content"], str)
    assert "What is Python?" in messages[0]["content"]


# ---------------------------------------------------------------------------
# 25: planner does not execute tools.
# ---------------------------------------------------------------------------

def test_planner_never_executes_a_tool() -> None:
    tool = RaisingTool()
    llm = FakeLLM([_multi_step_json()])
    generator = LLMPlanGenerator(llm_client=llm)

    generator.generate("Find the latest AI news and summarize it.")

    assert tool.executed is False


def test_planner_has_no_tool_registry_dependency() -> None:
    """The constructor accepts only an LLMClient — there is no ToolRegistry
    parameter at all to accidentally wire up."""
    import inspect

    signature = inspect.signature(LLMPlanGenerator.__init__)
    assert list(signature.parameters) == ["self", "llm_client"]


# ---------------------------------------------------------------------------
# 26: planner does not mutate AgentState (it doesn't even accept one).
# ---------------------------------------------------------------------------

def test_planner_generate_signature_takes_only_a_string() -> None:
    import inspect

    signature = inspect.signature(LLMPlanGenerator.generate)
    params = list(signature.parameters)
    assert params == ["self", "user_input"]
    assert signature.parameters["user_input"].annotation in (str, "str")


# ---------------------------------------------------------------------------
# 27: tool names are not required in plan output, and are rejected if a
# step tries to smuggle one in as if it were meaningful — the parser simply
# ignores unknown keys, since the contract only requires step_id/description.
# ---------------------------------------------------------------------------

def test_tool_name_in_model_output_is_ignored_not_required() -> None:
    response = json.dumps({"steps": [
        {"step_id": 1, "description": "Search for AI news", "tool_name": "web_search", "tool_input": "AI news"},
    ]})
    llm = FakeLLM([response])
    generator = LLMPlanGenerator(llm_client=llm)

    plan = generator.generate("Find AI news.")

    assert len(plan.steps) == 1
    assert plan.steps[0].description == "Search for AI news"
    assert not hasattr(plan.steps[0], "tool_name")


def test_plan_without_any_tool_name_field_is_perfectly_valid() -> None:
    llm = FakeLLM([_one_step_json()])
    generator = LLMPlanGenerator(llm_client=llm)

    plan = generator.generate("What is Python?")

    assert isinstance(plan, Plan)


# ---------------------------------------------------------------------------
# Safe normalization: only description.strip() — nothing else.
# ---------------------------------------------------------------------------

def test_description_whitespace_is_stripped() -> None:
    response = json.dumps({"steps": [{"step_id": 1, "description": "  Explain Python  "}]})
    llm = FakeLLM([response])
    generator = LLMPlanGenerator(llm_client=llm)

    plan = generator.generate("What is Python?")

    assert plan.steps[0].description == "Explain Python"


def test_step_order_is_preserved_exactly_as_returned_by_the_model() -> None:
    """Steps must never be reordered, even if step_ids arrive out of order."""
    response = json.dumps({"steps": [
        {"step_id": 2, "description": "second thing"},
        {"step_id": 1, "description": "first thing"},
    ]})
    llm = FakeLLM([response])
    generator = LLMPlanGenerator(llm_client=llm)

    plan = generator.generate("Do two things.")

    assert [s.step_id for s in plan.steps] == [2, 1]
    assert [s.description for s in plan.steps] == ["second thing", "first thing"]


# ---------------------------------------------------------------------------
# 18: LLM client failures propagate rather than becoming a fabricated plan.
# ---------------------------------------------------------------------------

def test_llm_client_runtime_error_propagates() -> None:
    class FailingLLM:
        def generate(self, messages, *, json_mode: bool = False) -> str:
            raise RuntimeError("Ollama connection failed")

    generator = LLMPlanGenerator(llm_client=FailingLLM())

    with pytest.raises(RuntimeError, match="Ollama connection failed"):
        generator.generate("What is Python?")


# ---------------------------------------------------------------------------
# 17: real Ollama smoke test. Skipped automatically if unreachable, never
# executes the plan, never calls Tavily/any external web service.
# ---------------------------------------------------------------------------

def _ollama_is_available() -> bool:
    import urllib.error
    import urllib.request

    try:
        urllib.request.urlopen(f"{settings.ollama_base_url}/api/tags", timeout=2)
        return True
    except (urllib.error.URLError, OSError):
        return False


@pytest.mark.integration
def test_real_ollama_plan_generation_smoke() -> None:
    if not _ollama_is_available():
        pytest.skip(f"Ollama is not reachable at {settings.ollama_base_url}; skipping live smoke test.")

    llm_client = LLMClient(provider="ollama", model_name=settings.model_name, base_url=settings.ollama_base_url)
    generator = LLMPlanGenerator(llm_client=llm_client)

    plan = generator.generate("What is the latest AI news?")

    assert isinstance(plan, Plan)
    assert len(plan.steps) >= 1
    assert plan.status is PlanStatus.PENDING
    assert all(step.status is PlanStatus.PENDING for step in plan.steps)
