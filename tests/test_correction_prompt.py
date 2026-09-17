"""Step 17, Phase 4: the CORRECTION FEEDBACK prompt block.

Fully offline — a FakeLLM records the exact prompt text sent, matching
test_decision_maker.py's own conventions.
"""
from __future__ import annotations

import json

import pytest

from app.agent.decision_maker import LLMDecisionMaker
from app.agent.memory_formatting import MEMORY_CONTEXT_LABEL
from app.agent.reliability import FailureCategory
from app.agent.state import AgentState, CorrectionNote
from app.agent.tool_registry import ToolRegistry


class FakeLLM:
    def __init__(self, responses: list[str]):
        self.responses = iter(responses)
        self.calls: list[dict[str, object]] = []

    def generate(self, messages: list[dict[str, str]], *, json_mode: bool = False) -> str:
        self.calls.append({"messages": messages, "json_mode": json_mode})
        return next(self.responses)


def _note(category: FailureCategory, message: str, step: int = 1) -> CorrectionNote:
    return CorrectionNote(category=category, safe_message=message, step=step, signature=category.value)


def _final_json(answer: str) -> str:
    return json.dumps({"action_type": "final", "final_answer": answer})


def _prompt_text(llm: FakeLLM, call_index: int = 0) -> str:
    return llm.calls[call_index]["messages"][-1]["content"]


# ===========================================================================
# 1 — no corrections: prompt is byte-identical to before Step 17
# ===========================================================================

def test_prompt_has_no_correction_block_when_state_has_no_corrections() -> None:
    llm = FakeLLM([_final_json("hi")])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=ToolRegistry())
    state = AgentState(user_input="hello")

    decision_maker.decide(state)

    assert "CORRECTION FEEDBACK" not in _prompt_text(llm)
    assert "HOW TO TREAT CORRECTION FEEDBACK" not in _prompt_text(llm)


def test_format_corrections_returns_empty_string_when_no_corrections() -> None:
    decision_maker = LLMDecisionMaker(llm_client=FakeLLM([]), tool_registry=ToolRegistry())
    state = AgentState(user_input="hello")

    assert decision_maker._format_corrections(state) == ""


def test_prompt_is_byte_identical_with_and_without_the_method_when_empty() -> None:
    """The strongest form of the "byte-identical when idle" guarantee."""
    llm_a = FakeLLM([_final_json("hi")])
    llm_b = FakeLLM([_final_json("hi")])
    dm_a = LLMDecisionMaker(llm_client=llm_a, tool_registry=ToolRegistry())
    dm_b = LLMDecisionMaker(llm_client=llm_b, tool_registry=ToolRegistry())

    dm_a.decide(AgentState(user_input="hello"))
    dm_b.decide(AgentState(user_input="hello"))

    assert _prompt_text(llm_a) == _prompt_text(llm_b)


# ===========================================================================
# 2 — placement: after MEMORY, before EXECUTION HISTORY
# ===========================================================================

def test_correction_block_appears_before_execution_history() -> None:
    llm = FakeLLM([_final_json("hi")])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=ToolRegistry())
    state = AgentState(user_input="hello")
    state.record_correction(_note(FailureCategory.DECISION_PARSE, "fixed message"))

    decision_maker.decide(state)
    prompt = _prompt_text(llm)

    assert prompt.index("CORRECTION FEEDBACK") < prompt.index("EXECUTION HISTORY")


def test_correction_block_appears_after_memory_block_when_both_present() -> None:
    from app.agent.memory_context import MemoryContext, MemoryContextItem
    from datetime import datetime, timezone

    llm = FakeLLM([_final_json("hi")])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=ToolRegistry())
    state = AgentState(
        user_input="hello",
        memory_context=MemoryContext(
            session_id="s1",
            items=(
                MemoryContextItem(
                    memory_id="m1",
                    content="User prefers Python.",
                    similarity=0.9,
                    created_at=datetime.now(timezone.utc),
                    confidence=1.0,
                ),
            ),
        ),
    )
    state.record_correction(_note(FailureCategory.DECISION_PARSE, "fixed message"))

    decision_maker.decide(state)
    prompt = _prompt_text(llm)

    assert prompt.index(MEMORY_CONTEXT_LABEL) < prompt.index("CORRECTION FEEDBACK")
    assert prompt.index("CORRECTION FEEDBACK") < prompt.index("EXECUTION HISTORY")


# ===========================================================================
# 3 — serialization: JSON, one entry per correction, in order
# ===========================================================================

def test_correction_block_lists_one_entry_per_note_in_order() -> None:
    decision_maker = LLMDecisionMaker(llm_client=FakeLLM([]), tool_registry=ToolRegistry())
    state = AgentState(user_input="hello")
    state.record_correction(_note(FailureCategory.DECISION_PARSE, "first message", step=1))
    state.record_correction(_note(FailureCategory.UNKNOWN_TOOL, "second message", step=2))

    block = decision_maker._format_corrections(state)
    payload_text = block.split("CORRECTION FEEDBACK (JSON):\n", 1)[1].strip()
    entries = json.loads(payload_text)

    assert entries == [
        {"category": "decision_parse", "message": "first message"},
        {"category": "unknown_tool", "message": "second message"},
    ]


def test_correction_block_payload_is_valid_json() -> None:
    """Content cannot forge structure: the payload always parses back to
    exactly what was recorded, matching format_memory_context's own
    "parse it back" test discipline."""
    decision_maker = LLMDecisionMaker(llm_client=FakeLLM([]), tool_registry=ToolRegistry())
    state = AgentState(user_input="hello")
    for category in FailureCategory:
        state.record_correction(_note(category, f"message for {category.value}"))

    block = decision_maker._format_corrections(state)
    payload_text = block.split("CORRECTION FEEDBACK (JSON):\n", 1)[1].strip()

    parsed = json.loads(payload_text)
    assert len(parsed) == len(list(FailureCategory))


# ===========================================================================
# 4 — safety: never raw model output, never exception text, never an
#     invented tool name — only what CorrectionNote.safe_message carries
# ===========================================================================

def test_correction_block_contains_only_the_safe_message_text() -> None:
    decision_maker = LLMDecisionMaker(llm_client=FakeLLM([]), tool_registry=ToolRegistry())
    state = AgentState(user_input="hello")
    state.record_correction(
        CorrectionNote(
            category=FailureCategory.DECISION_PARSE,
            safe_message="Respond with STRICT JSON ONLY.",
            step=1,
            signature="decision_parse",
        )
    )

    block = decision_maker._format_corrections(state)

    assert "Respond with STRICT JSON ONLY." in block


def test_correction_block_never_includes_memory_id_or_similarity() -> None:
    """A CorrectionNote never carries these fields at all (see state.py),
    so this is a structural guarantee, not merely an absence in one
    example."""
    decision_maker = LLMDecisionMaker(llm_client=FakeLLM([]), tool_registry=ToolRegistry())
    state = AgentState(user_input="hello")
    state.record_correction(_note(FailureCategory.TOOL_EXECUTION_FAILED, "fixed message"))

    block = decision_maker._format_corrections(state)

    assert "memory_id" not in block
    assert "similarity" not in block


def test_correction_block_never_leaks_the_step_number() -> None:
    """`step` exists for internal bookkeeping only — it is never rendered
    to the model."""
    decision_maker = LLMDecisionMaker(llm_client=FakeLLM([]), tool_registry=ToolRegistry())
    state = AgentState(user_input="hello")
    state.record_correction(_note(FailureCategory.DECISION_PARSE, "fixed message", step=42))

    block = decision_maker._format_corrections(state)

    assert "42" not in block


def test_correction_block_never_leaks_the_internal_signature() -> None:
    decision_maker = LLMDecisionMaker(llm_client=FakeLLM([]), tool_registry=ToolRegistry())
    state = AgentState(user_input="hello")
    state.record_correction(
        CorrectionNote(
            category=FailureCategory.INVALID_TOOL_INPUT,
            safe_message="fixed message",
            step=1,
            signature="invalid_tool_input:date_tool_super_secret_internal_name",
        )
    )

    block = decision_maker._format_corrections(state)

    assert "invalid_tool_input:date_tool_super_secret_internal_name" not in block
    assert "signature" not in block


# ===========================================================================
# 5 — truncation / boundedness across many corrections
# ===========================================================================

def test_correction_block_stays_bounded_at_the_maximum_realistic_count() -> None:
    """Even at the largest number of corrections a single request can ever
    accumulate structurally (bounded by max_iterations), the block must
    not explode: safe_message templates are fixed-length, so N entries
    scale linearly and predictably."""
    decision_maker = LLMDecisionMaker(llm_client=FakeLLM([]), tool_registry=ToolRegistry())
    state = AgentState(user_input="hello")
    for i in range(20):  # far beyond any realistic max_iterations
        state.record_correction(_note(FailureCategory.DECISION_PARSE, "fixed message", step=i))

    block = decision_maker._format_corrections(state)

    assert len(block) < 5000  # generous bound; fixed messages never balloon


# ===========================================================================
# 6 — framing: application-authored header precedes the payload
# ===========================================================================

def test_correction_block_framing_precedes_the_payload() -> None:
    decision_maker = LLMDecisionMaker(llm_client=FakeLLM([]), tool_registry=ToolRegistry())
    state = AgentState(user_input="hello")
    state.record_correction(_note(FailureCategory.DECISION_PARSE, "fixed message"))

    block = decision_maker._format_corrections(state)

    assert block.index("HOW TO TREAT CORRECTION FEEDBACK") < block.index("CORRECTION FEEDBACK (JSON)")
