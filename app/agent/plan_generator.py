"""Converts a user request into a structured Plan, using an LLM.

    User request -> LLMPlanGenerator -> Plan

This module produces plans; it never executes them, and it never decides
which tool (if any) will carry a step out. LLMPlanGenerator:
- does NOT execute tools — it has no ToolRegistry dependency at all.
- does NOT mutate AgentState — its only input is a plain string
  (`user_input`), its only output is a `Plan`; it doesn't even accept an
  AgentState.
- does NOT call AgentLoop.
- does NOT produce a final user-facing answer.
- does NOT decide which tool executes a step, or how/when — see
  app/agent/plan.py's `PlanStep`: it has no tool_name/tool_input field, on
  purpose.

Planner vs. LLMDecisionMaker (app/agent/decision_maker.py) — same
underlying LLM and provider, entirely different job:
- LLMPlanGenerator describes WHAT needs to happen: a plan is a static list
  of meaningful steps, decided once, up front, from the user's request
  alone.
- LLMDecisionMaker decides WHAT ACTION should happen NOW, called
  *repeatedly* by AgentLoop, informed by growing execution history — it
  remains the only thing that ever chooses a tool or produces a final
  answer.
- A Tool's `execute()` (app/tools/base.py) is HOW a chosen action actually
  runs.

Planner / action-decider / tool stay three separate concerns:
    Planner        = WHAT needs to happen
    LLMDecisionMaker = WHAT ACTION should happen now
    Tool           = HOW the action is executed

Integration with AgentLoop/AgentState.plan is explicitly NOT done here (see
the Step 10 report). AgentLoop still does not read state.plan; nothing
calls LLMPlanGenerator except tests. That wiring is deliberately deferred.
"""
from __future__ import annotations

import json
import logging

from app.agent.plan import Plan, PlanStep
from app.models.llm import LLMClient

logger = logging.getLogger(__name__)


class PlanGenerationError(Exception):
    """Raised when the model's raw output cannot be parsed into a valid
    Plan. Mirrors DecisionParseError's rationale (app/agent/decision_maker.py):
    a distinct exception for "the LLM said something we can't trust or act
    on," kept separate from ValueError — ValueError is reserved for invalid
    *input* to generate() (a blank user_input), a precondition failure the
    caller could have avoided, not a planner-output failure."""


class LLMPlanGenerator:
    """Implements the PlanGenerator protocol (app/agent/plan.py) using an LLM."""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def generate(self, user_input: str) -> Plan:
        if user_input is None or not str(user_input).strip():
            raise ValueError("user_input cannot be empty.")

        prompt = self._build_prompt(str(user_input).strip())
        raw_response = self.llm.generate(
            [{"role": "user", "content": prompt}],
            json_mode=True,
        )
        return self._parse_plan(raw_response)

    # -- Prompt construction --------------------------------------------------

    def _build_prompt(self, user_input: str) -> str:
        return f"""
You are a planning component. Your ONLY job is to decompose the user's
request into a small, ordered sequence of meaningful steps. You do not
execute anything, choose tools, or answer the request yourself.

Rules:
- Return STRICT JSON ONLY: one JSON object, no markdown fences, no extra
  text, no chain-of-thought.
- The object must contain a "steps" array.
- Every step must contain an integer "step_id" and a concise "description".
- step_id values must start at 1, be unique, and reflect step order.
- Use the minimum number of meaningful steps necessary. A simple request
  needs only one step — do not manufacture filler steps like "analyze the
  question" or "verify the result" that add no real content.
- Multi-part requests may need multiple steps, one per distinct piece of
  work actually required.
- Do not include tool names, tool inputs, or any other execution detail.
- Do not provide the final answer.
- Only plan work relevant to the user's request.

Required shape:
{{"steps": [{{"step_id": 1, "description": "..."}}, {{"step_id": 2, "description": "..."}}]}}

USER REQUEST:
{user_input}
""".strip()

    # -- Output parsing / validation -------------------------------------------

    def _parse_plan(self, raw_response: str) -> Plan:
        cleaned = (raw_response or "").strip()
        if not cleaned:
            raise PlanGenerationError("The model returned an empty response.")

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise PlanGenerationError(f"Model output was not valid JSON: {exc}") from exc

        if not isinstance(parsed, dict):
            raise PlanGenerationError("Model output was not a JSON object.")

        steps_raw = parsed.get("steps")
        if steps_raw is None:
            raise PlanGenerationError('Model output was missing "steps".')
        if not isinstance(steps_raw, list):
            raise PlanGenerationError('Model output\'s "steps" was not a list.')
        if not steps_raw:
            raise PlanGenerationError('Model output\'s "steps" list was empty.')

        # Order is preserved exactly as the model produced it — never sorted
        # or reordered (see the module docstring's "WHAT, not HOW" boundary
        # and the Step 10 report's normalization rules).
        steps = [self._parse_step(index, raw_step) for index, raw_step in enumerate(steps_raw)]

        try:
            return Plan(steps=steps)
        except ValueError as exc:  # e.g. duplicate step_id — Plan's own invariant, reused rather than duplicated
            raise PlanGenerationError(str(exc)) from exc

    def _parse_step(self, index: int, raw_step: object) -> PlanStep:
        if not isinstance(raw_step, dict):
            raise PlanGenerationError(f"Step at position {index} was not a JSON object.")

        step_id = raw_step.get("step_id")
        if not isinstance(step_id, int) or isinstance(step_id, bool):
            raise PlanGenerationError(
                f"Step at position {index} had a missing or non-integer step_id: {step_id!r}."
            )
        if step_id < 1:
            raise PlanGenerationError(f"Step at position {index} had step_id={step_id}, which must be >= 1.")

        description = raw_step.get("description")
        if not isinstance(description, str) or not description.strip():
            raise PlanGenerationError(f"Step {step_id} had a missing or blank description.")

        try:
            # The only normalization performed anywhere in this class:
            # stripping incidental whitespace from an otherwise-valid
            # description. Nothing is invented, reordered, merged, or split.
            return PlanStep(step_id=step_id, description=description.strip())
        except ValueError as exc:  # defense in depth; should be unreachable given the checks above
            raise PlanGenerationError(str(exc)) from exc
