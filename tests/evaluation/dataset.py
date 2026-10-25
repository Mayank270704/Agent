"""Step 29 — the permanent, versioned end-to-end capability-evaluation
dataset (v1).

This is a NEW, broader harness than the existing routing-decision
evaluators (app/agent/evaluation.py, app/agent/execution_evaluation.py):
those grade a single `decide()` call or a scripted-LLM full execution —
neither exercises the real HTTP layer, the real production `chat_service`
singleton, real Ollama, or real Tavily, and neither covers security,
memory, timeout, or telemetry at all. Nothing here duplicates their
functionality; `runner.py` reuses `app/agent/telemetry.py`'s `ListEventSink`
directly rather than reinventing an event capture mechanism.

Every case has EXPLICIT ground truth (Step 29 Phase 3) — never "the model's
own opinion of itself." `final_answer_check`, where set, is a deterministic,
structural predicate (substring/regex over the reply text) — never exact
string equality against LLM-generated prose, and never a second LLM-as-
judge call (out of scope: Step 29 adds no new model). Where no deterministic
predicate can honestly characterize a correct answer to a genuinely
open-ended question (e.g. "Explain how photosynthesis works"),
`final_answer_check` is deliberately left `None` and the case is graded
only on structural properties (non-empty, no error leakage) — this is
documented explicitly, not silently glossed over, in the Step 29 report.

Categories A-H are MODEL/TOOL-DEPENDENT: run through the real production
path (FastAPI TestClient -> real `app.main.chat_service` -> real
AgentOrchestrator -> real AgentLoop -> real LLMDecisionMaker -> real Ollama
-> real ToolExecutionGate -> real Tool, including real Tavily for selected
web_search calls), each case run for multiple trials (see runner.py) to
surface model variance, exactly as Step 27 established.

Categories I-N are MECHANISM-DEPENDENT, not model-quality-dependent
(tool-input validation, failure recovery, session memory, security,
timeout, telemetry): these are deterministic by construction, so a single
trial is sufficient and appropriate — running them 3x would only measure
Python's own determinism, not anything meaningful. Where a mechanism
already has exhaustive, passing coverage elsewhere in this suite (most of
L — security), this dataset defines a SMALL, REPRESENTATIVE case set that
proves the metric end-to-end through the real production singleton, and
the module docstring points at the exhaustive file(s) rather than
re-deriving that coverage here — see each category's own comment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class EvalCase:
    """One evaluation case with explicit, pre-declared ground truth.

    `requires_tool`/`expected_tool`/`requires_web` describe what a
    well-behaved agent SHOULD do; `final_answer_check`, if set, is a
    deterministic predicate over the final reply text. `category` is one
    of the Step 29 Phase 2 letters (A-N). `session_id`, when set, lets a
    case opt into session-scoped behavior (episodic/semantic memory,
    session_id-keyed conversation) — categories other than K/session-
    dependent L cases leave it `None` (anonymous, request-scoped).
    """

    case_id: str
    category: str
    message: str
    requires_tool: bool
    expected_tool: str | None = None
    requires_web: bool = False
    final_answer_check: Callable[[str], bool] | None = None
    session_id: str | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.requires_tool and not self.expected_tool:
            raise ValueError(f"case {self.case_id!r}: requires_tool=True needs an expected_tool.")
        if not self.requires_tool and self.expected_tool:
            raise ValueError(f"case {self.case_id!r}: expected_tool set without requires_tool=True.")
        if self.requires_web and self.expected_tool != "web_search":
            raise ValueError(f"case {self.case_id!r}: requires_web=True must pair with expected_tool='web_search'.")


def _contains_any(*substrings: str) -> Callable[[str], bool]:
    """Deterministic, case-insensitive "reply mentions one of these"
    check — the simplest honest predicate for a fact with few correct
    surface forms (a number, a name, a fixed answer)."""

    def check(reply: str) -> bool:
        lowered = reply.lower()
        return any(s.lower() in lowered for s in substrings)

    return check


def _non_empty_reasonable_length(reply: str) -> bool:
    """The weakest honest predicate this dataset uses: the reply is
    non-blank and long enough to be an actual answer, not an empty string
    or a one-character stub. Used ONLY for genuinely open-ended cases
    where no stronger deterministic check exists — see the module
    docstring's note on Category A's "final-answer correctness" caveat."""
    return isinstance(reply, str) and len(reply.strip()) >= 10


# ---------------------------------------------------------------------------
# A. Direct answers — should NOT require a tool.
# ---------------------------------------------------------------------------
CATEGORY_A: list[EvalCase] = [
    EvalCase("a1", "A", "What is the capital of France?", False, final_answer_check=_contains_any("paris")),
    EvalCase("a2", "A", "What is 12 multiplied by 8?", False, final_answer_check=_contains_any("96")),
    EvalCase(
        "a3", "A", "If a train travels 60 miles in 2 hours, what is its average speed in mph?", False,
        final_answer_check=_contains_any("30"),
    ),
    EvalCase(
        "a4", "A", "Explain what a for loop is in programming, in one or two sentences.", False,
        final_answer_check=_non_empty_reasonable_length,
        notes="Open-ended explanation; graded structurally only (non-empty, reasonable length) — see module docstring.",
    ),
    EvalCase(
        "a5", "A",
        "Reply with exactly the single word: acknowledged",
        False,
        final_answer_check=_contains_any("acknowledged"),
        notes="Instruction-following check.",
    ),
]

# ---------------------------------------------------------------------------
# B. Time — requires the current time.
# ---------------------------------------------------------------------------
CATEGORY_B: list[EvalCase] = [
    EvalCase("b1", "B", "What time is it right now?", True, "time"),
    EvalCase("b2", "B", "Can you tell me the current time?", True, "time"),
    EvalCase("b3", "B", "What's the time?", True, "time"),
]

# ---------------------------------------------------------------------------
# C. Date — requires the current date/day.
# ---------------------------------------------------------------------------
CATEGORY_C: list[EvalCase] = [
    EvalCase("c1", "C", "What is today's date?", True, "time", notes="TimeTool's output includes today's date — see app/agent/router.py's established E/time mapping."),
    EvalCase("c2", "C", "What day of the week is it today?", True, "time"),
    EvalCase("c3", "C", "What year is it?", True, "time"),
]

# ---------------------------------------------------------------------------
# D. Explicit web search commands — deterministically routed (Step 26/28).
# ---------------------------------------------------------------------------
CATEGORY_D: list[EvalCase] = [
    EvalCase("d1", "D", "Search the web for the current version of Python.", True, "web_search", requires_web=True),
    EvalCase("d2", "D", "Search online for the latest Django release.", True, "web_search", requires_web=True),
    EvalCase("d3", "D", "Find the current UK prime minister on the internet.", True, "web_search", requires_web=True),
    EvalCase("d4", "D", "Look up the current price of gold online.", True, "web_search", requires_web=True),
]

# ---------------------------------------------------------------------------
# E. Current information — no explicit "search" wording; the model must
# choose to use web_search on its own judgment (advisory hint only, Step 27
# — NOT deterministically routed). Genuinely the hardest, most model-
# dependent category in this dataset.
# ---------------------------------------------------------------------------
CATEGORY_E: list[EvalCase] = [
    EvalCase("e1", "E", "What are the latest developments in AI this year?", True, "web_search", requires_web=True),
    EvalCase("e2", "E", "What is the current price of Bitcoin?", True, "web_search", requires_web=True),
    EvalCase("e3", "E", "What is the latest version of the Python programming language?", True, "web_search", requires_web=True),
    EvalCase("e4", "E", "What's the current status of the James Webb Space Telescope mission?", True, "web_search", requires_web=True),
]

# ---------------------------------------------------------------------------
# F. Static/historical — should NOT trigger web_search.
# ---------------------------------------------------------------------------
CATEGORY_F: list[EvalCase] = [
    EvalCase("f1", "F", "What year was Python first released?", False, final_answer_check=_non_empty_reasonable_length),
    EvalCase("f2", "F", "When was the Eiffel Tower built?", False, final_answer_check=_non_empty_reasonable_length),
    EvalCase("f3", "F", "Who wrote Romeo and Juliet?", False, final_answer_check=_contains_any("shakespeare")),
    EvalCase("f4", "F", "What is the boiling point of water in Celsius at sea level?", False, final_answer_check=_contains_any("100")),
]

# ---------------------------------------------------------------------------
# G. Local/technical context — "search for"/"look up"/"find" referring to
# content the user already gave, not the internet (Step 27's target class).
# ---------------------------------------------------------------------------
CATEGORY_G: list[EvalCase] = [
    EvalCase(
        "g1", "G", "Search for the bug in this code: def add(a, b): retun a + b", False,
        final_answer_check=_non_empty_reasonable_length,
    ),
    EvalCase("g2", "G", "Look up the meaning of the word 'ephemeral'.", False, final_answer_check=_non_empty_reasonable_length),
    EvalCase(
        "g3", "G", "Find the value of x in this equation: 2x + 4 = 10", False,
        final_answer_check=_contains_any("3"),
    ),
    EvalCase("g4", "G", "Search for a good name for a pet goldfish.", False, final_answer_check=_non_empty_reasonable_length),
]

# ---------------------------------------------------------------------------
# H. Multi-part requests.
# ---------------------------------------------------------------------------
CATEGORY_H: list[EvalCase] = [
    EvalCase(
        "h1", "H", "What time is it, and also what is 5 + 7?", True, "time",
        notes="Mixed deterministic-tool + static-math request; existing precedence rules (Step 20/26) route to time first.",
    ),
    EvalCase(
        "h2", "H", "What is the capital of Japan, and can you search the web for its current mayor?", True, "web_search",
        requires_web=True, notes="Static fact + explicit web command; deterministic web route (Step 26) fires.",
    ),
    EvalCase(
        "h3", "H", "Explain what HTTP is, and tell me today's date.", True, "time",
        notes="Static explanation + deterministic-temporal; temporal precedence (Step 20) applies.",
    ),
]

MODEL_DEPENDENT_CASES: list[EvalCase] = (
    CATEGORY_A + CATEGORY_B + CATEGORY_C + CATEGORY_D + CATEGORY_E + CATEGORY_F + CATEGORY_G + CATEGORY_H
)

ALL_CATEGORIES: dict[str, list[EvalCase]] = {
    "A": CATEGORY_A, "B": CATEGORY_B, "C": CATEGORY_C, "D": CATEGORY_D,
    "E": CATEGORY_E, "F": CATEGORY_F, "G": CATEGORY_G, "H": CATEGORY_H,
}
