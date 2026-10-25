"""Milestone 44.1 — web-search observation formatting (reliability
hardening), app/agent/decision_maker.py's `_format_web_search_data`.

Context (see docs/MILESTONE_44_DEPLOYMENT.md's companion diagnosis, not
reproduced here): a read-only diagnosis found Tavily retrieval and routing
both work correctly, but the OLD generic JSON-stringify-and-truncate path
buried web_search's title/url fields inside deeply-escaped JSON and applied
one truncation cut across the whole serialized result list, so which result
survived depended on byte position, not relevance. This file tests the
replacement: a compact, labeled, per-result-bounded, total-bounded text
format — presentation only, no retrieval/routing change.
"""
from __future__ import annotations

import json

from app.agent.decision_maker import (
    LLMDecisionMaker,
    _MAX_WEB_OBSERVATION_CHARS,
    _MAX_WEB_RESULT_CONTENT_CHARS,
    _MAX_WEB_RESULTS_FORMATTED,
)
from app.agent.state import AgentState
from app.agent.tool_registry import ToolRegistry


class FakeLLM:
    """Replays a fixed sequence of responses; records every call. No
    network/Ollama involved — matches tests/test_decision_maker.py's own
    FakeLLM exactly."""

    def __init__(self, responses: list[str]):
        self.responses = iter(responses)
        self.calls: list[dict[str, object]] = []

    def generate(self, messages: list[dict[str, str]], *, json_mode: bool = False) -> str:
        self.calls.append({"messages": messages, "json_mode": json_mode})
        return next(self.responses)


def _decision_maker() -> LLMDecisionMaker:
    return LLMDecisionMaker(llm_client=FakeLLM(["{}"]), tool_registry=ToolRegistry())


def _realistic_results(n: int, *, content_len: int = 100) -> list[dict[str, object]]:
    return [
        {
            "title": f"Result {i} Title",
            "url": f"https://example.com/{i}",
            "content": f"content-{i}-" + ("x" * content_len),
            "score": 1.0 - (i * 0.01),
            "retrieved_at": "2026-10-01T00:00:00Z",
        }
        for i in range(1, n + 1)
    ]


# ===========================================================================
# 1. Readable, labeled evidence.
# ===========================================================================

def test_web_results_are_formatted_as_readable_labeled_evidence() -> None:
    dm = _decision_maker()
    data = _realistic_results(2)

    rendered = dm._format_web_search_data(data)

    assert rendered.startswith("WEB SEARCH RESULTS")
    assert "Result 1" in rendered
    assert "Result 2" in rendered
    assert "Title:" in rendered
    assert "URL:" in rendered
    assert "Content:" in rendered
    # Not the old deeply-escaped JSON shape.
    assert '\\"title\\"' not in rendered
    assert "\\u" not in rendered


# ===========================================================================
# 2. Title/URL survive formatting.
# ===========================================================================

def test_title_and_url_survive_formatting() -> None:
    dm = _decision_maker()
    data = [{"title": "LeetCode #1: Two Sum", "url": "https://example.com/two-sum", "content": "evidence"}]

    rendered = dm._format_web_search_data(data)

    assert "Title: LeetCode #1: Two Sum" in rendered
    assert "URL: https://example.com/two-sum" in rendered


def test_missing_title_or_url_is_omitted_not_shown_as_blank() -> None:
    dm = _decision_maker()
    data = [{"content": "evidence with no title or url"}]

    rendered = dm._format_web_search_data(data)

    assert "Title:" not in rendered
    assert "URL:" not in rendered
    assert "evidence with no title or url" in rendered


# ===========================================================================
# 3. Ordering preserved.
# ===========================================================================

def test_results_remain_ordered() -> None:
    dm = _decision_maker()
    data = _realistic_results(4)

    rendered = dm._format_web_search_data(data)

    positions = [rendered.index(f"Result {i}") for i in range(1, 5)]
    assert positions == sorted(positions)
    # The highest-ranked (first-supplied) result's own title appears before
    # every later result's title.
    assert rendered.index("Result 1 Title") < rendered.index("Result 2 Title")
    assert rendered.index("Result 2 Title") < rendered.index("Result 3 Title")


# ===========================================================================
# 4. Per-result content is bounded.
# ===========================================================================

def test_per_result_content_is_bounded() -> None:
    dm = _decision_maker()
    data = [{"title": "Long", "url": "https://example.com/1", "content": "y" * 10_000}]

    rendered = dm._format_web_search_data(data)

    assert "...[content truncated]" in rendered
    assert "y" * 10_000 not in rendered
    # The content portion for this one result must not exceed the per-result cap.
    content_section = rendered.split("Content:\n", 1)[1]
    assert len(content_section) <= _MAX_WEB_RESULT_CONTENT_CHARS + len("...[content truncated]") + 10


# ===========================================================================
# 5. Total observation context is bounded.
# ===========================================================================

def test_total_observation_context_is_bounded() -> None:
    dm = _decision_maker()
    # 5 results, each near the per-result cap -- comfortably exceeds the
    # total cap if nothing bounded the combined output.
    data = _realistic_results(5, content_len=_MAX_WEB_RESULT_CONTENT_CHARS)

    rendered = dm._format_web_search_data(data)

    assert len(rendered) <= _MAX_WEB_OBSERVATION_CHARS + 200  # + a little slack for the header/omission note


# ===========================================================================
# 6. Large result sets cannot create unbounded prompts.
# ===========================================================================

def test_large_result_sets_cannot_create_unbounded_prompts() -> None:
    dm = _decision_maker()
    data = _realistic_results(50, content_len=_MAX_WEB_RESULT_CONTENT_CHARS)

    rendered = dm._format_web_search_data(data)

    assert len(rendered) <= _MAX_WEB_OBSERVATION_CHARS + 200
    assert "more result(s) omitted for length" in rendered
    # Only the first _MAX_WEB_RESULTS_FORMATTED are ever even considered.
    assert f"Result {_MAX_WEB_RESULTS_FORMATTED + 1}" not in rendered


def test_result_count_beyond_the_formatted_cap_is_never_formatted_even_if_it_would_fit() -> None:
    """A defensive, independent cap on the NUMBER of results formatted
    (not just total characters) -- matches WebSearchTool's own
    max_results=5, but enforced here regardless of what any caller's data
    actually contains."""
    dm = _decision_maker()
    data = _realistic_results(_MAX_WEB_RESULTS_FORMATTED + 3, content_len=5)  # tiny content, would easily fit

    rendered = dm._format_web_search_data(data)

    assert f"Result {_MAX_WEB_RESULTS_FORMATTED}" in rendered
    assert f"Result {_MAX_WEB_RESULTS_FORMATTED + 1}" not in rendered
    assert "[+3 more result(s) omitted for length]" in rendered


# ===========================================================================
# 7. Unicode/escaped content handled correctly.
# ===========================================================================

def test_unicode_content_is_preserved_literally_not_escaped() -> None:
    dm = _decision_maker()
    data = [{"title": "LeetCode #1: Two Sum — Solved", "url": "https://example.com/1", "content": "π ≈ 3.14, emoji: 🎯"}]

    rendered = dm._format_web_search_data(data)

    assert "LeetCode #1: Two Sum — Solved" in rendered
    assert "π ≈ 3.14, emoji: 🎯" in rendered
    assert "\\u2014" not in rendered  # not JSON-escaped at this layer


# ===========================================================================
# 8. Existing non-web observations retain current behavior.
# ===========================================================================

def test_non_list_data_falls_back_to_existing_stringify_and_truncate_behavior() -> None:
    """A non-list `data` (None, a bare string, a dict) is not web_search's
    real shape -- falls back to the pre-44.1 generic path unchanged, byte
    for byte."""
    dm = _decision_maker()

    assert dm._format_web_search_data(None) == dm._stringify_and_truncate(None)
    assert dm._format_web_search_data("x" * 5000) == dm._stringify_and_truncate("x" * 5000)
    assert dm._format_web_search_data({"day_of_week": "Wednesday"}) == dm._stringify_and_truncate(
        {"day_of_week": "Wednesday"}
    )


def test_non_web_search_tool_observations_are_unaffected_by_this_formatter() -> None:
    """A `date`/`time` tool observation must go through _format_history's
    unchanged generic path -- _format_web_search_data is never even called
    for a non-web_search tool_name."""
    dm = _decision_maker()
    state = AgentState(user_input="what day is it")
    state.add_observation("date", success=True, data={"day_of_week": "Wednesday"})

    history = json.loads(dm._format_history(state))

    assert history["observations"][0]["data"] == json.dumps(
        {"day_of_week": "Wednesday"}, ensure_ascii=True, sort_keys=True
    )


# ===========================================================================
# Empty list is handled gracefully (edge case, same fallback path).
# ===========================================================================

def test_empty_result_list_falls_back_gracefully() -> None:
    dm = _decision_maker()
    assert dm._format_web_search_data([]) == dm._stringify_and_truncate([])


# ===========================================================================
# End-to-end: the formatted text actually reaches the full prompt built for
# the model, via the real _build_prompt/_format_history path (not just the
# formatter in isolation).
# ===========================================================================

def test_formatted_web_results_reach_the_full_prompt() -> None:
    llm = FakeLLM([json.dumps({"action_type": "final", "final_answer": "ok"})])
    decision_maker = LLMDecisionMaker(llm_client=llm, tool_registry=ToolRegistry())
    state = AgentState(user_input="Search the web for LeetCode problem #1.")
    state.record_tool_call("web_search", "leetcode problem #1")
    state.add_observation(
        "web_search",
        success=True,
        data=[
            {
                "title": "LeetCode #1: Two Sum",
                "url": "https://example.com/two-sum",
                "content": "Given nums and target, return indices that sum to target.",
            }
        ],
    )

    decision_maker.decide(state)

    prompt = llm.calls[0]["messages"][0]["content"]
    assert "WEB SEARCH RESULTS" in prompt
    assert "Title: LeetCode #1: Two Sum" in prompt
    assert "URL: https://example.com/two-sum" in prompt
    assert "Given nums and target, return indices that sum to target." in prompt


# ===========================================================================
# Regression: the highest-ranked result containing the relevant evidence
# must survive truncation even when the full result set, serialized the OLD
# way, would have exceeded the old 2000-character global limit.
# ===========================================================================

def test_highest_ranked_relevant_result_survives_even_when_total_data_is_large() -> None:
    """Direct regression for the Milestone 44.1 diagnosis finding: a
    highest-ranked result containing the actually-relevant evidence
    ("Two Sum... nums... target...") must still reach the model even when
    four lower-ranked, irrelevant-content results are also present and the
    combined raw data would dwarf the old 2000-char cap."""
    dm = _decision_maker()
    relevant = {
        "title": "LeetCode #1: Two Sum",
        "url": "https://example.com/two-sum",
        "content": (
            "Given an array of integers nums and an integer target, return "
            "indices of the two numbers such that they add up to target. "
            "Example: nums = [2, 7, 11, 15], target = 9, output = [0, 1]."
        ),
    }
    noise = [
        {"title": f"Unrelated Video {i}", "url": f"https://example.com/v{i}", "content": "transcript filler " * 100}
        for i in range(1, 5)
    ]
    data = [relevant, *noise]
    # Sanity check this scenario is actually large enough to have broken
    # the OLD single-cut JSON approach.
    assert len(json.dumps(data)) > 2000

    rendered = dm._format_web_search_data(data)

    assert "LeetCode #1: Two Sum" in rendered
    assert "https://example.com/two-sum" in rendered
    assert "nums = [2, 7, 11, 15], target = 9, output = [0, 1]" in rendered
    # It is the first-rendered result, consistent with "highest-ranked first".
    assert rendered.index("LeetCode #1: Two Sum") < rendered.index("Unrelated Video 1")
