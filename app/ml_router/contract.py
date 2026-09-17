"""The classification contract: the closed, four-label vocabulary this
classifier predicts, and the rule for assigning each label. This module
holds only the label enum and the documentation below — no model code,
no data. `dataset.py` and `classifier.py` both import `Intent` from here
so there is exactly one definition of the label set.

--------------------------------------------------------------------------
DIRECT
--------------------------------------------------------------------------
The request can be answered from the model's own static knowledge, with
no external/current-data tool and no local time/date tool. Includes:
concept explanations, definitions, math/reasoning, casual conversation,
writing requests, and — importantly — questions that merely CONTAIN a
recency-looking word ("current", "latest", "recent", "today") without
actually needing fresh external information:

    "What is machine learning?"                              -> DIRECT
    "What is 25 + 17?"                                        -> DIRECT
    "Explain the concept of electrical current."               -> DIRECT
    "What time complexity does quicksort have?"                -> DIRECT
    "What is a leap year?"                                      -> DIRECT
    "What is the most recent common ancestor in genetics?"      -> DIRECT

The last three are deliberate hard negatives: "current" (electrical
current, an unrelated noun sense), "time" (computational complexity, not
clock time), and "recent" (a static genetics term) each look like a
recency/tool signal but are not one. A classifier that merely pattern-
matches those words will mislabel these as WEB or TIME.

--------------------------------------------------------------------------
WEB
--------------------------------------------------------------------------
The request needs current, recent, external, or explicitly-requested web
information — something the model's static training data cannot be
trusted to answer correctly:

    "What is the latest AI news?"                               -> WEB
    "What is the current CEO of OpenAI?"                         -> WEB
    "Search the web for information about RAG."                  -> WEB
    "What are the recent developments at Google DeepMind?"       -> WEB

Compound requests (a static explanation PLUS a current-information ask)
are labeled WEB, not DIRECT: in the current production architecture, the
correct FIRST action for such a request is a tool call — the static half
is synthesized afterward, once the tool result is available (see
app/agent/loop.py's FINAL-synthesis behavior) — so the single label this
classifier assigns reflects "what should happen next," not "does the
whole request need external information."

    "Explain RAG and then tell me about recent developments in RAG."
                                                                  -> WEB

--------------------------------------------------------------------------
TIME
--------------------------------------------------------------------------
The request asks for the current local clock time — nothing else. A
request that merely contains the word "time" in a different sense (time
complexity, a historical time period, "what is Daylight Saving Time")
is DIRECT, not TIME:

    "What time is it?"                                           -> TIME
    "Can you tell me the current time?"                          -> TIME
    "What time complexity does merge sort have?"                 -> DIRECT (not TIME)
    "What is Daylight Saving Time?"                               -> DIRECT (not TIME)

--------------------------------------------------------------------------
DATE
--------------------------------------------------------------------------
The request asks for today's date, or the weekday of a SPECIFIED date —
a local, deterministic calendar computation, not a historical or
current-events question about a date:

    "What is today's date?"                                      -> DATE
    "What day was 25 December 2026?"                              -> DATE
    "What is the history of the Gregorian calendar?"              -> DIRECT (not DATE)
    "What happened on 12 September 2001?"                         -> WEB (not DATE — a
                                                                       historical/current-events
                                                                       question ABOUT a date,
                                                                       not a request for the
                                                                       date/weekday itself; this
                                                                       mirrors Router's own
                                                                       existing distinction in
                                                                       app/agent/router.py's
                                                                       `_deterministic_temporal_
                                                                       check`, route="web" for
                                                                       exactly this shape)

--------------------------------------------------------------------------
Why four labels, and why they mirror (but do not reuse) existing types
--------------------------------------------------------------------------
DIRECT/WEB/TIME/DATE map onto this codebase's existing FINAL/web_search/
time/date vocabulary (see app/agent/router.py's `Route` and
`RoutingHint`), because the classifier's eventual purpose (if ever
integrated) is to predict the same decision those existing mechanisms
make. `Intent` is deliberately a NEW, separate enum rather than an import
of `Route`/`RoutingHint` from app/agent/router.py — this package must not
import anything from app/agent/ (see the package docstring), so it
cannot share a type with production code without also sharing a
dependency edge that would blur the "not wired in" boundary.
"""
from __future__ import annotations

from enum import Enum


class Intent(Enum):
    """The closed, four-member label set this classifier predicts."""

    DIRECT = "DIRECT"
    WEB = "WEB"
    TIME = "TIME"
    DATE = "DATE"


INTENT_LABELS: tuple[str, ...] = tuple(intent.value for intent in Intent)
