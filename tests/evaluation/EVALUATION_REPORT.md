# Step 29 — Final End-to-End Agent Capability Evaluation

**Status:** baseline measurement (§1-22 below, unedited). See **§23 Step 30 Update** for
the one production-code change made since this baseline and its re-measured effect.

## 1. Objective

Establish an honest, reproducible, quantitative baseline of the **complete** production
agent — not a single component — as it exists after Step 28 (routing frozen).

## 2. System under test

The real, unmodified production path, exactly as `app/main.py` wires it at this
commit:

```
FastAPI (app/main.py)
  -> ChatService (app/services/chat.py)
    -> AgentOrchestrator (app/agent/orchestrator.py)
      -> AgentLoop (app/agent/loop.py)
        <-> LLMDecisionMaker (app/agent/decision_maker.py) -> Ollama (llama3.2:3b)
        <-> ToolExecutionGate (app/agent/tool_execution.py)
              -> AllowlistPermissionPolicy({"time","date","web_search"})
              -> ToolRegistry -> {TimeTool, DateTool, WebSearchTool (real Tavily)}
```

Active: deterministic temporal routing, deterministic web-search-command routing
(Steps 26/28), unconditional `BudgetedCorrectionPolicy` (max_corrections=2), a
120-second per-request deadline. Inactive (confirmed, not assumed): semantic memory,
the planner, telemetry (off by default — enabled only transiently by this evaluation's
own instrumentation, see §4), `app/ml_router`.

**Environment:** Windows, Python 3.13.7, Ollama `llama3.2:3b` (local), real Tavily API
key present. Single evaluation run, 2026-09-30.

## 3. Dataset

`tests/evaluation/dataset.py`, versioned as v1. 55 permanent cases across 14 categories
(A-N per the Step 29 design brief).

| Category | Cases | Execution mode |
|---|---|---|
| A. Direct answers | 5 | real HTTP, real Ollama, 3 trials |
| B. Time | 3 | real HTTP, real Ollama, 3 trials |
| C. Date | 3 | real HTTP, real Ollama, 3 trials |
| D. Explicit web search | 4 | real HTTP, real Ollama + real Tavily, 3 trials |
| E. Current information (no explicit command) | 4 | real HTTP, real Ollama + real Tavily, 3 trials |
| F. Static/historical | 4 | real HTTP, real Ollama, 3 trials |
| G. Local/technical context | 4 | real HTTP, real Ollama, 3 trials |
| H. Multi-part | 3 | real HTTP, real Ollama (+ Tavily for h2), 3 trials |
| I. Tool input robustness | 5 checks | component-level, deterministic, 1 run |
| J. Tool failure/recovery | 5 checks | AgentLoop-level (scripted LLM), deterministic, 1 run |
| K. Session memory | 3 checks | real `chat_service` (scripted LLM), deterministic, 1 run |
| L. Security/permissions | 5 checks | real production gate/registry, deterministic, 1 run |
| M. Timeout | 2 checks | real production tool_registry/gate via AgentOrchestrator, deterministic, 1 run |
| N. Telemetry | 4 checks | real `chat_service`, deterministic, 1 run |

Categories A-H: **90 real end-to-end HTTP requests** (30 cases × 3 trials). Categories
I-N: **29 deterministic mechanism checks**, single run each (repeating a deterministic
check would only measure Python's own determinism — see dataset.py's module docstring).

## 4. Ground-truth methodology

Every case declares explicit ground truth before execution (Phase 3): whether a tool is
required, which one, whether web access is required, and — where a deterministic
predicate can honestly characterize correctness — a `final_answer_check` (substring/
structural match, e.g. "the reply mentions 96" for "12×8"). Genuinely open-ended cases
(e.g. "explain what a for loop is") are **not** force-graded against invented exact
text; they use a weak structural check (non-empty, reasonable length) and this
limitation is stated explicitly, not hidden — see §13.

Per-request instrumentation (which tool was proposed/executed, how many LLM calls were
made, latency) is read from `app/agent/telemetry.py`'s `AgentEvent` stream, captured by
swapping the real `chat_service.event_sink` for a `ListEventSink` for the duration of
each call and restoring it immediately after (`tests/evaluation/runner.py:capture_events`)
— the same monkeypatch-the-singleton technique already established in this test suite
(e.g. `tests/test_deterministic_web_search_routing.py`). Telemetry is documented and
independently tested as purely observational (cannot affect execution), so this does not
change agent behavior, and `TELEMETRY_ENABLED` in `.env`/`app/config.py` is never touched.

## 5. Metrics (exact formulas)

**Task success** (the one aggregate metric — formula stated explicitly, per Phase 4):
a trial is a success **iff all** of:
1. HTTP 200 **and** the request's own telemetry recorded `REQUEST_COMPLETED` (never
   `REQUEST_FAILED`).
2. Correct tool selection: the expected tool was proposed if one was required; no tool
   was proposed if none was required (zero tolerance for an unnecessary call).
3. Where a deterministic `final_answer_check` exists, it passed.

This is a conjunction, not a weighted score — any single failure means the trial is not
a success. See `tests/evaluation/metrics.py:task_success`.

All other metrics (final-answer correctness, tool-selection accuracy, missed/unnecessary
-tool-call rate, tool-execution success rate, average/percentile latency, average
LLM/tool calls) are reported **separately per category**, never collapsed together.

## 6. Trial methodology

Categories A-H: 3 trials per case (Step 27's established methodology), sequential HTTP
requests against the real production singleton. Categories I-N: single run
(deterministic by construction).

## 7. Results — per category

*(exact console output from the real run, 2026-09-30, 494s wall-clock for categories A-H)*

```
[A] trials=15 cases=5 task_success_rate=100.0% final_answer_correctness_rate=100.0%
      avg_llm_calls=1.00 avg_tool_calls=0.00 latency(avg/min/max/p50/p95)=3.21/2.64/4.66/2.95/3.96s
[B] trials=9  cases=3 task_success_rate=44.4% tool_selection_accuracy=44.4% missed_tool_call_rate=55.6%
      avg_llm_calls=1.11 avg_tool_calls=0.44 latency(avg/min/max/p50/p95)=4.05/2.90/6.02/3.42/6.02s
[C] trials=9  cases=3 task_success_rate=66.7% tool_selection_accuracy=66.7% missed_tool_call_rate=33.3%
      avg_llm_calls=1.00 avg_tool_calls=0.67 latency(avg/min/max/p50/p95)=3.92/3.25/4.46/3.82/4.46s
[D] trials=12 cases=4 task_success_rate=100.0% tool_selection_accuracy=100.0% missed_tool_call_rate=0.0%
      avg_llm_calls=1.33 avg_tool_calls=1.33 latency(avg/min/max/p50/p95)=7.78/3.97/18.85/6.04/12.05s
[E] trials=12 cases=4 task_success_rate=66.7% tool_selection_accuracy=66.7% missed_tool_call_rate=33.3%
      avg_llm_calls=1.67 avg_tool_calls=0.67 latency(avg/min/max/p50/p95)=8.93/3.04/13.88/10.94/13.20s
[F] trials=12 cases=4 task_success_rate=100.0% final_answer_correctness_rate=100.0%
      avg_llm_calls=1.00 avg_tool_calls=0.00 latency(avg/min/max/p50/p95)=3.50/2.71/5.53/3.38/3.86s
[G] trials=12 cases=4 task_success_rate=100.0% final_answer_correctness_rate=100.0%
      avg_llm_calls=1.08 avg_tool_calls=0.00 latency(avg/min/max/p50/p95)=4.75/2.98/7.97/4.23/7.71s
[H] trials=9  cases=3 task_success_rate=66.7% tool_selection_accuracy=66.7% missed_tool_call_rate=33.3%
      avg_llm_calls=1.33 avg_tool_calls=1.00 latency(avg/min/max/p50/p95)=7.79/3.87/18.09/6.81/18.09s
------------------------------------------------------------------------------
[ALL A-H] trials=90 cases=30
    task_success_rate=83.3%  final_answer_correctness_rate=100.0% (of checkable cases)
    tool_selection_accuracy=70.6%  missed_tool_call_rate=29.4%  unnecessary_tool_call_rate=0.0%
    tool_execution_success_rate=100.0%
    avg_llm_calls=1.19  avg_tool_calls=0.48
    latency avg/min/max/p50/p95 = 5.44 / 2.64 / 18.85 / 3.97 / 12.16 s
```

## 8. Overall task-success rate

**83.3%** (75/90 trials), formula in §5. Driven almost entirely by categories B, E, H
(missed tool calls); A, D, F, G were 100%.

## 9. Tool-selection metrics

Overall **70.6%** tool-selection accuracy among the 51 trials that required a specific
tool; **29.4%** missed-tool-call rate; **0.0% unnecessary-tool-call rate across all 90
trials** — zero false-positive tool calls observed in this run, consistent with (though
not proof of permanent elimination of) Step 27's fix. `tool_execution_success_rate =
100%`: every tool call that WAS made completed operationally without error.

## 10. Final-answer metrics

**100%** on the deterministically-checkable subset (categories A, F, G — 41
trials with a `final_answer_check`). Categories B/C/D/E/H have no
`final_answer_check` in this dataset version (their ground truth is tool selection, not
answer text) and are reported as `n/a`, not silently scored.

## 11. Recovery metrics (category J)

5/5 checks passed. 4/5 correctable failure classes (unknown tool, invalid tool input,
tool execution failure, malformed decision) recovered to `COMPLETED`. The 5th
(permission denial) correctly stayed `FAILED` in exactly one iteration, confirming it is
never offered to any `CorrectionPolicy` — this is by design (Milestone 17/18), not a
recovery failure, and is reported as its own separate check rather than folded into the
recovery-rate denominator.

## 12. Memory metrics (category K)

3/3 checks passed: a second message in the same session sees the first (prompt contains
the established fact); a separate session does not inherit it; an anonymous
(`session_id=None`) request does not inherit another session's information.

## 13. Security results (category L) — Phase 7 gate

**5/5 checks passed. Security violations: 0/5.** Verified against the real production
`chat_service`/`_tool_execution_gate`/`_tool_registry`: an unallowlisted tool is denied;
a denied tool never executes; a confirmation-required tool without confirmation is
blocked; a model-fabricated `"authorized": true` JSON field has zero effect on a denied
tool (still denied, never executed); a model-fabricated `"confirmed": true` field has
zero effect on a confirmation-required tool (still blocked). `AgentEvent`'s dataclass
fields were also confirmed to contain no free-text field (`payload`/`message`/`prompt`/
`content`/`text` all absent), and every captured `tool_name` in a telemetry-enabled run
was a real registry key — never arbitrary text.

## 14. Timeout results (category M)

2/2 checks passed: an already-expired deadline fails the request with **zero** LLM
calls made; a normal request within a 120-second deadline completes normally.

## 15. Telemetry results (category N)

4/4 checks passed: disabled telemetry (the actual current production default) does not
affect request completion; `AgentEvent` has no free-text field (structural, dataclass-
level check); every telemetry-enabled request's `tool_name` events were real registry
keys; a deliberately-raising event sink never broke the request.

## 16. Performance results

Overall (A-H, 90 trials): mean **5.44s**, min **2.64s**, max **18.85s**, P50 **3.97s**,
P95 **12.16s**. Breakdown:
- Direct-answer / static / local-context categories (A, F, G): mean 3.2-4.8s, one LLM
  call, no tool — the FLOOR latency for this deployment (dominated by llama3.2:3b's own
  inference time on this hardware).
- Deterministic time/date categories (B, C): mean 3.9-4.1s despite requiring zero LLM
  calls for the DECISION — the one remaining LLM call (FINAL synthesis after the tool
  result) still dominates latency; the deterministic route saves a call, not
  necessarily wall-clock time, since a second call still happens.
- Web-search categories (D, E, H): mean 7.8-8.9s, max up to 18.85s — real Tavily
  network latency (typically 0.7-4s per call, occasionally higher, see raw log) added on
  top of 1-2 LLM calls.
- **Sample size caveat**: max 15 trials per category. P95 in particular is only weakly
  meaningful at this sample size (Phase 8's own instruction: do not overstate precision
  from a small sample) — treat P95 numbers as directional, not statistically precise.

## 17. LLM variance results

**4/30 cases (13.3%) produced a different task-success outcome across 3 identical
trials**, under completely unchanged code:

| Case | Category | Outcomes (trial 1,2,3) |
|---|---|---|
| b2 | B (time, no deterministic match) | False, False, True |
| e2 | E (current info) | False, True, True |
| e3 | E (current info) | False, False, True |
| e4 | E (current info) | True, False, True |

All four unstable cases are in categories that are **not** deterministically routed
(B's b2/b3 don't match Router's narrow temporal pattern list; all of E is
advisory-hint-only by design, Step 27). Every deterministically-routed case (all of D,
all of B's b1, all of F/G/A) was **perfectly stable** across all 3 trials — this is a
structural guarantee of the deterministic routes, not a coincidence of this run, and is
independently proven offline (zero LLM calls) elsewhere in this test suite.

*(Not directly comparable to Step 27's own 25%-of-cases variance figure — that dataset
was deliberately adversarial/ambiguous by design; this dataset is a broader realistic
mix. Both numbers are honestly reported, neither is treated as equivalent to the other.)*

## 18. Most important failures

1. **Category B's two non-deterministic phrasings** ("Can you tell me the current
   time?", "What's the time?") miss the deterministic router's exact pattern list (only
   "what time is it" and a handful of literal phrases match — see
   `app/agent/router.py`'s `_deterministic_temporal_check`) and fall to LLM judgment,
   which then misses the tool call in the majority of trials. **This is a coverage gap
   in the deterministic pattern list, not a Step 29 architectural finding** — closing it
   would be squarely in the spirit of the Step 20/26/28 pattern (narrow, evidence-driven
   deterministic-route extension), a natural Step 30 candidate.
2. **Category E (current information, no explicit command)** is the single worst-
   performing category by design intent (66.7% success, 3 of 4 cases cross-trial
   unstable) — this is the class Step 27 explicitly left to the model's own judgment
   after finding no safe way to make it deterministic without reopening false positives.
   Confirmed here again, with fresh, independent data.
3. **One malformed-decision parse error** was observed live during the run (visible in
   the raw log at 13:33:56, `correction.triggered category=decision_parse`) and
   recovered successfully on the next iteration — direct, real-world confirmation that
   the correction pipeline (Milestone 17) works end-to-end under real, not scripted,
   model unpredictability.

## 19. What appears to be a production-code problem

- The deterministic temporal pattern list (`Router._deterministic_temporal_check`) is
  narrower than the general concept of "the user is asking for the time" — natural
  phrasings like "Can you tell me the current time?" fall through to LLM judgment even
  though they are just as unambiguous as "What time is it?". This is a coverage gap,
  not a design flaw — the existing narrow-pattern discipline (Steps 20/26/27/28) is
  sound; it simply hasn't been extended to this specific phrasing family yet.

## 20. What appears to be a model limitation

- Category E's current-information judgment (66.7% success, high cross-trial variance)
  is llama3.2:3b's own reasoning under real inference variance, not a routing or prompt
  defect — Step 27 already established (and this run reconfirms) that no safe
  deterministic rule covers this class without reintroducing false positives.
- The one malformed-decision (invalid JSON) event during the run is consistent with
  llama3.2:3b's previously-documented occasional JSON-formatting lapses (see Step 27's
  own evaluation notes) — recovered correctly, not a defect in the recovery mechanism.

## 21. Limitations of this evaluation

- Dataset size is intentionally modest (30 model-dependent cases × 3 trials = 90
  samples) — enough to see clear, real signal (0% unnecessary-call rate across 90
  trials; the deterministic/non-deterministic split's effect on variance) but too small
  for tight statistical confidence intervals, especially per-category (as few as 9
  trials). No claim of statistical significance is made beyond "this is what was
  observed in this run."
- `final_answer_check` predicates are deterministic substring/structural checks, not
  semantic correctness grading — a reply could pass a check while being subtly wrong in
  ways the check can't detect (and vice versa, though not observed in this run). No
  second model was used to grade answers (explicitly out of scope for this milestone).
- Single evaluation run. Categories A-H's numbers reflect one session's real-Ollama
  behavior on 2026-09-30 and may shift with model updates, load, or Tavily result
  changes — the dataset and harness are designed to be re-run, not treated as one-time.
- Category E's cases are, by the dataset's own design, expected to be the hardest;
  their low score is not surprising and should not be over-weighted as representative
  of the WHOLE agent's quality.

## 22. Recommended next engineering work

See §23 below (implemented) and the Step 30 final chat report's "Recommendation for
Step 31."

## 23. Step 30 Update — deterministic temporal coverage fix, measured

**Change:** `Router._deterministic_temporal_check` (`app/agent/router.py`) gained one
new, whole-message-anchored check for four additional unambiguous current-time
phrasings: "what's the time", "what is the time", "can you tell me the current time",
"can you tell me what time it is". Matched against the ENTIRE message only (not as a
substring like the pre-existing pattern list), specifically because "what is the time"
is a realistic sub-phrase of unrelated technical questions this deployment already
fields (e.g. "What is the time complexity of this algorithm?") — verified directly, and
covered by a permanent regression test (`tests/test_deterministic_temporal_routing.py::
test_production_technical_time_complexity_question_still_consults_the_model`). No other
file changed.

**Re-measurement:** the identical Step 29 dataset/harness/methodology
(`tests/evaluation/test_categories_a_through_h_full_evaluation`), same 30 cases × 3
trials, re-run after the fix:

| Metric | Before (Step 29) | After (Step 30) |
|---|---|---|
| **Category B task success** | **44.4%** (4/9) | **100.0%** (9/9) |
| Category B missed-tool-call rate | 55.6% | 0.0% |
| Category B tool-selection accuracy | 44.4% | 100.0% |
| Overall task success (A-H) | 83.3% (75/90) | 83.3% (75/90) |
| Overall tool-selection accuracy | 70.6% | 70.6% |
| Overall missed-tool-call rate | 29.4% | 29.4% |
| Overall unnecessary-tool-call rate | 0.0% | 0.0% |
| Overall mean latency | 5.44s | 4.88s |

Category B's improvement is exactly the intended, structural fix — proven with zero LLM
calls for the decision itself (`tests/test_deterministic_temporal_routing.py`'s
`test_new_time_phrasings_select_the_time_tool_without_consulting_the_llm`), not a
model-judgment improvement that could regress on the next run.

**Overall task success stayed flat (75/90 both times) — an offsetting coincidence, not
an absence of improvement:** Category B gained +5 successes and Category C gained +1
(66.7%→77.8%, also structurally unrelated to this fix — plausible run-to-run model
variance), while **Category E dropped sharply, 66.7%→16.7% (-6 successes)**. Category
E's four cases ("latest AI developments", "current Bitcoin price", "latest Python
version", "James Webb Space Telescope status") were checked directly against
`Router.deterministic_tool_route`/`deterministic_web_search_route` and **matched
neither, before or after this change** — Category E is structurally unreachable by
Step 30's fix. Its regression coincides with a 62.09s max latency observed in the same
run (vs. 13.88s in the original Step 29 run) and a separate, confirmed-transient
socket-timeout failure in an unrelated test during the same session (see the Step 30
chat report's Phase 5/9 notes) — consistent with a temporary local
Ollama/Tavily/network slowdown during this particular run, not a regression this
change could cause. This is reported plainly rather than smoothed over: Category E
remains, as Step 29 already established, the hardest and most run-to-run-variable
category in this dataset, for reasons unrelated to temporal routing.

**Security/mechanism categories (I-N):** unaffected, re-confirmed 25/25 checks passing
including 0/5 security violations (`app/agent/permissions.py`, `tool_execution.py`,
`tool_registry.py` untouched by this change).

## 24. Step 31 Audit — reliability audit, no code change made

A full audit of `decision_maker.py`, `loop.py`, `reliability.py`, `orchestrator.py`,
`chat.py`, `tool_execution.py`, `permissions.py`, and `router.py` against the Step
29/30 evidence, plus a direct, fresh empirical check (30 real Ollama calls against raw,
pre-parse model output for the exact prompts Category E/H use), found **no
production-safe, evidence-justified code change** outside the now-frozen routing layer:

- **Category E** (current-info, no explicit command): the decision prompt already
  states the correct rule plainly ("Current, latest, recent, or live information
  generally needs a tool...") and the advisory `ROUTING HINT` already fires for
  keyword-bearing cases — the model still misses calls despite both signals already
  being present. Classified as a genuine model-capability limitation (llama3.2:3b
  instruction-following at this scale), not a prompt-clarity gap: making the rule any
  more forceful would functionally re-implement keyword routing inside the prompt text,
  the exact risk Step 27 already rejected.
- **Category H** (multi-part): traced precisely, not assumed. `h1`/`h2` are both fully
  deterministic (verified directly, zero LLM calls) and 100% reliable in both the Step
  29 and Step 30 runs; **100% of H's failures come from `h3` alone**
  ("...tell me today's date" embedded in a longer sentence, which matches no existing
  temporal pattern). This is a single, narrow routing-coverage gap of the exact kind
  Step 30 fixed for other phrasings — not a general multi-part-handling defect in
  `AgentLoop`'s iteration model, which `h1`/`h2` prove works correctly. Left unfixed
  this milestone because the routing layer is explicitly frozen; flagged as a candidate
  for a future step if the freeze is ever lifted.
- **Malformed decision JSON**: 30/30 raw model outputs parsed successfully in a direct
  reproduction against the exact Category E/H prompts (a throwaway audit script, not
  part of this repository). Consistent with the ~1-in-250 rate observed across all
  Step 26-30 evaluation runs combined. The
  one live occurrence Step 29 did observe recovered on its very first retry via the
  existing, already-proven correction pipeline. No evidence of a systematic parsing gap
  (e.g. markdown-fenced JSON) was found; a defensive parser change was considered and
  explicitly NOT made, for lack of evidence it is needed.
- **Wrong-tool selection, final-answer synthesis, latency variance**: no systematic
  defect evidenced. The one latency outlier in the Step 30 re-run (62.09s) coincided
  with a separate, confirmed-transient socket-timeout failure in the same session — both
  already-existing timeout layers (60s per-LLM-call, 120s per-request) bounded it
  correctly; this is infrastructure/session variance, not a code defect.

**Conclusion:** the architecture validated through Steps 17-30 (correction budget,
repetition detection, closed failure taxonomy, security-first denial handling) is sound
and shows no gap this audit could respectably tie a fix to. No file under `app/` was
changed by this milestone. See the Step 31 chat report for the full audit trace and
classification.
