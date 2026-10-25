# Architecture Status — Step 32 Freeze (extended through Step 39)

**Status:** frozen architectural baseline (Step 32), as verified against the actual
repository on 2026-09-30, with one explicitly-scoped subsystem — persistence
(Steps 36-39) — added since without reopening the freeze on routing, the LLM decision
pipeline, `ToolExecutionGate`, or permissions (see §20a). This document is read-only
with respect to behavior: producing/updating it changes no file under `app/`.

## 1. Project purpose

A locally-runnable, tool-using conversational agent: FastAPI service, custom (non-
LangChain, non-LangGraph) agent runtime, backed by a local LLM (Ollama, `llama3.2:3b`)
with a small set of allow-listed tools (current time, date lookup, web search via
Tavily). The defining architectural property is an explicit authority boundary — the
model *proposes* a tool call; the application, never the model, *authorizes* it.

## 2. Current architecture

```
FastAPI (app/main.py)
  -> ChatService (app/services/chat.py)
    -> AgentOrchestrator (app/agent/orchestrator.py)
      -> AgentLoop (app/agent/loop.py)
        <-> LLMDecisionMaker (app/agent/decision_maker.py) -> LLMClient -> Ollama
        <-> ToolExecutionGate (app/agent/tool_execution.py)
              -> AllowlistPermissionPolicy (app/agent/permissions.py)
              -> ToolRegistry -> {TimeTool, DateTool, WebSearchTool}
```

`app/main.py` is the single composition root: every collaborator above is constructed
exactly once, at import time, and reused across requests (`chat_service` is a
module-level singleton). `ChatService.ask()` builds a fresh `AgentOrchestrator` per
call (cheap, stateless-at-construction) to avoid a shared-mutable-state race under
FastAPI's sync-handler threadpool.

## 3. Production request flow (exact, traced against current code)

```
POST /chat  (app/main.py:chat(), a sync def — not async)
  -> chat_service.ask(message, session_id)
       - memory = session_store.get_memory(session_id)   [explicit session]
               or a FRESH InMemoryConversationMemory()    [session_id=None, anonymous]
       - execution_context = ExecutionContext(session_id=session_id)   # confirmed_tools
         always empty — no confirmation-granting mechanism exists anywhere in this
         codebase
       - request_id = uuid4(); event_emitter = EventEmitter(event_sink, ...) if
         event_sink is not None else None
       - deadline = time.monotonic() + settings.request_timeout_seconds  (120s default)
       - builds AgentOrchestrator:
           session_id given    -> also passes episodic_memory, memory_retriever,
                                   memory_extractor, memory_writer (the latter three
                                   are None unless SEMANTIC_MEMORY_ENABLED=true)
           session_id omitted  -> none of those four are passed; tool_registry/
                                   tool_execution_gate/correction_policy ARE still
                                   passed either way (no "anonymous = unauthorized"
                                   carve-out exists)
  -> orchestrator.process(message)
       - plan = None (no plan_generator was injected — see §13)
       - memory_context = retriever.retrieve(...) if a retriever was passed, else None
       - state = AgentState(user_input=..., messages=<prior turns>, plan=None,
                 memory_context=...)
       - loop.run(state):
           repeat (state.step incremented BEFORE each decision; bounded by
                   max_iterations=5 default AND the deadline, checked each iteration):
             decision = decision_maker.decide(state)
                 1. _deterministic_temporal_decision(state)      [Step 20/30, if
                    deterministic_temporal_routing=True — production: True]
                 2. _deterministic_web_decision(state)           [Step 26/28, if
                    deterministic_web_search_routing=True — production: True]
                 3. else: build the real decision prompt (tools block, advisory
                    ROUTING HINT, execution history, memory/correction blocks) and
                    call the real Ollama backend; parse+validate the JSON response
                    into an AgentDecision (untrusted until validated)
             if TOOL:
                 tool_execution_gate.execute(tool_name, tool_input, execution_context)
                     1. resolve  — ToolRegistry.get(tool_name); unregistered -> fail
                     2. authorize — AllowlistPermissionPolicy.evaluate(descriptor,
                        context) -> ALLOW / DENY / CONFIRM, reading ONLY app-owned
                        state, never anything model-derived
                     3. validate — the tool's own validate()/execute() ValueError
                        contract
                     4. confirm  — blocks if CONFIRM and context.is_confirmed() is
                        False (always False today — nothing populates it)
                     5. execute  — only now does Tool.execute() actually run (real
                        Tavily HTTP call for web_search; real system clock for time;
                        pure parsing for date)
                 -> Observation appended to state; on any failure, AgentLoop classifies
                    a FailureCategory and asks the injected CorrectionPolicy
                    (BudgetedCorrectionPolicy, unconditionally wired) for a verdict —
                    CORRECT (append a CorrectionNote, continue to the next NORMAL
                    iteration) or TERMINATE (fail the state). PERMISSION_DENIED and
                    CONFIRMATION_REQUIRED are structurally never offered to the policy
                    at all.
             if FINAL: state.status = COMPLETED, loop ends
       - on COMPLETED: commit the turn to conversation memory; IF session_id was
         supplied, record exactly one EpisodicMemoryRecord (deterministic summary, no
         LLM call); IF memory_extractor/writer were supplied (semantic memory enabled
         AND session_id supplied), extract and write semantic memory using the
         episodic record's id as provenance
  -> ChatResponse(reply=result.answer)  -> HTTP 200
     (ValueError -> 400; MemorySessionIsolationError -> 500; RuntimeError -> 502)
```

## 4. Component inventory

| # | Component | File(s) | Production path? | Deterministic / model | Security-sensitive | Status |
|---|---|---|---|---|---|---|
| 1 | API layer | `app/main.py` | Yes | deterministic | Boundary (input validation, error mapping) | A |
| 2 | ChatService | `app/services/chat.py` | Yes | deterministic | Session/context construction | A |
| 3 | Session memory | `app/agent/memory.py` (`InMemorySessionMemoryStore`) | Yes | deterministic | Session isolation | A |
| 4 | Conversation memory | `app/agent/memory.py` (`InMemoryConversationMemory`) | Yes | deterministic | Anonymous-request isolation | A |
| 5 | Episodic memory | `app/agent/episodic_memory.py` | Yes, write-only (session_id required) | deterministic | Provenance only, no prompt injection | B |
| 6 | Semantic memory (store/writer/retriever) | `app/agent/semantic_memory.py`, `memory_writer.py`, `memory_retriever.py` | Wired, gated off | deterministic mechanism over model-produced embeddings | Cross-session isolation (`MemorySessionIsolationError`) | B |
| 7 | Embeddings | `app/agent/embeddings.py` (Protocol), `local_embeddings.py` (impl) | Only if semantic memory enabled | model (embedding model) | None beyond semantic memory's own | B |
| 8 | Vector index | `app/agent/vector_index.py` | Only if semantic memory enabled | deterministic | None | B |
| 9 | Memory retrieval | `app/agent/memory_retriever.py`, `memory_context.py`, `memory_formatting.py` | Only if semantic memory enabled | deterministic search | Prompt-injection framing for recalled content | B |
| 10 | Memory writing/extraction | `app/agent/memory_writer.py`, `memory_extraction.py` | Only if semantic memory enabled | model (LLM extraction) + deterministic write | Cross-session write isolation | B |
| 11 | Router (deterministic routes) | `app/agent/router.py` | Yes | deterministic (regex/exact-match) | Proposes only — never authorizes | A |
| 12 | Router.decide() / RouterDecision (full classifier) | `app/agent/router.py` | No | model | N/A — unused | D |
| 13 | LLMDecisionMaker | `app/agent/decision_maker.py` | Yes | model (+ deterministic fast paths) | Output treated as untrusted, parsed/validated | A |
| 14 | AgentLoop | `app/agent/loop.py` | Yes | deterministic control flow | Sole `AgentState` mutator | A |
| 15 | Plan / PlanGenerator | `app/agent/plan.py`, `plan_generator.py` | No (never injected in the live `chat_service`) | model (generator) / deterministic (domain model) | N/A — unused | C |
| 16 | ToolRegistry | `app/agent/tool_registry.py` | Yes | deterministic | Sole source of tool truth | A |
| 17 | ToolExecutionGate | `app/agent/tool_execution.py` | Yes | deterministic | THE authorization boundary | A |
| 18 | Permissions | `app/agent/permissions.py` (`AllowlistPermissionPolicy`) | Yes | deterministic | Core security decision | A |
| 19 | Confirmation | `app/agent/permissions.py` (`ExecutionContext.confirmed_tools`) | Yes (mechanism); always empty in practice | deterministic | No population mechanism exists — no tool can currently be confirmed | B |
| 20 | Validation | tool's own `validate()`/`execute()` `ValueError` contract | Yes | deterministic | Precondition enforcement | A |
| 21 | Correction/reliability | `app/agent/reliability.py` (`BudgetedCorrectionPolicy`) | Yes, unconditional | deterministic | Never offered PERMISSION_DENIED/CONFIRMATION_REQUIRED | A |
| 22 | Timeout/deadline | `ChatService.ask()` + `AgentLoop`'s `deadline` param; `LLMClient`'s per-call 60s | Yes | deterministic | Bounds worst-case request lifetime | A |
| 23 | Telemetry/observability | `app/agent/telemetry.py` | Wired, OFF by default | deterministic, observational only | Closed field set, no free-text field | B |
| 24 | Logging | `app/main.py:configure_logging` | Yes | deterministic | Structured, non-sensitive fields only (audited) | A |
| 25 | Configuration | `app/config.py` | Yes | deterministic | Reads env only; no secret ever hard-coded | A |
| 26 | tests/evaluation | `tests/evaluation/` | N/A (test infra) | mixed | N/A | A (as test infra) |
| 27 | Experimental ML router | `app/ml_router/` | No — zero imports from any production file (verified) | model (TF-IDF + Logistic Regression) | N/A — unreachable from execution | D |
| 28 | External integrations | Ollama (local), Tavily (real API), OpenAI (optional alternate provider) | Yes (Ollama, Tavily); optional (OpenAI) | model / deterministic HTTP | Ollama local-only; Tavily key env-only | A/B |
| 29 | Deployment/runtime | `uvicorn`, in-process, single-worker assumption throughout (locks, LRU/SQLite stores) | Yes | deterministic | Process-local by default; optionally persisted locally (#30) | A (for current scale) |
| 30 | Persistence (Steps 36-39) | `app/persistence/` (`db.py`, `sqlite_memory.py`, `sqlite_episodic_memory.py`, `sqlite_semantic_memory.py`) | Yes, opt-in | deterministic | Parameterized SQL only; no new data crosses the existing security boundary — see §21 | A/B (opt-in) |
| 31 | Health check | `app/main.py` (`GET /health`) | Yes | deterministic | Read-only; reports persistence status only, never internal paths/errors to the caller | A |

## 5. Production foundation (Classification A)

Components exercised by every real request, with no gating flag, fully tested:
FastAPI layer, `ChatService`, session/conversation memory, `AgentOrchestrator`,
`AgentLoop`, `LLMDecisionMaker`, deterministic Router routes, `ToolRegistry`,
`ToolExecutionGate`, `AllowlistPermissionPolicy`, tool validation, unconditional
`BudgetedCorrectionPolicy`, timeout/deadline enforcement, logging, configuration, and
the three registered tools (`time`, `date`, `web_search`).

## 6. Production-limited components (Classification B)

Wired into the real path but narrower in scope or off by default:
- **Episodic memory** — created on every successfully completed, session-scoped
  request, but never read back into any prompt (`get_recent()` has zero production
  callers — confirmed by direct code search). Write-only provenance for semantic
  memory today.
- **Confirmation mechanism** — structurally present and enforced (a tool with
  `requires_confirmation=True` is genuinely blocked), but nothing in this codebase
  ever populates `ExecutionContext.confirmed_tools`. No production tool currently sets
  `requires_confirmation=True`, so this path is exercised only by tests.
- **Telemetry** — fully wired through every layer, default OFF
  (`TELEMETRY_ENABLED` unset). When on, purely observational (cannot affect
  execution — enforced by `SafeEventSink` plus a closed, free-text-free event schema).
- **Semantic memory** (`semantic_memory.py`, `embeddings.py`, `local_embeddings.py`,
  `vector_index.py`, `memory_retriever.py`, `memory_writer.py`, `memory_extraction.py`,
  `semantic_memory_wiring.py`) — **reclassified from C to B in Step 33** (see §14a):
  fully wired in `main.py`, activates for real when `SEMANTIC_MEMORY_ENABLED=true`,
  default `false`. Step 33's audit found and closed the one concrete activation-
  readiness gap (`min_similarity` was unset, i.e. no relevance floor), and added a
  real-embedding-model evaluation proving the mechanism genuinely works for its
  identified use case. Enabling it still requires installing
  `requirements-embeddings.txt` separately (not part of the core install) — if
  missing, the application fails to START, loudly, by design (see §14a).
- **External integrations** — Ollama and Tavily are both real, required dependencies
  for full functionality (Tavily failure degrades `web_search` gracefully via the
  correction pipeline, never crashes a request).

## 7. Prototype / partially wired components (Classification C)

Fully built and independently tested, but not exercised by the live `chat_service`
instance `app/main.py` constructs:
- **Planning** (`plan.py`, `plan_generator.py`) — `AgentLoop` genuinely reads
  `state.plan` when present (Step 11), but `ChatService.__init__` has no
  `plan_generator` parameter at all, so there is no way for the live composition root
  to supply one. `AgentOrchestrator.plan_generator` defaults to `None` and stays
  `None` in every real request.

## 8. Experimental components (Classification D)

- **`app/ml_router/`** (TF-IDF + Logistic Regression intent classifier, ~1053 lines) —
  verified zero imports from any file outside its own package. Structurally cannot
  execute, authorize, or confirm anything. A benchmarking artifact from Milestone 22,
  not an integration.
- **`Router.decide()`** / the full LLM-backed `RouterDecision` classifier — superseded
  by `LLMDecisionMaker` since Step 7 and never called in the production path. Only
  `Router.classify_hint()` (advisory) and the two deterministic route methods are used
  in production; `decide()` remains for its own tests and the historical A/B evaluation
  harness (`app/agent/evaluation.py`'s `NoHintRouter` comparisons), not live traffic.

## 9. Deferred components (Classification E)

Explicitly not present in this codebase in any form: a distributed/networked store
(Redis, a hosted database), a vector database (pgvector, FAISS, Chroma, Pinecone,
Qdrant, Weaviate), MCP, LangChain, LangGraph, a second/external paid LLM beyond the
optional OpenAI provider switch, a subscription/credit/billing system, multi-tenancy
(session_id is a conversation scope key, explicitly documented as never identity or
authentication).

**No longer deferred as of Steps 36-39: local, opt-in persistence.** Every memory
layer can now survive a process restart via a local SQLite file
(`MEMORY_BACKEND=sqlite`, default remains `memory` — see §21 for the full picture).
PostgreSQL/pgvector remain deferred and would only become relevant for a genuine
multi-process/horizontally-scaled deployment, which this architecture does not target
today (§16).

## 10. Known limitations (Classification F)

Verified, not speculative — see `tests/evaluation/EVALUATION_REPORT.md` for full
methodology and numbers:
- **Category E** (current-information queries with no explicit search command) —
  model-capability-bound; the decision prompt already states the rule and the
  advisory hint already fires, and the model still misses a material fraction of
  calls. Not fixable without either forcing the rule (reintroducing keyword-routing
  risk, explicitly rejected in Step 27) or a different model (out of scope).
- **A narrow, un-fixed routing gap** (`h3`-style: "...tell me today's date" embedded
  in a longer sentence) — identified precisely in Step 31, intentionally left
  unfixed because the routing layer is frozen.
- **Rare malformed decision JSON** (~1 in 250 real decide() calls across all
  evaluation runs to date; 0/30 in a direct Step 31 reproduction) — already
  gracefully recovered by the correction pipeline; not demonstrated as a systematic
  parser defect.
- **Latency variance** under sustained local Ollama load in a single long test
  session (one observed 62s outlier, one transient socket-timeout in an unrelated
  test) — both already bounded correctly by the existing 60s per-call / 120s
  per-request timeout layers; classified as infrastructure/session variance, not a
  code defect.
- **Semantic memory extraction-time `RuntimeError` propagation** (Step 33 audit
  finding, `tests/test_semantic_memory_evaluation.py::test_kb_extraction_time_
  ollama_failure_...`): if Ollama becomes unreachable specifically DURING the
  post-answer memory-extraction call (not the main decision call), the exception is
  not caught — the already-computed, correct answer is lost and the request surfaces
  as HTTP 502. Narrow (requires Ollama to fail in a specific few-hundred-millisecond
  window, only when a session_id is supplied AND semantic memory is enabled) and
  deliberately NOT fixed this milestone — documented per the Step 31 precedent
  (a genuine, narrow limitation over a speculative patch), and only reachable at all
  once semantic memory is enabled (default: not enabled).
- **Semantic memory's scope is session-only, not cross-session/cross-visit** — a new
  `session_id` (e.g. the user returns after closing their browser) starts with zero
  memory, identically to conversation and episodic memory. The concrete use case this
  subsystem serves (Step 33 Phase 3) is recall WITHIN one long-running session, past
  conversation memory's 20-message window — not a persistent, cross-visit user
  profile. Conflating the two would overstate what this activation provides.

## 11. Security boundary

Verified directly against the real production `chat_service`/`_tool_execution_gate`/
`_tool_registry` (Step 29's Category L, 5/5 checks, 0 violations) and by the broader
existing suite (`test_production_tool_authorization.py`, `test_tool_execution_gate.py`,
`test_security_audit_18c.py`):

| Question | Answer | Evidence |
|---|---|---|
| Can the model invent a tool? | No | `tool_registry.has(name)` checked before any `AgentDecision` is built; unregistered names never reach the gate. |
| Can the model self-authorize? | No | `AgentDecision` carries only `tool_name`/`tool_input`; no `authorized`/`permission` field is ever parsed from model JSON. |
| Can the model fabricate confirmation? | No | `is_confirmed()` reads only `ExecutionContext.confirmed_tools` (app-populated); a fabricated `"confirmed": true` field has zero effect — verified directly. |
| Do unallowlisted tools execute? | No | `AllowlistPermissionPolicy` denies anything outside `{"time","date","web_search"}`; verified against the real production policy. |
| Do invalid inputs bypass validation? | No | A tool's own `ValueError` contract is always the authority; `INVALID_TOOL_INPUT` is correctable but never bypassed. |
| Does permission denial reach execution? | No | `PermissionDeniedError` is raised before `tool.execute()` is ever called (structurally proven — an AST-level test asserts the deterministic dispatch path contains no `.execute(`/`.get(` call at all). |
| Can confirmation requirements be bypassed? | No | `ConfirmationRequiredError` blocks unconditionally when unconfirmed. |
| Can correction override permission/confirmation? | No | `PERMISSION_DENIED`/`CONFIRMATION_REQUIRED` are structurally never offered to any `CorrectionPolicy` by `AgentLoop`, and `BudgetedCorrectionPolicy` independently refuses them too (defense in depth). |

**Model ≠ Authorization ≠ Execution** holds throughout: the model only ever produces a
`(tool_name, tool_input)` proposal; `ToolExecutionGate` is the sole path to
`Tool.execute()`, and it is never bypassed by any code path in this repository.

## 12. Memory architecture

| Layer | Active? | Production path? | Default enabled? | Persistent? | Process-local? | Tested? | Security notes |
|---|---|---|---|---|---|---|---|
| Conversation memory | Yes | Yes | Yes (no flag) | **Optional** (`MEMORY_BACKEND=sqlite`; default `memory` = No) | Yes | Yes (`test_memory*.py`, `test_sqlite_memory.py`) | Anonymous requests isolated (no shared singleton) |
| Session memory | Yes | Yes | Yes (no flag) | **Optional** (`MEMORY_BACKEND=sqlite`) | Yes | Yes (`test_session_*.py`, `test_sqlite_memory.py`) | LRU-bounded, thread-safe, `session_id` is scope-only, never identity |
| Episodic memory | Yes (write-only) | Yes, session-scoped only | Yes (no flag) | **Optional** (`MEMORY_BACKEND=sqlite`) | Yes | Yes (`test_episodic_memory*.py`, `test_sqlite_episodic_memory.py`) | Never read back into a prompt; session-partitioned |
| Semantic memory | No (opt-in, Step 33) | Yes, when enabled | **No** (`SEMANTIC_MEMORY_ENABLED=false`) | **Optional** (`MEMORY_BACKEND=sqlite`, Step 38) | Yes | Yes, extensively (`test_semantic_memory*.py`, `test_memory_retriever.py`, `test_memory_writer_concurrency.py`, `test_semantic_memory_evaluation.py`, `test_sqlite_semantic_memory.py`) | `MemorySessionIsolationError` on any cross-session leak attempt; requires separate dependency install; `min_similarity=0.3` (Step 33) bounds retrieval relevance; embedding-model identity checked at startup when persisted (§21) |
| Embeddings | No (unless semantic memory enabled) | Gated | No | N/A (not a store) | Yes | Yes (`test_embeddings.py`, `test_local_embeddings.py`) | Local model only, no API key, no network call |
| Vector index | No (unless semantic memory enabled) | Gated | No | **Optional** (`MEMORY_BACKEND=sqlite`, Step 38) | Yes | Yes (`test_vector_index.py`, `test_sqlite_semantic_memory.py`) | N/A |

**Persistence (Steps 36-39) is opt-in and off by default** (`MEMORY_BACKEND=memory`
reproduces every milestone through Step 35 exactly — no behavior change, no required
SQLite path). When `MEMORY_BACKEND=sqlite`, all five layers above persist to one local
SQLite file across process restarts. See §21 for the full picture: what is/isn't
persisted, schema versioning, restart behavior, and current scope/limitations.

## 13. Planning architecture

- `PlanGenerator` is NOT production-wired: `ChatService.__init__` has no
  `plan_generator` parameter, so there is no way for `app/main.py`'s composition root
  to supply one to the live `chat_service`.
- `AgentLoop` genuinely DOES use `state.plan` when one is present (advances/fails plan
  steps) — this is real, tested code, just never reached in production because no
  caller ever attaches a plan to `AgentState` in the live path.
- Multi-step behavior that DOES exist in production today comes entirely from
  `AgentLoop`'s own iterate-decide-observe cycle (up to `max_iterations=5`), not from
  planning — proven directly in this milestone's audit (Step 31): the two fully
  deterministic multi-part evaluation cases (`h1`, `h2`) both resolve correctly across
  multiple iterations with zero planning involvement.
- Known planner limitation (pre-existing, not re-verified this milestone): plan-step
  completion is inferred from a single successful tool call per step — a step
  genuinely needing more than one tool call would be marked complete after the first.
- **Recommendation: remains deferred.** No evidence from this or prior audits suggests
  production requests need multi-step planning beyond what `AgentLoop`'s existing
  iteration already provides.

## 14. Model/tool architecture

- **Ollama + Llama 3.2 3B** (local, default, no paid API required) supplies: general
  knowledge for FINAL answers, reasoning for static/technical questions, and the
  TOOL-vs-FINAL decision itself for every case not covered by a deterministic route.
- **Deterministic mechanisms** (`Router`'s temporal and web-search-command routes) —
  zero LLM calls, 100% reliable by construction — handle the narrow, unambiguous
  slice of tool selection: exact current-time phrasings, explicit
  web/online/internet-qualified search commands, and date-for-a-specified-date
  lookups.
- **Current-information retrieval**: exclusively via the `web_search` tool (real
  Tavily Search API call) — the model has no other source of information newer than
  its training data; there is no RAG/semantic-memory-backed current-info path active
  in production.
- **Local deterministic tools** (`time`, `date`) never call the model or the network —
  pure local computation.
- No second model, no paid API beyond the already-optional OpenAI provider switch, and
  no model change were introduced by this or the preceding four milestones.

### 14a. Semantic memory's model roles (Step 33 audit)

Semantic memory is about durable, USER-STATED facts recalled within one session — it
is not RAG over external knowledge, and does not change the current-information
statement above (that remains `web_search`'s job alone).

| Question | Answer |
|---|---|
| Does llama3.2:3b perform memory extraction? | **Yes** — `LLMMemoryExtractor` (`memory_extraction.py`) calls the SAME shared `LLMClient`/Ollama instance as the main reasoning path, via a separate, dedicated prompt. One extra LLM call per successfully completed, session-scoped request when enabled. |
| Does llama3.2:3b receive retrieved memory context? | **Yes** — `LLMDecisionMaker._format_memory()` renders it into the SAME single user-role decision prompt, framed explicitly as untrusted data (never a separate system/developer role, never a separate call). |
| Where exactly does it enter the prompt? | `app/agent/decision_maker.py:_build_prompt()` → `_format_memory()`, rendering the `MEMORY CONTEXT (JSON — ...)` block built by `memory_formatting.py`. |
| Is semantic retrieval independent of llama3.2:3b? | **Yes** — retrieval is pure vector math (embed query via `all-MiniLM-L6-v2`, cosine similarity search); llama3.2:3b is never consulted for retrieval itself, only for (a) the extraction call on write and (b) synthesizing the final answer that consumes retrieved context. |
| Ollama unavailable? | Main path: `RuntimeError` propagates uncaught → HTTP 502 (unchanged, pre-existing). Extraction path: **also propagates uncaught** (only `MemoryExtractionError`, a parse/shape failure, is caught) — see the limitation noted in §10. |
| `all-MiniLM-L6-v2` unavailable (not installed)? | `LocalEmbeddingProvider.__init__` raises `EmbeddingModelLoadError` at `main.py` IMPORT time when `SEMANTIC_MEMORY_ENABLED=true` — the **entire application fails to start**, loudly, by design (not merely "the feature is disabled"). |
| Embedding compute fails per-call? | `EmbeddingComputeError`/`EmbeddingError` caught on both read and write paths (`AgentOrchestrator`), degrades to "no memory this turn" / "candidates discarded," logged at WARNING — the main answer is unaffected. |
| Memory extraction fails (malformed output)? | `MemoryExtractionError` caught, degrades to "storing no semantic memory" — the already-computed main answer is returned unaffected. |
| Vector retrieval fails? | Pure local computation; a session mismatch raises `MemorySessionIsolationError` (never silently degraded — a security signal, not an infrastructure failure) → HTTP 500 via `main.py`'s dedicated handler. |

## 15. Testing architecture

| Category | Location | Notes |
|---|---|---|
| Unit / component tests | `tests/*.py` (67 files) | Tools, state, permissions, gate, registry, reliability, memory layers, telemetry, router, decision maker, etc. |
| Integration tests (`@pytest.mark.integration`) | Scattered across `tests/*.py` + `tests/evaluation/` | Require reachable Ollama and/or a real `TAVILY_API_KEY`; excluded from the default suite via `pytest.ini`'s marker |
| Security tests | `test_production_tool_authorization.py`, `test_permissions.py`, `test_tool_execution_gate.py`, `test_tool_authority_boundary.py`, `test_security_audit_18c.py`, `tests/evaluation/test_capability_evaluation.py`'s Category L | |
| Mechanism tests | `tests/evaluation/test_capability_evaluation.py` Categories I/J/K/M/N | Tool-input robustness, recovery, session memory, timeout, telemetry — all deterministic, single-run |
| Evaluation dataset (end-to-end) | `tests/evaluation/{dataset,runner,metrics}.py` | 55 permanent cases across 14 categories; real production path; versioned |
| Recovery tests | `test_tool_failure_correction.py`, `test_agent_loop_correction.py`, `test_reliability*.py`, `test_production_correction_wiring.py` | |
| Memory tests | `test_memory*.py`, `test_session_*.py`, `test_episodic_memory*.py`, `test_semantic_memory*.py`, `test_anonymous_memory_isolation.py`, `test_semantic_memory_evaluation.py` (Step 33 — the real-embedding-model retrieval-quality and activation-readiness evaluation) | |
| Persistence tests (Steps 36-39) | `test_sqlite_memory.py`, `test_sqlite_episodic_memory.py`, `test_sqlite_semantic_memory.py`, `test_persistence_schema.py`, `test_memory_backend_config.py`, `test_persistence_production_readiness.py` | CRUD, FIFO/LRU/retention parity with the in-memory implementations, session isolation, dedup/supersession, restart-across-processes, schema migration, concurrency, SQL-injection-shaped input, `/health` endpoint, startup validation, failure behavior — all deterministic, no real Ollama |
| Timeout tests | `test_request_timeout.py` | |
| Telemetry tests | `test_telemetry.py`, `test_telemetry_integration.py`, `test_telemetry_production.py` | |
| Routing tests | `test_router.py`, `test_deterministic_temporal_routing.py`, `test_deterministic_web_search_routing.py`, `test_routing_evaluation.py`, `test_web_intent_ambiguity_evaluation.py`, `test_web_search_routing_evaluation.py` | |

**Historical numbers** (Step 33, 2026-09-30, before persistence existed):
non-integration suite 2061 passed, 29 deselected; integration suite 29 passed
(1358.04s); `compileall`: clean.

**Current verified numbers** (Step 39, after persistence Steps 36-39): non-integration
suite **2253 passed, 38 deselected**; `compileall`: clean. (The real-Ollama/Tavily
integration suite was not re-run this milestone — no routing/model/tool code changed;
see §20a.)

## 16. Runtime/deployment assumptions

Single-process, single-worker assumption throughout: every store — in-memory
(`InMemorySessionMemoryStore`, `InMemoryEpisodicMemory`, `InMemorySemanticMemory`) or,
since Step 36-39, SQLite-backed — uses in-process locks (`threading.Lock`) around its
own connection/dict and is not safe for multi-process/multi-worker deployment as-is;
SQLite's own file-level concurrency handles multiple THREADS within one process (the
actual deployment model here — see below) but was never evaluated for multiple
separate OS processes sharing the file. FastAPI's `/chat` handler is a synchronous
`def`, run by Starlette's threadpool; concurrency safety within one process relies on
each store's own locking, verified for every memory layer (session/episodic/semantic/
vector) in both backends. Runs via `uvicorn` locally; no Docker, no CI/CD, no cloud
deployment configuration exists in this repository.

## 17. Current verified metrics

From `tests/evaluation/EVALUATION_REPORT.md` (Step 29 baseline, re-measured Step 30):
overall task success 83.3% (75/90 trials); unnecessary-tool-call rate 0.0% across 90
trials; tool execution success rate 100%; Category B (time) task success 100.0% (post
Step 30, up from 44.4%); security violations 0/5; recovery 5/5; memory 3/3; timeout
2/2; telemetry 4/4; mean latency 4.88s (post Step 30), P50 3.97s, P95 12.16s
(directional, small sample). Test suite (post Step 33): 2061 non-integration
passed, 29 deselected; 29 integration passed; `compileall` clean. Step 33's own
semantic-memory-specific evaluation (`tests/test_semantic_memory_evaluation.py`):
12/12 passed — real-embedding-model semantic-match quality (0.338-0.645 related vs.
0.013-0.075 unrelated similarity), 7/7 retrieval-quality cases correctly classified
at `min_similarity=0.3`, and a real-Ollama end-to-end proof of the concrete use case.

## 18. Explicit non-goals (this milestone and generally)

LangChain/LangGraph integration; wiring `app/ml_router` into production; changing
semantic memory's DEFAULT (still off — Step 33 made it activation-ready and validated,
it did not flip the default); persistent/distributed storage; a vector database; MCP;
paid API integrations beyond the existing optional OpenAI switch; a subscription/credit
system; multi-tenant SaaS identity; further routing-layer changes; any change to
`ToolExecutionGate`, permissions, or the LLM decision pipeline.

## 19. Future extension points

Identified, not scheduled: (a) ~~a shared/persistent backing store~~ — **done, Steps
36-39** (see §21); a genuine multi-process/horizontally-scaled deployment (PostgreSQL +
pgvector behind the same Protocols) remains the next step IF that scale is ever
actually required — not scheduled, no evidence of need today (§16); (b) an operator
deployment actually turning `SEMANTIC_MEMORY_ENABLED=true` on, now that Step 33 has
closed the one concrete readiness gap found (`min_similarity`) and measured the
mechanism genuinely working for its identified use case — still each deployment's own
choice, not a default change; (c) the `h3`-style routing gap, if the routing freeze is
ever lifted; (d) a deliberate decision on model capability (larger/different local
model, or an external one) if Category E's limitation is ever prioritized; (e) wiring
`PlanGenerator` if genuine multi-step-planning need (beyond what `AgentLoop`'s
iteration already covers) is demonstrated; (f) fixing the extraction-time `RuntimeError`
propagation gap (§10), if semantic memory sees real deployment and the race is observed
in practice.

## 20. Architecture freeze statement

As of Step 32, the agent foundation was frozen: the routing layer (Step 30), the LLM
decision pipeline, `ToolExecutionGate`, permissions, and the correction/reliability
system are validated, tested, and not to be modified without a new, separately-scoped,
evidence-driven milestone. That freeze held through Step 33 — no routing, security,
execution-gate, or decision-pipeline file was touched. Step 33 made exactly one
narrow, evidence-driven exception WITHIN its own explicitly-scoped subsystem: semantic
memory (Classification C→B) gained one new, additive, disabled-by-default
configuration field (`SEMANTIC_MEMORY_MIN_SIMILARITY`) and a real-embedding-model
evaluation, closing its one identified activation-readiness gap — the DEFAULT
(disabled) and the frozen subsystems remain untouched. Remaining Classification C/D/E
components stay exactly as built — available, tested where applicable, not activated.

### 20a. The freeze held through Steps 36-39 too

Persistence (Steps 36-39) is a genuinely new, separately-scoped subsystem under §21 —
exactly the kind of work this freeze statement already anticipated ("further work
should select a genuinely new subsystem"), not an exception to it. Verified directly:
zero lines changed in `app/agent/router.py`, `decision_maker.py`, `permissions.py`,
`tool_execution.py`, `tool_registry.py`, `loop.py`, or any routing/security/execution
file across all four persistence milestones. The DEFAULT (`MEMORY_BACKEND=memory`)
reproduces pre-Step-36 behavior exactly. Further work should continue to select a
genuinely new subsystem rather than re-open either this freeze or persistence's own
now-complete scope.
## 21. Persistence architecture (Steps 36-39)

**Status: implemented, tested, opt-in.** Not a future/deferred item (see the former
§9/§12 wording, now corrected) — every memory layer can survive a process restart
today, behind one configuration flag, default off.

### Backend selection

- `MEMORY_BACKEND` (`app/config.py`): `memory` (default) or `sqlite`. Invalid values
  fail application startup immediately and loudly — never silently interpreted as
  either backend.
- `memory` reproduces every milestone through Step 35 exactly: process-local
  dicts/lists, lost on restart, no SQLite path touched, no `.db` file created, `sqlite3`
  never imported beyond Python's own stdlib presence.
- `sqlite` persists ALL FIVE memory layers (conversation, session, episodic, semantic
  records, vectors) to **one local SQLite file** (`SQLITE_DB_PATH`, default
  `./data/conversations.db`, parent directory created automatically). Each layer opens
  its own connection to that same file; there is no cross-process server, no network
  listener, no additional runtime dependency (Python's built-in `sqlite3` module only).

### What is persisted

Exactly what each existing in-memory Protocol already represents, and nothing more:
conversation turns (role + content), session recency metadata, episodic event records
(type, summary, timestamp, small structured metadata), semantic memory records
(content, confidence, active/inactive state, source-event provenance), and their
embedding vectors (tagged with the embedding model that produced them).

### What is deliberately NOT persisted

`AgentState`, tool calls/observations/errors, raw model decision JSON, telemetry
events, or any secret/credential — none of these have a column anywhere in the schema;
this was true of the in-memory stores already and remains true of their SQLite
counterparts (verified structurally, by test, for every table — see
`tests/test_sqlite_*.py`). The existing semantic-memory security gate
(`rejection_reason()`, unchanged since Step 16F) still runs before anything reaches a
table, persisted or not.

### Schema versioning / migrations

One SQLite file, one `schema_meta` table recording an integer version (currently 3).
Numbered, incremental migrations (`app/persistence/db.py`) bring an older database
forward automatically and losslessly on open; a database recorded at a version newer
than the running code supports — or with no migration path defined for a step in
between — fails application startup loudly (`SchemaVersionError`), never silently
reinterpreted or downgraded.

### Restart behavior

Verified directly (subprocess-to-subprocess tests, `tests/test_persistence_production_
readiness.py` and the Step 36/37/38 test files): a value written by one process and
read by a genuinely separate later process round-trips correctly for every layer,
including semantic similarity search. Session isolation remains structural (every
query is parameterized and partition-scoped) and is unaffected by persistence.

### Health check

`GET /health` (Step 39, new — `GET /` is unaffected) reports application liveness,
which backend is configured, whether the SQLite file is currently reachable (a real
read against an already-open connection, not just "did startup succeed"), and whether
its recorded schema version is supported. Deliberately O(1): it never scans semantic or
vector data, regardless of how much a deployment has accumulated.

### Current scope and limitations

- Single-file, single-machine SQLite — matches this architecture's existing
  single-process/single-worker assumption (§16); not a solution for horizontal scaling.
- Brute-force cosine search over persisted vectors, unchanged from the in-memory
  implementation (no ANN, no vector-database infrastructure) — appropriate at the
  per-session scale this layer targets (§12); not benchmarked past that.
- No encryption at rest; relies on normal filesystem permissions, the same trust
  boundary `.env` already relies on.
- **PostgreSQL + pgvector remain future-only**, and only relevant if a genuine
  multi-process/horizontally-scaled deployment is ever actually required — no such
  requirement exists today, and nothing in this milestone schedules one (§19).

