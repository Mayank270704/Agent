# AI Agent Chatbot

A locally-runnable, tool-using conversational agent built on FastAPI, with a
custom (non-LangChain, non-LangGraph) agent runtime: a bounded decision/tool
execution loop, an explicit tool-authorization boundary, self-correction,
and a layered (mostly opt-in) memory system.

This README describes the project **as the repository currently implements
it**. Where a component exists in code but is not part of the live request
path, that is called out explicitly rather than implied.

## What the system does

The service exposes a single chat endpoint. A user sends a message; the
agent — backed by a local LLM (Ollama + Llama 3.2 3B by default) — decides,
turn by turn, whether to answer directly or call one of a small set of
registered tools (web search, current time, date/weekday lookup), observes
the tool's result, and continues until it produces a final answer or a
bounded number of iterations/time is exhausted. Every tool call passes
through an explicit authorization gate before it is allowed to run — the
model *proposes*, it never *authorizes*.

## Architecture overview

```
FastAPI (app/main.py)
    |
ChatService (app/services/chat.py)
    |
AgentOrchestrator (app/agent/orchestrator.py)
    |
AgentState  --->  AgentLoop (app/agent/loop.py)
                       |   \
                       |    \--> LLMDecisionMaker (app/agent/decision_maker.py) -> LLMClient (Ollama)
                       |
                       +--> ToolExecutionGate (app/agent/tool_execution.py)
                                 |
                          resolve -> authorize (PermissionPolicy) -> validate -> confirm -> execute
                                 |
                              ToolRegistry -> Tool.execute() -> ToolResult
                                 |
                              Observation (fed back into AgentState)
```

The loop repeats decide → (optionally) execute-tool → observe until the
model returns a FINAL decision, a failure is not recoverable, `max_iterations`
is reached, or the per-request deadline passes. `AgentOrchestrator` then
wraps the result, optionally commits it to conversation/episodic/semantic
memory, and returns an `AgentResult` to `ChatService`.

## Production request flow

This is the actual, verified path a live `POST /chat` request takes today
(`app/main.py`'s module-level `chat_service` singleton):

```
FastAPI POST /chat
  -> ChatService.ask(message, session_id)
       - resolves conversation memory (per-session store, or a fresh
         request-scoped memory if session_id is omitted)
       - builds a fresh ExecutionContext, request_id, and deadline
       - builds a fresh AgentOrchestrator for this call
  -> AgentOrchestrator.process(message)
       - builds a fresh AgentState
       - AgentLoop.run(state):
           repeat:
             LLMDecisionMaker.decide(state)  -> AgentDecision (TOOL | FINAL)
             if TOOL:
               ToolExecutionGate.execute(tool_name, tool_input, context)
                 resolve (ToolRegistry) -> authorize (PermissionPolicy)
                 -> validate -> confirm -> Tool.execute() -> ToolResult
               -> Observation appended to state
             if FINAL: loop ends
           until FINAL, an unrecoverable failure, max_iterations, or deadline
       - on COMPLETED: commits the turn to conversation memory
         (and, only when session_id was supplied: records one episodic
         memory event, and — only if semantic memory is enabled — extracts
         and writes semantic memory)
  -> AgentResult.answer returned as the HTTP response body
```

Self-correction (`BudgetedCorrectionPolicy`, `app/agent/reliability.py`) is
wired unconditionally: recoverable failures (e.g. invalid tool input, a
transient tool error) can be corrected in-loop, up to a budget, before the
request is failed. `PERMISSION_DENIED` and `CONFIRMATION_REQUIRED` are
never eligible for correction — they terminate the loop.

## Core components

- **AgentLoop** (`app/agent/loop.py`) — generic, bounded control flow. Knows
  nothing about specific tools or how decisions are made; only orchestrates
  decide → execute → observe, enforces `max_iterations` and an optional
  cooperative wall-clock deadline, and applies the injected correction
  policy on failure.
- **LLMDecisionMaker** (`app/agent/decision_maker.py`) — the reasoning
  layer. Turns `AgentState` into an `AgentDecision` (FINAL or TOOL) via a
  prompt to the LLM. Only decides — never executes a tool or mutates state.
  Model output is untrusted text: it is parsed and strictly validated, and
  an invented/unregistered tool name is rejected.
- **ToolRegistry** (`app/agent/tool_registry.py`) — the single source of
  truth for which tools exist and their metadata (`ToolDescriptor`:
  name, description, input schema, capability, risk level).
- **ToolExecutionGate** (`app/agent/tool_execution.py`) — the one
  authoritative boundary that turns a proposed `(tool_name, tool_input)`
  into a real `tool.execute()` call: resolve → authorize → validate →
  confirm → execute.
- **PermissionPolicy** (`app/agent/permissions.py`) — decides ALLOW / DENY /
  CONFIRM for a tool call. Pure and LLM-independent; never reads anything
  the model produced. Production uses `AllowlistPermissionPolicy` with an
  explicit, application-owned allow-list (`time`, `date`, `web_search`).
- **CorrectionPolicy** (`app/agent/reliability.py`) — pure function of
  `(state, failure)` deciding whether a recoverable failure gets one more
  in-loop attempt (`BudgetedCorrectionPolicy`, budget of 2) or fails the
  request.
- **ConversationMemory** (`app/agent/memory.py`) — persistent, per-session,
  in-process store of raw user/assistant turns (message-count bounded). One
  instance per session via `InMemorySessionMemoryStore`; anonymous
  (`session_id` omitted) requests get a fresh, request-scoped instance that
  is discarded when the call returns — never shared across callers.
- **SessionMemory** — the `session_id -> ConversationMemory` mapping
  (`InMemorySessionMemoryStore`), owned by `ChatService`.
- **EpisodicMemory** (`app/agent/episodic_memory.py`) — one compact,
  structured record per successfully completed *session-scoped* request
  (deterministic summary, no LLM call). Foundation for semantic memory
  provenance; not retrieved back into prompts today.
- **Semantic Memory** (`app/agent/semantic_memory.py`,
  `memory_retriever.py`, `memory_writer.py`, `memory_extraction.py`,
  `local_embeddings.py`, `vector_index.py`, `semantic_memory_wiring.py`) —
  a full embed → store → retrieve → prompt-inject pipeline is implemented
  and tested, and **is wired into `app/main.py`'s composition root**, but
  only activates when `SEMANTIC_MEMORY_ENABLED=true` (default `false`). See
  [Memory architecture](#memory-architecture) below.
- **Observability** (`app/agent/telemetry.py`) — a typed `AgentEvent` model
  plus an injectable `EventSink`. Observational only (cannot affect
  execution); a failing sink can never break a request. Default OFF
  (`TELEMETRY_ENABLED=false`); when on, events render as structured log
  lines via `LoggingEventSink`.
- **Request timeout** — `AgentLoop` accepts an absolute
  `time.monotonic()` deadline (`REQUEST_TIMEOUT_SECONDS`, default `120`),
  computed once per `ChatService.ask()` call and checked cooperatively
  between loop iterations (an in-flight LLM/tool call is not interrupted
  mid-call). Separate from `LLMClient`'s own per-call HTTP timeout (60s).
- **Tools** (`app/tools/`) — see [Available tools](#available-tools).

## Available tools

Registered and allow-listed in production (`app/main.py`):

| Tool | What it does | Capability / risk | Confirmation |
|---|---|---|---|
| `time` | Returns the current local date/time from the runtime clock (date, time, weekday, timezone, ISO timestamp). Takes no input. | READ / LOW | No |
| `date` | Parses a free-text date and returns its weekday and calendar breakdown. | READ / LOW | No |
| `web_search` | Searches the web via the Tavily Search API for current information, with basic time-range boosting for time-sensitive queries. | EXTERNAL_NETWORK / MEDIUM | No |

`Router` (`app/agent/router.py`) is **not** part of the live decision path
(see [Experimental / non-production components](#experimental--non-production-components)),
except for two narrow, deterministic pieces `LLMDecisionMaker` uses as
*authoritative* shortcuts — see [Web-search routing](#web-search-routing)
below for the second one. Both bypass the LLM only for their own narrow,
unambiguous class; everything else still goes through the LLM and the same
`ToolExecutionGate`.

## Web-search routing

Three layers decide whether `web_search` runs, in order of how much they
trust the model's own judgment:

1. **Deterministic temporal routing** (`deterministic_temporal_routing=True`,
   Step 20, extended Step 30) — unambiguous "what time/date is it" phrasing
   routes straight to `time`/`date`, never to `web_search`, with no LLM
   call. Unaffected by anything below; checked first. A small set of
   additional current-time phrasings ("what's the time", "what is the
   time", "can you tell me the current time", "can you tell me what time
   it is") were added in Step 30 after the Step 29 end-to-end evaluation
   measured them falling through to unreliable LLM judgment — matched only
   as the ENTIRE message, never as a substring, specifically so a
   technical question like "What is the time complexity of this
   algorithm?" is never misrouted to the local clock.
2. **Deterministic web-search-command routing** (`deterministic_web_search_
   routing=True`, Step 26, extended Step 28) — an explicit, imperative
   instruction that also names "web"/"online"/"internet" (e.g. "search the
   web for X", "look up X online", "find X online", "find X on the
   internet") routes straight to `web_search`, with no LLM call. A bare,
   unqualified "search for X" or "look up X" — with no web/
   online/internet reference — is deliberately **not** included: adversarial
   testing during this milestone found real false positives from that
   broader phrasing on ordinary, non-web requests ("search for the bug in
   this function", "look up to your role model"). Negated phrasing ("don't
   search the web for X") is also excluded. See `Router.deterministic_web_
   search_route` (`app/agent/router.py`) for the exact pattern set.
3. **Everything else — advisory hint + full LLM judgment.** A bare recency
   keyword ("latest", "recent", "current") or the explicit phrase "search
   the web" still gets an advisory `ROUTING HINT` line in the prompt
   (`Router.classify_hint`, Step 8C) — the model decides, and can be, and
   sometimes is, wrong. A bare "search for X"/"look up X"/"look for X" (no
   web/online/internet reference) does **not** get that hint either (Step
   27): a real-model diagnostic measured the hint itself — not model
   nondeterminism alone — as the dominant cause of unnecessary `web_search`
   calls on exactly that class (33% with the hint active vs. 0% with it
   disabled, same cases, same model), so `Router._WEB_INTENT_PHRASES` was
   narrowed to drop those three bare phrases, matching the same qualifier
   discipline the deterministic route (above) already used. This reduces,
   but does not eliminate, incorrect tool selection — see
   [Known limitations](#known-limitations).

## LLM setup

- **Provider**: [Ollama](https://ollama.com), running locally — the default
  and only provider this project requires no paid API key for.
- **Model**: `llama3.2:3b` (Llama 3.2, 3B parameters) by default.
- **Why**: a small model that runs entirely on a local machine (CPU-friendly),
  keeps the core setup free of any paid LLM API dependency, and is fast
  enough for an interactive chat loop with multiple decision iterations per
  request.
- **Not required for the core setup**: an OpenAI API key. `LLM_PROVIDER=openai`
  is also implemented (`app/models/llm.py`) as an alternative provider, but
  it is not the default and requires `OPENAI_API_KEY` to be set.

## Tavily / web search

- The `web_search` tool calls the [Tavily](https://tavily.com) Search API.
- It is only invoked when the agent's decision loop chooses to call
  `web_search` — typically for queries about current/recent information.
- **Requires an API key.** Without `TAVILY_API_KEY` configured, calling
  `web_search` raises a configuration error (surfaced to the user as a
  generic failure message, per the security architecture below — the key
  name itself is never echoed back to the caller).
- Configure it via `.env`:
  ```
  TAVILY_API_KEY=your-real-key-here
  ```
- Never put a real key in documentation, source code, or `.env.example`.

## Installation (Windows)

These commands are verified against this repository's actual layout and
dependency files.

```powershell
# 1. Clone and enter the project
git clone <this-repo-url>
cd chatbot

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Install core dependencies
pip install -r requirements.txt

# 4. (Optional) install semantic-memory dependencies — only needed if you
#    plan to enable SEMANTIC_MEMORY_ENABLED or run its integration tests
pip install -r requirements-embeddings.txt

# 5. (Optional) install the experimental ML router's dependencies — only
#    needed for training/evaluating/testing app/ml_router/, not for
#    running the application
pip install -r requirements-ml-router.txt

# 6. Set up Ollama (local LLM runtime)
#    Download and install from https://ollama.com, then pull the model:
ollama pull llama3.2:3b

# 7. Configure environment
copy .env.example .env
# then edit .env and fill in any values you need (see table below) —
# the defaults work out of the box for the local Ollama path with no
# tools that require external API keys other than web_search
```

## Environment variables

Every variable below is declared in `.env.example` and actually read by
`app/config.py` — nothing invented, nothing omitted (enforced by
`tests/test_env_example.py`).

| Variable | Purpose | Required? | Safe example |
|---|---|---|---|
| `APP_NAME` | Display name used as the FastAPI app title and the `GET /` response body. | No (default: `Chatbot API`) | `Chatbot API` |
| `APP_PORT` | Port `uvicorn` binds to when running via `python -m app.main`. | No (default: `8000`) | `8000` |
| `LOG_LEVEL` | Root logging level (`debug`/`info`/`warning`/`error`/`critical`). | No (default: `info`) | `info` |
| `LLM_PROVIDER` | `ollama` (local, default) or `openai`. | No (default: `ollama`) | `ollama` |
| `MODEL_NAME` | Model name passed to the LLM provider. | No (default: `llama3.2:3b`) | `llama3.2:3b` |
| `OLLAMA_BASE_URL` | Base URL of the local Ollama server. | No (default: `http://localhost:11434`) | `http://localhost:11434` |
| `OLLAMA_MODEL` | Ollama model tag (kept alongside `MODEL_NAME`). | No (default: `llama3.2:3b`) | `llama3.2:3b` |
| `OLLAMA_TEMPERATURE` | Optional Ollama sampling temperature (Milestone 44.1). Unset sends no `options` field at all — identical to every request before this setting existed. | No (default: unset — Ollama's own default) | *(leave empty, or e.g. `0.2`)* |
| `OLLAMA_SEED` | Optional Ollama sampling seed (Milestone 44.1), for reproducing an identical prompt's output run-to-run. | No (default: unset — normal non-reproducible sampling) | *(leave empty, or e.g. `42`)* |
| `OPENAI_API_KEY` | OpenAI API key. Only used/required if `LLM_PROVIDER=openai`. | Conditional | *(leave empty unless using OpenAI)* |
| `TAVILY_API_KEY` | API key for the `web_search` tool's Tavily calls. | Required for `web_search` to work | *(leave empty until you have a real key)* |
| `TELEMETRY_ENABLED` | `true`/`false` — turns on structured event logging. | No (default: `false`) | `false` |
| `REQUEST_TIMEOUT_SECONDS` | Total wall-clock seconds allowed for one `/chat` request. | No (default: `120`) | `120` |
| `SEMANTIC_MEMORY_ENABLED` | `true`/`false` — activates the semantic memory pipeline. Off by default; **not part of the standard production setup**. | No (default: `false`) | `false` |
| `EMBEDDING_MODEL_NAME` | Local sentence-embedding model used if semantic memory is enabled. | No (default: `sentence-transformers/all-MiniLM-L6-v2`) | `sentence-transformers/all-MiniLM-L6-v2` |
| `EMBEDDING_DEVICE` | `cpu` or `cuda`, for the local embedding model. | No (default: `cpu`) | `cpu` |
| `SEMANTIC_MEMORY_MAX_RECORDS_PER_SESSION` | Per-session cap on stored semantic memory records. | No (default: `200`) | `200` |
| `SEMANTIC_MEMORY_MIN_SIMILARITY` | Cosine-similarity floor (`[-1.0, 1.0]`) a memory must meet to be retrieved. Calibrated (Step 33) against the real embedding model — see [Memory architecture](#memory-architecture). | No (default: `0.3`) | `0.3` |
| `MEMORY_BACKEND` | `memory` (default, process-local, lost on restart) or `sqlite` (persists conversation/session/episodic/semantic/vector memory to a local file across restarts — see [Persistence](#persistence-steps-36-39)). Any other value fails startup loudly. | No (default: `memory`) | `memory` |
| `SQLITE_DB_PATH` | Local SQLite file path. Only used when `MEMORY_BACKEND=sqlite`; parent directory is created automatically. Never commit the resulting `.db` file (already in `.gitignore`). | No (default: `./data/conversations.db`) | `./data/conversations.db` |
| `CORS_ALLOWED_ORIGINS` | Comma-separated list of browser origins allowed to call this API cross-origin. Default names the local Next.js dev frontend explicitly (`http://localhost:3000`, `http://127.0.0.1:3000`) — never a wildcard. Add a deployed frontend's real origin here. | No (default: `http://localhost:3000,http://127.0.0.1:3000`) | `http://localhost:3000,http://127.0.0.1:3000` |

Never commit a real `.env` file — it is already excluded via `.gitignore`.

## Running the application

Backend (FastAPI):
```powershell
# Ensure Ollama is running locally, then either:
python -m app.main
# or, equivalently:
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API is then available at `http://localhost:8000`.

Frontend (Next.js, `frontend/` — see [Frontend](#frontend) below):
```powershell
cd frontend
npm install
npm run dev
```

The chat UI is then available at `http://localhost:3000`, talking to the backend
above. Run both at once for the full product experience.

## Deployment / Running Locally

This project's only verified, supported deployment target is **local /
single-machine** (your own computer, or one server you control) — the
real local model (Ollama + `llama3.2:3b`) is a hard requirement, and this
repository runs single-process/single-worker throughout (see
`docs/ARCHITECTURE_STATUS.md` §16). It is not claimed to be cloud- or
container-deployable; see `docs/MILESTONE_44_DEPLOYMENT.md` for the full
reasoning. **Total mandatory cost: ₹0** — local Ollama inference, local
SQLite, and local Next.js/FastAPI all run with no billing; Tavily web
search is the one optional external service, with free-tier limits.

Production-style startup (no `--reload`, no `npm run dev`):
```powershell
# Backend
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Frontend (build once, then serve the build)
cd frontend
npm run build
npm run start
```

Verify before use: `curl http://localhost:8000/health` should return
`{"status": "ok", ...}`, and `http://localhost:3000` should load and
complete a real chat round trip.

**Note:** `NEXT_PUBLIC_API_BASE_URL` is baked into the frontend's static
JS at `npm run build` time, not read at `npm run start` time — set it in
the environment *before* building if the backend isn't at the default
`http://localhost:8000`, and rebuild after changing it.

See `docs/MILESTONE_44_DEPLOYMENT.md` for the full deployment topology,
Ollama setup, SQLite persistence setup, a security checklist, and
troubleshooting.

## API

```
GET  /              identification payload
GET  /health         application/persistence health
POST /chat           send a message, get the agent's reply
GET  /docs            interactive, try-it-out API documentation (Swagger UI)
GET  /openapi.json    machine-readable OpenAPI schema
```

### `GET /`

Returns a simple identification payload. Unchanged since the project's first
milestone — kept minimal on purpose; see `GET /health` for actual status reporting.

Response:
```json
{"message": "Chatbot API"}
```

### `GET /health`

A minimal liveness/readiness probe (Step 39). Returns `200` when healthy, `503` when
the configured persistence backend is unreachable or its schema is unsupported. Cheap
by design — never scans semantic/vector data regardless of how much a deployment has
accumulated, and never calls Ollama or Tavily.

Response (`MEMORY_BACKEND=memory`, the default):
```json
{"status": "ok", "application": "running", "memory_backend": "memory", "persistence": "not_configured"}
```

Response (`MEMORY_BACKEND=sqlite`, healthy):
```json
{"status": "ok", "application": "running", "memory_backend": "sqlite", "persistence": "available", "schema_version": 3, "schema_supported": true}
```

### `POST /chat`

Request:
```json
{
  "message": "What is a transformer?",
  "session_id": "demo-session"
}
```

- `message` — required, non-empty string, at most 8000 characters.
- `session_id` — **optional**. When provided, the conversation (and, when
  enabled, episodic/semantic memory) is scoped to that key — **the same
  session_id preserves conversation/session state** across requests, resuming
  the same conversation. It is a **conversation-scope key only, never an
  identity or authentication mechanism**. **Omitting session_id creates an
  isolated, request-scoped conversation context** with no memory of prior
  turns, which leaves no state behind once the response is returned. At most
  200 characters when supplied.

Response:
```json
{"reply": "A transformer is a neural network architecture..."}
```

The response shape is always exactly `{"reply": string}` — regardless of whether the
agent answered directly or used a tool first; a tool call, if any, happens entirely
server-side.

Error responses (always JSON, always `{"detail": ...}`):
- `400` — empty message, blank `session_id`, or other invalid business-rule input.
- `422` — malformed request body, a missing required field, or `message`/`session_id`
  exceeding its length limit.
- `500` — an internal server-side error (a defensive memory-consistency check, or any
  other unexpected failure) — never includes internal detail (file paths, stack
  traces, secrets) in the response body.
- `502` — an upstream dependency (e.g. Ollama) failed or was unreachable.

### Interactive documentation

`GET /docs` serves FastAPI's auto-generated Swagger UI — request/response schemas,
field descriptions, and a try-it-out console, kept accurate to the real behavior
above automatically (same Pydantic models the API itself validates against). `GET
/openapi.json` serves the underlying machine-readable schema, useful for generating a
typed client.

## Frontend

A minimal chat UI lives in `frontend/` (Next.js App Router + React + TypeScript +
Tailwind CSS) — a thin client over the `POST /chat` API above; it implements no
agent logic of its own.

```powershell
cd frontend
npm install
cp .env.local.example .env.local   # optional -- defaults to http://localhost:8000
npm run dev
```

Open `http://localhost:3000`. Type a message and press Enter (Shift+Enter for a new
line) or press Send. The same browser tab resumes its conversation across a page
refresh (a `session_id` is generated once and kept in `localStorage`); "New Chat"
starts a fresh, isolated `session_id`. A status dot in the header reflects `GET
/health` (checked once on load, never polled).

- **Configuration**: `NEXT_PUBLIC_API_BASE_URL` (see `frontend/.env.local.example`)
  — the only environment variable the frontend reads, and the only thing that is
  ever public (`NEXT_PUBLIC_*` is inlined into the browser bundle by Next.js; no
  secret ever belongs in a variable with that prefix). Defaults to
  `http://localhost:8000`.
- **CORS**: the backend's `CORS_ALLOWED_ORIGINS` (see
  [Environment variables](#environment-variables) above) must include whatever
  origin the frontend is actually served from — the shipped default already covers
  `npm run dev`'s own `http://localhost:3000`.
- **What it shows**: only what the API actually returns — the chat transcript and a
  simple online/degraded/offline status. No tool/reasoning/memory/admin panel exists
  in the UI, because the backend exposes no API for any of those.

See `docs/MILESTONE_42_FRONTEND.md` for the full architecture, and
`frontend/README.md` for frontend-specific scripts.

## Testing

```powershell
# Full non-integration suite (no network, no live Ollama/Tavily required)
python -m pytest -m "not integration" -q

# Integration tests only — require a reachable Ollama instance and/or a
# real TAVILY_API_KEY; excluded from the default suite for that reason
python -m pytest -m "integration" -q

# Everything
python -m pytest -q

# Compile check (syntax-level sanity check across the whole codebase)
python -m compileall app tests
```

As of Step 42 (frontend/demo UI), on this machine:
**2275 passed, 38 deselected** for the backend non-integration suite (up
from 2268/38 after Step 41 — Step 42's only backend change was
configuration-backed CORS, `CORS_ALLOWED_ORIGINS`/`CORSMiddleware`, plus
its focused tests; no agent/routing/model/memory/persistence code
changed, so the real-Ollama/Tavily integration suite was not re-run this
milestone), and a clean `compileall` run with no errors. The frontend
(`frontend/`) has its own focused suite — **16 passed** (`npm test`,
Vitest, `lib/` request/error/session-id logic) — plus a clean `npm run
lint`, `npm run typecheck`, and `npm run build`; see
`docs/MILESTONE_42_FRONTEND.md` for the full results.

Step 43 added a real browser end-to-end suite
(`frontend/e2e/`, Playwright) that drives the actual built UI against the
real backend and a real local Ollama — **24 passed** (`npm run e2e` from
`frontend/`; requires the backend running separately, e.g. `python -m
app.main` or `uvicorn app.main:app`). It covers connectivity, the normal
LLM/time/date/web-search flows, multi-turn session memory, New Chat
isolation, refresh persistence, invalid input, simulated backend-down and
HTTP-error handling, CORS, and security sanity checks. See
`docs/MILESTONE_43_E2E.md` for the full matrix and results.

The last full
backend integration run (Step 34): **29 passed** for the integration
suite (run against a real local
Ollama instance, a real Tavily API key, and the real local embedding model
— this took roughly 22-23 minutes due to genuine local LLM inference
across several repeated-trial evaluation suites; one earlier run saw a
single transient socket-timeout failure unrelated to any code change,
confirmed by an immediate isolated re-run passing cleanly in 199s — a
known
characteristic of long, network-dependent local LLM sessions, not a
regression). Integration tests are excluded from the default suite via the
`integration` pytest marker (`pytest.ini`) precisely because they require
live external dependencies (Ollama running locally,
and/or Tavily network access) that are not assumed to be present in every
environment.

`tests/evaluation/` (Step 29) is a separate, permanent, versioned
end-to-end capability evaluation — not a pass/fail gate — covering 14
categories (direct answers, time/date, explicit web search, current
information, static/historical, local-context, multi-part, tool-input
robustness, failure/recovery, session memory, security, timeout,
telemetry) through the real production path with repeated trials for
model-dependent cases. Run it directly with
`python -m pytest tests/evaluation/ -m "not integration" -q` (mechanism
checks) or add `-m "integration"` for the full real-Ollama/Tavily
evaluation. See `tests/evaluation/EVALUATION_REPORT.md` for the
methodology and a real run's measured results, including honest
limitations and cross-trial model variance — this is a baseline
measurement, not a claim that the agent has been tuned to it.

See also `docs/ARCHITECTURE_STATUS.md` for the full component-by-component
architecture inventory and freeze statement (Step 32), and
`docs/MODEL_CAPABILITY_EVALUATION.md` for a dedicated evaluation of
`llama3.2:3b`'s own capability boundary within this architecture (Step 34)
— 17 categories (direct answers, reasoning, arithmetic, tool selection,
tool-input quality, correction, synthesis, memory usage, instruction
following, adversarial resistance, and more), with an explicit model
capability classification and strategy decision.

## Security architecture

The central boundary in this system: **the LLM proposes a tool call; it
never authorizes one.**

```
LLMDecisionMaker  -->  "I'd like to call web_search with {...}"  (a proposal)
        |
        v
ToolExecutionGate.execute(tool_name, tool_input, ExecutionContext)
    1. resolve     — look the tool up in ToolRegistry (the single source
                      of truth for what tools exist)
    2. authorize    — PermissionPolicy.evaluate(descriptor, context) ->
                       ALLOW / DENY / CONFIRM, using only application-owned
                       state (an explicit allow-list in production) — never
                       anything the model, user, or memory produced
    3. validate     — the tool's own input validation
    4. confirm      — block if the policy said CONFIRM and no trusted
                       confirmation exists for this request
    5. execute      — only now does Tool.execute() actually run
```

Nothing downstream of `LLMDecisionMaker` ever reads a model-produced field
named `authorized`, `permission`, `confirmed`, or similar — the parsed
`AgentDecision` only carries `tool_name` and `tool_input`, so there is no
field for the model to set that could grant itself authorization even if
it tried.

Other security-relevant properties, verified in the code:

- **`.env` protection**: `.env` is git-ignored; only `.env.example` (with
  empty placeholders for every secret field) is tracked.
- **No secrets in source**: API keys are read exclusively from environment
  variables via `app/config.py`; nothing is hard-coded.
- **Failure messages are sanitized**: a failed request returns a generic
  "I could not complete this request" message to the caller — internal
  detail (e.g. which configuration variable was missing, the full list of
  registered tool names) is logged server-side only, never echoed in the
  HTTP response.
- **Request/session limits**: `ConversationMemory` is bounded by a
  message-count cap; anonymous requests get an isolated, discarded-after-use
  memory instance (no cross-caller leakage); each request has an absolute
  wall-clock deadline (`REQUEST_TIMEOUT_SECONDS`) and the loop has a hard
  `max_iterations` bound (default 5).
- **Telemetry cannot leak sensitive content**: `AgentEvent` has a closed,
  typed field set with no free-text payload field — raw prompts, model
  output, tool input/output, and memory content cannot flow through it.

## Memory architecture

Four distinct layers exist in the codebase. **Only conversation memory and
(when a `session_id` is supplied) episodic memory are exercised by every
live request; semantic memory is fully built and wired but disabled by
default.**

| Layer | Module | Lifetime | Active in production? |
|---|---|---|---|
| Conversation memory | `app/agent/memory.py` | Per session (or per anonymous request, then discarded) | **Yes**, always |
| Session memory | `app/agent/memory.py` (`InMemorySessionMemoryStore`) | Lifetime of the `ChatService` instance | **Yes**, always (it's what makes `session_id` resumable) |
| Episodic memory | `app/agent/episodic_memory.py` | Partitioned by session | **Yes, only when `session_id` is supplied** — anonymous requests record no episode |
| Semantic memory | `app/agent/semantic_memory.py` + retriever/writer/extractor/embeddings/vector index | Partitioned by session, capped per session | **No — disabled by default, but activation-ready (Step 33).** Fully implemented, tested, and wired into `app/main.py`'s composition root behind `SEMANTIC_MEMORY_ENABLED` (default `false`). When enabled, it also requires installing `requirements-embeddings.txt` — missing it means the app fails to **start**, loudly, by design. |

By default (`MEMORY_BACKEND=memory`), all memory stores here are **in-process** (no
database, no external store) and are lost on process restart — **including semantic
memory's own state**. Setting `MEMORY_BACKEND=sqlite` persists every layer above to a
local file instead — see [Persistence](#persistence-steps-36-39) below.

## Persistence (Steps 36-39)

**Opt-in, off by default.** `MEMORY_BACKEND=memory` (the default) is byte-identical to
every milestone before Step 36 — no SQLite path is touched, no `.db` file is created.

Setting `MEMORY_BACKEND=sqlite` persists **all five** memory layers above —
conversation, session, episodic, semantic records, and their embedding vectors — to
**one local SQLite file** (`SQLITE_DB_PATH`, default `./data/conversations.db`) across
process restarts. No server, no network listener, no extra dependency — just Python's
built-in `sqlite3` module.

- **What's persisted**: exactly what each layer's existing in-memory Protocol already
  represents (conversation turns, episodic event summaries, semantic facts with their
  provenance and confidence, and their vectors) — nothing more.
- **What's deliberately never persisted**: `AgentState`, tool calls/observations, raw
  model decision JSON, telemetry events, or any secret/credential. The existing
  semantic-memory security gate (content/instruction/secret rejection) still runs
  before anything reaches a table.
- **Schema/migrations**: one versioned schema (`app/persistence/db.py`), numbered
  migrations applied automatically and losslessly on open; a database at an
  unsupported version fails startup loudly rather than being silently reinterpreted.
- **Restart behavior**: verified directly, process-to-process — a value written by one
  process and read by a later, separate process round-trips correctly for every layer,
  including semantic similarity search. Session isolation stays structural and is
  unaffected.
- **`GET /health`**: reports whether the configured backend is reachable and its
  schema version supported — see [API usage](#api-usage).
- **Scope/limitations**: single SQLite file, matching this project's existing
  single-process assumption (not a horizontal-scaling solution); the same brute-force
  cosine search as the in-memory implementation (no ANN/vector database); no
  encryption at rest (same filesystem-permission trust boundary as `.env`).
  PostgreSQL + pgvector remain a future-only option, relevant only if a genuine
  multi-process/horizontally-scaled deployment is ever actually required — not
  scheduled, no evidence of need today.

See `docs/ARCHITECTURE_STATUS.md` §21 for the complete picture.

### Semantic memory: concrete use case and activation (Step 33)

**Implemented ≠ enabled.** Semantic memory has been fully built since early in this
project and remains implemented; Step 33 is what made it genuinely *activation-ready*
and *evaluated*, without changing its default (still off).

- **The use case it solves that conversation memory cannot**: conversation memory is a
  20-message FIFO window *per session* — a preference stated early in a long
  conversation is evicted from it. Semantic memory keeps it retrievable for the rest
  of that same session, and — unlike conversation memory, which has no search
  capability at all — retrieves it even when a later question uses **completely
  different wording** than how the fact was originally stated (embedding similarity,
  not keyword/text matching). It does **not** provide cross-session/cross-visit memory
  — a new `session_id` still starts with nothing, exactly like conversation and
  episodic memory.
- **What changed in Step 33**: an audit found that the retrieval wiring had no
  relevance floor (`min_similarity` was unset, i.e. `-1.0` — accepts anything). A
  direct measurement against the real embedding model (`all-MiniLM-L6-v2`) found
  unrelated-query similarities of 0.01–0.08 against a stored fact, and
  semantically-related-but-differently-worded queries at 0.34–0.65 — a wide, clean
  gap. `SEMANTIC_MEMORY_MIN_SIMILARITY` (default `0.3`) closes that gap; see
  `tests/test_semantic_memory_evaluation.py` for the full evaluation, including a
  real-Ollama, real-embeddings, end-to-end proof of the use case above.
- **Model roles**: retrieval itself never calls Llama 3.2 3B — it is pure vector math
  against the local embedding model. Llama 3.2 3B is used for two separate things:
  extracting candidate facts from a completed turn (one extra LLM call, via a
  dedicated prompt, only on a successfully completed session-scoped request), and
  synthesizing the final answer that may consume retrieved memory context (rendered
  into the same untrusted-data-framed prompt block described in
  [Security architecture](#security-architecture)).
- **Known limitation**: if Ollama becomes unreachable specifically during the
  post-answer extraction call (not the main decision call), that failure is not
  caught and the already-computed correct answer is lost, surfacing as HTTP 502. Rare
  and narrow (requires Ollama to fail in a specific window, and only reachable when
  semantic memory is enabled) — documented, not patched, per this project's
  established preference for accurate documentation over speculative fixes.

## Known limitations

Verified against the codebase and its own in-code capability-assessment
notes (`app/main.py`, `app/agent/decision_maker.py`), not speculative:

- **Local 3B model limitations**: `llama3.2:3b` is small; its tool-vs-final
  decisions and final-answer synthesis are noticeably less reliable than a
  larger hosted model, particularly on ambiguous phrasing.
- **Tool-selection under-calling was measured directly**: before the
  deterministic temporal routing fix, an internal capability assessment
  measured a 76.5% missed-tool-call rate for time/date queries even with
  an advisory routing hint in the prompt (and 0% unnecessary calls). That
  specific class (unambiguous "what time/date is it") is now handled by a
  deterministic, non-LLM shortcut; ambiguous phrasing and other tool
  categories (e.g. `web_search`) still depend entirely on the LLM's own
  judgment and are not guaranteed.
- **Web search under-calling — partially mitigated (Step 26), not
  eliminated**: an explicit search *command* naming "web"/"online"/
  "internet" is now handled deterministically (see
  [Web-search routing](#web-search-routing)) — a real before/after
  evaluation against the local model (`tests/test_web_search_routing_
  evaluation.py`, 16-case dataset) measured `missed_tool_call_rate` drop
  from 40% to 20% and `tool_selection_accuracy` rise from 60% to 80% on
  that run, with 100% of explicit-command cases now caught with zero
  missed LLM calls (this varies run to run; the local model's output is
  nondeterministic — see that test file for exact numbers and reproduction
  steps). Outside an explicit command — a bare recency keyword or an
  ordinary current-information *question* ("What is the latest AI news?")
  — whether the model calls `web_search` is still entirely its own,
  non-deterministic judgment, advisory-hinted but not guaranteed; the same
  evaluation still measured missed calls in that class. A bare, unqualified
  "search for X"/"look up X" (no web/online/internet reference) is also
  still LLM-judged, deliberately: it was excluded from the deterministic
  route after adversarial testing found false positives, and (Step 27) is
  now also excluded from the advisory hint for the same measured reason —
  a real-model diagnostic isolated the hint itself as the dominant cause of
  unnecessary `web_search` calls on that class (33%→0% across repeated
  trials on the same cases with only the hint toggled). A permanent,
  broader before/after evaluation (`tests/test_web_intent_ambiguity_
  evaluation.py`, 20-case dataset, 3 trials per arm) measured a smaller but
  consistent improvement on the full mix of categories: average
  `unnecessary_tool_call_rate` 8.9%→4.4%, average accuracy 88.3%→90.0%.
  That same evaluation also measured real cross-trial variance — 5 of 20
  cases produced a *different* decision across 3 identical trials under
  unchanged code — confirming a meaningful share of residual incorrect
  selections is irreducible llama3.2:3b nondeterminism, not something a
  further prompt or routing change can fully close without a larger or
  different model.
- **Semantic memory is disabled by default** and is not part of the standard,
  assumed production setup — Step 33 made it activation-ready and evaluated (see
  [Memory architecture](#memory-architecture)), but the default is unchanged, and an
  extraction-time-Ollama-failure edge case remains a known, documented, un-patched
  limitation for deployments that do enable it.
- **Planner (`PlanGenerator`) is not production-wired**: it is fully built
  and independently tested, but `app/main.py`'s live `chat_service` does
  not pass a `plan_generator`, so no request goes through plan generation
  today.
- **Single-instance only, persistence is opt-in**: `MEMORY_BACKEND=sqlite`
  (Steps 36-39) persists conversation/episodic/semantic/vector memory
  across process restarts (default remains `memory`, non-persistent — see
  [Persistence](#persistence-steps-36-39)); neither backend supports
  multi-instance/horizontal-scaling deployment as-is.
- **No UI confirmation workflow**: `PermissionPolicy`'s `CONFIRM` outcome
  and `ExecutionContext.confirmed_tools` exist structurally, but nothing in
  this codebase currently populates a confirmation — a tool the policy
  marks `CONFIRM` will always block, since there is no path for that
  confirmation to be supplied yet.

## Experimental / non-production components

- **ML router** (`app/ml_router/`) — an independent local intent classifier
  (TF-IDF + Logistic Regression, Milestone 22) that predicts DIRECT / WEB /
  TIME / DATE for a message. It is a **benchmarking artifact, not an
  integration**: nothing in `app/agent/`, `app/services/`, or
  `app/main.py` imports anything from this package, and it has no code
  path to a `Tool`, `ToolRegistry`, `PermissionPolicy`, or
  `ExecutionContext` — it structurally cannot execute, authorize, or
  confirm anything. Its dependencies (`requirements-ml-router.txt`) are
  not part of the core install and are not required to run the
  application.
- **Router** (`app/agent/router.py`) — a regex + LLM classifier kept in the
  codebase from an earlier architecture. `AgentOrchestrator` does not use
  it for decision-making; the only production use is its narrow,
  deterministic, LLM-free temporal-phrase matcher (see
  [Available tools](#available-tools)).

## Project status

**Not claimed to be "production-ready."** What is true today, verified
against the repository:

- The core request path (FastAPI → ChatService → AgentOrchestrator →
  AgentLoop → LLMDecisionMaker → ToolExecutionGate → Tools) is implemented,
  tested, and functions end-to-end against a local Ollama instance.
- Tool authorization, self-correction, request timeouts, and conversation
  memory are live in every request.
- 2275 backend non-integration tests pass; the codebase compiles cleanly.
  The frontend's own focused suite (16 tests, Vitest) and `npm run
  build`/`lint`/`typecheck` all pass too — see
  `docs/MILESTONE_42_FRONTEND.md`.
- Persistence (`MEMORY_BACKEND=sqlite`) and `GET /health` are implemented
  and tested (Steps 36-39), opt-in, off by default — see
  [Persistence](#persistence-steps-36-39).
- The backend/agent architecture was audited end-to-end and frozen at
  Milestone 40 (`docs/FINAL_PRODUCTION_AUDIT.md`: 0 P0, 0 P1 findings).
  Milestone 41 polished the HTTP API itself (request-size validation,
  OpenAPI documentation, a consistent JSON error shape) without changing
  that architecture — see `docs/MILESTONE_41_API_POLISH.md`. Milestone 42
  added a minimal chat frontend (`frontend/`) and the one backend change
  it required (configuration-backed CORS) — architecture remains frozen.
- A minimal chat UI (`frontend/`, Next.js + React + TypeScript + Tailwind)
  now exists and consumes the API above directly — see
  [Frontend](#frontend).
- The full product (browser → frontend → API → agent → Ollama/tools →
  response) was verified end-to-end at Milestone 43: 24 real-browser
  Playwright tests pass against the real backend and a real local
  `llama3.2:3b`, alongside the existing backend/frontend regression suites
  — see `docs/MILESTONE_43_E2E.md`.
- Semantic memory, the planner, and the ML router are built and
  independently tested but are either disabled by default or entirely
  unwired from the live request path, by design.
- Known reliability gaps stem primarily from the small local model's own
  decision quality (see [Known limitations](#known-limitations)), not from
  the surrounding architecture.
