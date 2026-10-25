# Milestone 41 — API / Product Polish

Audited and polished against the actual repository on 2026-10-01, following directly
from Milestone 40's verdict (A. PRODUCTION-READY FOUNDATION, 0 P0, 0 P1).

## 1. Scope

Make the existing, frozen backend's HTTP API clean, documented, predictable, and
ready for a frontend client — without touching agent architecture, routing, memory,
persistence, the model, or any security-boundary component. Inspected fresh:
`app/main.py`, `app/services/chat.py`, `app/config.py`, `tests/test_api.py`,
`README.md`, `docs/ARCHITECTURE_STATUS.md`, `docs/FINAL_PRODUCTION_AUDIT.md` — not
assumed from prior reports.

## 2. Current API Surface

Confirmed by source read and live `TestClient` exercise before any change:

| Endpoint | Status before this milestone |
|---|---|
| `GET /` | Identification payload, `{"message": settings.app_name}` |
| `GET /health` | Step 39's liveness/readiness probe, already correct |
| `POST /chat` | `{"message": str, "session_id": str\|None}` → `{"reply": str}`; no length limits on either field |
| `GET /docs` | Reachable (FastAPI default), minimal metadata (title only, no description, no field descriptions/examples) |
| `GET /openapi.json` | Reachable, same minimal-metadata state |
| CORS | **Not configured** — confirmed via `grep -rn "CORSMiddleware\|allow_origins" app/` returning nothing |

## 3. API Contract

**Unchanged.** `POST /chat` still accepts exactly `{"message": "...", "session_id":
"..."}` and returns exactly `{"reply": "..."}` — no field was added, renamed, or
removed. The only contract change is an upper length bound on two existing fields
(§4), which is backward compatible with every existing caller (no legitimate chat
message or session key was ever near 8000/200 characters).

## 4. Request Validation

Audited before changing anything:

| Case | Before | After |
|---|---|---|
| `message` missing | 422 (Pydantic, required field) | unchanged |
| `message` empty (`""`) | 400 (`ChatService.ask()`'s own check) | unchanged |
| `message` whitespace-only | 400 | unchanged |
| `message` extremely large | **accepted, unbounded** — no limit anywhere in the stack | **422** if over 8000 characters |
| `session_id` absent | isolated, request-scoped conversation | unchanged |
| `session_id` empty/blank | 400 | unchanged |
| `session_id` extremely large | **accepted, unbounded** | **422** if over 200 characters |
| Malformed JSON body | 422 (FastAPI default) | unchanged (verified) |
| Unexpected extra fields | silently ignored | unchanged (verified — forward-compatible for a frontend client; not weakened to `extra="forbid"`, since unknown fields are structurally inert — `ChatRequest` only ever reads `.message`/`.session_id`) |

**Finding**: nothing upstream of `ChatRequest` limited request body size at all
(Starlette/uvicorn apply no default limit) — an unbounded `message`/`session_id` was
a genuine, open resource-exhaustion vector for a public-facing endpoint. This is the
one validation gap closed this milestone, via `pydantic.Field(max_length=...)` —
pure API-layer validation, zero architecture change.

**Limits are not arbitrary**: `MAX_MESSAGE_LENGTH = 8000` follows the same reasoning
already used elsewhere in this codebase for bounded text (`MAX_CANDIDATE_CONTENT_
CHARS=300`, `_MAX_OBSERVATION_DATA_CHARS=2000`), sized larger since a chat message is
the primary request, not a derived artifact — ~2000 tokens, comfortable for a long
question while bounding worst-case payload size. `MAX_SESSION_ID_LENGTH = 200` is
generous for any realistic session-key scheme (UUIDs are 36 characters) while
bounding it as a SQLite `TEXT PRIMARY KEY` / dict key. Both are internal constants in
`app/main.py`, not env-configurable — no new setting was introduced, consistent with
"do not invent speculative configurability" for a value nothing yet asks to tune.

## 5. Error Handling

All eight scenarios tested live (`TestClient`) before and after:

| Scenario | HTTP status | Response shape | Leaks anything? |
|---|---|---|---|
| A. Normal request | 200 | `{"reply": "..."}` | No |
| B. Invalid request body (malformed JSON) | 422 | FastAPI's standard validation-error JSON | No |
| C. Empty/invalid message | 400 | `{"detail": "User message cannot be empty."}` | No |
| D. Tool failure (precondition, e.g. missing `TAVILY_API_KEY`) | **200** | `{"reply": "I could not complete this request..."}` | **No — not even the env var name.** See finding below. |
| E. Model failure (`RuntimeError`, Ollama unreachable) | 502 | `{"detail": "Ollama connection failed at <base_url>: ..."}` | Only the non-secret base URL (pre-existing, unchanged, reviewed in Milestone 40) |
| F. Request timeout | 200 | `{"reply": "I could not complete this request..."}` | No (pre-existing `AgentOrchestrator` behavior, unchanged) |
| G. SQLite failure | 503 (`/health`) or **500** (`/chat`, new catch-all) | JSON `{"detail": "Internal server error."}` | No — exception type logged server-side only |
| H. Semantic-memory failure | Degrades silently to "no memory this turn"; answer unaffected | 200 | No (pre-existing, unchanged — `EmbeddingError` caught narrowly in `AgentOrchestrator`) |

**Finding, corrected during test-writing** (documented honestly, not hidden): a tool
*precondition* failure (a `ValueError` raised inside `WebSearchTool.execute()`, e.g. a
missing API key) is **not** the same `ValueError` path `/chat`'s own `except
ValueError` branch handles — that branch only ever sees a `ValueError` from
`ChatService.ask()`'s own input validation (an empty message), raised *before*
`AgentOrchestrator` exists. A tool-level `ValueError` is instead caught inside
`AgentLoop`, classified `INVALID_TOOL_INPUT`, and run through the production
`BudgetedCorrectionPolicy`; since the identical failure repeats on retry, it
terminates as `FAILED` after two attempts and `AgentOrchestrator._failure_answer`
(an existing Step 17 hardening) returns the fixed, generic, safe string — **never**
the `ValueError`'s own text — as an ordinary 200 reply. This is *more* conservative
than a 400 with the env var name would have been. No code change was needed; the
architecture already handles this correctly. Verified by a new regression test.

**Change made**: a global `@app.exception_handler(Exception)` catch-all, so whatever
escapes every endpoint's own handling (previously: Starlette's default bare
plain-text `"Internal Server Error"`, inconsistent with every other error path's
JSON shape) now returns the same `{"detail": "Internal server error."}` JSON as every
other 5xx. Verified directly that this does **not** intercept `HTTPException`s a
route raises on purpose (FastAPI resolves the more specific handler first) — existing
400/500/502 paths are byte-for-byte unaffected (regression-tested).

## 6. Health Endpoint

Reviewed against Step 39's implementation — found already correct, unchanged in
behavior:

- Distinguishes application liveness, configured backend, persistence reachability,
  and schema support — confirmed via source read and the 19 existing, still-passing
  tests in `tests/test_persistence_production_readiness.py`.
- Healthy → 200; degraded/unreachable SQLite → 503 — confirmed.
- `MEMORY_BACKEND=memory` → always healthy, never touches SQLite — confirmed.
- No semantic/vector scan — confirmed (one cheap, already-open-connection read only).
- No Ollama call, no Tavily call — confirmed by source read (no import of either
  client in the handler).
- No state mutation — confirmed (`health_check()` is a pure `SELECT`).

Only OpenAPI metadata (`summary`/`description`/`responses`) was added to the route
decorator — zero behavioral change.

## 7. OpenAPI / Swagger

Before: title only (`"Chatbot API"`), no description, no endpoint summaries, no field
descriptions or examples. A frontend developer would have had to read the response
JSON ad hoc to understand `session_id`'s semantics.

After: app-level `description` (what the API does, how to use `/chat`/`/health`);
per-endpoint `summary`/`description`/`response_description`/`responses` (documenting
400/422/500/502 for `/chat`, 200/503 for `/health`); `Field(description=...,
examples=...)` on `message`, `session_id`, and `reply`. Verified live: `/openapi.json`
exposes exactly `/`, `/health`, `/chat` with non-empty descriptions; `/docs` renders
successfully. No new documentation framework — pure FastAPI/Pydantic metadata, as
required.

## 8. CORS Decision

**Not configured. Deliberately deferred to Step 42.**

No frontend origin is known yet — Step 42 (frontend/demo) has not happened, so there
is no real port/domain to allow-list. Adding `allow_origins=["*"]` was explicitly
forbidden and would be insecure for a POST endpoint that mutates session state; adding
a specific hardcoded origin now would be guessing (the frontend's dev server port,
framework, and eventual deployed domain are all still unknown); adding a
configuration-driven mechanism (e.g. a new `CORS_ALLOWED_ORIGINS` setting) with no
known value to put in it would be speculative plumbing for a need that does not exist
yet — exactly what this milestone's change policy forbids ("do not add a CORS
subsystem if there is no actual need yet"). Confirmed live: `POST /chat` returns no
`Access-Control-Allow-Origin` header for a cross-origin-shaped request today
(regression-tested). **Recommendation for Step 42**: add `CORSMiddleware` at that
point, configured from a new env var holding the frontend's actual origin(s) — not
a wildcard.

## 9. Frontend Readiness

Verified directly, this session:

- Stable `POST /chat` contract — unchanged request/response shape.
- Predictable JSON response — always `{"reply": string}` on success.
- `session_id` support — resumes conversation when reused; isolated when omitted
  (live-tested, both directions).
- Predictable HTTP errors — every error path now returns JSON with a `detail` field
  (400/422/500/502), consistently.
- CORS decision documented (§8) rather than silently absent.
- No browser-incompatible assumptions — plain JSON over HTTP, no cookies, no
  non-standard headers required.
- No server-side state accidentally shared between users — confirmed (anonymous
  isolation, cross-session isolation both live-tested; `ChatService.ask()` builds a
  fresh `AgentOrchestrator`/`AgentState` per call, no shared mutable state across
  FastAPI's threadpool).

## 10. Security Review

| Check | Result |
|---|---|
| Oversized input | **Closed this milestone** — 422 above 8000/200 chars |
| Malformed JSON | 422, no leak (verified) |
| Unexpected fields | Ignored, structurally inert (verified) |
| `session_id` abuse | Bounded (200 chars); still never used for authorization (unchanged — `app/agent/permissions.py`) |
| Error leakage | No stack traces, no file paths beyond the pre-existing, already-reviewed Ollama `base_url` in 502 text (Milestone 40 §8) |
| Secret leakage | `TAVILY_API_KEY`/`OPENAI_API_KEY` never appear in any response body — verified live for the tool-failure path specifically this milestone |
| Debug leakage | None found; no `logger.debug` on the request path |
| Unrestricted CORS | Not present — confirmed absent, not wildcarded |
| Accidental internal state exposure | None — response schema remains exactly `{"reply": ...}` / `{"detail": ...}` |

No authentication was added — explicitly out of scope per the milestone's own rules
("a future productization decision, not a reason to redesign the backend now").

## 11. Performance

Lightweight measurements only, this session:

| Operation | Result |
|---|---|
| `GET /` | avg 3.9ms, max 24.5ms over 20 calls |
| `GET /health` (memory backend) | avg 2.6ms, max 3.6ms over 20 calls |
| `POST /chat` (FakeLLM, framework+agent overhead only) | avg 3.1ms, max 5.1ms over 15 calls |

No obvious API-level bottleneck. Real end-to-end latency remains dominated entirely by
actual Ollama/Tavily network and inference time, unchanged and not re-measured (no
model/routing change this milestone).

## 12. Tests

15 new focused tests added to `tests/test_api.py` (preserving all 11 existing ones,
unmodified): oversized/at-limit `message`/`session_id`, malformed JSON, extra-field
tolerance, the corrected tool-failure finding (§5), the new catch-all 500 (with a
dedicated `raise_server_exceptions=False` client to observe real-server behavior),
confirmation that existing `HTTPException` paths are unaffected by the catch-all, a
timeout test through the real HTTP layer (previously only tested at the
`ChatService` level), OpenAPI/`\/docs` surface checks, and a no-CORS-headers check.
`/health`'s SQLite-backend behavior was re-run (19 existing tests,
`tests/test_persistence_production_readiness.py`) rather than duplicated — unaffected
by this milestone's changes.

## 13. Files Changed

- `app/main.py` — the only production file touched: `Field(max_length=...)` on
  `ChatRequest.message`/`session_id`; `FastAPI(description=...)`; `summary`/
  `description`/`responses` on all three route decorators; one new
  `@app.exception_handler(Exception)`. No other file under `app/` was modified.
- `tests/test_api.py` — 15 new tests appended; nothing removed or altered.
- `README.md` — API section rewritten per §8 of the milestone brief's requested
  format; test-count references updated.
- `docs/MILESTONE_41_API_POLISH.md` — this document (new).

`docs/ARCHITECTURE_STATUS.md` was **not** modified — no architecture behavior
changed.

## 14. Git / Secret Audit

- `.env`: confirmed absent from `git ls-files` and `git status` — untracked,
  untouched.
- No `.db`/`.db-wal`/`.db-shm` file tracked or present.
- No test was deleted or had its assertions weakened — `git diff --stat` on
  `tests/test_api.py` shows additions only.
- No architecture file (`router.py`, `decision_maker.py`, `permissions.py`,
  `tool_execution.py`, `tool_registry.py`, `loop.py`, any `app/persistence/*` or
  `app/agent/*memory*` file) was touched — confirmed via `git diff --stat`.
- Nothing committed, nothing pushed.

## 15. Known Limitations

- CORS remains unconfigured by design (§8) — must be added in Step 42 once the
  frontend's real origin is known.
- No authentication exists or was added — explicitly deferred, per the milestone's
  own rules, to a future productization decision.
- The 502 response for an Ollama connection failure includes the (non-secret)
  `base_url` — pre-existing, unchanged, already reviewed and accepted in Milestone 40.
- `message`/`session_id` length limits are fixed constants, not environment-
  configurable — no evidence yet that any deployment needs to tune them; adding
  configurability now would be speculative.

None of the above is blocking.

## 16. Final Verdict

### A. API READY FOR FRONTEND

**The backend architecture remains frozen.** No agent/routing/memory/persistence/
security file was changed this milestone — only `app/main.py`'s HTTP-layer surface
(validation bounds, OpenAPI metadata, one consistency-only exception handler) and its
tests/documentation. The API's contract, error shapes, and documentation are now
accurate, bounded, and consistent, with CORS explicitly and deliberately deferred to
Step 42 rather than guessed at.
