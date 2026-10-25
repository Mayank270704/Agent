# Milestone 43 — Full Product End-to-End Testing

Verified against the actual repository on 2026-10-01, directly following Milestone 42
(Frontend / Demo UI — `docs/MILESTONE_42_FRONTEND.md`, verdict A). This is a
**validation milestone**: it proves the existing product works as a whole. No
architecture changed. No new features were added.

## 1. Objective

Validate the complete product exactly as a real user would experience it:

```
Browser → Next.js Frontend (localhost:3000) → HTTP (cross-origin) → FastAPI
(localhost:8000) → ChatService → AgentOrchestrator → AgentLoop → Decision /
ToolExecutionGate → Ollama (llama3.2:3b) / time / web_search → Response → Frontend
```

Confirm this path works end-to-end with the real local model and real tools, not
mocks, and only fix genuine P0/P1 defects if the testing surfaces any — minimally,
with regression coverage. Everything else (non-blocking limitations, pre-existing
model behavior) is documented, not "fixed."

## 2. Environment

- OS: Windows 11, PowerShell/Git Bash
- Python 3.13.7, pytest 9.1.1
- Node v22.19.0, npm 11.6.0
- Backend: `python -m uvicorn app.main:app --host 127.0.0.1 --port 8000` (started
  manually; confirmed via `GET /health` → `{"status":"ok","application":"running",
  "memory_backend":"memory","persistence":"not_configured"}`)
- `settings.memory_backend == "memory"` (default, no SQLite backend in this run)
- Ollama: running locally, `llama3.2:3b` present (`curl http://localhost:11434/api/tags`)
- `TAVILY_API_KEY`: set (confirmed via `settings.tavily_api_key` boolean only, never
  the value)
- Frontend dev server: `npm run dev` on `localhost:3000` (also auto-started by
  Playwright's `webServer` config with `reuseExistingServer: true` when not already
  running)
- No prior E2E infrastructure existed before this milestone (confirmed by searching
  for Playwright/Cypress/Puppeteer config before installing anything)

## 3. Architecture path tested

Every automated E2E test (except the deliberately-mocked error-handling tests in
§13) exercises the full real path: real Chromium browser → real cross-origin fetch
→ real FastAPI process → real `ChatService`/`AgentOrchestrator`/`AgentLoop` → real
`ToolExecutionGate` → real Ollama (`llama3.2:3b`) and, for the one web-search test,
the real Tavily API → real HTTP response → real React render. Nothing in the main
product-flow tests is faked.

## 4. E2E test matrix

| # | Category | Covered by | Real backend/model? |
|---|---|---|---|
| A | Frontend↔API connectivity | `e2e/connectivity.spec.ts` | yes |
| B | Normal LLM flow (no tool) | `e2e/llm-flow.spec.ts` | yes |
| C | Time tool | `e2e/llm-flow.spec.ts` + Layer-2 script (§9) | yes |
| D | Date tool | `e2e/llm-flow.spec.ts` + Layer-2 script (§9) | yes |
| E | Explicit web search (minimal) | `e2e/web-search.spec.ts` + Layer-2 script | yes (real Tavily) |
| F | Multi-turn session | `e2e/session.spec.ts` | yes |
| G | New Chat session isolation | `e2e/session.spec.ts` | yes |
| H | Refresh persistence | `e2e/session.spec.ts` | yes |
| I | Invalid input (empty/whitespace/oversized) | `e2e/validation-and-errors.spec.ts` | yes (client-side, no call made) |
| J | Backend-unavailable handling | `e2e/validation-and-errors.spec.ts` | simulated via `page.route()` abort |
| K | HTTP 4xx/5xx handling | `e2e/validation-and-errors.spec.ts` | simulated via `page.route()` fulfill |
| L | Malformed response handling | `e2e/validation-and-errors.spec.ts` | simulated via `page.route()` fulfill |
| M | CORS | `e2e/security-and-cors.spec.ts` + manual `curl -X OPTIONS` (§11) | yes |
| N | Health (no aggressive polling) | `e2e/connectivity.spec.ts` (test 2) + code inspection (one-shot `useEffect`, no interval) | yes |
| O | Security sanity | `e2e/security-and-cors.spec.ts` | yes |
| P | Real Ollama + llama3.2:3b flow | woven through B/C/D/F/G/H above, plus `tests/test_agent_integration.py::test_real_ollama_smoke_decision_without_tools` (§9) | yes |

J/K/L deliberately use Playwright's `page.route()` network interception rather than
stopping the real backend process — this follows the brief's own instruction not to
change production architecture just to make a test convenient, and reuses the exact
same frontend error-handling code path (`lib/api.ts`'s `ApiError` branches) that a
genuine outage would hit.

## 5. Automated test results

**Playwright browser E2E** (`frontend/`, `npx playwright test`, Chromium,
`workers: 1`): **24 passed, 0 failed** (1.7 min). Breakdown:

- `connectivity.spec.ts` — 4/4
- `llm-flow.spec.ts` — 3/3
- `security-and-cors.spec.ts` — 5/5
- `session.spec.ts` — 3/3
- `validation-and-errors.spec.ts` — 8/8
- `web-search.spec.ts` — 1/1

First run surfaced 5 failures, all test-authoring bugs in the brand-new spec files
themselves (zero product defects) — see §15/§16.

**Layer-2 backend-internal verification** (one-time script, not a permanent test
file — the public API deliberately never exposes `tool_calls`, by design, so this
confirms routing correctness directly against the real production `ChatService`
composition): ran 4 real prompts through the exact same `AgentOrchestrator`
construction `ChatService.ask()` itself uses.

| Prompt | Tool(s) called | Status |
|---|---|---|
| "What time is it right now?" | `['time']` | COMPLETED |
| "What is today's date?" | `['time']` | COMPLETED |
| "Search the web for the current version number of the Next.js framework." | `['web_search', 'web_search']` | COMPLETED |
| "In one short sentence, what is the capital of France?" | `[]` | COMPLETED |

Confirms deterministic routing works exactly as documented (date queries route
through the `time` tool by design, not a separate `date` tool — see
`docs/ARCHITECTURE_STATUS.md`). The double `web_search` call is a known, pre-existing
behavior — see §17.

**Existing real-Ollama integration test**:
`tests/test_agent_integration.py::test_real_ollama_smoke_decision_without_tools -m
integration` — **1 passed** in 6.92s.

## 6. Manual smoke-test results

Captured real rendered screenshots (Chromium, via Playwright's `page.screenshot()`,
reviewed visually) rather than relying solely on automated assertions, to check
layout/spacing/responsiveness that assertions don't cover:

1. **Desktop empty state** (1280×800) — header, sidebar with session id and New Chat,
   centered empty-state copy, input bar. Clean, matches the design brief, "Online"
   status correctly shown in green.
2. **Desktop conversation** (1280×800) — sent "What time is it right now?", received
   a real reply ("It is currently 15:04:49 IST.") matching the system's actual time.
   User/assistant bubbles correctly aligned right/left.
3. **Mobile viewport** (390×844, iPhone-sized) — sidebar collapses above the
   conversation, input bar and bubbles reflow correctly, no horizontal scroll, no
   clipped content.
4. **Error banner** (network-failure simulation) — red banner, safe copy ("Unable to
   reach the agent. Check that the backend is running."), visible Retry button, no
   raw error/stack trace.

No visual defects found. Screenshots are local, throwaway verification artifacts
(not committed).

## 7. Backend regression result

```
python -m pytest -m "not integration" -q
```

**2275 passed, 38 deselected, 1 warning, 60.60s** — identical to the pre-Step-43
baseline (confirmed at the start of this milestone before any change was made).
Step 43's only backend-adjacent action was running tests; no `app/` source file was
modified.

```
python -m compileall -q app tests
```

Clean, no errors.

## 8. Frontend test/build/typecheck/lint results

- `npm test` (Vitest): **16 passed** (`lib/session.test.ts` 6, `lib/api.test.ts` 10)
  — unchanged from Milestone 42; the `data-testid` additions did not touch `lib/`.
- `npm run lint`: 0 problems.
- `npm run typecheck`: 0 errors (including the new `e2e/*.ts` files, which fall under
  the project's existing `tsconfig.json` `**/*.ts` include).
- `npm run build`: clean production build (Turbopack), static pages generated, no
  warnings.

## 9. Ollama real-model result

All of B/C/D/F/G/H's Playwright tests, plus the Layer-2 script (§5) and the existing
`test_real_ollama_smoke_decision_without_tools` integration test, ran against the
real local `llama3.2:3b` — none were mocked. No timeouts occurred (backend's own
`REQUEST_TIMEOUT_SECONDS=120` was never approached; individual turns completed in
2-10 seconds). Latency was recorded, not treated as a failure condition, per the
brief.

## 10. Web-search result

Exactly 2 genuine end-to-end web-search invocations were made this milestone (1 via
`e2e/web-search.spec.ts`, 1 via the Layer-2 script) — deliberately minimal, per the
brief's Tavily-quota instruction. Both returned real, non-empty, on-topic answers
(current Next.js version info) with no "I can't search the web" refusal, confirming
the `web_search` tool and `TAVILY_API_KEY` are genuinely wired and functional through
the full stack. Note: each of those 2 invocations internally issued **2** real Tavily
API calls (4 total) — see §17 for why, and why this is not a Step 43 defect.

## 11. CORS result

Real cross-origin browser requests (`localhost:3000` → `localhost:8000`) were
verified two ways:

1. **Live, via Playwright** (`security-and-cors.spec.ts`): the actual `POST /chat`
   and `GET /health` responses observed by the real browser carry
   `access-control-allow-origin: http://localhost:3000` exactly (never `*`), and no
   `access-control-allow-credentials` header (matches `allow_credentials=False` in
   `app/main.py`).
2. **Direct, via `curl -X OPTIONS`** against the real running backend (to verify the
   preflight handshake itself, since Chromium's own preflight-response caching
   — `Access-Control-Max-Age: 600` — made live-capturing the OPTIONS request from
   Playwright non-deterministic across isolated browser contexts; see §17):
   ```
   HTTP/1.1 200 OK
   access-control-allow-methods: GET, POST
   access-control-max-age: 600
   access-control-allow-headers: Accept, Accept-Language, Content-Language, Content-Type
   access-control-allow-origin: http://localhost:3000
   ```
   Confirms the preflight handshake itself is correct and non-wildcard.

No wildcard origin anywhere, in either check.

## 12. Session-isolation result

- **F (multi-turn)**: a second message in the same session correctly recalled
  context stated in the first ("My favorite programming language is Python" →
  later correctly repeated back "Python").
- **G (New Chat isolation)**: clicking New Chat produces a different `session_id`
  (confirmed via `localStorage` inspection), clears the visible transcript, and the
  new session does **not** inherit the old session's context (a fact stated in
  session A, "teal", was correctly absent from session B's answer).
- **H (refresh persistence)**: the same `session_id` survives a page reload
  (`localStorage` read before/after reload matched exactly), the cached transcript
  reappears, and the backend still recalls context from before the refresh (a number
  stated pre-refresh was correctly repeated back post-refresh).

All three passed against the real backend/model, not a mock.

## 13. Error-handling result

- **J (backend unavailable)**: a simulated network failure (`page.route()` abort)
  produces the safe message "Unable to reach the agent. Check that the backend is
  running.", clears the loading state, and Retry — once the simulated outage is
  lifted — successfully resends the same pending message and gets a real reply. The
  UI is never left stuck.
- **K (HTTP 4xx/5xx)**: a simulated 422 surfaces the backend's own validation
  `msg` text, not raw JSON; a simulated 500 surfaces "Internal server error." with no
  stack trace — matching `app/main.py`'s catch-all handler and `lib/api.ts`'s
  `readErrorDetail()` exactly.
- **L (malformed response)**: a 200 response with a non-JSON body, and a 200
  response with valid JSON but missing the `reply` field, are both correctly
  classified as "The agent returned an unexpected response." — the app never
  crashes and remains usable immediately after.

## 14. Security sanity result

- No secret-looking values (`TAVILY_API_KEY`, API key prefixes, `.env` content,
  Python tracebacks, private-key markers) found in page content or any network
  response body during a real conversation flow.
- No `dangerouslySetInnerHTML` anywhere in `app/`, `components/`, or `lib/`
  (verified by a source scan, not just a spot check).
- No `NEXT_PUBLIC_*` environment variable other than the documented
  `NEXT_PUBLIC_API_BASE_URL` is read anywhere in the frontend source.
- CORS is never a wildcard (§11); `allow_credentials` is `False`.
- Cross-session leakage: covered by §12's New Chat isolation result (session B never
  saw session A's data).
- Git/artifact audit (§18): no `.env`, API keys, databases, `node_modules`, `.next`,
  or generated secrets/logs are tracked.

## 15. Failures and classification

The first full Playwright run produced 5 failures. All are classified **P3**
(cosmetic/test-authoring issues in brand-new Step 43 test code, zero product
defects):

1. `connectivity.spec.ts` — `getByRole("heading", { name: "AI Agent" })` matched
   two elements (the header `<h1>AI Agent</h1>` and the empty-state `<h2>Your local
   AI agent</h2>`, both containing the substring). Test-locator bug.
2. `connectivity.spec.ts` — asserted the Send button is enabled immediately after a
   reply arrives, without accounting for the input having been cleared on submit (so
   there is correctly nothing to send). Incorrect test assumption, not a product bug
   — the button is disabled because the box is empty, by design.
3. `security-and-cors.spec.ts` — the CORS test required observing a live OPTIONS
   preflight response via Playwright's response listener; Chromium's own preflight
   caching (`Access-Control-Max-Age: 600`) made this non-deterministic to capture
   live. Test-design issue — see §17/§11 for the replacement approach.
4. & 5. `validation-and-errors.spec.ts` (J and L) — the same "Send button should be
   enabled" assumption as #2, in the backend-unavailable and malformed-response
   tests.

No P0 or P1 issues were found anywhere in the product itself.

## 16. Fixes made if any

- Fixed all 5 test-file issues above (`frontend/e2e/connectivity.spec.ts`,
  `frontend/e2e/security-and-cors.spec.ts`, `frontend/e2e/validation-and-errors.spec.ts`)
  by correcting the test assertions/locators, not the product. Re-ran: **24/24
  passed**.
- `frontend/.gitignore` gap: Playwright's own generated output directories
  (`test-results/`, `playwright-report/`, `blob-report/`) were not excluded. Added
  them (mirrors the existing `node_modules`/`.next` pattern). No product code change.
- **Zero changes to `app/` (backend) or to any component/library's runtime
  behavior.** The only non-test frontend changes this milestone are the purely
  additive `data-testid` attributes added to existing components before any test was
  written (verified via a full `npm test`/`lint`/`typecheck` pass immediately after
  adding them, before writing a single E2E test) — these are test hooks, not
  behavior changes.

## 17. Known limitations

- **`web_search` fires twice per triggering user turn.** Confirmed via the Layer-2
  script (§5): a single user message that routes to `web_search` results in
  `['web_search', 'web_search']` — the agent's own iterative loop refines its query
  once before producing a final answer. This is pre-existing `AgentLoop`/decision
  logic, unchanged by this milestone, and out of scope to alter under the
  architecture freeze. It roughly doubles real Tavily quota cost per web-search
  turn — worth knowing when budgeting API usage, not a defect.
- **Indirect multi-turn coreference is occasionally unreliable** — a known
  `llama3.2:3b` limitation, already documented in `docs/FINAL_PRODUCTION_AUDIT.md`
  §11 and `docs/MILESTONE_42_FRONTEND.md` §9. This milestone's session tests use
  direct "repeat back exactly X" phrasing specifically to avoid depending on the
  model's unreliable indirect-reference handling, consistent with that prior finding.
- **CORS preflight (OPTIONS) is verified via a direct `curl` check, not captured
  live through Playwright**, because of Chromium's own preflight-response caching
  (§11/§15). The actual cross-origin POST/GET responses the browser receives ARE
  verified live.
- **`npm run e2e` requires the backend to be started manually first** (different
  toolchain/process; `playwright.config.ts`'s `webServer` only auto-starts the
  frontend, by design, with a comment explaining why).
- No CI pipeline runs any of this automatically — out of scope for this milestone
  (Step 44, deployment/packaging, is next).

## 18. Final verdict

**A. E2E VERIFIED — PRODUCTION FOUNDATION READY**

The complete product — real browser, real frontend, real cross-origin API call, real
agent orchestration, real local Ollama model, real Tavily web search, real session
persistence — was exercised end-to-end and works as designed. All 5 failures
encountered were test-authoring issues in the brand-new Step 43 suite itself, fixed
without touching product code; the full backend regression (2275/38/0) and frontend
suite (16/16 + clean lint/typecheck/build) are unchanged. No P0 or P1 issues were
found. Backend and frontend architecture remain frozen, exactly as left at the end of
Milestone 42.

**Recommended next milestone: Step 44 — Deployment / Production Packaging** (not
started in this session).
