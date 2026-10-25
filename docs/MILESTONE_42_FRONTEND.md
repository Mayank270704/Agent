# Milestone 42 — Frontend / Demo UI

Built and verified against the actual repository on 2026-10-01, directly following
Milestone 41 (API/Product Polish — `docs/MILESTONE_41_API_POLISH.md`, verdict A).

## 1. Frontend Architecture

`frontend/` — a new, independent Next.js (App Router) + React 19 + TypeScript +
Tailwind CSS v4 project, scaffolded via `create-next-app` and extended. It is a
**thin client only**: no agent logic, no routing/tool/memory decisions, no direct
model access — every capability comes from the existing `POST /chat` / `GET /health`
API. The backend's own `app/` package (Python) was not restructured or duplicated.

```
frontend/
  app/                 Next.js App Router entry (layout.tsx, page.tsx, globals.css)
  components/          UI components (presentational + one orchestrator)
  lib/                 API client, session-id handling, shared types — no JSX here
  lib/*.test.ts         Vitest unit tests for lib/
  .env.local.example    documents NEXT_PUBLIC_API_BASE_URL
```

No frontend existed before this milestone (verified by searching the repository for
`package.json`/`next.config.*`/`app/`(Next)/`pages/`/`src/` before writing anything —
only the backend's own `app/` Python package was found).

## 2. Component Structure

- **`ChatApp`** (`components/ChatApp.tsx`) — the one orchestrator. Owns all client
  state (`sessionId`, `messages`, `loading`, `error`, `pendingMessage`,
  `connectionStatus`), resolves the session on mount, persists the visible
  transcript to `sessionStorage` (UI cache only — see §4), and renders the layout
  (header, sidebar, conversation, input).
- **`Sidebar`** — "New Chat" button + truncated session id, stacks above the
  conversation on narrow screens, becomes a left column at `md:` and up.
- **`MessageBubble`** — one user or assistant bubble; styling only, no logic.
- **`TypingIndicator`** — the loading state shown while waiting for a reply.
- **`ChatInput`** — textarea + Send button; Enter-to-send, Shift+Enter-for-newline,
  a client-side `maxLength` matching the backend's own limit (see §3), disabled
  while a request is in flight.
- **`StatusIndicator`** — a dot + text label driven only by `GET /health`'s
  `status` field (see §6).

No component makes its own `fetch()` call — every request goes through `lib/api.ts`.

## 3. API Integration

`lib/api.ts` is the **only** place that calls the backend:

- `sendChatMessage({ message, session_id })` → `POST /chat`, JSON body,
  `Content-Type: application/json`, a client-side `AbortController` timeout
  (130s — slightly above the backend's own default 120s `REQUEST_TIMEOUT_SECONDS`,
  a safety net for a genuinely stuck connection, not a duplicate of the backend's
  own cooperative deadline, which the backend architecture owns and this milestone
  does not touch).
- `getHealth()` → `GET /health`, returns the parsed body (including a 503/degraded
  response, which is still a successful parse) or `null` on a genuine network
  failure.
- Non-2xx responses: the backend's own `detail` field (a plain string, or the first
  message of a Pydantic validation-error array for 422s) is surfaced directly —
  Milestone 41 already audited that this field never contains secrets, paths, or
  stack traces, so the frontend trusts it as safe, user-facing text.
- `NEXT_PUBLIC_API_BASE_URL` (`frontend/.env.local.example`) configures the base
  URL; defaults to `http://localhost:8000` if unset, so `npm run dev` works out of
  the box against the documented local backend setup with zero configuration.

`ChatRequest`/`ChatResponse`/`HealthResponse`/`Message`/`ApiError` (`lib/types.ts`)
mirror exactly what the backend's own Pydantic models send/accept — no field
represents a capability the backend doesn't have.

## 4. Session Handling

`lib/session.ts`: `session_id` is a client-generated `crypto.randomUUID()` (with a
non-cryptographic fallback for an environment without it), stored in
`localStorage` under one key, generated once and reused across requests —
resuming the same backend-side conversation on a page refresh. `session_id` is
never derived from personal information. "New Chat" generates and persists a new,
distinct id; the previous session's backend-side state is left exactly as-is — no
destructive endpoint exists or is called.

A **separate**, `sessionStorage`-keyed cache (keyed by `session_id`) holds the
*visible transcript only*, so a refresh doesn't show an empty screen while the
(already-correct) backend history is the actual source of truth. This is
explicitly a UI convenience, not a second memory system: it is never read by
anything except `ChatApp`, and starting a new chat abandons it entirely.

## 5. Error Handling

Every failure path produces a safe, understandable message, never raw internal
detail:

| Scenario | User sees |
|---|---|
| Network/backend unreachable | "Unable to reach the agent. Check that the backend is running." |
| Request aborted by the client-side timeout | "The request took too long and was cancelled. Please try again." |
| 4xx/5xx with a backend `detail` | The backend's own (already-sanitized) detail text |
| Malformed/unexpected success body | "The agent returned an unexpected response." |

On any error: the user's own message bubble is kept, a dismissable error banner
appears with a **Retry** button (re-sends the same pending text without duplicating
the bubble or re-typing), loading state clears, and the conversation is otherwise
untouched. No fake assistant response is ever synthesized.

## 6. CORS Decision

**Configured** this milestone — the minimum backend change required, now that a
real, concrete frontend origin exists (Milestone 41 correctly deferred this when
none did).

- New setting: `CORS_ALLOWED_ORIGINS` (`app/config.py`, `_parse_cors_origins`),
  comma-separated, default `http://localhost:3000,http://127.0.0.1:3000` — the
  Next.js dev server's own default origin, in both the `localhost` and `127.0.0.1`
  forms (browsers treat them as different origins).
- `app/main.py`: `CORSMiddleware` added with `allow_origins=list(settings.
  cors_allowed_origins)` (never `["*"]`), `allow_credentials=False` (this API uses
  no cookies/auth — `session_id` travels in the JSON body), `allow_methods=["GET",
  "POST"]` (exactly what this API exposes), `allow_headers=["Content-Type"]`
  (exactly what the frontend's JSON POST needs preflight clearance for).
- Verified live (real `uvicorn` backend process, real cross-origin request with
  `Origin: http://localhost:3000`, real Ollama call): `access-control-allow-origin:
  http://localhost:3000` present on the actual response, absent for an
  unrecognized origin (`http://evil.example.com`), preflight `OPTIONS` succeeds.
- Deploying the frontend elsewhere later only requires adding its real origin to
  `CORS_ALLOWED_ORIGINS` — never widening the policy.

## 7. Accessibility

- Every interactive control is a semantic `<button>` or labeled `<textarea>`
  (`ChatInput`'s textarea has an associated, screen-reader-only `<label>`).
- Full keyboard operation: Tab reaches every control, Enter sends, Shift+Enter
  inserts a newline, focus returns to the input after sending.
- Visible focus rings on every interactive element (`focus:ring-*` Tailwind
  utilities) — never suppressed.
- Status is never color-only: `StatusIndicator` pairs its colored dot with a text
  label ("Online"/"Degraded"/"Backend unavailable"); the error banner is text, not
  an icon alone, and carries `role="alert"`.
- Message bubbles carry a screen-reader-only prefix ("You said:"/"Agent replied:")
  so a screen reader user can distinguish turns without relying on bubble position/
  color alone.
- No accessibility library was added — the project had none before, and these are
  all native HTML/ARIA primitives.

## 8. Responsive Behavior

Single Tailwind layout, no separate mobile markup: the sidebar is a horizontal bar
above the conversation on narrow screens (`flex-col`, default) and becomes a left
column at `md:` and wider (`md:flex-row`); session detail text is hidden below
`md:` to keep the mobile bar compact while the "New Chat" button stays reachable at
every width; the conversation area uses `h-dvh` (dynamic viewport height, correct
on mobile browsers whose chrome resizes the viewport) and `overflow-y-auto` so long
transcripts scroll independently of the fixed input bar; message bubbles cap at
`85%`/`75%` of the container width and wrap long content (`whitespace-pre-wrap
break-words`) so neither very long single words nor very long replies break the
layout.

## 9. Testing

No frontend testing framework existed before this milestone (confirmed by
inspection). Added **Vitest** (`vitest@3.2.7`, chosen specifically for
`@types/node@^20` compatibility — the scaffold's own pinned version — avoiding an
unrelated dependency bump) + **jsdom** (`jsdom@26.1.0`, for `localStorage`/`window`
in tests), deliberately minimal: 16 focused unit tests on `lib/` only (pure
request/response/error-handling and session-id logic), no component-rendering or
E2E browser automation stack — that is left for Step 43 (Full Product End-to-End
Testing) rather than added here "merely for this milestone."

```
npm test        -> 16 passed (lib/session.test.ts: 6, lib/api.test.ts: 10)
npm run lint    -> clean, 0 problems
npm run typecheck (tsc --noEmit) -> clean
npm run build   -> compiles, type-checks, and statically prerenders successfully
```

Beyond the automated suite, this milestone also verified live, end-to-end, with the
**real** backend and **real** Ollama (not mocked): `npm run dev` + `uvicorn
app.main:app`, a real cross-origin `POST /chat` from `Origin: http://localhost:3000`
returning a real model answer with the correct CORS header; two-turn same-session
continuity (confirmed the mechanism works — `ConversationMemory` correctly stores
and replays both turns; a less-direct recall prompt was answered incorrectly by
llama3.2:3b on one trial, which a more direct phrasing then answered correctly —
this is the already-documented "conversation coreference: MODEL LIMITATION, not an
architecture defect" finding from Milestone 40 §11, re-confirmed, not a new
frontend/backend bug); a separate `session_id` correctly had no access to the
first session's content; the rendered page HTML contains every expected UI element
("AI Agent", "New Chat", the empty-state copy, "Ask anything…"); the default
backend URL was confirmed correctly inlined into the production client bundle.

## 10. Security

- No API key (Tavily/OpenAI), no Ollama credential, and no filesystem/database path
  appears anywhere in `frontend/` source — verified by a direct grep across
  `*.ts`/`*.tsx`/`*.json` (excluding `node_modules`/`.next`), confirmed clean.
- `.env.local` (the real, local-only config file) is git-ignored; only
  `.env.local.example` (a template with a non-secret default URL) is trackable —
  required an explicit `!.env.local.example` negation in `frontend/.gitignore`,
  since the scaffold's own `.env*` pattern would otherwise have swallowed it too
  (caught and fixed this milestone).
- The only `NEXT_PUBLIC_*` variable is the API base URL — a public value by
  necessity (any `NEXT_PUBLIC_*` variable is inlined into the browser bundle); no
  secret was ever placed behind that prefix or any other.
- No `dangerouslySetInnerHTML` anywhere — message content is rendered as plain
  React text content, never interpreted as HTML.
- CORS is configuration-backed and origin-specific, never a wildcard (§6).
- `node_modules/`, `.next/`, `tsconfig.tsbuildinfo` all confirmed excluded from
  what `git add` would track (dry-run verified).

## 11. Files Changed

**New** (`frontend/`, entire directory): Next.js scaffold
(`package.json`, `tsconfig.json`, `next.config.ts`, `eslint.config.mjs`,
`postcss.config.mjs`, `app/layout.tsx`, `app/page.tsx`, `app/globals.css`,
`app/favicon.ico`) plus this milestone's own code (`components/*.tsx`,
`lib/*.ts`, `lib/*.test.ts`, `vitest.config.ts`, `.env.local.example`,
project-specific `README.md`, `.gitignore` with the `.env.local.example`
exception).

**Backend** (the minimum required): `app/config.py` (+`_parse_cors_origins`,
`cors_allowed_origins` setting), `app/main.py` (+`CORSMiddleware`
registration), `.env.example` (+`CORS_ALLOWED_ORIGINS` documentation),
`tests/test_env_example.py` (+1 entry in the consumed-vars contract),
`tests/test_api.py` (+8 CORS tests). No other backend file touched — see §12 for
the direct diff-stat proof.

**Not changed**: anything under `app/agent/`, `app/persistence/`,
`app/services/chat.py`, `app/semantic_memory_wiring.py`, or any routing/model/
memory/persistence/security file.

## 12. Known Limitations

- `GET /health` is checked once on initial load only (per the explicit
  requirement) — no manual retry control exists yet if it was checked before the
  backend finished starting; a page refresh re-checks it.
- The frontend's `MAX_MESSAGE_LENGTH` (8000) is a local constant mirroring
  `app/main.py`'s own value, not fetched dynamically — the two must be kept in
  sync by hand if the backend's limit ever changes (no shared schema file exists
  between the two projects).
- `npm audit` reports 2 moderate-severity advisories in `@vitest/mocker` (a
  path-traversal/arbitrary-file-read issue in Vitest's dev-only test mocker,
  https://github.com/advisories/GHSA-82fw-gwwq-j7x9). This is a **devDependency
  only** — never part of the production build or shipped bundle — and the fix
  requires `vitest@5`, which reintroduces the `@types/node@^22||>=24` peer
  conflict this milestone deliberately avoided (an unrelated dependency bump,
  against this milestone's own "do not upgrade unrelated dependencies" rule).
  Documented, not fixed, per the explicit "if warnings are harmless, document
  them" instruction.
- No authentication exists — explicitly out of scope for this milestone.
- No streaming/WebSocket — the frontend consumes the existing synchronous
  `POST /chat` exactly as built; the backend exposes no streaming endpoint to
  consume.

None of the above is blocking.

## 13. Final Verdict

### A. FRONTEND READY FOR E2E

**Backend architecture remains frozen.** Verified directly: zero lines changed in
`app/agent/router.py`, `decision_maker.py`, `permissions.py`, `tool_execution.py`,
`tool_registry.py`, `loop.py`, `reliability.py`, `app/persistence/*`, or
`app/semantic_memory_wiring.py` this milestone. The only backend change was the
minimum required to connect the real, now-known frontend origin
(configuration-backed CORS, never a wildcard). The frontend builds, type-checks,
lints, and tests cleanly, and was verified live end-to-end against the real
backend and real Ollama.

**The next milestone is Step 43 — Full Product End-to-End Testing.**
