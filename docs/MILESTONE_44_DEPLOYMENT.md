# Milestone 44 — Deployment / Production Packaging

Verified against the actual repository on 2026-10-01, directly following Milestone 43
(Full Product E2E Testing — `docs/MILESTONE_43_E2E.md`, verdict A). This is a
**packaging and reproducibility milestone**: it documents and validates how to run
the existing, already-frozen product reliably — it does not change the agent,
the model, or any architecture.

## 1. Objective

Make the project reproducible to set up and run — locally, or on a single machine
the operator controls — at **₹0 mandatory cost**, without introducing Docker,
Kubernetes, paid cloud services, or a paid LLM API, and without claiming a cloud
deployment path the architecture does not actually support.

## 2. Current architecture

Unchanged from Milestones 40-43 (frozen): Next.js frontend → FastAPI backend →
`ChatService`/`AgentOrchestrator`/`AgentLoop` → `ToolExecutionGate` → local Ollama
(`llama3.2:3b`) and the `time`/`date`/`web_search` tools (Tavily, optional). Backend
is **single-process, single-worker** throughout — every memory store (in-memory or
SQLite) uses in-process locking and was never evaluated for multiple OS processes
sharing one database file (`docs/ARCHITECTURE_STATUS.md` §16, unchanged by this
milestone). This fact alone is what bounds the realistic deployment topology below —
it is not a new finding, just the one this milestone's packaging decisions must
respect.

## 3. Supported deployment topology

**A. Local/demo deployment (verified, supported):**
```
Browser → Next.js (dev server, npm run dev) → FastAPI (uvicorn --reload) →
Ollama → llama3.2:3b
```
This is what Milestones 42/43 already ran and verified end-to-end.

**B. Single-machine production-style deployment (verified this milestone):**
```
Browser → Next.js (production build, npm run build && npm run start) →
FastAPI (uvicorn, no --reload) → Ollama → llama3.2:3b (+ optional SQLite
persistence)
```
Same machine, same process topology as A, just without the development-only
conveniences (hot reload, dev server). This is the realistic "deploy it somewhere
you control" target, and the one validated in §12/§17 below.

**C. Cloud deployment — future option only, not supported today.** A genuinely free
cloud host that can run Ollama + `llama3.2:3b` persistently (the model needs
several GB of RAM and sustained CPU, not a serverless/ephemeral free tier) does not
exist among the common ₹0 options (Vercel serverless, Render/Railway free tiers,
GitHub Pages, Netlify). The **frontend alone** could be deployed free to a static/
Next.js host (Vercel, Netlify) today — it is a stateless client — but that is not
"the complete product," since it would still need a reachable backend+Ollama host,
which is the actual blocker. This repository does **not** claim complete-stack cloud
deployability; see §15.

## 4. Local setup

See the root README's "Installation (Windows)" section (unchanged by this
milestone) for the full first-time setup (`venv`, `pip install -r
requirements.txt`, `.env` from `.env.example`). This milestone adds the
production-style *startup* commands (§12) on top of that existing setup, not a
new setup path.

## 5. Ollama setup

1. Install from <https://ollama.com> (local, free, no account/API key required).
2. Verify it's running: `curl http://localhost:11434/api/tags` — should return a
   JSON list of installed models (confirmed working in this environment during
   this milestone's validation).
3. Verify `llama3.2:3b` is present: check that list for `"name":"llama3.2:3b"`.
4. If missing, pull it: `ollama pull llama3.2:3b` (a few GB download, one-time,
   fully offline after that).
5. Start the backend (§6) — it does **not** start or manage Ollama itself; Ollama
   must already be running as its own local process/service.
6. Verify the backend can reach it: `GET /health` does **not** check Ollama
   (by design — see §11), so the real check is a live request:
   `curl -X POST http://localhost:8000/chat -H "Content-Type: application/json"
   -d "{\"message\": \"hello\"}"` should return a real `{"reply": ...}`, not a
   `502`.

No model is downloaded into, or shipped with, this repository — `ollama pull` is
the operator's own one-time step against their own local Ollama installation.

## 6. Backend setup

Production-style startup (validated this milestone):
```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
(No `--reload` — that flag is a development convenience only, and watches the
filesystem for changes, which a production/demo run doesn't need.) `python -m
app.main` (the other documented form) is equivalent for local/demo use.

## 7. Frontend setup

Production-style startup (validated this milestone):
```powershell
cd frontend
npm run build
npm run start
```
`npm run build` produced a clean, warning-free production build (Turbopack,
static pages). `npm run start` was then run for real and verified two ways: a
direct `curl http://localhost:3000/` (HTTP 200, real page HTML), and a real
Chromium browser session (via the existing Playwright setup) that loaded the page
and completed one real chat round trip against the real backend/Ollama.

**Important, verified directly this milestone:** `NEXT_PUBLIC_API_BASE_URL` is
inlined into the static JS bundle at **build** time (confirmed by grepping the
built `.next/static/chunks/*.js` output for the literal backend URL string) — it
is *not* re-read at `npm run start` time. A deployment pointing at a non-default
backend address must set this variable before `npm run build`, and rebuild after
changing it.

## 8. Environment variables

Full table already exists in the root README's "Environment variables" section
(unchanged by this milestone — still accurate and still enforced by
`tests/test_env_example.py`). Summary for deployment purposes:

**Required (no safe default that makes the product actually work):** none —
every variable has a working default for the local-Ollama path with no optional
tools. `TAVILY_API_KEY` is "required" only in the narrow sense that `web_search`
specifically needs it; the rest of the product works without it.

**Deployment-relevant, commonly set:**
- `OLLAMA_BASE_URL` / `MODEL_NAME` — only if Ollama isn't at the default
  `http://localhost:11434` or a different model tag is pulled.
- `CORS_ALLOWED_ORIGINS` — must name the real deployed frontend origin(s) if not
  `localhost:3000`/`127.0.0.1:3000` (§9).
- `MEMORY_BACKEND` / `SQLITE_DB_PATH` — only if opting into persistence (§10).

**Optional, independent of deployment:** `TAVILY_API_KEY` (web search),
`SEMANTIC_MEMORY_ENABLED` and its related variables, `TELEMETRY_ENABLED`,
`REQUEST_TIMEOUT_SECONDS`.

**Never required:** an Ollama API key (Ollama has none), `OPENAI_API_KEY` (only
read when `LLM_PROVIDER=openai`, which is not the deployed default — confirmed by
reading `app/models/llm.py`: the `OpenAI` client and its key check only construct
on that branch).

Confirmed by inspection: `.env.example` and `frontend/.env.local.example` contain
placeholders/safe defaults only, no real secret; the only `NEXT_PUBLIC_*` variable
anywhere in the frontend source is `NEXT_PUBLIC_API_BASE_URL`, which is a plain
URL, never a secret (re-confirmed this milestone — same static-scan result as
Milestone 43 §14).

## 9. CORS configuration

Unchanged, origin-specific, never a wildcard (`app/main.py`'s
`CORSMiddleware(allow_origins=list(settings.cors_allowed_origins), ...)`). For a
real single-machine deployment where the frontend is reached at a non-localhost
address (e.g. the machine's LAN IP, or a domain pointed at it), set
`CORS_ALLOWED_ORIGINS` to that exact origin — re-verified working this milestone:
the production frontend build (§7) still received correct, non-wildcard
`access-control-allow-origin: http://localhost:3000` headers on both the
preflight and the actual request.

## 10. SQLite / persistence configuration

`MEMORY_BACKEND=sqlite` was exercised fresh this milestone end-to-end (not just
re-reading the Step 36-39 audit) against a throwaway database file
(`data/deploy_check.db`, deleted after the test, never tracked — `/data/` and
`*.db*` are already in `.gitignore`):

1. **Startup**: started with `MEMORY_BACKEND=sqlite SQLITE_DB_PATH=./data/
   deploy_check.db` — the parent directory and the `.db`/`.db-wal`/`.db-shm` files
   were created automatically.
2. **Health**: `GET /health` returned `{"status":"ok", ..., "persistence":
   "available", "schema_version":3, "schema_supported":true}`.
3. **A real chat turn** was sent ("My favorite fruit is mango...", tagged with a
   session id) and answered normally.
4. **Restart persistence**: the process was killed and a genuinely new process
   started against the same database file. `GET /health` was healthy again
   immediately, and a follow-up request on the same session id correctly recalled
   "mango" — proving the conversation survived the restart, not just that the
   file existed.
5. **DB files remain untracked**: confirmed via `git status`/`git check-ignore`
   both before and after this test — nothing under `data/` was ever staged.
6. **No destructive migration behavior**: not re-tested from scratch this
   milestone (would require constructing an old-schema database on purpose) —
   this is already covered by permanent, existing test coverage
   (`tests/test_persistence_schema.py`, `tests/test_persistence_production_
   readiness.py`) per Steps 36-39's audit, reused rather than duplicated here.

PostgreSQL/pgvector was **not** introduced — SQLite remains sufficient for the
documented single-machine topology, exactly as `docs/ARCHITECTURE_STATUS.md` §19
already concluded ("not scheduled, no evidence of need today").

## 11. Health verification

`GET /health` is unchanged (Step 39): O(1), never scans vector/semantic data,
never calls Ollama or Tavily. This milestone re-confirms, by direct test, that
this means an **Ollama outage is invisible to `/health`** — `/health` reports
`"status":"ok"` even while `/chat` would correctly return `502` (§11 of this
document's companion test, see §17). This is intentional, existing, documented
behavior (`/health` answers "is the application and its own configured
persistence backend alive", not "is every upstream dependency reachable") — not
a gap introduced or found by this milestone, just verified directly rather than
assumed.

## 12. Production-style startup

Combined sequence validated this milestone, in order:
```powershell
# 1. Ollama already running locally (separate process/service)
curl http://localhost:11434/api/tags

# 2. Backend, production-style (no --reload)
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 3. Verify
curl http://localhost:8000/health

# 4. Frontend, production build + serve
cd frontend
npm run build
npm run start

# 5. Verify in a browser: http://localhost:3000
```
All five steps were actually run this milestone (§17), not just documented.

## 13. Security checklist

- No `.env` tracked (`git check-ignore -v .env` confirms it's ignored; re-verified
  this milestone).
- No API keys, tokens, or secret values appear in any tracked file (`.env.example`
  and `frontend/.env.local.example` contain placeholders only — re-read this
  milestone).
- No database file (`data/*.db*`) tracked — re-verified this milestone via the
  live SQLite test in §10.
- No `node_modules`, no `.next` tracked (verified via `git check-ignore -v`,
  consistent with Milestone 42/43's own audits).
- No generated test artifacts tracked (`frontend/test-results/`,
  `playwright-report/`, `blob-report/` — added to `.gitignore` in Milestone 43,
  re-confirmed still correctly ignored).
- CORS is never a wildcard (§9).
- The Tavily key stays server-side only — `app/tools/web_search.py` reads
  `settings.tavily_api_key` (backend-only `os.getenv`); nothing in `frontend/`
  reads, proxies, or echoes it. Re-confirmed by the same static source scan
  Milestone 43 ran (`frontend/e2e/security-and-cors.spec.ts`'s "no unexpected
  `NEXT_PUBLIC_*`" test, still passing — §17).
- No Ollama API key is ever expected or read (§8).
- An Ollama/upstream failure surfaces as a safe `502 {"detail": "Ollama
  connection failed at <base_url>: <connection error>"}` — the base URL is the
  operator's own configured (non-secret) value, and the connection-error text is
  the OS/network layer's own message, never a stack trace or file path (verified
  live, §17).

No security control was weakened for deployment convenience; CORS, the
`502`-never-`500` upstream-failure handling, and the catch-all exception handler
are all pre-existing (Milestones 40-42), only re-verified here.

## 14. Cost model

**Mandatory cost: ₹0.**

- **Local Ollama inference** — no API billing; `llama3.2:3b` runs entirely on the
  operator's own machine/CPU, one-time download via `ollama pull`.
- **Local SQLite** — a plain file on local disk; no hosting/database service cost.
- **Local Next.js/FastAPI processes** — run on the operator's own machine; no
  hosting cost for the topology this milestone documents (A/B in §3).
- **Tavily web search** — the one *optional* external service. Free-tier limits
  apply (check Tavily's own current terms; not reproduced here since it is a
  third-party policy this repository doesn't control) — the product remains
  fully functional without it; only the `web_search` tool specifically needs it.

Nothing in this repository requires OpenAI, Anthropic, or Gemini API access, or
any paid SaaS, to run the documented topology.

## 15. Cloud deployment limitations

Stated plainly, per this milestone's explicit instruction not to overclaim:
**the complete system (including real local-model inference) is not
cloud-deployable within a genuinely free tier today.** `llama3.2:3b` needs
several GB of RAM and sustained CPU time that serverless/ephemeral free compute
(Vercel functions, free Render/Railway web services, GitHub Actions runners) does
not provide persistently or affordably. The frontend by itself *could* be
deployed free to a static/Next.js host, but that only relocates the UI — it still
needs a reachable backend+Ollama host, which remains the actual constraint. A
genuinely free, complete-stack cloud path was not found during this milestone's
inspection, and none is claimed. `docs/ARCHITECTURE_STATUS.md` §16/§19 already
reached the same conclusion (single-process/single-worker assumption; "a genuine
multi-process/horizontally-scaled deployment... remains the next step IF that
scale is ever actually required — not scheduled, no evidence of need today") —
this milestone does not change that assessment, only restates it in deployment
terms.

## 16. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `POST /chat` returns `502` | Ollama isn't running, or `OLLAMA_BASE_URL` is wrong | Start Ollama; verify `curl $OLLAMA_BASE_URL/api/tags` |
| `GET /health` is `"ok"` but chat still fails | Expected — `/health` never checks Ollama (§11) | Use a real `/chat` request, not `/health`, to verify the model path |
| Browser console shows a CORS error | The frontend's actual origin isn't in `CORS_ALLOWED_ORIGINS` | Add the exact origin (scheme+host+port) to `.env` and restart the backend |
| Frontend built but still calls the wrong backend URL | `NEXT_PUBLIC_API_BASE_URL` is baked in at build time (§7) | Set it before `npm run build`, then rebuild |
| `web_search` fails / agent says it can't search | `TAVILY_API_KEY` unset | Set it in `.env`; the rest of the product works without it |
| Backend won't start with `MEMORY_BACKEND=sqlite` | Bad/unwritable `SQLITE_DB_PATH`, or an incompatible old schema | Check directory permissions; a genuinely unsupported schema fails loudly with `SchemaVersionError` by design (§10) — it is never silently downgraded |
| `Address already in use` on port 8000/3000 | Another process (often a leftover dev server from a previous session) is bound to the port | Find and stop it (`netstat -ano \| findstr :8000`, then stop that PID) before starting a fresh one |

## 17. Validation results

Exact commands and results from this milestone's own run:

- Backend regression: `python -m pytest -m "not integration" -q` → **2275 passed,
  38 deselected**, unchanged from Milestone 43's baseline.
- `python -m compileall -q app tests` → clean.
- Frontend: `npm test` → **16 passed**; `npm run typecheck` → 0 errors; `npm run
  lint` → 0 problems; `npm run build` → clean production build.
- Ollama verified running, `llama3.2:3b` present (`curl http://localhost:11434/
  api/tags`).
- Backend started production-style (`uvicorn app.main:app`, no `--reload`) and
  verified healthy (`GET /health` → `200`).
- Frontend built and started production-style (`npm run build && npm run
  start`); verified via `curl` (HTTP 200) and a real Chromium browser session
  (Playwright) that loaded the page and completed one real chat turn.
- **One real chat request** (static question, no tool): `"capital of Japan?"` →
  `"Tokyo"`.
- **One real time-tool request**: `"What time is it right now?"` → a correct,
  real current-time answer.
- **Web-search request**: deliberately **not** re-run this milestone — Milestone
  43 (same session, same Tavily key) already performed genuine, successful
  `web_search` verification minutes earlier; re-running it here would only spend
  additional Tavily quota for no new information, which the brief explicitly
  warns against ("do not waste Tavily quota... only if necessary").
- **Ollama-unavailable graceful failure**: a throwaway backend instance pointed
  at an unreachable `OLLAMA_BASE_URL` returned `502` with a safe, non-stack-trace
  JSON body; the process did not crash and continued serving `/health` normally.
- **SQLite persistence**: full startup → health → chat → restart → recall cycle
  verified live (§10).
- **CORS**: re-verified correct (non-wildcard, exact origin) against the
  production frontend build specifically, not just the dev server.

## 18. Final verdict

**A. DEPLOYMENT READY — LOCAL / SINGLE-MACHINE**

The complete product runs reproducibly, at ₹0 mandatory cost, in a production-style
configuration (no dev-only flags) on a single machine: real Ollama inference, real
tool use, optional real SQLite persistence across restarts, a production Next.js
build served statically, and CORS correctly scoped to the real frontend origin.
Ollama unavailability degrades gracefully (`502`, no crash) rather than failing
the whole process. No P0/P1 issues were found. Cloud deployment of the complete
stack is explicitly **not** claimed — see §15 — because the architecture's own
single-process/single-worker/local-model assumptions (unchanged, by design) do not
fit a genuinely free cloud tier today; this is a documented limitation, not a
defect.

**Recommended next milestone: Step 45 — GitHub / Project Presentation &
Documentation** (not started in this session).
