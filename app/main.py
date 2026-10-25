from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.agent.episodic_memory import EpisodicMemory, InMemoryEpisodicMemory
from app.agent.memory import InMemorySessionMemoryStore, SessionMemoryStore
from app.agent.permissions import AllowlistPermissionPolicy
from app.agent.reliability import BudgetedCorrectionPolicy
from app.agent.semantic_memory import MemorySessionIsolationError
from app.agent.telemetry import EventSink, LoggingEventSink, SafeEventSink
from app.agent.tool_execution import ToolExecutionGate
from app.agent.tool_registry import ToolRegistry
from app.config import settings
from app.models.llm import LLMClient
from app.persistence import db as persistence_db
from app.persistence.sqlite_episodic_memory import SQLiteEpisodicMemory
from app.persistence.sqlite_memory import SQLiteSessionMemoryStore
from app.semantic_memory_wiring import build_semantic_memory
from app.services.chat import ChatService
from app.tools.date import DateTool
from app.tools.time import TimeTool
from app.tools.web_search import WebSearchTool

def configure_logging(level: str) -> None:
    """Activate Python's logging pipeline for this process (Production
    Readiness Audit, P0-1). `settings.log_level` (app/config.py) has
    existed since before this fix but was dead configuration — nothing
    anywhere called `logging.basicConfig()`/`logging.config.dictConfig()`,
    so every existing `logger.info(...)`/`logger.warning(...)` call in
    this codebase (tool authorization/denial, corrections, deterministic
    routing decisions, and the Milestone 19 `LoggingEventSink` telemetry
    path) was silently discarded in a real deployment.

    `force=True` is required, not decorative: this is the composition
    root, imported exactly once per process, but SOME other component
    (a test harness, an ASGI server's own logging setup, a library) may
    already have attached a handler to the root logger before this line
    runs. Without `force=True`, `basicConfig` is a silent no-op whenever
    the root logger already has ANY handler — which would reintroduce
    the exact "looks configured, silently isn't" failure mode this fix
    exists to close. `force=True` guarantees this call is authoritative
    regardless of import order.

    The format is deliberately minimal — timestamp, level, logger name,
    message — and renders ONLY what each log call already explicitly
    passes as arguments. It adds no new content to any log line: every
    existing call site in this codebase was already audited (Milestone
    18-C) to log structured, non-sensitive fields (tool names, category
    enums, booleans, counts) and never a raw prompt, model output, tool
    input/output, memory content, or secret — this function does not
    change what is logged, only whether it is ever emitted.
    """
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )


configure_logging(settings.log_level)

logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    # Milestone 41 — API/product polish: a concise, consumer-facing
    # description for /docs and /openapi.json. Deliberately short and
    # behavioral (what a client does with this API), not an architecture
    # writeup — see docs/ARCHITECTURE_STATUS.md for that.
    description=(
        "A tool-using conversational agent backed by a local LLM (Ollama). "
        "POST /chat to send a message; optionally supply session_id to "
        "resume a conversation. GET /health reports application and "
        "persistence status. See /docs for interactive, try-it-out API "
        "documentation."
    ),
)

# Milestone 42 — the minimum backend change the frontend milestone needs:
# allow the actual Next.js dev frontend (frontend/, port 3000 by default)
# to call this API cross-origin. Configuration-backed
# (`settings.cors_allowed_origins`, CORS_ALLOWED_ORIGINS), never
# `allow_origins=["*"]` — Milestone 41 deferred CORS entirely because no
# frontend origin was known yet; now one is, so the default names it
# explicitly instead of opening the API to every origin.
#
# Least-privilege on every other axis too: `allow_credentials=False` (this
# API uses no cookies/auth — session_id travels in the JSON body, never a
# cookie), `allow_methods` limited to the two methods this API actually
# exposes (GET for `/` and `/health`, POST for `/chat`), `allow_headers`
# limited to the one header the frontend's JSON POST actually needs
# preflight clearance for.
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_allowed_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# Step 16I: the composition root. Built exactly once, at import time --
# `chat_service` itself was already built this way before this step, so
# this changes WHAT is constructed, never WHEN or how many times.
#
# One LLMClient, shared between ChatService (for the agent's own
# reasoning) and the semantic memory bundle's LLMMemoryExtractor (for
# extracting durable facts from a completed turn) -- previously
# ChatService quietly built its own LLMClient internally when none was
# supplied; that default is reproduced here explicitly so it can be
# handed to both consumers instead of constructed twice.
_llm_client = LLMClient(
    provider=settings.llm_provider,
    model_name=settings.model_name,
    api_key=settings.openai_api_key,
    base_url=settings.ollama_base_url,
    # Milestone 44.1 -- both None/unset by default (settings.ollama_
    # temperature/ollama_seed), in which case this is byte-identical to
    # every milestone before 44.1. See app/config.py for what each does
    # and why neither is flipped on by default.
    temperature=settings.ollama_temperature,
    seed=settings.ollama_seed,
)

# None whenever SEMANTIC_MEMORY_ENABLED is unset or false (the default) --
# see app/semantic_memory_wiring.py. A construction failure here (e.g. an
# unloadable model when ENABLED) is deliberately NOT caught: this module
# adds no try/except around it, so a real embedding provider that cannot
# load fails application STARTUP loudly rather than letting the process
# boot into a state where semantic memory looks configured but silently
# never works.
_semantic_memory = build_semantic_memory(
    enabled=settings.semantic_memory_enabled,
    model_name=settings.embedding_model_name,
    device=settings.embedding_device,
    max_records_per_session=settings.semantic_memory_max_records_per_session,
    # Step 33: closes a gap found auditing activation readiness — the
    # retriever's own class default (-1.0) applies NO relevance floor, so
    # a session with few stored memories would surface its least-bad match
    # regardless of relevance. See app/config.py's
    # `semantic_memory_min_similarity` for the measured calibration this
    # value (0.3 by default) is based on. No effect while semantic memory
    # is disabled (the default) -- this line is never reached in that case.
    min_similarity=settings.semantic_memory_min_similarity,
    llm_client=_llm_client,
    # Step 38: reuses the SAME MEMORY_BACKEND/SQLITE_DB_PATH settings Steps
    # 36/37 already introduced -- no separate semantic-memory backend flag.
    # No effect while semantic memory is disabled (the default).
    memory_backend=settings.memory_backend,
    sqlite_db_path=settings.sqlite_db_path,
)

# Milestone 18-A: the tool-authorization boundary, activated for real.
# Built ONCE, here, at import time — the same composition-root discipline
# `_llm_client`/`_semantic_memory` above already follow — and reused for
# every request; `ChatService.ask()` passes this SAME registry and gate
# into a fresh `AgentOrchestrator` on every call (see app/services/chat.py),
# so authorization is never rebuilt, and never bypassed, per request.
#
# The allow-list below is an EXPLICIT, application-owned enumeration of
# the only tools this deployment currently approves — never derived from
# `_tool_registry.list_tools()` or any other "whatever happens to be
# registered" source (see app/agent/permissions.py's module docstring on
# why AllowlistPermissionPolicy holds no registry reference at all). A
# tool registered below but NOT named here is denied; a tool registered
# LATER (by a future milestone) and not added here is denied by the exact
# same mechanism — extending the registry has zero effect on what this
# policy authorizes.
_tool_registry = ToolRegistry()
_tool_registry.register(WebSearchTool())
_tool_registry.register(TimeTool())
_tool_registry.register(DateTool())

# Step 36: conversation/session persistence backend selection. Default
# "memory" (settings.memory_backend) reproduces every milestone before 36
# exactly -- a fresh, process-local InMemorySessionMemoryStore. "sqlite"
# selects the persistent backend instead; a failure to open/create that
# database (bad path, permissions, disk full) is deliberately NOT caught
# here, for the same reason `_semantic_memory`'s construction above isn't:
# a configured-but-unavailable persistence backend must fail application
# startup loudly, never silently fall back to the in-memory store (Step 36
# requirement: "do not silently fall back from SQLite to memory").
if settings.memory_backend == "sqlite":
    _session_store: SessionMemoryStore = SQLiteSessionMemoryStore(db_path=settings.sqlite_db_path)
else:
    _session_store = InMemorySessionMemoryStore()

# Step 37: episodic memory persistence, selected by the SAME MEMORY_BACKEND
# flag (no separate setting) -- "sqlite" selects SQLiteEpisodicMemory,
# sharing the same db_path/database file as _session_store above (a
# SEPARATE connection to it; see app/persistence/db.py's migration-2
# comment for why episodic memory tracks its own session bookkeeping
# rather than the conversation/session store's `sessions` table). Failure
# to open/create the database is NOT caught here either, for the identical
# reason as _session_store above.
if settings.memory_backend == "sqlite":
    _episodic_memory: EpisodicMemory = SQLiteEpisodicMemory(db_path=settings.sqlite_db_path)
else:
    _episodic_memory = InMemoryEpisodicMemory()

_permission_policy = AllowlistPermissionPolicy({"time", "date", "web_search"})

_tool_execution_gate = ToolExecutionGate(_tool_registry, _permission_policy)

# Production correction wiring (Capability Assessment finding #1): the
# already-built, already-tested, already-18-C-audited Milestone 17
# self-correction mechanism, activated for real requests. Built ONCE,
# here, at import time — the same composition-root discipline every other
# collaborator on this page follows — and reused for every request,
# exactly like `_tool_execution_gate` above (unconditional, no settings
# flag: unlike telemetry, there is no deployment-relevant reason to run
# WITHOUT correction — a malformed decision or a transient tool failure
# should not need an operator to opt in to recovering from it).
#
# Defaults are used unchanged (max_corrections=2, include_tool_error_text
# =False) — see app/agent/reliability.py's module docstring for why those
# specific values were chosen; nothing about wiring this in requires
# changing them. PERMISSION_DENIED and CONFIRMATION_REQUIRED remain
# structurally non-correctable regardless of this policy's configuration
# — AgentLoop never offers them to ANY policy (see app/agent/loop.py),
# and this policy independently refuses them too, as defense in depth.
_correction_policy = BudgetedCorrectionPolicy()

# Milestone 19: the observability composition point. Default OFF
# (settings.telemetry_enabled, unset unless TELEMETRY_ENABLED is set) --
# disabled means `_event_sink` is None and every downstream `event_emitter`
# stays None all the way through ChatService -> AgentOrchestrator ->
# AgentLoop/LLMDecisionMaker, so no AgentEvent is ever constructed and
# behavior is byte-identical to every milestone before 19. When enabled,
# events render as human-readable log lines via LoggingEventSink -- the
# derived view, not a second competing telemetry stream (see
# app/agent/telemetry.py's module docstring) -- wrapped in SafeEventSink
# so a bug in this rendering can NEVER fail a real request; this is
# defense in depth, since `EventEmitter` itself already applies the
# identical wrapping unconditionally.
_event_sink: EventSink | None = SafeEventSink(LoggingEventSink()) if settings.telemetry_enabled else None

# Disabled semantic memory passes None for all three -- byte-identical to
# every milestone before 16I, and to `ChatService()`'s own defaults. The
# tool registry/gate, by contrast, are ALWAYS supplied now: unlike
# semantic memory, tool execution has no "disabled by default" mode --
# every real tool call in this deployment goes through authorization.
chat_service = ChatService(
    llm_client=_llm_client,
    session_store=_session_store,
    episodic_memory=_episodic_memory,
    memory_retriever=_semantic_memory.retriever if _semantic_memory is not None else None,
    memory_extractor=_semantic_memory.extractor if _semantic_memory is not None else None,
    memory_writer=_semantic_memory.writer if _semantic_memory is not None else None,
    tool_registry=_tool_registry,
    tool_execution_gate=_tool_execution_gate,
    event_sink=_event_sink,
    correction_policy=_correction_policy,
    # Capability Assessment finding #2: make the ALREADY-deterministic
    # time/date cases authoritative instead of merely advisory. The
    # measured problem was under-calling (76.5% missed-tool-call rate with
    # the advisory hint active, 0% unnecessary calls), and Router's
    # LLM-free matcher already identifies this narrow class with
    # certainty. Scope is exactly that class: the general Router is NOT
    # authoritative, ambiguous temporal phrasing still goes to the LLM,
    # and the selected tool still runs through the unchanged
    # ToolExecutionGate pipeline -- see app/agent/decision_maker.py's
    # `_deterministic_temporal_decision`.
    deterministic_temporal_routing=True,
    # Step 26 — the same fix, applied to the most consequential remaining
    # under-calling gap: llama3.2:3b frequently ignores an explicit
    # instruction to search ("search the web for X", "look up X online"), even
    # with the existing advisory ROUTING HINT (Router.classify_hint /
    # RoutingHint.TOOL_LIKELY) already in the prompt. Scope is deliberately
    # narrow, matching the temporal fix's own discipline: only an explicit,
    # imperative search COMMAND is authoritative (see Router.
    # deterministic_web_search_route); a bare recency keyword ("latest",
    # "current") or an ordinary current-information QUESTION with no
    # command phrasing still goes through the unchanged advisory hint and
    # the LLM's own judgment. The selected tool still runs through the
    # unchanged ToolExecutionGate pipeline -- see app/agent/decision_maker.py's
    # `_deterministic_web_decision`.
    deterministic_web_search_routing=True,
)


# Milestone 41 — API/product polish: upper-bound length validation.
# Nothing upstream of ChatRequest currently limits request body size at
# all (Starlette/uvicorn apply no default limit), so an unbounded
# `message` was a genuine, open resource-exhaustion vector for a
# public-facing endpoint -- this closes exactly that gap, additively,
# with no change to any existing accepted input (every existing test's
# message/session_id is far under both bounds).
#
# MAX_MESSAGE_LENGTH mirrors the SPIRIT of two existing bounded-text
# constants elsewhere in this codebase (app/agent/memory_extraction.py's
# MAX_CANDIDATE_CONTENT_CHARS=300 for a distilled fact,
# app/agent/decision_maker.py's _MAX_OBSERVATION_DATA_CHARS=2000 for one
# tool observation) but is deliberately larger: a chat message is the
# PRIMARY request, not a derived artifact, so it needs more headroom. 8000
# characters (~2000 tokens at the common ~4-chars/token heuristic)
# comfortably fits a long, detailed question while staying well short of
# risking the model's practical context budget once tool descriptions,
# execution history, and (optionally) memory context are also added to
# the prompt.
MAX_MESSAGE_LENGTH = 8000
# A session_id is a short, caller-chosen key (a UUID, a slug, ...), never
# user-authored prose -- 200 characters is generous for any realistic
# scheme while bounding it as a SQLite TEXT PRIMARY KEY / in-memory dict
# key (MEMORY_BACKEND=sqlite, app/persistence/).
MAX_SESSION_ID_LENGTH = 200


class ChatRequest(BaseModel):
    message: str = Field(
        max_length=MAX_MESSAGE_LENGTH,
        description="The user's chat message.",
        examples=["What is a transformer?"],
    )
    # Step 13: optional conversation-scope key, NOT identity/auth (see
    # app/agent/memory.py). Omitted -> a fresh, isolated, request-scoped
    # conversation, exactly like before Step 13.
    session_id: str | None = Field(
        default=None,
        max_length=MAX_SESSION_ID_LENGTH,
        description=(
            "Optional conversation-scope key — NOT an identity or authentication "
            "mechanism. Reusing the same session_id resumes the same conversation "
            "(and, when enabled, episodic/semantic memory). Omitting it starts an "
            "isolated, request-scoped conversation with no memory of prior turns "
            "and leaves no state behind."
        ),
        examples=["demo-session"],
    )


class ChatResponse(BaseModel):
    reply: str = Field(description="The agent's final answer to the user's message.")


@app.get(
    "/",
    summary="API root",
    description="Minimal identification endpoint — confirms the service is reachable.",
)
def read_root() -> dict[str, str]:
    return {"message": settings.app_name}


@app.get(
    "/health",
    summary="Application and persistence health",
    description=(
        "Reports whether the application is running, which memory backend is "
        "configured (memory or sqlite), and — only when a SQLite backend is "
        "configured — whether it is currently reachable and its schema version "
        "is supported. Returns HTTP 200 when healthy, 503 when the configured "
        "SQLite backend is unreachable or its schema is unsupported. Does not "
        "call Ollama or Tavily, and never scans semantic/vector data — this "
        "check stays cheap regardless of how much data exists."
    ),
    responses={
        200: {"description": "Healthy — application running, persistence (if configured) reachable."},
        503: {"description": "Degraded — the configured SQLite backend is unreachable or its schema is unsupported."},
    },
)
def health() -> JSONResponse:
    """Step 39: a minimal, CHEAP liveness/readiness probe — new endpoint,
    `GET /` above is untouched (backward compatible; no existing caller of
    `GET /` is affected by this addition).

    Distinguishes exactly the four things a persistence-aware health check
    needs to, no more:
      - the application process is running (trivially true if this handler
        runs at all; echoed explicitly as `application: "running"` so a
        caller does not have to infer it from a 200 alone)
      - whether a persistent backend is even CONFIGURED
        (`memory_backend`)
      - whether that backend is actually reachable right now
        (`persistence: "available"` vs `"unavailable"`) -- proven by a
        REAL read against the already-open connection
        (`SQLiteSessionMemoryStore.health_check()`), not merely "did
        construction succeed at startup"
      - whether its recorded schema version is one this running code
        supports (`schema_supported`)

    Deliberately does NOT touch episodic or semantic/vector storage: all
    five persistent stores share ONE physical database file in production
    (see app/persistence/db.py), so one cheap, already-open-connection read
    against the session store is a representative, O(1) probe for the
    whole file's reachability -- checking all five separately would add
    cost for no additional signal, and a semantic/vector-specific scan is
    exactly the "expensive check on every health request" this endpoint is
    required to avoid.

    Returns HTTP 200 when healthy, 503 when the configured persistence
    backend is unreachable or its schema is unsupported -- the conventional
    contract for a liveness/readiness probe consumed by an orchestrator.
    """
    if settings.memory_backend != "sqlite":
        # No persistent backend configured -- nothing to probe, and
        # critically, in-memory mode never opens a `sqlite3.Connection` or
        # touches `settings.sqlite_db_path` at all (see app/main.py's
        # backend-selection block above): SQLite stays a non-required path
        # exactly as Step 39 requires.
        return JSONResponse(
            status_code=200,
            content={
                "status": "ok",
                "application": "running",
                "memory_backend": settings.memory_backend,
                "persistence": "not_configured",
            },
        )

    try:
        schema_version = _session_store.health_check()
    except Exception as exc:  # noqa: BLE001 - any failure here means "unhealthy"; the
        # specific exception type (sqlite3.Error, SchemaVersionError, ...)
        # is reported by name only, never str(exc) -- a raw SQLite error
        # message can include the filesystem path, which does not belong
        # in a response body a client can read (same discipline as the
        # MemorySessionIsolationError handler below).
        return JSONResponse(
            status_code=503,
            content={
                "status": "degraded",
                "application": "running",
                "memory_backend": settings.memory_backend,
                "persistence": "unavailable",
                "detail": type(exc).__name__,
            },
        )

    schema_supported = schema_version == persistence_db.CURRENT_SCHEMA_VERSION
    return JSONResponse(
        status_code=200 if schema_supported else 503,
        content={
            "status": "ok" if schema_supported else "degraded",
            "application": "running",
            "memory_backend": settings.memory_backend,
            "persistence": "available",
            "schema_version": schema_version,
            "schema_supported": schema_supported,
        },
    )


@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send a message to the agent",
    description=(
        "Sends one message to the agent and returns its final answer. Supply "
        "session_id to resume a conversation (and, when enabled, episodic/"
        "semantic memory) across requests; omit it for an isolated, "
        "request-scoped conversation. The agent may call a registered tool "
        "(current time, date lookup, web search) before answering, entirely "
        "server-side — the response is always the final answer text, never an "
        "intermediate tool call."
    ),
    response_description="The agent's final answer.",
    responses={
        400: {"description": "Invalid request: empty message, or a blank session_id."},
        422: {"description": "Malformed request body, or message/session_id exceeds its length limit."},
        500: {"description": "Internal server error (a server-side invariant violation, or an unexpected failure)."},
        502: {"description": "An upstream dependency (e.g. Ollama) failed or was unreachable."},
    },
)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        # An explicit-but-blank session_id (e.g. "" or "   ") is rejected by
        # InMemorySessionMemoryStore itself as a ValueError -> the existing
        # 400 handler below already covers it; no new validation is added
        # here (see the Step 13 report on avoiding duplicated validation).
        reply = chat_service.ask(request.message, session_id=request.session_id)
    except MemorySessionIsolationError as exc:
        # Step 16I: a cross-session memory integrity violation is neither
        # the caller's fault (not a 400 -- nothing about THIS request was
        # invalid) nor an ordinary upstream-dependency failure (not the
        # generic 502 below) -- it is a server-side invariant violation,
        # so it gets its own branch and its own status code. It MUST be
        # handled here, before the `except ValueError` branch: this
        # exception IS a ValueError subclass (see app/agent/semantic_memory.py),
        # so without this branch ordered first it would silently be
        # reported as an ordinary 400 "bad request" -- exactly the
        # degradation the 16I design explicitly forbids.
        #
        # The response detail is a fixed, generic string, never
        # `str(exc)` -- that message names both a memory_id and two
        # session_ids (see MemorySessionIsolationError's raise sites),
        # and none of that belongs in a response body a client can read.
        # The server-side log line is similarly restricted to the
        # exception's TYPE, not its text, for the same reason.
        logger.error("Memory session isolation violation (%s).", type(exc).__name__)
        raise HTTPException(status_code=500, detail="Internal memory consistency error.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return ChatResponse(reply=reply)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Milestone 41 — a safety net for whatever escapes every endpoint's own
    handling, so the error response SHAPE stays consistent and predictable
    for a client (always JSON, always `{"detail": ...}`) instead of
    Starlette's own default plain-text 500.

    This changes nothing about what is caught or how: `/chat`'s own
    try/except (MemorySessionIsolationError / ValueError / RuntimeError ->
    HTTPException) still runs first and is completely unaffected — FastAPI
    resolves the more specific `HTTPException` handler before this
    catch-all one (verified directly: an `HTTPException` raised inside a
    route still reaches the client with its own status/detail, unchanged).
    This handler only ever receives what escapes that entirely — a
    genuinely unexpected failure (e.g. a raw `sqlite3.Error` if the
    configured SQLite file becomes unavailable mid-request, a future bug).

    The exception's TYPE NAME is logged server-side for diagnosis; its
    message is never included in the response body, which could echo
    internal detail (a file path, a query, a stack-trace-shaped string) —
    the identical discipline the existing `MemorySessionIsolationError`
    handler above already applies to its own exception.
    """
    logger.error("Unhandled exception escaped request handling (%s).", type(exc).__name__)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.app_port, reload=True)
