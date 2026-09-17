from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.agent.permissions import AllowlistPermissionPolicy
from app.agent.reliability import BudgetedCorrectionPolicy
from app.agent.semantic_memory import MemorySessionIsolationError
from app.agent.telemetry import EventSink, LoggingEventSink, SafeEventSink
from app.agent.tool_execution import ToolExecutionGate
from app.agent.tool_registry import ToolRegistry
from app.config import settings
from app.models.llm import LLMClient
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

app = FastAPI(title=settings.app_name)

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
    llm_client=_llm_client,
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
)


class ChatRequest(BaseModel):
    message: str
    # Step 13: optional conversation-scope key, NOT identity/auth (see
    # app/agent/memory.py). Omitted -> ChatService's single legacy
    # conversation, exactly like before Step 13.
    session_id: str | None = None


class ChatResponse(BaseModel):
    reply: str


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": settings.app_name}


@app.post("/chat", response_model=ChatResponse)
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.app_port, reload=True)
