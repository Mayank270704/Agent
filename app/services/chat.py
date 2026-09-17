"""Thin service wrapper around AgentOrchestrator for the FastAPI layer.

Step 12: this class used to keep its own `conversation` list, but nothing
ever read it — it was appended to on every turn and never passed to the
LLM (see the Step 1 architecture audit). That was replaced with a real,
working `ConversationMemory` (app/agent/memory.py) — the ONE authoritative
source of cross-request conversation history.

Step 13 — session-scoped memory: `ask()` gained an optional `session_id`.

    session_id given      -> SessionMemoryStore.get_memory(session_id)
    session_id omitted    -> a FRESH, request-scoped ConversationMemory,
                              built per `ask()` call and discarded when that
                              call returns (see `_resolve_memory`)

`session_id` is a conversation-scope key ONLY — not identity, not
authentication (see app/agent/memory.py's module docstring). Two different
explicit session IDs can never share memory; that guarantee lives entirely
in `InMemorySessionMemoryStore` (app/agent/memory.py), not duplicated here.

--------------------------------------------------------------------------
Anonymous requests are request-scoped, not a shared global conversation
--------------------------------------------------------------------------
Until the anonymous-memory-isolation fix, `session_id=None` resolved to a
single `self._default_memory` instance attribute — one conversation shared
by EVERY caller who omitted a session_id, for the entire lifetime of the
`ChatService` (and `app/main.py`'s `chat_service` is a module-level
singleton, so in the deployed server that meant one transcript shared
across all anonymous callers, growing until restart). The capability
assessment reproduced the consequence live: an unrelated anonymous request
was answered with content carried over from an earlier anonymous request.

`session_id=None` now resolves to a FRESH `InMemoryConversationMemory` per
`ask()` call, which is dropped as soon as that call returns. An anonymous
request therefore starts with no history and leaves none behind — it
cannot inherit another anonymous request's messages, and cannot leak its
own into the next one. There is deliberately no instance attribute holding
it: a shared attribute is precisely what the defect was.

This does NOT give anonymous requests a durable identity — it removes the
accidental one they had. An anonymous request still records no episode and
performs no semantic retrieval/write (see the Step 14 / 16E rules below,
which are unchanged): those remain reserved for explicit sessions, because
they are the parts that would persist BEYOND the request, and a request
that deliberately has no cross-request identity must not create durable
cross-request state. The Milestone 19 `request_id` is likewise unaffected:
it identifies one request for telemetry correlation and is never used as,
or promoted to, a conversation identity.

Why a fresh AgentOrchestrator is built per `ask()` call rather than reused
from `__init__` (as before Step 13): AgentOrchestrator reads `self.memory`
at call time, so the OLD single-orchestrator design would have needed this
method to mutate `self.orchestrator.memory` on every call to point at the
right session. FastAPI runs a synchronous route handler (like `/chat`'s) in
a thread pool, so two concurrent requests for two different sessions could
genuinely race on that shared mutable attribute. Building a new, cheap,
stateless-at-construction AgentOrchestrator per call (sharing only the
already-stateless `self.llm`) avoids that hazard entirely — see the Step 13
report for the concurrency reasoning in full.

Step 14 correction — episodic memory is now LIVE in this class, not just at
the AgentOrchestrator level:

    self.episodic_memory: ONE InMemoryEpisodicMemory, owned by this
    ChatService instance (same explicit-ownership pattern as
    self.session_store — never a module-level singleton).

    session_id given    -> `ask()` passes BOTH `memory` (this session's
                            ConversationMemory) AND `episodic_memory` +
                            `session_id` into the fresh AgentOrchestrator.
                            A successful execution therefore commits a
                            conversation turn AND records one episode,
                            both tagged to that same session_id.

    session_id omitted  -> the request-scoped conversation memory
                            described above is used, and
                            `episodic_memory`/`session_id` are
                            NOT passed to AgentOrchestrator at all, so
                            `orchestrator.episodic_memory` stays None and
                            NO episodic record is ever created for this
                            path. This is deliberate (see the Step 14
                            correction report): there is no real session
                            identity to tag a record with here, and
                            inventing one (e.g. "default"/"anonymous")
                            would misrepresent an anonymous, unscoped
                            conversation as if it were a real session.

Step 16E-C extends that same split to semantic memory retrieval: the
optional `memory_retriever` is passed to AgentOrchestrator ONLY on the
explicit-session path. The `session_id=None` path deliberately performs no
semantic retrieval at all, for the identical reason it records no episode —
there is no honest session to scope a search to, and retrieval scoped to an
invented shared identity would be a cross-user memory leak, not a feature.

Step 16E-D adds the write half (`memory_extractor` + `memory_writer`) under
the identical rule: they are passed to AgentOrchestrator ONLY on the
explicit-session path, so a `session_id=None` request extracts nothing and
writes nothing. Writing a durable fact under an invented shared identity
would be strictly worse than reading one — it would persist one user's
information where another could later retrieve it.

Milestone 18-A — tool authorization, wired for BOTH session paths, unlike
memory: `tool_registry`/`tool_execution_gate` are injected once (owned by
this instance, or ultimately by app/main.py's composition root) and passed
into `AgentOrchestrator` on EVERY `ask()` call, `session_id` present or
not. This is a deliberate difference from the memory rule above: memory
and episodic recording need a real session identity to be attributable to,
so they are withheld without one; tool execution has no such requirement
— an anonymous, unscoped request still executes real tools and must be
authorized exactly as a scoped one is. There is no "authorization only
applies to sessions" carve-out.

Both new constructor parameters default to `None`, matching every other
optional collaborator here (`memory_retriever`, `memory_extractor`, ...):
`ChatService()` with neither supplied is byte-for-byte the pre-Milestone-
18-A behavior — `AgentOrchestrator` falls back to its own internally
constructed default `ToolRegistry` (see app/agent/orchestrator.py) and
`AgentLoop` runs with no gate, exactly as it always has. This class makes
no decision about tool authorization itself: it neither builds a
`ToolRegistry` nor a `PermissionPolicy` nor a `ToolExecutionGate` — those
are the composition root's job (see app/main.py) — it only forwards
whatever it was given, plus one thing it DOES construct itself: a fresh
`ExecutionContext` per `ask()` call (see below).

`ExecutionContext` is built HERE, fresh, on every single `ask()` call,
rather than once in `__init__` or once in app/main.py — deliberately,
because it carries `session_id`, which varies per request. Building it
fresh also means no confirmation or authorization state can ever leak
between two different sessions, or between two calls for the same
session: `confirmed_tools` starts empty on every context (there is no
mechanism anywhere in this codebase that adds to it), so there is nothing
to leak in the first place, but a fresh instance per call is the property
that keeps that true even if a future policy needed a real confirmed set.
`session_id` is passed through as descriptive metadata ONLY — see
app/agent/permissions.py's module docstring for why `AllowlistPermission
Policy.evaluate()` never reads it to make a decision. A session_id must
never itself grant privilege, and it does not: the allow-list this
context is evaluated against is fixed, application-owned configuration,
entirely independent of which session is asking.
"""
from __future__ import annotations

import uuid

from app.agent.episodic_memory import InMemoryEpisodicMemory
from app.agent.memory import ConversationMemory, InMemoryConversationMemory, InMemorySessionMemoryStore
from app.agent.memory_extraction import MemoryExtractor
from app.agent.memory_retriever import MemoryRetriever
from app.agent.memory_writer import MemoryWriter
from app.agent.orchestrator import AgentOrchestrator
from app.agent.permissions import ExecutionContext
from app.agent.reliability import CorrectionPolicy
from app.agent.telemetry import EventEmitter, EventSink
from app.agent.tool_execution import ToolExecutionGate
from app.agent.tool_registry import ToolRegistry
from app.config import settings
from app.models.llm import LLMClient


class ChatService:
    def __init__(
        self,
        llm_client: LLMClient | None = None,
        memory_retriever: MemoryRetriever | None = None,
        memory_extractor: MemoryExtractor | None = None,
        memory_writer: MemoryWriter | None = None,
        tool_registry: ToolRegistry | None = None,
        tool_execution_gate: ToolExecutionGate | None = None,
        event_sink: EventSink | None = None,
        correction_policy: CorrectionPolicy | None = None,
        deterministic_temporal_routing: bool = False,
    ):
        # Optional DI, matching every other component's pattern in this
        # codebase (AgentOrchestrator, Router, LLMDecisionMaker, ...) —
        # added so tests can inject a fake LLM client deterministically;
        # main.py's real ChatService() call is unaffected (still builds the
        # real LLMClient from settings, exactly as before Step 13).
        self.llm = llm_client or LLMClient(
            provider=settings.llm_provider,
            model_name=settings.model_name,
            api_key=settings.openai_api_key,
            base_url=settings.ollama_base_url,
        )
        self.session_store = InMemorySessionMemoryStore()
        # NOTE: there is deliberately no `_default_memory` instance
        # attribute here. Anonymous (`session_id=None`) requests get a
        # fresh, request-scoped ConversationMemory built inside
        # `_resolve_memory` instead — see the module docstring for the
        # defect a shared attribute caused.
        # Step 14 correction: one episodic store per ChatService instance,
        # explicitly owned here — never a module-level global. Shared
        # across all explicit sessions on THIS instance (records are
        # partitioned internally by session_id, same as session_store
        # partitions ConversationMemory) but never shared across separate
        # ChatService instances.
        self.episodic_memory = InMemoryEpisodicMemory()
        # Step 16E-C: semantic memory retrieval is INJECTED, never
        # constructed here, and defaults to None. A default retriever would
        # mean building an embedding provider + vector index that nothing
        # ever writes to (the write path is a later milestone), so every
        # request would pay for an embedding call and a search guaranteed to
        # return zero results. main.py's live ChatService() passes nothing,
        # so the deployed agent's behavior and prompt are unchanged.
        self.memory_retriever = memory_retriever
        # Step 16E-D: the write half, injected the same way and for the
        # same reason. Both default to None, so main.py's live
        # ChatService() neither retrieves nor writes semantic memory and
        # the deployed agent is unchanged.
        self.memory_extractor = memory_extractor
        self.memory_writer = memory_writer
        # Milestone 18-A: pure pass-through, opt-in, default None — see the
        # module docstring. Neither is validated against the other here
        # (e.g. "does tool_execution_gate wrap THIS tool_registry?"); that
        # consistency is the composition root's responsibility (app/main.py),
        # exactly as it already is for `memory_retriever` vs. `memory_writer`
        # sharing one store (see app/semantic_memory_wiring.py).
        self.tool_registry = tool_registry
        self.tool_execution_gate = tool_execution_gate
        # Milestone 19: pure pass-through, opt-in, default None. Never
        # constructed here — app/main.py's composition root decides
        # whether telemetry exists at all, exactly like tool_registry/
        # tool_execution_gate above.
        self.event_sink = event_sink
        # Production-correction wiring: pure pass-through, opt-in, default
        # None — identical pattern to tool_execution_gate/event_sink above.
        # This class makes no decision about WHICH policy or budget to use
        # (that is the composition root's job, app/main.py) and builds no
        # CorrectionPolicy of its own; it only forwards whatever it was
        # given to AgentOrchestrator on every ask() call, exactly like
        # tool_execution_gate. With correction_policy=None (the default),
        # AgentLoop's behavior is unchanged from before this wiring — see
        # app/agent/reliability.py's module docstring.
        self.correction_policy = correction_policy
        # Deterministic temporal routing: pure pass-through, opt-in,
        # default False — identical pattern to correction_policy above.
        # This class makes no routing decision itself; it only forwards
        # the composition root's choice (app/main.py) to every
        # AgentOrchestrator it builds.
        self.deterministic_temporal_routing = deterministic_temporal_routing

    def ask(self, user_message: str, session_id: str | None = None) -> str:
        if user_message is None or not user_message.strip():
            raise ValueError("User message cannot be empty.")

        cleaned_message = user_message.strip()
        memory = self._resolve_memory(session_id)
        # Milestone 18-A: built fresh on EVERY call — see the module
        # docstring for why this must never be cached or shared across
        # requests/sessions. `session_id` is carried as descriptive
        # metadata only; `confirmed_tools` is always empty here because
        # nothing in this codebase has a mechanism to populate it (no UI
        # confirmation workflow exists — see app/agent/permissions.py).
        execution_context = ExecutionContext(session_id=session_id)
        # Milestone 19: exactly one request_id per ask() call, generated
        # HERE — the same true request boundary `execution_context` above
        # already uses — never derived from, or equal to, `session_id`
        # (see app/agent/telemetry.py's module docstring on why a shared,
        # often-None session identity must never stand in for a per-call
        # request identity). Generated unconditionally (cheap, no I/O)
        # even when `self.event_sink` is None; the EventEmitter itself is
        # only ever constructed when a real sink exists, so disabled
        # telemetry constructs no AgentEvent anywhere in this call.
        request_id = str(uuid.uuid4())
        event_emitter = (
            EventEmitter(self.event_sink, request_id, session_id) if self.event_sink is not None else None
        )
        if session_id is None:
            # Legacy path (Step 13): no real session identity to tag an
            # episode with, so episodic recording stays off entirely.
            # Tool authorization is NOT part of that carve-out (see the
            # module docstring) — the gate/registry are still forwarded.
            orchestrator = AgentOrchestrator(
                llm_client=self.llm,
                memory=memory,
                tool_registry=self.tool_registry,
                tool_execution_gate=self.tool_execution_gate,
                execution_context=execution_context,
                event_emitter=event_emitter,
                correction_policy=self.correction_policy,
                deterministic_temporal_routing=self.deterministic_temporal_routing,
            )
        else:
            orchestrator = AgentOrchestrator(
                llm_client=self.llm,
                memory=memory,
                episodic_memory=self.episodic_memory,
                memory_retriever=self.memory_retriever,
                memory_extractor=self.memory_extractor,
                memory_writer=self.memory_writer,
                session_id=session_id,
                tool_registry=self.tool_registry,
                tool_execution_gate=self.tool_execution_gate,
                execution_context=execution_context,
                event_emitter=event_emitter,
                correction_policy=self.correction_policy,
                deterministic_temporal_routing=self.deterministic_temporal_routing,
            )
        result = orchestrator.process(cleaned_message)
        return result.answer

    def _resolve_memory(self, session_id: str | None) -> ConversationMemory:
        """Explicit session -> that session's persistent memory, from the
        store, exactly as before. Anonymous -> a brand-new, bounded
        ConversationMemory that lives only for this one `ask()` call.

        Called exactly once per `ask()`, so "a fresh instance per call" is
        structural here rather than conventional: nothing retains the
        returned object after `ask()` returns, and no attribute on `self`
        ever points at it. `InMemoryConversationMemory` is bounded by its
        own `max_messages` default (app/agent/memory.py) — an anonymous
        conversation cannot grow without limit even within a single
        request.
        """
        if session_id is None:
            return InMemoryConversationMemory()
        return self.session_store.get_memory(session_id)
