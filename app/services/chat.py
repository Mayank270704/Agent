"""Thin service wrapper around AgentOrchestrator for the FastAPI layer.

Step 12: this class used to keep its own `conversation` list, but nothing
ever read it — it was appended to on every turn and never passed to the
LLM (see the Step 1 architecture audit). That was replaced with a real,
working `ConversationMemory` (app/agent/memory.py) — the ONE authoritative
source of cross-request conversation history.

Step 13 — session-scoped memory: `ask()` gained an optional `session_id`.

    session_id given      -> SessionMemoryStore.get_memory(session_id)
    session_id omitted    -> self._default_memory (this instance's own
                              single legacy conversation — the exact Step 12
                              behavior, preserved for any caller that never
                              adopts sessions)

`session_id` is a conversation-scope key ONLY — not identity, not
authentication (see app/agent/memory.py's module docstring). Two different
explicit session IDs can never share memory; that guarantee lives entirely
in `InMemorySessionMemoryStore` (app/agent/memory.py), not duplicated here.

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

    session_id omitted  -> exactly the pre-correction Step 13 behavior:
                            the legacy `_default_memory` conversation is
                            used, and `episodic_memory`/`session_id` are
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
"""
from __future__ import annotations

from app.agent.episodic_memory import InMemoryEpisodicMemory
from app.agent.memory import ConversationMemory, InMemoryConversationMemory, InMemorySessionMemoryStore
from app.agent.memory_extraction import MemoryExtractor
from app.agent.memory_retriever import MemoryRetriever
from app.agent.memory_writer import MemoryWriter
from app.agent.orchestrator import AgentOrchestrator
from app.config import settings
from app.models.llm import LLMClient


class ChatService:
    def __init__(
        self,
        llm_client: LLMClient | None = None,
        memory_retriever: MemoryRetriever | None = None,
        memory_extractor: MemoryExtractor | None = None,
        memory_writer: MemoryWriter | None = None,
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
        # Preserves pre-Step-13 behavior for any caller that never passes a
        # session_id: one conversation, owned by this ChatService instance,
        # never touching the session store (so it can never collide with a
        # real session_id, however it's spelled).
        self._default_memory = InMemoryConversationMemory()
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

    def ask(self, user_message: str, session_id: str | None = None) -> str:
        if user_message is None or not user_message.strip():
            raise ValueError("User message cannot be empty.")

        cleaned_message = user_message.strip()
        memory = self._resolve_memory(session_id)
        if session_id is None:
            # Legacy path (Step 13): no real session identity to tag an
            # episode with, so episodic recording stays off entirely.
            orchestrator = AgentOrchestrator(llm_client=self.llm, memory=memory)
        else:
            orchestrator = AgentOrchestrator(
                llm_client=self.llm,
                memory=memory,
                episodic_memory=self.episodic_memory,
                memory_retriever=self.memory_retriever,
                memory_extractor=self.memory_extractor,
                memory_writer=self.memory_writer,
                session_id=session_id,
            )
        result = orchestrator.process(cleaned_message)
        return result.answer

    def _resolve_memory(self, session_id: str | None) -> ConversationMemory:
        if session_id is None:
            return self._default_memory
        return self.session_store.get_memory(session_id)
