"""Cross-request conversation memory — explicitly distinct from AgentState.

    AgentState (app/agent/state.py):
        WORKING execution state for ONE agent execution/request.
        Lifetime: one request.
        Contains: the current user_input, execution history (tool_calls,
        observations, errors), plan, final_answer, status.

    ConversationMemory (this module):
        PERSISTENT conversation history ACROSS requests.
        Lifetime: many requests — as long as the owning object (today: one
        ChatService instance) stays alive.
        Contains ONLY externally meaningful conversation turns: user
        messages and final assistant answers. Never contains tool calls,
        observations, planner internals, raw model decision JSON, or
        errors — that information already has a home (AgentState /
        AgentResult) and does not belong in conversation history.

This module is intentionally the smallest possible foundation: one Protocol
plus one in-process implementation, with a simple message-COUNT limit (no
token counting, no summarization, no importance scoring, no retrieval). No
database, no files, no vector store, no embeddings — all explicitly future
milestones. Because every caller depends only on the `ConversationMemory`
Protocol, the in-process implementation can later be swapped for a
Redis-/Postgres-backed one without touching the rest of the agent.

Future layers (not implemented here):
    ConversationMemory -> Episodic Memory -> Semantic Memory -> Embeddings
        -> Vector Store -> RAG

Concurrency note: `InMemoryConversationMemory` is NOT thread-safe and is not
made so here — that is a deliberate non-goal for this step (see the Step 12
report). There is no module-level mutable state anywhere in this file: every
`InMemoryConversationMemory()` instance owns its own list, and it must be
explicitly constructed and injected by a caller (see
app/agent/orchestrator.py's `memory` parameter) — never a hidden singleton.

--------------------------------------------------------------------------
Step 13 — SessionMemoryStore: session_id -> ConversationMemory
--------------------------------------------------------------------------

A single `ConversationMemory` is one conversation. Real usage needs many
independent conversations at once — one per SESSION. `SessionMemoryStore`
is the (equally small) mapping from a `session_id` string to its own
`ConversationMemory`, so that Session A's history and Session B's history
can never mix.

A session_id is NOT identity or authentication — it is only a conversation
scope, supplied by the caller (see app/main.py's `ChatRequest.session_id`).
Nothing here checks who the caller is; that is an explicit non-goal (see
the Step 13 report).

`InMemorySessionMemoryStore` is process-local, non-persistent, and
disappears on restart — like `InMemoryConversationMemory`, it is not
thread-safe and not suitable for a multi-process deployment. It is not yet
a production distributed session store; a later milestone could replace it
with a Redis- or database-backed store that satisfies the same
`SessionMemoryStore` Protocol without changing anything above it (see
app/services/chat.py, the only current caller).

There is no module-level store instance anywhere in this file — a
`SessionMemoryStore` must be explicitly constructed and owned by a caller
(today: one `ChatService` instance), exactly like `ConversationMemory`
itself.

--------------------------------------------------------------------------
Milestone 23 — bounded session count + atomic get-or-create
--------------------------------------------------------------------------
Each session's OWN message history was already bounded
(`max_messages_per_session`), but the NUMBER of distinct session_ids the
store retains was not — a long-running process accumulating one entry
per session_id ever seen is unbounded memory growth. `max_sessions`
bounds that too, via LRU: `get_memory` marks a session as most-recently-
used on every access (new OR existing), and once the session COUNT
exceeds `max_sessions`, the least-recently-used session is evicted —
never an arbitrary or random choice, and never a session that was just
touched. This preserves the existing rule that an EXPLICIT session
persists across requests: it is only ever evicted once it has gone
LEAST-recently-used among every other session concurrently held, which
requires `max_sessions` distinct sessions to be more active more
recently — normal usage of a bounded number of concurrent conversations
never evicts anything.

`get_memory`'s check-then-create was also not atomic under concurrent
first requests for the same brand-new session_id (a real, if narrow,
race under FastAPI's threadpool-executed sync handlers). A single
`threading.Lock` around the check-then-create-and-touch section closes
it; the lock is held only for in-memory dict/OrderedDict bookkeeping —
never across an LLM call, a tool call, or any other user code, which all
happen well outside this class entirely (see app/services/chat.py:
`get_memory` returns before any of that begins).
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Protocol, runtime_checkable


@runtime_checkable
class ConversationMemory(Protocol):
    """Cross-request conversation history: user/assistant turns only.

    Deliberately a Protocol, matching this codebase's other collaborator
    contracts (Tool, DecisionMaker, PlanGenerator) — an implementation just
    needs these four methods, no inheritance required.
    """

    def add_user_message(self, message: str) -> None:
        ...

    def add_assistant_message(self, message: str) -> None:
        ...

    def get_messages(self) -> list[dict[str, str]]:
        ...

    def clear(self) -> None:
        ...


class InMemoryConversationMemory:
    """The only implementation for now: a plain in-process list.

    - Non-persistent: history is lost when the process restarts.
    - Roles are controlled entirely by which method is called
      (`add_user_message` / `add_assistant_message`) — there is no way for
      a caller to inject an arbitrary role string.
    - `get_messages()` returns a fresh copy (a new list of new dicts) on
      every call, so a caller can never mutate memory state through the
      returned value, and messages are stored as new dicts (never the
      caller's own object) so the reverse is also true.
    - Eviction (`max_messages`): once adding a message would exceed the
      limit, the OLDEST messages are dropped first — a simple FIFO,
      message-COUNT limit. Newest messages are always preserved. This is
      intentionally simple: no token counting, no summarization.
    """

    def __init__(self, max_messages: int = 20):
        if max_messages < 1:
            raise ValueError("max_messages must be >= 1.")
        self._max_messages = max_messages
        self._messages: list[dict[str, str]] = []

    def add_user_message(self, message: str) -> None:
        self._append("user", message)

    def add_assistant_message(self, message: str) -> None:
        self._append("assistant", message)

    def get_messages(self) -> list[dict[str, str]]:
        return [dict(message) for message in self._messages]

    def clear(self) -> None:
        self._messages.clear()

    def _append(self, role: str, message: str) -> None:
        if not isinstance(message, str) or not message.strip():
            raise ValueError(f"{role} message cannot be empty.")
        self._messages.append({"role": role, "content": message})
        overflow = len(self._messages) - self._max_messages
        if overflow > 0:
            del self._messages[:overflow]


@runtime_checkable
class SessionMemoryStore(Protocol):
    """Maps a session_id to its own, isolated ConversationMemory."""

    def get_memory(self, session_id: str) -> ConversationMemory:
        ...

    def clear_session(self, session_id: str) -> None:
        ...


class InMemorySessionMemoryStore:
    """The only implementation for now: a plain in-process dict of
    session_id -> InMemoryConversationMemory.

    - `get_memory(session_id)` is get-or-create: the first call for a given
      (normalized) session_id creates a fresh InMemoryConversationMemory;
      every later call with the SAME session_id returns that exact SAME
      instance. A different session_id always gets a different instance.
    - session_id validation/normalization happens HERE, and only here (the
      one consistent location — see the Step 13 report): it must be a
      non-empty string once leading/trailing whitespace is stripped, and
      the stripped form is what's used as the dict key, so `"A"` and
      `" A "` refer to the same session. `ChatService` and the FastAPI
      layer do not duplicate this check; they simply let it propagate as a
      ValueError, exactly like every other invalid-input case in this
      codebase.
    - `clear_session(session_id)` removes that session's entry entirely
      (a no-op if it was never created). The NEXT `get_memory()` call for
      that same session_id then creates a genuinely fresh, empty
      ConversationMemory — a caller still holding a reference to the OLD
      memory object keeps seeing its old history, but it is no longer
      reachable through the store.
    - `max_sessions` bounds the NUMBER of distinct sessions retained,
      via LRU eviction — see the module docstring's Milestone 23 section.
    """

    def __init__(self, max_messages_per_session: int = 20, max_sessions: int = 1000):
        if max_messages_per_session < 1:
            raise ValueError("max_messages_per_session must be >= 1.")
        if max_sessions < 1:
            raise ValueError("max_sessions must be >= 1.")
        self._max_messages_per_session = max_messages_per_session
        self._max_sessions = max_sessions
        # OrderedDict, not dict: `move_to_end` gives an O(1) way to track
        # recency for LRU eviction. Insertion order alone (a plain dict)
        # cannot express "this EXISTING session was just used again".
        self._sessions: OrderedDict[str, ConversationMemory] = OrderedDict()
        self._lock = threading.Lock()

    def get_memory(self, session_id: str) -> ConversationMemory:
        normalized = self._normalize(session_id)  # no shared state touched; safe outside the lock
        with self._lock:
            existing = self._sessions.get(normalized)
            if existing is not None:
                self._sessions.move_to_end(normalized)  # mark most-recently-used
                return existing

            memory = InMemoryConversationMemory(max_messages=self._max_messages_per_session)
            self._sessions[normalized] = memory
            if len(self._sessions) > self._max_sessions:
                # popitem(last=False) removes the OLDEST (least-recently-
                # used) entry — never the one just inserted, since a
                # brand-new session is always the most-recently-used one.
                self._sessions.popitem(last=False)
            return memory

    def clear_session(self, session_id: str) -> None:
        normalized = self._normalize(session_id)
        with self._lock:
            self._sessions.pop(normalized, None)

    def _normalize(self, session_id: str) -> str:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string.")
        return session_id.strip()
