"""Persistent (disk-backed) implementations of existing agent-memory
Protocols — Step 36 (conversation/session), extended by Step 37 (episodic)
and Step 38 (semantic memory + vectors).

Every class here implements a Protocol already defined in app/agent/
(`ConversationMemory`, `SessionMemoryStore`, `EpisodicMemory`,
`SemanticMemoryStore`, `VectorIndex`) with no change to those modules and
no new Protocol introduced. This mirrors the composition-root pattern
already used for semantic memory (see app/semantic_memory_wiring.py): a
concrete collaborator is swapped in at app/main.py, and nothing above the
Protocol boundary (ChatService, AgentOrchestrator, AgentLoop,
LLMDecisionMaker, ToolExecutionGate) changes at all.

All five persistent stores share ONE SQLite database file (selected by
MEMORY_BACKEND=sqlite, app/config.py) when enabled; each manages its own
connection to that file.
"""
from app.persistence.sqlite_episodic_memory import SQLiteEpisodicMemory
from app.persistence.sqlite_memory import SQLiteConversationMemory, SQLiteSessionMemoryStore
from app.persistence.sqlite_semantic_memory import SQLiteSemanticMemory, SQLiteVectorIndex

__all__ = [
    "SQLiteConversationMemory",
    "SQLiteSessionMemoryStore",
    "SQLiteEpisodicMemory",
    "SQLiteSemanticMemory",
    "SQLiteVectorIndex",
]
