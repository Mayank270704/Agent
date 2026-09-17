"""Semantic memory composition factory (Step 16I).

This module is the ONLY place that assembles the concrete semantic-memory
stack — the real (or, in tests, fake) `EmbeddingProvider`, the
`VectorIndex`, the `SemanticMemoryStore`, and the three collaborators that
depend on all three: `SemanticMemoryWriter`, `SemanticMemoryRetriever`,
and `LLMMemoryExtractor`. Nothing upstream of this module needs to change
for 16I to exist: every one of those five classes was already built
(16A–16H) against Protocols, injected rather than constructed by its
consumer. This module is simply the first caller that actually chooses
concrete implementations for the production application, in one place,
so that choice is never duplicated or allowed to drift.

    Settings (app/config.py)
           |
           v
    build_semantic_memory(...)   <-- this module
           |
           v
    SemanticMemoryBundle(retriever, extractor, writer)
           |
           v
    ChatService(memory_retriever=..., memory_extractor=..., memory_writer=...)

--------------------------------------------------------------------------
Why a separate module rather than doing this inline in app/main.py
--------------------------------------------------------------------------
Two independent reasons:

1. Testability. `app/main.py` builds its `chat_service` at IMPORT time
   (`chat_service = ChatService(...)` at module scope), and dozens of
   existing tests `import app.main`. If the real stack were assembled
   inline there, every one of those tests would either need the real
   embedding model available or would need `app.main` itself mocked
   around. Keeping assembly in its own function means tests can call
   `build_semantic_memory(...)` directly with a fake `provider_factory`
   and verify the wiring with no model, no torch, and no change to how
   `app/main.py` is tested.
2. Single point of truth for the sharing invariant (see below). If this
   logic were duplicated (e.g. once for a hypothetical CLI entry point and
   once for the FastAPI app), the two copies could drift apart and quietly
   end up building two separate stacks instead of one.

--------------------------------------------------------------------------
The sharing invariant — the one thing this module exists to guarantee
--------------------------------------------------------------------------
`SemanticMemoryWriter` and `SemanticMemoryRetriever` MUST be constructed
from the exact same `SemanticMemoryStore`, `VectorIndex`, and
`EmbeddingProvider` instances. This is not a style preference: if the
writer and retriever held two different index objects, the application
would write successfully (no error anywhere) and retrieve nothing (also
no error anywhere) — a silently broken feature that every existing
per-component test would still pass, because each component in isolation
would be behaving correctly. This module is the one place that instance
identity is decided, and its own test suite (`tests/test_semantic_memory_
wiring.py`) asserts identity directly (`bundle.writer.vector_index is
bundle.retriever.vector_index`, etc.) rather than only testing behavior
that happens to depend on it.

--------------------------------------------------------------------------
Default OFF, and what "disabled" means structurally
--------------------------------------------------------------------------
`build_semantic_memory(enabled=False, ...)` returns `None` and — this is
the important part — does so WITHOUT calling `provider_factory` at all.
Not "calls it and discards the result," not "constructs it lazily
elsewhere": the function returns before the concrete embedding provider
(the real one, by default `LocalEmbeddingProvider`) is ever referenced.
That is what makes it true that a disabled deployment imports no ML
library and downloads no model — the heavy import already lives inside
`LocalEmbeddingProvider.__init__` (16H), and this module simply never
calls it when disabled, so the two facts compose into "disabled costs
nothing" without this module needing any special-case import guard of its
own.

--------------------------------------------------------------------------
Dimension: never hard-coded
--------------------------------------------------------------------------
`InMemoryVectorIndex(dimension=provider.dimension)` — read FROM the
constructed provider, always. Hard-coding 384 (all-MiniLM-L6-v2's current
dimension) would silently break the moment `EMBEDDING_MODEL_NAME` names a
different model; reading it from the live instance is what lets this
module stay correct for any provider satisfying the Protocol.

--------------------------------------------------------------------------
Retention: the writer is the ONLY retention mechanism
--------------------------------------------------------------------------
`InMemorySemanticMemory()` is constructed with NO cap — deliberately,
always. `SemanticMemoryWriter(..., max_records_per_session=...)` is the
sole owner of eviction (16F). Capping the store as well would let two
independent eviction policies disagree about which record dies, and — far
worse — the store has no reference to the vector index, so a store-level
eviction cannot also remove the corresponding vector, stranding it (the
retriever would then skip it as a stale hit — silently degrading recall,
never crashing, and easy to miss). One owner, one mechanism.

--------------------------------------------------------------------------
What this module does NOT do
--------------------------------------------------------------------------
- No retries, no fallback provider, no partial degradation. A construction
  failure inside `provider_factory` (see `EmbeddingModelLoadError`, 16H)
  propagates unchanged — this module adds no try/except of its own. The
  design decision is that a startup that cannot build a working real
  provider must fail loudly, not boot in some quieter, half-working state.
- No caching, no persistence, no singleton at module scope. Every call to
  `build_semantic_memory(...)` builds an independent stack; `app/main.py`
  calls it exactly once, at import, and keeps the one bundle it gets back.
- No decision about WHETHER to enable semantic memory. `enabled` is an
  explicit, required argument — reading `Settings` and deciding is the
  caller's job (`app/main.py`), not this module's.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from app.agent.embeddings import EmbeddingProvider
from app.agent.local_embeddings import LocalEmbeddingProvider
from app.agent.memory_extraction import LLMMemoryExtractor, MemoryExtractor
from app.agent.memory_retriever import MemoryRetriever, SemanticMemoryRetriever
from app.agent.memory_writer import MemoryWriter, SemanticMemoryWriter
from app.agent.semantic_memory import InMemorySemanticMemory
from app.agent.vector_index import InMemoryVectorIndex
from app.models.llm import LLMClient

ProviderFactory = Callable[..., EmbeddingProvider]
"""A callable built with `model_name` and `device` keyword arguments,
returning an `EmbeddingProvider`. Defaults to `LocalEmbeddingProvider`
itself (whose constructor accepts exactly those two, among others), which
is why the default needs no wrapping lambda. Tests substitute a small
lambda that ignores both arguments and returns a
`DeterministicEmbeddingProvider` instead, so the wiring logic itself is
exercised with no real model."""


@dataclass(frozen=True)
class SemanticMemoryBundle:
    """The three collaborators ChatService's existing dependency-injection
    points already accept — nothing more.

    Deliberately just these three fields, matching exactly the three
    keyword arguments `ChatService.__init__` has accepted since 16E-D
    (`memory_retriever`, `memory_extractor`, `memory_writer`). The
    underlying store, index, and provider are NOT exposed here: they are
    reachable through the writer/retriever's own public attributes
    (`.semantic_memory`, `.vector_index`, `.embedding_provider`) for
    anything that genuinely needs them — most concretely, this module's
    own tests, which assert those three are identical objects across
    `writer` and `retriever`. Adding separate `store`/`index`/`provider`
    fields here would create a second way to reach the same objects, and
    therefore a second place a future edit could update inconsistently.

    Frozen, matching every other record type in this codebase.
    """

    retriever: MemoryRetriever
    extractor: MemoryExtractor
    writer: MemoryWriter


def build_semantic_memory(
    *,
    enabled: bool,
    model_name: str,
    device: str,
    max_records_per_session: int,
    llm_client: LLMClient,
    provider_factory: ProviderFactory = LocalEmbeddingProvider,
) -> SemanticMemoryBundle | None:
    """Assemble the semantic-memory stack, or decline to.

    Every argument is REQUIRED and keyword-only — there is deliberately no
    default that could make this function "usually build something."
    `app/main.py` is expected to pass every value explicitly, sourced from
    `Settings`, so the composition root's behavior is always a direct,
    readable function of configuration rather than of this function's own
    opinions.

    Returns `None` when `enabled` is `False` — never an "empty" or
    "no-op" bundle. `ChatService`'s memory collaborators are `... | None`
    (16E-C/16E-D) precisely so that "not configured" is representable
    without inventing a null-object stack that would still have to satisfy
    every Protocol.

    `provider_factory` exists ONLY for tests. Production code should never
    pass it — the default (`LocalEmbeddingProvider`) is what makes this
    the function that activates REAL semantic embeddings, which is the
    entire point of Step 16I. It is called as
    `provider_factory(model_name=model_name, device=device)`.

    Raises whatever the underlying constructors raise, unmodified:
    `EmbeddingModelLoadError` (or a test double's own exception) if the
    provider cannot be built, `ValueError` if `SemanticMemoryWriter`
    rejects `max_records_per_session`, `ValueError` if `llm_client` is
    `None` (from `LLMMemoryExtractor`). This function adds no try/except
    of its own — see the module docstring's "What this module does NOT
    do" section for why that is deliberate.
    """
    if not isinstance(enabled, bool):
        raise ValueError("enabled must be a bool.")

    if not enabled:
        # Returns before `provider_factory` is even referenced, let alone
        # called — see the module docstring's "Default OFF" section for
        # why that ordering is the whole point.
        return None

    provider = provider_factory(model_name=model_name, device=device)

    # Dimension is read FROM the constructed provider — never hard-coded
    # (see the module docstring).
    index = InMemoryVectorIndex(dimension=provider.dimension)

    # Deliberately uncapped — SemanticMemoryWriter is the sole retention
    # owner (see the module docstring).
    store = InMemorySemanticMemory()

    # The writer and retriever share `store`, `provider`, and `index` by
    # construction here — this is the ONE place that decides it, and it is
    # not repeated or re-derived anywhere else in this function.
    writer = SemanticMemoryWriter(
        store,
        provider,
        index,
        max_records_per_session=max_records_per_session,
    )
    retriever = SemanticMemoryRetriever(store, provider, index)
    extractor = LLMMemoryExtractor(llm_client)

    return SemanticMemoryBundle(retriever=retriever, extractor=extractor, writer=writer)
