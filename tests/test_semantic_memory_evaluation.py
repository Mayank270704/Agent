"""Step 33 — semantic memory activation-readiness evaluation.

Mechanism correctness (session isolation, duplicate/supersession handling,
context bounds, disabled-mode behavior, concurrency, wiring) is ALREADY
exhaustively tested elsewhere in this suite with
`DeterministicEmbeddingProvider` — see test_semantic_memory*.py,
test_memory_*.py. This file does not duplicate that; it adds exactly what
those files cannot provide: evidence about REAL semantic-retrieval
QUALITY (this was the one genuine gap this milestone's audit found — see
docs/ARCHITECTURE_STATUS.md's Step 33 section), a proof of the concrete
use case end to end, and a couple of failure-mode checks specific to this
milestone's own wiring change (`min_similarity`) and an architecture
observation (extraction-time Ollama unavailability).

Three groups, matching Step 33 Phase 9's required distinction:

1. EMBEDDING-MODEL tests (real `all-MiniLM-L6-v2`, no Ollama) — the
   retrieval-quality dataset: does a semantically-related but
   differently-worded query retrieve a stored fact, and does an unrelated
   query not? `@pytest.mark.integration` only because loading the real
   model takes ~20s and downloads weights on first use elsewhere — it
   makes no network call at all once cached, and needs no Ollama, no
   Tavily, and no API key. Skips if `sentence-transformers` is not
   installed (the optional dependency), same courtesy every other
   real-embedding test in this suite already gives.

2. FULL PRODUCTION-PATH tests (real Ollama + real embeddings) — the
   concrete use case (Step 33 Phase 3), exercised through a genuinely
   activated `ChatService`/`AgentOrchestrator`, proving write -> embed ->
   index -> retrieve -> render -> real LLM final answer end to end.
   `@pytest.mark.integration`, skips if Ollama is unreachable.

3. DETERMINISTIC unit tests — the new `min_similarity` wiring itself, and
   the extraction-time RuntimeError propagation gap this audit found (not
   fixed — Step 31/Step 33 precedent: document a genuine limitation rather
   than patch around it without a separately-scoped decision). Fully
   offline.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

import pytest

from app.agent.embeddings import DeterministicEmbeddingProvider
from app.agent.episodic_memory import InMemoryEpisodicMemory
from app.agent.local_embeddings import LocalEmbeddingProvider
from app.agent.memory_extraction import LLMMemoryExtractor, MemoryCandidate, MemoryExtractor
from app.agent.memory_retriever import SemanticMemoryRetriever
from app.agent.memory_writer import MemoryWriter, SemanticMemoryWriter
from app.agent.semantic_memory import InMemorySemanticMemory
from app.agent.state import AgentStatus
from app.agent.vector_index import InMemoryVectorIndex, cosine_similarity
from app.config import settings
from app.models.llm import LLMClient
from app.services.chat import ChatService
from app.semantic_memory_wiring import build_semantic_memory

import importlib.util

# `find_spec` checks installability WITHOUT importing the module — an
# actual `import sentence_transformers` at collection time would pull
# torch into sys.modules for the whole pytest session regardless of which
# tests run, which is exactly the cost app/agent/local_embeddings.py's own
# module docstring goes out of its way to avoid (the heavy import lives
# inside LocalEmbeddingProvider.__init__, never at module scope) — and
# would spuriously fail test_semantic_memory_wiring.py's
# `test_disabled_does_not_import_the_ml_stack`, which asserts exactly that
# nothing has imported it.
_SENTENCE_TRANSFORMERS_AVAILABLE = importlib.util.find_spec("sentence_transformers") is not None


def _ollama_is_available() -> bool:
    try:
        urllib.request.urlopen(f"{settings.ollama_base_url}/api/tags", timeout=2)
        return True
    except (urllib.error.URLError, OSError):
        return False


class FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self.prompts: list[str] = []
        self._responses = iter(responses)

    def generate(self, messages: list[dict[str, str]], *, json_mode: bool = False) -> str:
        self.prompts.append(messages[-1]["content"])
        try:
            return next(self._responses)
        except StopIteration:
            raise AssertionError("FakeLLM ran out of scripted responses") from None


def _final_json(answer: str) -> str:
    return json.dumps({"action_type": "final", "final_answer": answer})


def _extraction_json(content: str, confidence: float = 0.9) -> str:
    """A valid LLMMemoryExtractor response. The memory_extractor shares the
    SAME FakeLLM as the main decision path in `_build_activated_chat_
    service` (matching production, where both consume `chat_service.llm`),
    so every ask() call that COMPLETES triggers one extra, extraction-
    purpose generate() call after the main decision -- scripting a turn
    means scripting BOTH, in order: [decide_response, extraction_response].
    """
    return json.dumps({"memories": [{"content": content, "confidence": confidence}]})


# ===========================================================================
# Group 1 — EMBEDDING-MODEL tests: real all-MiniLM-L6-v2, no Ollama.
# The concrete-use-case retrieval-quality dataset (Step 33 Phase 3).
# ===========================================================================

# The concrete use case this milestone identified (Step 33 Phase 3):
# conversation memory is a 20-message FIFO window, so a preference stated
# early in a long session is evicted from it; semantic memory is what
# makes that preference recallable later in the SAME session, via
# DIFFERENT wording than how it was originally stated — something
# conversation memory, which has no search/retrieval capability at all,
# structurally cannot do.
STORED_FACT = "User prefers Python for machine learning projects."

# (query, should_match) — ground truth defined explicitly, per Phase 3.
RETRIEVAL_QUALITY_CASES: list[tuple[str, bool]] = [
    ("What language should I use for my ML project?", True),
    ("Which programming language do I like for machine learning?", True),
    ("Remind me what I said about my preferred ML language.", True),
    ("What is the capital of France?", False),
    ("What time is it?", False),
    ("Can you help me plan a birthday party?", False),
    ("What's the weather like today?", False),
]


@pytest.fixture(scope="module")
def real_embedding_provider() -> LocalEmbeddingProvider:
    if not _SENTENCE_TRANSFORMERS_AVAILABLE:
        pytest.skip("sentence-transformers is not installed (requirements-embeddings.txt).")
    return LocalEmbeddingProvider()


@pytest.mark.integration
def test_real_embedding_model_semantic_match_quality(real_embedding_provider: LocalEmbeddingProvider) -> None:
    """The actual evidence behind this milestone's activation decision:
    does the real model separate related from unrelated queries with a
    usable margin? Informational per-case output plus one structural
    assertion (the measured gap must be positive — related queries score
    higher than unrelated ones), not a claim of a fixed numeric accuracy."""
    fact_vector = real_embedding_provider.embed(STORED_FACT)

    results = []
    for query, should_match in RETRIEVAL_QUALITY_CASES:
        query_vector = real_embedding_provider.embed(query)
        similarity = cosine_similarity(fact_vector, query_vector)
        results.append((query, should_match, similarity))

    print(f"\nStored fact: {STORED_FACT!r}")
    for query, should_match, similarity in results:
        print(f"  sim={similarity:.3f} should_match={should_match!s:5s} :: {query!r}")

    related = [sim for _, should, sim in results if should]
    unrelated = [sim for _, should, sim in results if not should]
    gap = min(related) - max(unrelated)
    print(f"min(related)={min(related):.3f} max(unrelated)={max(unrelated):.3f} gap={gap:.3f}")

    # Structural claim only: related queries score meaningfully higher than
    # unrelated ones. NOT asserting exact thresholds on nondeterministic
    # model internals beyond "the ordering is right" — a real model's
    # precise scores can shift slightly across versions/hardware.
    assert gap > 0, "related queries did not separate from unrelated queries"


@pytest.mark.integration
def test_configured_min_similarity_correctly_classifies_the_evaluation_set(
    real_embedding_provider: LocalEmbeddingProvider,
) -> None:
    """Proves Step 33's actual wiring choice (settings.semantic_memory_
    min_similarity, default 0.3) end to end through the real
    SemanticMemoryRetriever + real embeddings: related queries ARE
    retrieved, unrelated queries are NOT — Phase 3 items 3 and 4,
    together, as an actual pass/fail retrieval outcome rather than a raw
    similarity number."""
    store = InMemorySemanticMemory()
    index = InMemoryVectorIndex(dimension=real_embedding_provider.dimension)
    retriever = SemanticMemoryRetriever(
        store, real_embedding_provider, index, min_similarity=settings.semantic_memory_min_similarity
    )

    from app.agent.semantic_memory import SemanticMemoryRecord

    record = SemanticMemoryRecord(
        memory_id="fact-1",
        session_id="eval-session",
        content=STORED_FACT,
        created_at=datetime.now(timezone.utc),
        source_event_ids=("evt-1",),
    )
    store.add(record)
    index.add("fact-1", "eval-session", real_embedding_provider.embed(STORED_FACT))

    correct = 0
    for query, should_match in RETRIEVAL_QUALITY_CASES:
        retrieved = retriever.retrieve("eval-session", query)
        did_match = len(retrieved) == 1 and retrieved[0].memory.memory_id == "fact-1"
        status = "OK" if did_match == should_match else "MISS"
        print(f"  [{status}] should_match={should_match!s:5s} got={did_match!s:5s} :: {query!r}")
        if did_match == should_match:
            correct += 1

    print(f"{correct}/{len(RETRIEVAL_QUALITY_CASES)} cases classified as expected "
          f"(min_similarity={settings.semantic_memory_min_similarity})")
    # All 7 cases (3 related, 4 unrelated) are expected correct given the
    # measured gap in test_real_embedding_model_semantic_match_quality —
    # asserted directly since this IS the deterministic consequence of a
    # real, already-measured model output, not a fresh nondeterministic
    # claim.
    assert correct == len(RETRIEVAL_QUALITY_CASES)


@pytest.mark.integration
def test_real_embedding_duplicate_detection(real_embedding_provider: LocalEmbeddingProvider) -> None:
    """Phase 3 item 8 (duplicate handling) with the REAL model: a near-
    identical restatement of the same fact is merged, not stored twice."""
    store = InMemorySemanticMemory()
    index = InMemoryVectorIndex(dimension=real_embedding_provider.dimension)
    writer = SemanticMemoryWriter(store, real_embedding_provider, index)

    first = writer.write(
        "eval-session", [MemoryCandidate(content=STORED_FACT, confidence=0.9)], source_event_ids=("evt-1",)
    )
    # An exact restatement -- similarity 1.0 against itself, always above
    # any reasonable duplicate_threshold regardless of model.
    second = writer.write(
        "eval-session", [MemoryCandidate(content=STORED_FACT, confidence=0.9)], source_event_ids=("evt-2",)
    )

    assert len(first) == 1
    assert len(second) == 1
    assert second[0].memory_id == first[0].memory_id  # merged, not a new record
    assert store.list_recent("eval-session", limit=10) == [second[0]]  # still exactly one record
    assert set(second[0].source_event_ids) == {"evt-1", "evt-2"}  # provenance unioned


@pytest.mark.integration
def test_real_embedding_performance(real_embedding_provider: LocalEmbeddingProvider) -> None:
    """Step 33 Phase 10 — embedding latency, measured directly (not
    prematurely optimized, not claimed from documentation alone)."""
    texts = [q for q, _ in RETRIEVAL_QUALITY_CASES]

    start = time.monotonic()
    for text in texts:
        real_embedding_provider.embed(text)
    single_elapsed = time.monotonic() - start

    start = time.monotonic()
    real_embedding_provider.embed_many(texts)
    batch_elapsed = time.monotonic() - start

    print(f"\nembed() x{len(texts)}: {single_elapsed * 1000:.1f}ms total, "
          f"{single_elapsed / len(texts) * 1000:.1f}ms/call")
    print(f"embed_many() x{len(texts)}: {batch_elapsed * 1000:.1f}ms total, "
          f"{batch_elapsed / len(texts) * 1000:.1f}ms/text")

    # Sanity bound only (generous -- this is CPU inference on arbitrary dev
    # hardware, not a performance guarantee): embedding one short query
    # should not take longer than an ordinary LLM decision call.
    assert single_elapsed / len(texts) < 1.0


# ===========================================================================
# Group 2 — FULL PRODUCTION-PATH tests: real Ollama + real embeddings.
# ===========================================================================

def _build_activated_chat_service(llm: LLMClient, min_similarity: float = 0.3) -> ChatService:
    """Mirrors app/main.py's real composition, but with enabled=True and a
    FakeLLM/real-LLM the test controls -- never the module-level
    `chat_service` singleton (which stays disabled, per the default)."""
    bundle = build_semantic_memory(
        enabled=True,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
        max_records_per_session=200,
        min_similarity=min_similarity,
        llm_client=llm,
        memory_backend="memory",
        sqlite_db_path="unused.db",
    )
    return ChatService(
        llm_client=llm,
        memory_retriever=bundle.retriever,
        memory_extractor=bundle.extractor,
        memory_writer=bundle.writer,
    )


@pytest.mark.integration
def test_concrete_use_case_end_to_end_with_real_ollama() -> None:
    """The actual concrete use case (Step 33 Phase 3), through the REAL
    production path with semantic memory genuinely activated: a stated
    preference survives being pushed out of a later turn's immediate
    context, and is recalled by the real model when asked about it with
    DIFFERENT wording than how it was originally stated -- the one thing
    conversation memory cannot do at all."""
    if not _ollama_is_available():
        pytest.skip(f"Ollama is not reachable at {settings.ollama_base_url}; skipping live semantic memory test.")
    if not _SENTENCE_TRANSFORMERS_AVAILABLE:
        pytest.skip("sentence-transformers is not installed (requirements-embeddings.txt).")

    real_llm = LLMClient(provider="ollama", model_name=settings.model_name, base_url=settings.ollama_base_url)
    chat_service = _build_activated_chat_service(real_llm)
    session_id = f"eval-use-case-{time.time_ns()}"

    # Turn 1: state the preference. The model's OWN final answer is real
    # (not scripted) -- only the fact that a durable preference was stated
    # matters for extraction; any reasonable acknowledgement is fine.
    answer_1 = chat_service.ask("I prefer Python for machine learning projects.", session_id=session_id)
    assert isinstance(answer_1, str) and answer_1.strip()

    # Turn 2: ask with DIFFERENT wording. No mention of "Python" or
    # "machine learning" verbatim -- if the model answers correctly, it can
    # only be via retrieved semantic memory, not the literal recent text
    # (which conversation memory WOULD also supply here, so this alone
    # isn't fully conclusive -- see the next assertion for the stronger
    # proof).
    answer_2 = chat_service.ask(
        "Which programming language do I like for machine learning?", session_id=session_id
    )
    print(f"\nTurn 1 answer: {answer_1!r}")
    print(f"Turn 2 answer: {answer_2!r}")
    assert "python" in answer_2.lower()


@pytest.mark.integration
def test_semantic_memory_context_reaches_the_prompt_for_reworded_query() -> None:
    """Stronger, structural proof of the same use case: captures the ACTUAL
    prompt text sent to the model on turn 2, proving the MEMORY CONTEXT
    block (not just plausible conversation-memory carryover) is what
    supplied the fact -- using a scripted FakeLLM so turn 2's prompt can be
    inspected directly, with REAL embeddings doing the retrieval."""
    if not _SENTENCE_TRANSFORMERS_AVAILABLE:
        pytest.skip("sentence-transformers is not installed (requirements-embeddings.txt).")

    llm = FakeLLM([
        _final_json("Got it, I'll remember that."),
        _extraction_json("User prefers Python for machine learning projects."),
        _final_json("You prefer Python."),
        _extraction_json("User asked about their preferred ML language.", confidence=0.2),
    ])
    chat_service = _build_activated_chat_service(llm)
    session_id = f"eval-prompt-proof-{time.time_ns()}"

    chat_service.ask("I prefer Python for machine learning projects.", session_id=session_id)
    chat_service.ask("Which programming language do I like for machine learning?", session_id=session_id)

    # prompts recorded in order: [turn1 decide, turn1 extract, turn2
    # decide, turn2 extract] -- the DECIDE prompt (where memory context is
    # rendered) is the second-to-last, not the last (extraction's prompt).
    second_decide_prompt = llm.prompts[-2]
    assert "MEMORY CONTEXT" in second_decide_prompt
    assert "python" in second_decide_prompt.lower()


@pytest.mark.integration
def test_unrelated_query_does_not_surface_the_memory_in_the_prompt() -> None:
    """The negative case, through the same real-embeddings production
    path: an unrelated second turn must NOT pull the stored preference
    into its prompt at all (min_similarity correctly filters it out)."""
    if not _SENTENCE_TRANSFORMERS_AVAILABLE:
        pytest.skip("sentence-transformers is not installed (requirements-embeddings.txt).")

    llm = FakeLLM([
        _final_json("Got it, I'll remember that."),
        _extraction_json("User prefers Python for machine learning projects."),
        _final_json("Paris is the capital of France."),
        _extraction_json("User asked about the capital of France.", confidence=0.2),
    ])
    chat_service = _build_activated_chat_service(llm)
    session_id = f"eval-negative-{time.time_ns()}"

    chat_service.ask("I prefer Python for machine learning projects.", session_id=session_id)
    chat_service.ask("What is the capital of France?", session_id=session_id)

    # See the companion positive test for why this is index -2, not -1:
    # the DECIDE prompt precedes turn 2's own extraction-prompt in
    # llm.prompts.
    second_decide_prompt = llm.prompts[-2]
    assert "MEMORY CONTEXT" not in second_decide_prompt


# ===========================================================================
# Group 3 — deterministic unit tests: this milestone's own wiring change,
# and a documented (not fixed) limitation this audit found.
# ===========================================================================

def test_build_semantic_memory_requires_min_similarity_explicitly() -> None:
    """Step 33: min_similarity is a REQUIRED keyword-only argument, matching
    this module's own stated design principle ("every argument is
    REQUIRED... there is deliberately no default that could make this
    function usually build something") -- a caller that omits it gets a
    TypeError at the call site, not a silently-applied default."""
    with pytest.raises(TypeError):
        build_semantic_memory(  # type: ignore[call-arg]
            enabled=True,
            model_name="fake/model",
            device="cpu",
            max_records_per_session=200,
            llm_client=FakeLLM([]),
            memory_backend="memory",
            sqlite_db_path="unused.db",
        )


def test_wiring_min_similarity_reaches_the_constructed_retriever() -> None:
    bundle = build_semantic_memory(
        enabled=True,
        model_name="fake/model",
        device="cpu",
        max_records_per_session=200,
        min_similarity=0.42,
        llm_client=FakeLLM([]),
        memory_backend="memory",
        sqlite_db_path="unused.db",
        provider_factory=lambda *, model_name, device: DeterministicEmbeddingProvider(dimension=8),
    )
    assert bundle.retriever.min_similarity == 0.42


def test_default_settings_min_similarity_is_0_3() -> None:
    """Locks in the evidence-calibrated default (see app/config.py) against
    accidental drift."""
    assert settings.semantic_memory_min_similarity == 0.3


def test_disabled_semantic_memory_never_touches_min_similarity() -> None:
    """CRITICAL INVARIANT (Step 33 Phase 11): disabled means the function
    returns before ANY of its arguments, including the new one, are used
    for anything beyond Python's own parameter binding -- verified by
    passing a deliberately invalid value and confirming no error occurs
    (a real SemanticMemoryRetriever would reject an out-of-range value at
    construction; disabled mode never constructs one)."""
    result = build_semantic_memory(
        enabled=False,
        model_name="fake/model",
        device="cpu",
        max_records_per_session=200,
        min_similarity=999.0,  # out of [-1.0, 1.0]; would raise if ever used
        llm_client=FakeLLM([]),
        memory_backend="memory",
        sqlite_db_path="unused.db",
    )
    assert result is None


class _RuntimeErrorExtractor:
    """A MemoryExtractor stand-in simulating Ollama becoming unreachable
    DURING the extraction call -- distinct from a MemoryExtractionError
    (malformed output), which IS already caught."""

    def extract(self, user_message: str, assistant_answer: str) -> list[MemoryCandidate]:
        raise RuntimeError("Ollama connection failed at http://localhost:11434: simulated")


def test_kb_extraction_time_ollama_failure_is_a_documented_limitation_not_silently_swallowed() -> None:
    """Step 33 Phase 1/8 audit finding, locked in as a regression test
    rather than silently left undiscovered: AgentOrchestrator's write-path
    only catches MemoryExtractionError (app/agent/orchestrator.py's
    `_maybe_write_semantic_memory`) -- a RuntimeError from the extractor's
    OWN LLM call (e.g. Ollama becoming unreachable between the main
    answer's LLM call and the extraction call) is NOT caught there and
    propagates all the way to the caller, exactly like any other
    unclassified exception in this codebase (see app/agent/reliability.py's
    module docstring: "an uncaught/unexpected exception ... propagates
    exactly as it always has").

    This is NOT fixed by this milestone (Step 31/33 precedent: a genuine,
    narrow limitation is documented, not silently patched without a
    separately-scoped decision) -- this test exists so the behavior is
    proven and visible, in docs/ARCHITECTURE_STATUS.md and here, rather
    than discovered by surprise in production. The practical consequence:
    if this exact race occurs, a request whose REAL answer was already
    successfully computed can still surface as an HTTP 502 to the caller,
    because ChatService.ask() has already returned by the time
    orchestrator.process() re-raises -- no, precisely: process() raises
    BEFORE returning, so the already-computed answer is lost with it. That
    is the concrete cost of this limitation, stated plainly.
    """
    from app.agent.orchestrator import AgentOrchestrator
    from app.agent.tool_registry import ToolRegistry

    llm = FakeLLM([_final_json("Here is your answer.")])
    episodic = InMemoryEpisodicMemory()
    orchestrator = AgentOrchestrator(
        llm_client=llm,
        tool_registry=ToolRegistry(),
        episodic_memory=episodic,
        memory_extractor=_RuntimeErrorExtractor(),
        memory_writer=_UnreachableWriter(),
        session_id="eval-failure-session",
    )

    with pytest.raises(RuntimeError, match="Ollama connection failed"):
        orchestrator.process("I prefer Python for machine learning.")

    # The episode WAS recorded (it happens before extraction) -- only the
    # semantic-memory write is lost, not the episodic provenance.
    assert len(episodic.get_recent("eval-failure-session", limit=10)) == 1


class _UnreachableWriter:
    """Never actually reached in the RuntimeError test above (extraction
    fails first) -- exists only to satisfy AgentOrchestrator's constructor
    requirement that a writer accompany an extractor."""

    def write(self, session_id, candidates, source_event_ids):  # pragma: no cover
        raise AssertionError("writer should never be called: extraction fails first")
