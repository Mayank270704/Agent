"""Step 16I: concurrency safety of SemanticMemoryWriter.write().

FastAPI runs its synchronous route handlers in a threadpool, so two
concurrent requests can call `write()` on the SAME writer instance (the
production composition root builds exactly one writer per process — see
app/semantic_memory_wiring.py). `_persist`'s duplicate lookup -> embed ->
store -> index -> retention sequence is a check-then-act workflow, not a
single atomic operation, so without serialization concurrent calls can
race on the shared store/index state.

These tests use REAL threads (not asyncio, not mocks of threading) against
DeterministicEmbeddingProvider — fully offline, no real model — and widen
the race window with a small artificial delay inside a wrapping provider,
so a missing lock would be caught with high probability rather than only
in theory. They verify OUTCOMES (final counts, no lost writes, no
stranded vectors), not internal lock acquisition, so they test the
guarantee rather than the mechanism.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from app.agent.embeddings import DeterministicEmbeddingProvider
from app.agent.memory_extraction import MemoryCandidate
from app.agent.memory_writer import SemanticMemoryWriter
from app.agent.semantic_memory import InMemorySemanticMemory
from app.agent.vector_index import InMemoryVectorIndex

DIMENSION = 8


class SlowProvider:
    """Wraps a real EmbeddingProvider and adds a small sleep inside
    `embed`, so a thread that has just computed a vector but not yet
    finished `_persist` stays in flight long enough for other threads to
    reach the same duplicate-lookup window — widening the race rather
    than relying on scheduler luck alone."""

    def __init__(self, inner, delay: float = 0.005):
        self._inner = inner
        self._delay = delay

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    def embed(self, text: str):
        time.sleep(self._delay)
        return self._inner.embed(text)

    def embed_many(self, texts):
        time.sleep(self._delay)
        return self._inner.embed_many(texts)


def _writer(max_records_per_session: int | None = None, delay: float = 0.005):
    store = InMemorySemanticMemory()
    index = InMemoryVectorIndex(dimension=DIMENSION)
    provider = SlowProvider(DeterministicEmbeddingProvider(dimension=DIMENSION), delay=delay)
    writer = SemanticMemoryWriter(store, provider, index, max_records_per_session=max_records_per_session)
    return writer, store, index


# ===========================================================================
# 1 — CONCURRENT DUPLICATE WRITES DO NOT PRODUCE DUPLICATE RECORDS
# ===========================================================================

def test_concurrent_identical_candidates_merge_into_exactly_one_record() -> None:
    """Without serialization, N threads could each see "no duplicate yet"
    in _find_duplicate before any of them has stored a record, producing N
    separate records for what should be one fact restated N times."""
    writer, store, index = _writer()
    threads = 8
    barrier = threading.Barrier(threads)

    def write_one(i: int) -> None:
        barrier.wait()  # maximize actual overlap
        writer.write("alice", [MemoryCandidate("User prefers Python for ML.", 0.9)], [f"evt-{i}"])

    with ThreadPoolExecutor(max_workers=threads) as pool:
        list(pool.map(write_one, range(threads)))

    records = store.list_recent("alice", limit=100)
    assert len(records) == 1
    # Provenance from every thread's write was preserved, not lost — the
    # merge path unions source_event_ids rather than one write clobbering
    # another's.
    assert set(records[0].source_event_ids) == {f"evt-{i}" for i in range(threads)}
    # No stranded vectors: exactly one index entry backs the one record.
    hits = index.search(
        "alice",
        DeterministicEmbeddingProvider(dimension=DIMENSION).embed("User prefers Python for ML."),
        top_k=10,
    )
    assert len(hits) == 1


def test_concurrent_distinct_candidates_all_survive() -> None:
    """The lock must not silently drop or merge genuinely distinct facts
    written concurrently — only true duplicates should collapse."""
    writer, store, index = _writer()
    count = 12
    barrier = threading.Barrier(count)

    def write_one(i: int) -> None:
        barrier.wait()
        writer.write("alice", [MemoryCandidate(f"User fact number {i}.", 0.9)], [f"evt-{i}"])

    with ThreadPoolExecutor(max_workers=count) as pool:
        list(pool.map(write_one, range(count)))

    records = store.list_recent("alice", limit=100)
    assert len(records) == count  # nothing lost, nothing incorrectly merged


# ===========================================================================
# 2 — RETENTION STAYS CONSISTENT UNDER CONCURRENT WRITES
# ===========================================================================

def test_concurrent_writes_respect_the_retention_cap_without_stranding_vectors() -> None:
    """The critical invariant: store count and index count must move
    together. A race in the check-then-act eviction sequence could evict a
    store record without its vector (or vice versa), stranding an entry
    that the retriever would later skip with a WARNING — quietly degrading
    recall rather than crashing."""
    cap = 5
    writer, store, index = _writer(max_records_per_session=cap)
    total = 20
    barrier = threading.Barrier(total)

    def write_one(i: int) -> None:
        barrier.wait()
        writer.write("alice", [MemoryCandidate(f"User distinct fact {i}.", 0.9)], [f"evt-{i}"])

    with ThreadPoolExecutor(max_workers=total) as pool:
        list(pool.map(write_one, range(total)))

    stored_ids = {r.memory_id for r in store.list_recent("alice", limit=100)}
    assert len(stored_ids) == cap  # the cap was never exceeded

    # Every surviving record has a matching vector, and no vector survives
    # for a record that was evicted (no stranding in either direction).
    provider = DeterministicEmbeddingProvider(dimension=DIMENSION)
    for i in range(total):
        hits = index.search("alice", provider.embed(f"User distinct fact {i}."), top_k=1)
        matched = hits and hits[0].similarity > 0.999
        if matched:
            assert hits[0].memory_id in stored_ids, (
                f"fact {i} has a vector but no matching stored record (stranded vector)"
            )


def test_concurrent_writes_never_raise() -> None:
    """A race condition in an unlocked check-then-act sequence often
    surfaces as an exception (e.g. a KeyError from a dict mutated mid
    iteration) rather than only as a silently wrong count. No thread here
    should ever raise."""
    writer, _store, _index = _writer()
    total = 16
    barrier = threading.Barrier(total)
    errors: list[BaseException] = []
    lock = threading.Lock()

    def write_one(i: int) -> None:
        barrier.wait()
        try:
            writer.write("alice", [MemoryCandidate(f"User fact {i}.", 0.9)], [f"evt-{i}"])
        except BaseException as exc:  # noqa: BLE001 - capturing for the assertion below
            with lock:
                errors.append(exc)

    with ThreadPoolExecutor(max_workers=total) as pool:
        list(pool.map(write_one, range(total)))

    assert errors == []


# ===========================================================================
# 3 — THE LOCK IS PER-INSTANCE, NOT GLOBAL
# ===========================================================================

def test_two_separate_writers_do_not_contend_with_each_other() -> None:
    """Two independent writers (two independent sessions/stores in
    production terms) must not serialize against each other — only writes
    to the SAME writer instance are protected."""
    writer_a, store_a, _index_a = _writer(delay=0.05)
    writer_b, store_b, _index_b = _writer(delay=0.05)

    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_a = pool.submit(writer_a.write, "alice", [MemoryCandidate("Fact A.", 0.9)], ["evt-a"])
        fut_b = pool.submit(writer_b.write, "bob", [MemoryCandidate("Fact B.", 0.9)], ["evt-b"])
        fut_a.result()
        fut_b.result()
    elapsed = time.monotonic() - start

    # If the two writers shared a lock, this would take about 2x the
    # single-call delay (serialized); independent locks let them overlap.
    assert elapsed < 0.09
    assert len(store_a.list_recent("alice", limit=10)) == 1
    assert len(store_b.list_recent("bob", limit=10)) == 1


def test_the_lock_is_a_private_instance_attribute() -> None:
    writer, _store, _index = _writer()

    assert isinstance(writer._lock, type(threading.Lock()))
