"""Tests for the memory context contract (Step 16E-A).

Pure data + one adapter: no Ollama, no Tavily, no network, no embeddings,
no real retrieval. Most tests construct MemoryContextItem/MemoryContext
directly; the adapter tests build RetrievedMemory objects by hand rather
than running a retriever, keeping the conversion under test in isolation.
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import pytest

from app.agent.memory_context import (
    MemoryContext,
    MemoryContextItem,
    build_memory_context,
)
from app.agent.memory_retriever import RetrievedMemory
from app.agent.semantic_memory import SemanticMemoryRecord


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _item(memory_id: str = "m1", **overrides: object) -> MemoryContextItem:
    fields: dict[str, object] = {
        "memory_id": memory_id,
        "content": f"fact {memory_id}",
        "similarity": 0.9,
        "created_at": _now(),
        "confidence": 1.0,
    }
    fields.update(overrides)
    return MemoryContextItem(**fields)  # type: ignore[arg-type]


def _record(memory_id: str = "m1", session_id: str = "A", **overrides: object) -> SemanticMemoryRecord:
    fields: dict[str, object] = {
        "memory_id": memory_id,
        "session_id": session_id,
        "content": f"fact {memory_id}",
        "created_at": _now(),
        "source_event_ids": ("evt-1",),
    }
    fields.update(overrides)
    return SemanticMemoryRecord(**fields)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 1/2: creating items and contexts.
# ---------------------------------------------------------------------------

def test_create_one_context_item() -> None:
    created = _now()
    item = MemoryContextItem(
        memory_id="m1",
        content="User prefers Python for ML.",
        similarity=0.87,
        created_at=created,
        confidence=0.8,
    )

    assert item.memory_id == "m1"
    assert item.content == "User prefers Python for ML."
    assert item.similarity == pytest.approx(0.87)
    assert item.created_at == created
    assert item.confidence == pytest.approx(0.8)


def test_create_context_with_multiple_items() -> None:
    context = MemoryContext(session_id="A", items=(_item("m1"), _item("m2"), _item("m3")))

    assert len(context.items) == 3
    assert [i.memory_id for i in context.items] == ["m1", "m2", "m3"]


def test_context_records_its_session() -> None:
    context = MemoryContext(session_id="A", items=(_item(),))

    assert context.session_id == "A"


def test_context_session_id_is_normalized() -> None:
    context = MemoryContext(session_id="  A  ", items=())

    assert context.session_id == "A"


# ---------------------------------------------------------------------------
# 3/4: similarity and metadata preserved through the adapter.
# ---------------------------------------------------------------------------

def test_adapter_preserves_similarity_exactly() -> None:
    retrieved = [RetrievedMemory(memory=_record("m1"), similarity=0.4242)]

    context = build_memory_context("A", retrieved)

    assert context.items[0].similarity == pytest.approx(0.4242)


def test_adapter_preserves_relevant_memory_metadata() -> None:
    created = _now() - timedelta(days=30)
    record = _record("m1", content="User prefers Python.", created_at=created, confidence=0.75)

    context = build_memory_context("A", [RetrievedMemory(memory=record, similarity=0.5)])
    item = context.items[0]

    assert item.memory_id == "m1"
    assert item.content == "User prefers Python."
    assert item.created_at == created
    assert item.confidence == pytest.approx(0.75)


def test_adapter_produces_exactly_one_item_per_input() -> None:
    retrieved = [RetrievedMemory(memory=_record(f"m{i}"), similarity=0.5) for i in range(4)]

    context = build_memory_context("A", retrieved)

    assert len(context.items) == 4


def test_adapter_stamps_the_requested_session() -> None:
    context = build_memory_context("  A  ", [RetrievedMemory(memory=_record("m1", session_id="A"), similarity=0.5)])

    assert context.session_id == "A"


# ---------------------------------------------------------------------------
# 5: empty context is valid and ordinary.
# ---------------------------------------------------------------------------

def test_empty_context_is_valid() -> None:
    context = MemoryContext(session_id="A", items=())

    assert context.items == ()
    assert len(context.items) == 0


def test_items_defaults_to_empty() -> None:
    context = MemoryContext(session_id="A")

    assert context.items == ()


def test_adapter_with_no_retrieved_memories_returns_empty_context() -> None:
    context = build_memory_context("A", [])

    assert context.session_id == "A"
    assert context.items == ()


# ---------------------------------------------------------------------------
# 6/8: immutability and no mutable collection leakage.
# ---------------------------------------------------------------------------

def test_context_item_is_immutable() -> None:
    item = _item()

    with pytest.raises(FrozenInstanceError):
        item.content = "changed"  # type: ignore[misc]


def test_context_is_immutable() -> None:
    context = MemoryContext(session_id="A", items=(_item(),))

    with pytest.raises(FrozenInstanceError):
        context.session_id = "B"  # type: ignore[misc]


def test_items_are_stored_as_a_tuple_even_when_a_list_is_passed() -> None:
    context = MemoryContext(session_id="A", items=[_item("m1")])

    assert isinstance(context.items, tuple)


def test_mutating_the_callers_list_after_construction_does_not_affect_the_context() -> None:
    caller_list = [_item("m1")]
    context = MemoryContext(session_id="A", items=caller_list)

    caller_list.append(_item("m-sneaky"))
    caller_list[0] = _item("m-tampered")

    assert [i.memory_id for i in context.items] == ["m1"]


def test_items_tuple_has_no_mutating_methods() -> None:
    context = MemoryContext(session_id="A", items=(_item(),))

    assert not hasattr(context.items, "append")
    with pytest.raises(TypeError):
        context.items[0] = _item("m2")  # type: ignore[index]


def test_mutating_the_callers_list_after_adapter_call_does_not_affect_the_context() -> None:
    retrieved = [RetrievedMemory(memory=_record("m1"), similarity=0.5)]

    context = build_memory_context("A", retrieved)
    retrieved.append(RetrievedMemory(memory=_record("m2"), similarity=0.9))

    assert [i.memory_id for i in context.items] == ["m1"]


# ---------------------------------------------------------------------------
# 7: validation.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_value", ["", "   ", None, 123])
def test_item_rejects_invalid_memory_id(bad_value: object) -> None:
    with pytest.raises(ValueError):
        _item(memory_id=bad_value)


@pytest.mark.parametrize("bad_value", ["", "   ", None, 123])
def test_item_rejects_invalid_content(bad_value: object) -> None:
    with pytest.raises(ValueError):
        _item(content=bad_value)


@pytest.mark.parametrize("bad_value", [1.5, -1.5, "0.5", True, None])
def test_item_rejects_invalid_similarity(bad_value: object) -> None:
    with pytest.raises(ValueError):
        _item(similarity=bad_value)


@pytest.mark.parametrize("boundary", [-1.0, 0.0, 1.0])
def test_item_accepts_similarity_boundaries(boundary: float) -> None:
    assert _item(similarity=boundary).similarity == boundary


@pytest.mark.parametrize("bad_value", [-0.01, 1.01, "0.5", True, None])
def test_item_rejects_invalid_confidence(bad_value: object) -> None:
    with pytest.raises(ValueError):
        _item(confidence=bad_value)


def test_item_rejects_naive_datetime() -> None:
    with pytest.raises(ValueError):
        _item(created_at=datetime.now())


def test_item_rejects_non_datetime_created_at() -> None:
    with pytest.raises(ValueError):
        _item(created_at="2026-01-01T00:00:00Z")


@pytest.mark.parametrize("bad_value", ["", "   ", None, 123])
def test_context_rejects_invalid_session_id(bad_value: object) -> None:
    with pytest.raises(ValueError):
        MemoryContext(session_id=bad_value, items=())  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_items", ["not a sequence", 123, None])
def test_context_rejects_non_sequence_items(bad_items: object) -> None:
    with pytest.raises(ValueError):
        MemoryContext(session_id="A", items=bad_items)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_item", [None, "m1", {"memory_id": "m1"}, 42])
def test_context_rejects_items_that_are_not_context_items(bad_item: object) -> None:
    with pytest.raises(ValueError):
        MemoryContext(session_id="A", items=(bad_item,))  # type: ignore[arg-type]


def test_context_rejects_a_raw_semantic_record_as_an_item() -> None:
    """A storage type must not be smuggled in where a projected item
    belongs."""
    with pytest.raises(ValueError):
        MemoryContext(session_id="A", items=(_record("m1"),))  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_value", ["", "   ", None, 123])
def test_adapter_rejects_invalid_session_id(bad_value: object) -> None:
    with pytest.raises(ValueError):
        build_memory_context(bad_value, [])  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_retrieved", ["not a sequence", 123, None])
def test_adapter_rejects_non_sequence_retrieved(bad_retrieved: object) -> None:
    with pytest.raises(ValueError):
        build_memory_context("A", bad_retrieved)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_entry", [None, "m1", 42])
def test_adapter_rejects_entries_that_are_not_retrieved_memories(bad_entry: object) -> None:
    with pytest.raises(ValueError):
        build_memory_context("A", [bad_entry])  # type: ignore[list-item]


def test_adapter_rejects_a_raw_semantic_record_as_an_entry() -> None:
    with pytest.raises(ValueError):
        build_memory_context("A", [_record("m1")])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# 9: session safety -- the container's session is a VERIFIED claim.
# ---------------------------------------------------------------------------

def test_adapter_raises_when_a_record_belongs_to_a_different_session() -> None:
    retrieved = [RetrievedMemory(memory=_record("m-rogue", session_id="B"), similarity=0.9)]

    with pytest.raises(ValueError, match="session isolation violation"):
        build_memory_context("A", retrieved)


def test_adapter_raises_if_any_record_in_a_batch_is_from_another_session() -> None:
    retrieved = [
        RetrievedMemory(memory=_record("m1", session_id="A"), similarity=0.9),
        RetrievedMemory(memory=_record("m2", session_id="A"), similarity=0.8),
        RetrievedMemory(memory=_record("m-rogue", session_id="B"), similarity=0.7),
    ]

    with pytest.raises(ValueError, match="session isolation violation"):
        build_memory_context("A", retrieved)


def test_adapter_does_not_silently_drop_the_wrong_session_memory() -> None:
    """A wrong-session record must fail loudly rather than be filtered
    out, so the upstream integrity bug cannot hide."""
    retrieved = [
        RetrievedMemory(memory=_record("m1", session_id="A"), similarity=0.9),
        RetrievedMemory(memory=_record("m-rogue", session_id="B"), similarity=0.7),
    ]

    with pytest.raises(ValueError):
        build_memory_context("A", retrieved)


def test_session_mismatch_check_tolerates_whitespace_differences() -> None:
    retrieved = [RetrievedMemory(memory=_record("m1", session_id=" A "), similarity=0.5)]

    context = build_memory_context("A", retrieved)

    assert [i.memory_id for i in context.items] == ["m1"]


def test_context_carries_exactly_one_session_not_one_per_item() -> None:
    """Session lives on the container; repeating it per item would create
    a second place for the same truth to be stated."""
    context = build_memory_context("A", [RetrievedMemory(memory=_record("m1"), similarity=0.5)])

    assert context.session_id == "A"
    assert not hasattr(context.items[0], "session_id")


# ---------------------------------------------------------------------------
# 10: deterministic ordering -- input order preserved, never re-sorted.
# ---------------------------------------------------------------------------

def test_adapter_preserves_input_order_exactly() -> None:
    retrieved = [
        RetrievedMemory(memory=_record("first"), similarity=0.9),
        RetrievedMemory(memory=_record("second"), similarity=0.5),
        RetrievedMemory(memory=_record("third"), similarity=0.1),
    ]

    context = build_memory_context("A", retrieved)

    assert [i.memory_id for i in context.items] == ["first", "second", "third"]


def test_adapter_does_not_re_sort_by_similarity() -> None:
    """Ranking is not this layer's job: even a deliberately unsorted input
    comes out in the same order it went in."""
    retrieved = [
        RetrievedMemory(memory=_record("low"), similarity=0.1),
        RetrievedMemory(memory=_record("high"), similarity=0.9),
        RetrievedMemory(memory=_record("mid"), similarity=0.5),
    ]

    context = build_memory_context("A", retrieved)

    assert [i.memory_id for i in context.items] == ["low", "high", "mid"]
    assert [i.similarity for i in context.items] == pytest.approx([0.1, 0.9, 0.5])


def test_context_preserves_item_order_as_given() -> None:
    context = MemoryContext(session_id="A", items=(_item("z"), _item("a"), _item("m")))

    assert [i.memory_id for i in context.items] == ["z", "a", "m"]


def test_building_the_same_context_twice_is_deterministic() -> None:
    created = _now()
    retrieved = [RetrievedMemory(memory=_record("m1", created_at=created), similarity=0.5)]

    assert build_memory_context("A", retrieved) == build_memory_context("A", retrieved)


# ---------------------------------------------------------------------------
# 11: no vectors / no storage internals exposed.
# ---------------------------------------------------------------------------

def test_context_item_exposes_no_vector_or_embedding_details() -> None:
    item = _item()

    for absent in ["vector", "embedding", "dimension", "query_vector", "embedding_model", "vector_index"]:
        assert not hasattr(item, absent)


def test_context_item_exposes_no_storage_lifecycle_fields() -> None:
    """`active` and `source_event_ids` are storage/episodic concerns and
    must not cross this boundary."""
    item = _item()

    assert not hasattr(item, "active")
    assert not hasattr(item, "source_event_ids")


def test_context_item_does_not_carry_the_semantic_record_itself() -> None:
    item = _item()

    assert not hasattr(item, "memory")
    assert not hasattr(item, "record")


def test_adapter_output_does_not_leak_the_record_through_any_field() -> None:
    record = _record("m1")
    context = build_memory_context("A", [RetrievedMemory(memory=record, similarity=0.5)])

    for value in vars(context.items[0]).values():
        assert not isinstance(value, SemanticMemoryRecord)


def test_context_exposes_only_session_id_and_items() -> None:
    context = MemoryContext(session_id="A", items=(_item(),))

    assert set(vars(context).keys()) == {"session_id", "items"}


def test_context_item_exposes_only_the_five_projected_fields() -> None:
    item = _item()

    assert set(vars(item).keys()) == {"memory_id", "content", "similarity", "created_at", "confidence"}


# ---------------------------------------------------------------------------
# 12: no retrieval behavior lives in the context object.
# ---------------------------------------------------------------------------

def test_context_has_no_retrieval_methods() -> None:
    context = MemoryContext(session_id="A", items=(_item(),))

    for absent in ["retrieve", "search", "embed", "add", "get", "clear"]:
        assert not hasattr(context, absent)


def test_context_item_has_no_retrieval_methods() -> None:
    item = _item()

    for absent in ["retrieve", "search", "embed"]:
        assert not hasattr(item, absent)


def test_context_can_be_built_with_no_retriever_store_or_index_present() -> None:
    """The contract stands alone: constructing it requires none of the
    retrieval collaborators, proving no retrieval happens inside it."""
    context = MemoryContext(
        session_id="A",
        items=(MemoryContextItem("m1", "a fact", 0.5, _now(), 1.0),),
    )

    assert context.items[0].content == "a fact"


# ---------------------------------------------------------------------------
# No prompt formatting is performed at this layer (Step 16E-A boundary).
# ---------------------------------------------------------------------------

def test_no_prompt_formatting_helpers_exist_on_the_contract() -> None:
    context = MemoryContext(session_id="A", items=(_item(),))

    for absent in ["to_prompt", "render", "format", "as_text", "to_messages", "to_system_message"]:
        assert not hasattr(context, absent)


def test_content_is_stored_verbatim_without_decoration() -> None:
    """No labels, numbering, XML tags or preamble are added anywhere."""
    context = build_memory_context("A", [RetrievedMemory(memory=_record("m1", content="User prefers Python."), similarity=0.5)])

    assert context.items[0].content == "User prefers Python."
