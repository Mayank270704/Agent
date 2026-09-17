"""Milestone 22: app/ml_router/ — the independent, unwired ML intent
classifier. Proves the dataset, classifier, and evaluation machinery work
correctly in isolation, and proves the package's non-integration boundary
holds structurally, not just by convention.

Fully offline: no network, no LLM, no Ollama, no Tavily. Training a fresh
classifier from the dataset takes well under a second (TF-IDF + Logistic
Regression over ~370 short strings), so these tests train small fresh
models rather than depending on a pre-saved artifact — the one exception
being the save/load roundtrip tests, which explicitly exercise
persistence.
"""
from __future__ import annotations

import math

import pytest

from app.ml_router.classifier import IntentClassifier, IntentPrediction
from app.ml_router.contract import INTENT_LABELS, Intent
from app.ml_router.dataset import (
    IntentExample,
    assert_no_text_overlap,
    build_dataset,
    class_distribution,
    split_dataset,
)
from app.ml_router.evaluate import evaluate, evaluate_challenge_set, MILESTONE_20_CHALLENGE_SET

CHALLENGE_HOLDOUT = frozenset(
    {"RAG", "OpenAI", "Google DeepMind", "SpaceX", "Elon Musk", "quantum computing", "Python"}
)


@pytest.fixture(scope="module")
def dataset() -> list[IntentExample]:
    return build_dataset(holdout_topics=CHALLENGE_HOLDOUT)


@pytest.fixture(scope="module")
def splits(dataset: list[IntentExample]):
    return split_dataset(dataset)


@pytest.fixture(scope="module")
def trained_classifier(splits) -> IntentClassifier:
    train, _val, _test = splits
    return IntentClassifier().fit(train)


# ===========================================================================
# Dataset — volume, balance, contract compliance
# ===========================================================================

def test_dataset_size_is_within_the_target_range() -> None:
    examples = build_dataset()
    assert 400 <= len(examples) <= 800


def test_dataset_every_label_is_a_valid_intent(dataset: list[IntentExample]) -> None:
    valid = set(INTENT_LABELS)
    assert all(example.label in valid for example in dataset)


def test_dataset_all_four_classes_are_present_with_reasonable_volume(dataset: list[IntentExample]) -> None:
    counts = class_distribution(dataset)
    assert set(counts) == set(INTENT_LABELS)
    for label, count in counts.items():
        assert count >= 30, f"{label} has only {count} examples"


def test_dataset_contains_no_duplicate_text(dataset: list[IntentExample]) -> None:
    texts = [example.text.strip().lower() for example in dataset]
    assert len(texts) == len(set(texts))


def test_dataset_is_deterministic_across_builds() -> None:
    """Same seed, same templates -> byte-identical output every time."""
    first = build_dataset()
    second = build_dataset()
    assert [(e.text, e.label) for e in first] == [(e.text, e.label) for e in second]


def test_intent_example_rejects_blank_text() -> None:
    with pytest.raises(ValueError):
        IntentExample(text="   ", label="DIRECT")


def test_intent_example_rejects_an_unknown_label() -> None:
    with pytest.raises(ValueError):
        IntentExample(text="hello", label="MAYBE")


# ===========================================================================
# Dataset splits — integrity and leakage
# ===========================================================================

def test_split_sizes_are_non_trivial(splits) -> None:
    train, val, test = splits
    assert len(train) > 300
    assert len(val) > 50
    assert len(test) > 50


def test_split_has_no_text_overlap(splits) -> None:
    train, val, test = splits
    assert_no_text_overlap(train, val, test)  # must not raise


def test_split_overlap_check_actually_detects_a_planted_duplicate() -> None:
    """Proves assert_no_text_overlap is a real check, not a vacuous one."""
    a = [IntentExample(text="What time is it?", label="TIME")]
    b = [IntentExample(text="What time is it?", label="TIME")]
    with pytest.raises(AssertionError):
        assert_no_text_overlap(a, b)


def test_split_is_deterministic(dataset: list[IntentExample]) -> None:
    first = split_dataset(dataset)
    second = split_dataset(dataset)
    assert [e.text for e in first[0]] == [e.text for e in second[0]]
    assert [e.text for e in first[1]] == [e.text for e in second[1]]
    assert [e.text for e in first[2]] == [e.text for e in second[2]]


def test_split_preserves_every_example_exactly_once(dataset: list[IntentExample]) -> None:
    train, val, test = split_dataset(dataset)
    assert len(train) + len(val) + len(test) == len(dataset)


def test_split_is_roughly_stratified_by_class(dataset: list[IntentExample]) -> None:
    train, _val, test = split_dataset(dataset)
    train_dist = class_distribution(train)
    test_dist = class_distribution(test)
    for label in INTENT_LABELS:
        # every class present in the full dataset appears in both splits
        assert train_dist[label] > 0
        assert test_dist[label] > 0


def test_challenge_set_entities_are_absent_from_the_holdout_dataset(dataset: list[IntentExample]) -> None:
    """The Milestone 20 challenge set's named entities must not appear
    anywhere in the training/eval dataset when holdout_topics is used —
    this is what makes the challenge set a genuine external test."""
    for example in dataset:
        for topic in CHALLENGE_HOLDOUT:
            assert topic.lower() not in example.text.lower(), f"{topic!r} leaked into training: {example.text!r}"


# ===========================================================================
# Classifier — training, prediction, determinism
# ===========================================================================

def test_predict_before_fit_raises() -> None:
    classifier = IntentClassifier()
    with pytest.raises(RuntimeError):
        classifier.predict("What time is it?")


def test_fit_on_empty_list_raises() -> None:
    classifier = IntentClassifier()
    with pytest.raises(ValueError):
        classifier.fit([])


@pytest.mark.parametrize(
    "text,expected",
    [
        ("What time is it?", "TIME"),
        ("What's the current time?", "TIME"),
        ("What is today's date?", "DATE"),
        ("What day is it?", "DATE"),
        ("What is machine learning?", "DIRECT"),
        ("What is 25 + 17?", "DIRECT"),
        ("What are the latest developments in AI?", "WEB"),
        ("Search the web for information about robotics.", "WEB"),
    ],
)
def test_predict_returns_correct_label_for_canonical_examples(
    trained_classifier: IntentClassifier, text: str, expected: str
) -> None:
    prediction = trained_classifier.predict(text)
    assert prediction.label == expected


def test_predict_returns_an_intent_prediction_with_valid_fields(trained_classifier: IntentClassifier) -> None:
    prediction = trained_classifier.predict("What time is it?")
    assert isinstance(prediction, IntentPrediction)
    assert prediction.label in INTENT_LABELS
    assert 0.0 <= prediction.confidence <= 1.0
    assert set(prediction.probabilities) == set(INTENT_LABELS)


def test_probabilities_sum_to_approximately_one(trained_classifier: IntentClassifier) -> None:
    prediction = trained_classifier.predict("Explain gradient descent.")
    total = sum(prediction.probabilities.values())
    assert math.isclose(total, 1.0, abs_tol=1e-6)


def test_confidence_equals_the_probability_of_the_predicted_label(trained_classifier: IntentClassifier) -> None:
    prediction = trained_classifier.predict("What day is 25 December 2026?")
    assert prediction.confidence == prediction.probabilities[prediction.label]


def test_predict_proba_matches_predict(trained_classifier: IntentClassifier) -> None:
    text = "What is the latest news about robotics?"
    assert trained_classifier.predict_proba(text) == trained_classifier.predict(text).probabilities


def test_prediction_is_deterministic_across_repeated_calls(trained_classifier: IntentClassifier) -> None:
    text = "What are the recent developments in fusion energy?"
    first = trained_classifier.predict(text)
    second = trained_classifier.predict(text)
    assert first == second


def test_two_independently_trained_classifiers_from_the_same_data_agree(splits) -> None:
    """Same seed, same training data -> identical model, identical
    predictions — training has no hidden randomness."""
    train, _val, _test = splits
    a = IntentClassifier().fit(train)
    b = IntentClassifier().fit(train)
    text = "What is the latest news about robotics?"
    assert a.predict(text) == b.predict(text)


@pytest.mark.parametrize("bad_input", ["", "   ", None])
def test_predict_rejects_empty_or_none_input(trained_classifier: IntentClassifier, bad_input) -> None:
    with pytest.raises(ValueError):
        trained_classifier.predict(bad_input)


def test_predict_strips_surrounding_whitespace(trained_classifier: IntentClassifier) -> None:
    a = trained_classifier.predict("What time is it?")
    b = trained_classifier.predict("   What time is it?   ")
    assert a.label == b.label


# ===========================================================================
# Serialization / persistence
# ===========================================================================

def test_save_before_fit_raises(tmp_path) -> None:
    classifier = IntentClassifier()
    with pytest.raises(RuntimeError):
        classifier.save(tmp_path / "model.joblib")


def test_load_from_a_missing_path_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        IntentClassifier.load(tmp_path / "does_not_exist.joblib")


def test_save_then_load_roundtrip_produces_identical_predictions(trained_classifier: IntentClassifier, tmp_path) -> None:
    path = tmp_path / "roundtrip.joblib"
    trained_classifier.save(path)

    loaded = IntentClassifier.load(path)

    probe_texts = [
        "What time is it?",
        "What is today's date?",
        "What is machine learning?",
        "What are the latest developments in AI?",
    ]
    for text in probe_texts:
        assert loaded.predict(text) == trained_classifier.predict(text)


# ===========================================================================
# Evaluation utilities
# ===========================================================================

def test_evaluate_on_empty_examples_raises(trained_classifier: IntentClassifier) -> None:
    with pytest.raises(ValueError):
        evaluate(trained_classifier, [])


def test_evaluate_report_shape(trained_classifier: IntentClassifier, splits) -> None:
    _train, _val, test = splits
    report = evaluate(trained_classifier, test)

    assert report.n_samples == len(test)
    assert 0.0 <= report.accuracy <= 1.0
    assert {m.label for m in report.per_class} == set(INTENT_LABELS)
    assert set(report.confusion) == set(INTENT_LABELS)
    for true_label in INTENT_LABELS:
        assert set(report.confusion[true_label]) == set(INTENT_LABELS)


def test_confusion_matrix_row_sums_equal_class_support(trained_classifier: IntentClassifier, splits) -> None:
    _train, _val, test = splits
    report = evaluate(trained_classifier, test)
    test_dist = class_distribution(test)
    for true_label in INTENT_LABELS:
        row_sum = sum(report.confusion[true_label].values())
        assert row_sum == test_dist[true_label]


def test_held_out_test_accuracy_is_reasonably_high(trained_classifier: IntentClassifier, splits) -> None:
    """Not a strict regression gate on an exact number (the dataset is
    template-generated, so this could be trivially gamed) — a coarse
    sanity floor that the pipeline is actually learning something."""
    _train, _val, test = splits
    report = evaluate(trained_classifier, test)
    assert report.accuracy >= 0.80


def test_milestone_20_challenge_set_is_all_web_labels() -> None:
    """Structural sanity check on the challenge set itself."""
    assert len(MILESTONE_20_CHALLENGE_SET) == 20
    assert all(label == "WEB" for _text, label in MILESTONE_20_CHALLENGE_SET)


def test_evaluate_challenge_set_report_shape(trained_classifier: IntentClassifier) -> None:
    report = evaluate_challenge_set(trained_classifier)
    assert len(report.results) == 20
    assert 0.0 <= report.accuracy <= 1.0
    assert len(report.misclassified) == len(report.results) - round(report.accuracy * len(report.results))


# ===========================================================================
# Package boundary — this module is NOT wired into production
# ===========================================================================

def test_ml_router_package_is_not_imported_by_any_production_agent_module() -> None:
    """Structural, not conventional: greps the actual production source
    files for any import of app.ml_router. Milestone 22 is explicit that
    this must remain unwired until a future, separate integration
    decision."""
    import pathlib

    production_files = [
        "app/main.py",
        "app/services/chat.py",
        "app/agent/orchestrator.py",
        "app/agent/loop.py",
        "app/agent/decision_maker.py",
        "app/agent/router.py",
        "app/agent/tool_execution.py",
        "app/agent/permissions.py",
        "app/agent/tool_registry.py",
    ]
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    for relative_path in production_files:
        source = (repo_root / relative_path).read_text(encoding="utf-8")
        assert "ml_router" not in source, f"{relative_path} references ml_router"


def test_ml_router_does_not_import_from_the_production_agent_or_execution_path() -> None:
    """The reverse direction: app/ml_router/ itself must not import
    anything from the tool-execution/permission/orchestration path."""
    import pathlib

    forbidden_modules = (
        "app.agent.tool_execution",
        "app.agent.permissions",
        "app.agent.loop",
        "app.agent.orchestrator",
        "app.agent.tool_registry",
        "app.services.chat",
    )
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    ml_router_dir = repo_root / "app" / "ml_router"
    for path in ml_router_dir.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        for forbidden in forbidden_modules:
            assert forbidden not in source, f"{path.name} imports {forbidden}"


def test_intent_classifier_has_no_execute_or_authorize_shaped_method() -> None:
    """Structural proof the classifier cannot act on its own output: no
    method whose name suggests execution/authorization exists on the
    class at all."""
    forbidden_substrings = ("execute", "authorize", "confirm", "run_tool", "invoke")
    members = dir(IntentClassifier)
    for member in members:
        lowered = member.lower()
        assert not any(forbidden in lowered for forbidden in forbidden_substrings), member
