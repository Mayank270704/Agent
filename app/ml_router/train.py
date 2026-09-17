"""Training/evaluation entry point (Milestone 22, Phases 3-6).

    python -m app.ml_router.train

Builds the dataset (holding out the Milestone 20 challenge set's named
entities), splits it, trains the classifier, evaluates it on val/test,
runs the external Milestone 20 challenge set, runs the Phase 6
robustness probes, saves the trained artifact, and prints a full report.

This module has no side effect on production code: it writes only to
app/ml_router/artifacts/ (created if missing, matching the pattern
app/agent/local_embeddings.py already uses for its own cache directory),
and imports nothing from app/agent/ or app/services/.
"""
from __future__ import annotations

from app.ml_router.classifier import IntentClassifier, DEFAULT_ARTIFACT_PATH
from app.ml_router.dataset import (
    assert_no_text_overlap,
    build_dataset,
    class_distribution,
    save_dataset,
    split_dataset,
)
from app.ml_router.evaluate import (
    evaluate,
    evaluate_challenge_set,
    format_report,
    probe_robustness,
    ROBUSTNESS_PROBES,
)

# The exact entities the Milestone 20 challenge set names, held out of
# dataset GENERATION entirely (not merely filtered afterward) — see
# app/ml_router/evaluate.py's module docstring.
CHALLENGE_SET_HOLDOUT_TOPICS = frozenset(
    {"RAG", "OpenAI", "Google DeepMind", "SpaceX", "Elon Musk", "quantum computing", "Python"}
)


def main() -> None:
    examples = build_dataset(holdout_topics=CHALLENGE_SET_HOLDOUT_TOPICS)
    save_dataset(examples)

    print("=" * 78)
    print("DATASET")
    print("=" * 78)
    print(f"total examples: {len(examples)}")
    print(f"class distribution: {class_distribution(examples)}")

    train, val, test = split_dataset(examples)
    assert_no_text_overlap(train, val, test)
    print(f"train/val/test sizes: {len(train)}/{len(val)}/{len(test)}")
    print(f"train distribution: {class_distribution(train)}")
    print(f"val distribution:   {class_distribution(val)}")
    print(f"test distribution:  {class_distribution(test)}")
    print("no train/val/test text overlap: confirmed")

    classifier = IntentClassifier()
    classifier.fit(train)

    print()
    print("=" * 78)
    print("VALIDATION SET")
    print("=" * 78)
    val_report = evaluate(classifier, val)
    print(format_report(val_report))

    print()
    print("=" * 78)
    print("HELD-OUT TEST SET")
    print("=" * 78)
    test_report = evaluate(classifier, test)
    print(format_report(test_report))

    classifier.save(DEFAULT_ARTIFACT_PATH)
    print()
    print(f"Saved trained artifact to {DEFAULT_ARTIFACT_PATH}")

    print()
    print("=" * 78)
    print("MILESTONE 20 EXTERNAL CHALLENGE SET (20 queries, all true label WEB)")
    print("=" * 78)
    challenge_report = evaluate_challenge_set(classifier)
    print(f"accuracy: {challenge_report.accuracy:.1%} ({len(challenge_report.results) - len(challenge_report.misclassified)}/{len(challenge_report.results)})")
    print()
    print("per-query results:")
    for r in challenge_report.results:
        mark = "PASS" if r.correct else "FAIL"
        print(f"  [{mark}] predicted={r.predicted:<7} conf={r.confidence:.2f}  {r.query}")

    if challenge_report.misclassified:
        print()
        print("misclassified:")
        for r in challenge_report.misclassified:
            print(f"  expected={r.expected} got={r.predicted} (conf={r.confidence:.2f}): {r.query}")

    print()
    print("=" * 78)
    print("PHASE 6 ROBUSTNESS PROBES (qualitative — no pass/fail)")
    print("=" * 78)
    predictions = probe_robustness(classifier)
    for probe, prediction in zip(ROBUSTNESS_PROBES, predictions):
        probs = ", ".join(f"{label}={p:.2f}" for label, p in sorted(prediction.probabilities.items(), key=lambda kv: -kv[1]))
        print(f"  {probe!r}")
        print(f"    -> {prediction.label} (confidence={prediction.confidence:.2f})  [{probs}]")


if __name__ == "__main__":
    main()
