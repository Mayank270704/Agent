"""Evaluation utilities (Milestone 22, Phase 5) and the Milestone 20
external challenge set (Phase 5/6).

--------------------------------------------------------------------------
MILESTONE_20_CHALLENGE_SET
--------------------------------------------------------------------------
The exact 20 queries from Milestone 20's web-intent diagnostic
(tests/... — the diagnostic itself was a throwaway script, never
committed; these 20 strings are transcribed verbatim from that
milestone's report). All 20 are true label WEB, since that diagnostic
was specifically measuring web_search under-calling. None of these exact
strings appear in `app/ml_router/dataset.py`'s generated training data,
and `train.py` additionally holds out the specific entities they name
(RAG, OpenAI, Google DeepMind, SpaceX, Elon Musk, quantum computing,
Python) from generation entirely, so this is a genuine external
challenge set, not merely a held-out split of the same distribution.
"""
from __future__ import annotations

from dataclasses import dataclass

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from app.ml_router.classifier import IntentClassifier, IntentPrediction
from app.ml_router.contract import INTENT_LABELS
from app.ml_router.dataset import IntentExample

MILESTONE_20_CHALLENGE_SET: tuple[tuple[str, str], ...] = (
    # Family 1: explicit web/search commands
    ("Search the web for information about RAG.", "WEB"),
    ("Find online what the current stock price of NVIDIA is.", "WEB"),
    ("Look up the latest developments in quantum computing.", "WEB"),
    ("Find current information about retrieval augmented generation.", "WEB"),
    ("Search for the latest news about SpaceX.", "WEB"),
    # Family 2: recency/current-information requests
    ("What are the latest developments in AI?", "WEB"),
    ("What has happened recently in the tech industry?", "WEB"),
    ("What is the current state of the AI industry?", "WEB"),
    ("What is happening right now in the stock market?", "WEB"),
    ("What are today's top headlines?", "WEB"),
    ("What has happened in AI this week?", "WEB"),
    # Family 3: current entities/events
    ("What is the latest news about OpenAI?", "WEB"),
    ("What are the recent developments at Google DeepMind?", "WEB"),
    ("Tell me about current events in artificial intelligence.", "WEB"),
    ("What is Elon Musk doing currently?", "WEB"),
    ("What is the latest AI news?", "WEB"),
    # Family 4: compound requests
    ("Explain RAG and then tell me about recent developments in RAG.", "WEB"),
    ("What is a transformer, and what are the latest transformer models released this year?", "WEB"),
    ("Explain what machine learning is, then give me today's AI news.", "WEB"),
    ("What is Python, and what is the latest version of Python released?", "WEB"),
)

# The Phase 6 adversarial/robustness set — deliberately NOT used for
# training or for a pass/fail metric. Reported qualitatively (Phase 6:
# "check where the classifier becomes uncertain," not "add rules until
# it passes").
ROBUSTNESS_PROBES: tuple[str, ...] = (
    "what is the latest version of python",
    "what is Python",
    "what is the current time",
    "what is current machine learning research",
    "tell me what happened today",
    "what happened today in my life",
    "search the web for RAG",
    "explain RAG",
    "explain RAG and find recent developments",
    "what is the date",
    "what date was Python created",
)


@dataclass(frozen=True)
class ClassMetrics:
    label: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class EvaluationReport:
    n_samples: int
    accuracy: float
    per_class: list[ClassMetrics]
    confusion: dict[str, dict[str, int]]  # confusion["true_label"]["predicted_label"] = count
    labels_order: list[str]


def evaluate(classifier: IntentClassifier, examples: list[IntentExample]) -> EvaluationReport:
    """Standard held-out evaluation: accuracy, per-class precision/
    recall/F1, and a full confusion matrix. Deterministic — classifier
    inference has no randomness."""
    if not examples:
        raise ValueError("Cannot evaluate on an empty example list.")

    y_true = [example.label for example in examples]
    y_pred = [classifier.predict(example.text).label for example in examples]

    labels_order = list(INTENT_LABELS)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_order, zero_division=0
    )
    per_class = [
        ClassMetrics(label=label, precision=float(p), recall=float(r), f1=float(f), support=int(s))
        for label, p, r, f, s in zip(labels_order, precision, recall, f1, support)
    ]

    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)

    cm = confusion_matrix(y_true, y_pred, labels=labels_order)
    confusion = {
        true_label: {pred_label: int(cm[i][j]) for j, pred_label in enumerate(labels_order)}
        for i, true_label in enumerate(labels_order)
    }

    return EvaluationReport(
        n_samples=len(examples),
        accuracy=accuracy,
        per_class=per_class,
        confusion=confusion,
        labels_order=labels_order,
    )


@dataclass(frozen=True)
class ChallengeResult:
    query: str
    expected: str
    predicted: str
    confidence: float
    correct: bool


@dataclass(frozen=True)
class ChallengeReport:
    results: list[ChallengeResult]
    accuracy: float
    misclassified: list[ChallengeResult]


def evaluate_challenge_set(
    classifier: IntentClassifier, challenge_set: tuple[tuple[str, str], ...] = MILESTONE_20_CHALLENGE_SET
) -> ChallengeReport:
    results = []
    for query, expected in challenge_set:
        prediction = classifier.predict(query)
        results.append(
            ChallengeResult(
                query=query,
                expected=expected,
                predicted=prediction.label,
                confidence=prediction.confidence,
                correct=prediction.label == expected,
            )
        )
    accuracy = sum(1 for r in results if r.correct) / len(results)
    misclassified = [r for r in results if not r.correct]
    return ChallengeReport(results=results, accuracy=accuracy, misclassified=misclassified)


def probe_robustness(classifier: IntentClassifier, probes: tuple[str, ...] = ROBUSTNESS_PROBES) -> list[IntentPrediction]:
    """Runs each adversarial probe through the classifier and returns raw
    predictions (label + full probability distribution) for qualitative
    inspection — no pass/fail judgment is made here."""
    return [classifier.predict(probe) for probe in probes]


def format_report(report: EvaluationReport) -> str:
    lines = [
        f"n_samples={report.n_samples}  accuracy={report.accuracy:.1%}",
        "",
        f"{'label':<8} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>8}",
    ]
    for m in report.per_class:
        lines.append(f"{m.label:<8} {m.precision:>10.1%} {m.recall:>10.1%} {m.f1:>10.1%} {m.support:>8}")
    lines.append("")
    lines.append("confusion matrix (rows=true, cols=predicted):")
    header = "true\\pred".ljust(10) + "".join(label.ljust(8) for label in report.labels_order)
    lines.append(header)
    for true_label in report.labels_order:
        row = report.confusion[true_label]
        lines.append(true_label.ljust(10) + "".join(str(row[pred]).ljust(8) for pred in report.labels_order))
    return "\n".join(lines)
