"""The classifier itself (Milestone 22, Phase 4): TF-IDF vectorization +
multinomial Logistic Regression, via scikit-learn. Deliberately not
embeddings, not a Transformer — see the package docstring for why a
strong, fast, explainable baseline comes first.

--------------------------------------------------------------------------
This class is intent-signal only
--------------------------------------------------------------------------
`IntentClassifier.predict`/`predict_proba` return a label and a
probability distribution — plain data. Nothing on this class has a
reference to a `Tool`, a `ToolRegistry`, a `PermissionPolicy`, or an
`ExecutionContext`, and nothing on this class can construct one (see the
package docstring for the full non-integration boundary). A caller that
wanted to act on this classifier's output would still have to go through
the SAME `ToolExecutionGate` pipeline every other proposed action does —
this class has no way around that, structurally, because it never touches
anything on that path in the first place.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app.ml_router.contract import INTENT_LABELS
from app.ml_router.dataset import IntentExample

DEFAULT_ARTIFACT_PATH = Path(__file__).resolve().parent / "artifacts" / "intent_classifier_v1.joblib"

# Fixed for reproducible training — identical input data always produces
# an identical trained model.
_RANDOM_STATE = 20260922


@dataclass(frozen=True)
class IntentPrediction:
    """One classifier verdict: the predicted label, its own probability,
    and the full distribution over all four labels (for confidence-based
    downstream decisions, e.g. "only trust this above 0.6"). Frozen and
    field-limited on purpose — the same "plain data, no side effects"
    discipline app/agent/telemetry.py's `AgentEvent` uses, for the same
    reason: nothing here should be mistaken for an authorization or
    execution signal."""

    label: str
    confidence: float
    probabilities: dict[str, float]


class IntentClassifier:
    """A trained (or trainable) TF-IDF + Logistic Regression pipeline
    over the four-label `Intent` vocabulary (app/ml_router/contract.py).

    Preprocessing is deliberately simple and deterministic: scikit-learn's
    default TF-IDF tokenizer (lowercasing, basic word-boundary tokens,
    unigrams+bigrams), no stemming, no external NLP library. Bigrams are
    included specifically because several of the hard negatives in the
    dataset (app/ml_router/dataset.py's `_DIRECT_HARD_NEGATIVES`) are
    distinguished from their WEB/TIME/DATE counterparts only by two-word
    phrases ("time complexity" vs "current time"; "ocean currents" vs
    "current news").
    """

    def __init__(self) -> None:
        self._pipeline: Pipeline = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        ngram_range=(1, 2),
                        min_df=1,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=_RANDOM_STATE,
                        class_weight="balanced",  # dataset is imbalanced (TIME/DATE << DIRECT/WEB)
                    ),
                ),
            ]
        )
        self._fitted = False

    def fit(self, examples: list[IntentExample]) -> "IntentClassifier":
        if not examples:
            raise ValueError("Cannot fit on an empty example list.")
        texts = [example.text for example in examples]
        labels = [example.label for example in examples]
        self._pipeline.fit(texts, labels)
        self._fitted = True
        return self

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("IntentClassifier has not been fit or loaded yet.")

    def predict(self, text: str) -> IntentPrediction:
        """Return the classifier's verdict for `text`. Deterministic:
        the same fitted model + the same text always returns the same
        prediction (LogisticRegression.predict is not stochastic at
        inference time)."""
        self._require_fitted()
        if text is None or not str(text).strip():
            raise ValueError("text cannot be empty.")

        cleaned = str(text).strip()
        proba_row = self._pipeline.predict_proba([cleaned])[0]
        classes = list(self._pipeline.named_steps["clf"].classes_)
        probabilities = {label: float(p) for label, p in zip(classes, proba_row)}
        # Re-derive the label from probabilities (argmax) rather than
        # calling .predict() a second time, so `label` and
        # `probabilities` can never disagree with each other.
        best_label = max(probabilities, key=probabilities.get)
        return IntentPrediction(
            label=best_label,
            confidence=probabilities[best_label],
            probabilities=probabilities,
        )

    def predict_proba(self, text: str) -> dict[str, float]:
        """Convenience accessor: just the probability distribution."""
        return self.predict(text).probabilities

    def save(self, path: Path = DEFAULT_ARTIFACT_PATH) -> None:
        self._require_fitted()
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipeline, path)

    @classmethod
    def load(cls, path: Path = DEFAULT_ARTIFACT_PATH) -> "IntentClassifier":
        if not path.exists():
            raise FileNotFoundError(
                f"No trained classifier artifact at {path}. Run app/ml_router/train.py first."
            )
        instance = cls()
        instance._pipeline = joblib.load(path)
        instance._fitted = True
        return instance
