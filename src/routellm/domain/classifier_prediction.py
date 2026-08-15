"""Structured result of a statistical classification prediction."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ClassifierPrediction:
    """The result of classifying one prompt with the trained classifier.

    ``confidence`` is the maximum class score from ``predict_proba``. It is
    an uncalibrated probability estimate and must not be treated as a trusted
    probability until the classifier's scores are validated (SPEC section 20).
    When the classifier exposes no probability estimate (for example, Linear
    SVM decision margins), ``confidence`` is ``None`` and ``scores`` holds the
    raw decision scores, which must never be rendered as probabilities.
    ``scores`` lists every class with its score, ordered by score descending,
    so the decision stays explainable.
    """

    text: str
    category: str
    confidence: float | None
    scores: tuple[tuple[str, float], ...]
