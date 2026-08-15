"""Classifier protocol boundary for interchangeable benchmark candidates.

Routing code must not import or depend on any candidate implementation. It
sees only this protocol, so swapping the trained candidate behind the
interface never changes routing policy.
"""

from typing import Protocol

from routellm.domain.classifier_prediction import ClassifierPrediction


class FittedClassifier(Protocol):
    """Interface every fitted classifier candidate must satisfy."""

    classes: tuple[str, ...]

    def predict(self, text: str) -> ClassifierPrediction:
        """Classify a single prompt."""

    def predict_batch(self, texts: tuple[str, ...]) -> list[ClassifierPrediction]:
        """Classify many prompts in one vectorized pass."""
