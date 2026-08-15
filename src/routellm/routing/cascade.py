"""Cascaded routing policy combining rule evidence and calibrated confidence.

Milestone 4 policy, following the "cheapest reliable mechanism" principle:
rule evidence that is precise on held-out data wins; otherwise the calibrated
classifier is used above its validated confidence threshold; anything else
routes to ``unknown``/``fallback`` without ever presenting uncertainty as
certainty (SPEC sections 6, 20, and 38).

``threshold`` and ``override_categories`` are not hard-coded here: they are
selected at training time from the validation split by ``routellm cascade``
and stored in the cascade model. On the current dataset
(``data/datasets/prompts.csv``, seed 42) that run selected threshold 0.35
(validation macro F1 0.925) and override categories coding, creative_writing,
math, summarization, and translation; the cascade measured 0.973 accuracy and
0.976 macro F1 on the held-out test split. These numbers are data-dependent
and regenerated on every training run.
"""

from dataclasses import dataclass

from routellm.domain.classifier_prediction import ClassifierPrediction
from routellm.domain.signal import Signal
from routellm.routing import policy

SOURCE_RULES = "rules"
SOURCE_CLASSIFIER = "classifier"
SOURCE_FALLBACK = "fallback"


@dataclass(frozen=True)
class CascadeOutcome:
    """The result of one cascaded routing decision."""

    category: str
    source: str
    confidence: float | None


def cascade_outcomes(
    rule_categories: tuple[str, ...],
    predictions: tuple[ClassifierPrediction, ...],
    *,
    threshold: float,
    override_categories: frozenset[str],
) -> tuple[CascadeOutcome, ...]:
    """Apply the cascade policy to aligned rule and classifier results.

    For each prompt: a rule category that is override-safe wins immediately;
    otherwise a calibrated confidence at or above ``threshold`` accepts the
    classifier's category; anything else falls back to ``unknown``.
    """
    if len(rule_categories) != len(predictions):
        raise ValueError("rule_categories and predictions must have equal length.")
    if not 0 <= threshold <= 1:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}.")
    outcomes = []
    for rule_category, prediction in zip(rule_categories, predictions):
        if rule_category in override_categories:
            outcomes.append(CascadeOutcome(rule_category, SOURCE_RULES, None))
        elif prediction.confidence is not None and prediction.confidence >= threshold:
            outcomes.append(
                CascadeOutcome(prediction.category, SOURCE_CLASSIFIER, prediction.confidence)
            )
        else:
            outcomes.append(CascadeOutcome("unknown", SOURCE_FALLBACK, prediction.confidence))
    return tuple(outcomes)


def apply_cascade(
    signals: tuple[Signal, ...],
    prediction: ClassifierPrediction,
    *,
    threshold: float,
    override_categories: frozenset[str],
) -> CascadeOutcome:
    """Apply the cascade policy to one prompt's signals and prediction."""
    rule_category = policy.decide_category(signals)
    return cascade_outcomes(
        (rule_category,),
        (prediction,),
        threshold=threshold,
        override_categories=override_categories,
    )[0]
