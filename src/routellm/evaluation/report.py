"""Computation of classification evaluation metrics."""

from dataclasses import dataclass
from statistics import fmean

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


@dataclass(frozen=True)
class ClassMetrics:
    """Per-class precision, recall, F1, and support."""

    category: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class EvaluationReport:
    """Aggregate metrics over a set of predictions."""

    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class: tuple[ClassMetrics, ...]
    classes: tuple[str, ...]
    confusion: tuple[tuple[int, ...], ...]
    mean_latency_ms: float | None
    low_confidence_rate: float | None
    low_confidence_threshold: float
    n_prompts: int


def compute_metrics(
    y_true: tuple[str, ...],
    y_pred: tuple[str, ...],
    confidences: tuple[float | None, ...],
    classes: tuple[str, ...],
    low_confidence_threshold: float = 0.80,
    elapsed_seconds: float | None = None,
) -> EvaluationReport:
    """Compute evaluation metrics from actual predictions.

    ``confidences`` holds the maximum predicted probability per prompt and is
    used to report the low-confidence rate at ``low_confidence_threshold``.
    A ``None`` confidence means no probability estimate is available for that
    prompt (for example, Linear SVM margins); the low-confidence rate is then
    reported as ``None`` (n/a) instead of a number.
    """
    if len(y_true) != len(y_pred) or len(y_pred) != len(confidences):
        raise ValueError("y_true, y_pred, and confidences must have equal length.")
    if not (0 <= low_confidence_threshold <= 1):
        raise ValueError(
            f"low_confidence_threshold must be between 0 and 1, got {low_confidence_threshold}."
        )
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(classes),
        average=None,
        zero_division=0,
    )
    per_class = tuple(
        ClassMetrics(
            category=category,
            precision=float(precision_value),
            recall=float(recall_value),
            f1=float(f1_value),
            support=int(support_value),
        )
        for category, precision_value, recall_value, f1_value, support_value in zip(
            classes, precision, recall, f1, support
        )
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(classes))
    n_prompts = len(y_true)
    if n_prompts and any(confidence is None for confidence in confidences):
        low_confidence_rate = None
    elif n_prompts:
        low_confidence_rate = sum(
            confidence < low_confidence_threshold for confidence in confidences
        ) / n_prompts
    else:
        low_confidence_rate = 0.0
    mean_latency_ms = (
        1000.0 * elapsed_seconds / n_prompts
        if elapsed_seconds is not None and n_prompts
        else None
    )
    return EvaluationReport(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_precision=fmean(float(value) for value in precision),
        macro_recall=fmean(float(value) for value in recall),
        macro_f1=fmean(float(value) for value in f1),
        per_class=per_class,
        classes=classes,
        confusion=tuple(tuple(int(value) for value in row) for row in matrix),
        mean_latency_ms=mean_latency_ms,
        low_confidence_rate=low_confidence_rate,
        low_confidence_threshold=low_confidence_threshold,
        n_prompts=n_prompts,
    )
