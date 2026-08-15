"""Measured precision of rule-based decisions (SPEC section 48).

Rule decisions are not inherently high confidence. Before keyword rules may
bypass or override statistical classification, their precision must be
measured on held-out examples.
"""

from routellm.evaluation.report import compute_metrics
from routellm.preprocessing import preprocessor
from routellm.routing import policy
from routellm.signals import engine
from routellm.signals.keyword_rules import SIGNAL_RULES


def rule_categories(
    texts: tuple[str, ...],
    rules: dict[str, tuple[tuple[str, ...], ...]] = SIGNAL_RULES,
) -> tuple[str, ...]:
    """Predict a category for each text using only the keyword rules."""
    return tuple(
        policy.decide_category(engine.detect_signals(preprocessor.tokenize(text), rules))
        for text in texts
    )


def rule_precision_by_category(
    y_true: tuple[str, ...],
    y_pred: tuple[str, ...],
    classes: tuple[str, ...],
) -> dict[str, float]:
    """Return measured rule precision per category on a held-out split."""
    report = compute_metrics(
        y_true,
        y_pred,
        (None,) * len(y_true),
        classes,
        low_confidence_threshold=1.0,
    )
    return {metrics.category: metrics.precision for metrics in report.per_class}


def select_override_categories(
    precision_by_category: dict[str, float],
    min_precision: float,
) -> frozenset[str]:
    """Return categories precise enough to bypass the classifier.

    ``unknown`` is never override-safe: rule silence is not rule evidence.
    """
    if not 0 <= min_precision <= 1:
        raise ValueError(f"min_precision must be in [0, 1], got {min_precision}.")
    return frozenset(
        category
        for category, precision in precision_by_category.items()
        if category != "unknown" and precision >= min_precision
    )
