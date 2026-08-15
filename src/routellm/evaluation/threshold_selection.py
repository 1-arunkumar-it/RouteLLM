"""Selection of the confidence threshold from held-out validation data.

The threshold is not assumed (SPEC section 20): it is the value that
maximizes cascade macro F1 on the validation split. Ties resolve to the
higher threshold, which is the more conservative choice.
"""

from routellm.domain.classifier_prediction import ClassifierPrediction
from routellm.evaluation.report import compute_metrics
from routellm.routing.cascade import cascade_outcomes

THRESHOLD_GRID = tuple(round(index * 0.05, 2) for index in range(21))


def select_threshold(
    rule_categories: tuple[str, ...],
    predictions: tuple[ClassifierPrediction, ...],
    y_true: tuple[str, ...],
    classes: tuple[str, ...],
    override_categories: frozenset[str],
) -> tuple[float, float]:
    """Choose the threshold maximizing cascade macro F1 on a held-out split.

    Returns the chosen threshold and its validation macro F1.
    """
    if len(rule_categories) != len(predictions) or len(predictions) != len(y_true):
        raise ValueError("rule_categories, predictions, and y_true must have equal length.")
    if not rule_categories:
        raise ValueError("cannot select a threshold from an empty validation set.")
    best_threshold = THRESHOLD_GRID[0]
    best_f1 = -1.0
    for threshold in THRESHOLD_GRID:
        outcomes = cascade_outcomes(
            rule_categories,
            predictions,
            threshold=threshold,
            override_categories=override_categories,
        )
        y_pred = tuple(outcome.category for outcome in outcomes)
        report = compute_metrics(
            y_true,
            y_pred,
            (None,) * len(y_true),
            classes,
            low_confidence_threshold=1.0,
        )
        if report.macro_f1 > best_f1 or (
            report.macro_f1 == best_f1 and threshold > best_threshold
        ):
            best_f1 = report.macro_f1
            best_threshold = threshold
    return best_threshold, best_f1
