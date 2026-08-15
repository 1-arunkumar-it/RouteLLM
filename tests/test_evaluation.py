"""Tests for evaluation metric computation."""

import pytest

from routellm.evaluation.report import compute_metrics


def test_metrics_match_hand_computed_values():
    y_true = ("coding", "math", "translation")
    y_pred = ("coding", "coding", "translation")
    confidences = (0.9, 0.6, 0.95)
    report = compute_metrics(y_true, y_pred, confidences, ("coding", "math", "translation"))
    assert report.n_prompts == 3
    assert report.accuracy == pytest.approx(2 / 3)
    assert report.macro_precision == pytest.approx((0.5 + 0.0 + 1.0) / 3)
    assert report.macro_recall == pytest.approx((1.0 + 0.0 + 1.0) / 3)
    assert report.confusion == ((1, 0, 0), (1, 0, 0), (0, 0, 1))
    math_metrics = report.per_class[1]
    assert math_metrics.category == "math"
    assert math_metrics.precision == 0.0
    assert math_metrics.support == 1


def test_low_confidence_rate_counts_below_threshold():
    confidences = (0.9, 0.7, 0.95, 0.6)
    report = compute_metrics(
        ("a", "a", "a", "a"),
        ("a", "a", "a", "a"),
        confidences,
        ("a", "b"),
        low_confidence_threshold=0.8,
    )
    assert report.low_confidence_rate == pytest.approx(0.5)


def test_macro_averages_include_all_configured_classes():
    y_true = ("coding", "translation")
    y_pred = ("coding", "coding")
    report = compute_metrics(
        y_true,
        y_pred,
        (0.9, 0.6),
        ("coding", "math", "translation"),
    )
    assert len(report.per_class) == 3
    math_metrics = report.per_class[1]
    assert math_metrics.category == "math"
    assert math_metrics.support == 0
    assert math_metrics.precision == 0.0
    assert report.macro_precision == pytest.approx((0.5 + 0.0 + 0.0) / 3)
    assert report.macro_recall == pytest.approx((1.0 + 0.0 + 0.0) / 3)
    assert report.macro_f1 == pytest.approx((2 / 3 + 0.0 + 0.0) / 3)


def test_none_confidences_yield_na_low_confidence_rate():
    report = compute_metrics(
        ("a", "a"),
        ("a", "a"),
        (0.9, None),
        ("a", "b"),
    )
    assert report.low_confidence_rate is None


def test_mean_latency_is_derived_from_elapsed_time():
    report = compute_metrics(
        ("a",) * 10,
        ("a",) * 10,
        (0.9,) * 10,
        ("a", "b"),
        elapsed_seconds=0.01,
    )
    assert report.mean_latency_ms == pytest.approx(1.0)


def test_metric_input_lengths_must_match():
    with pytest.raises(ValueError, match="equal length"):
        compute_metrics(("a",), ("a", "b"), (0.9,), ("a",))


def test_threshold_must_be_in_unit_interval():
    with pytest.raises(ValueError, match="threshold"):
        compute_metrics(
            ("a",),
            ("a",),
            (0.9,),
            ("a", "b"),
            low_confidence_threshold=1.5,
        )
