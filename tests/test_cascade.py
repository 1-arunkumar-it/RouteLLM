"""Tests for the cascaded routing policy and cascade model (Milestone 4)."""

import pytest
from joblib import dump

from routellm.classification.cascade_model import (
    CascadeModel,
    fit_cascade,
    load_cascade_model,
    save_cascade_model,
)
from routellm.classification.dataset import Dataset
from routellm.configuration.cascade import CascadeConfig
from routellm.domain.classifier_prediction import ClassifierPrediction
from routellm.domain.signal import Signal
from routellm.evaluation.rule_metrics import (
    rule_categories,
    rule_precision_by_category,
    select_override_categories,
)
from routellm.evaluation.threshold_selection import select_threshold
from routellm.routing.cascade import (
    SOURCE_CLASSIFIER,
    SOURCE_FALLBACK,
    SOURCE_RULES,
    apply_cascade,
    cascade_outcomes,
)
from routellm.signals.keyword_rules import SIGNAL_RULES


def _prediction(category: str, confidence: float) -> ClassifierPrediction:
    return ClassifierPrediction(
        text="x",
        category=category,
        confidence=confidence,
        scores=((category, confidence),),
    )


def _dataset(categories=("coding", "math", "translation"), per_class=15) -> Dataset:
    texts = []
    labels = []
    for category in categories:
        for index in range(per_class):
            texts.append(f"sample {category} prompt {index}")
            labels.append(category)
    return Dataset(texts=tuple(texts), categories=tuple(labels))


# --- CascadeConfig validation ---


def test_cascade_config_defaults_are_valid():
    config = CascadeConfig()
    assert config.rule_override_min_precision == 0.90
    assert config.threshold is None
    assert config.calibration_cv == 5
    assert config.calibration_method == "sigmoid"


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_cascade_config_rejects_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="threshold"):
        CascadeConfig(threshold=threshold)


def test_cascade_config_rejects_invalid_override_precision():
    with pytest.raises(ValueError, match="rule_override_min_precision"):
        CascadeConfig(rule_override_min_precision=1.5)


def test_cascade_config_rejects_invalid_calibration_cv():
    with pytest.raises(ValueError, match="calibration_cv"):
        CascadeConfig(calibration_cv=1)


def test_cascade_config_rejects_unknown_calibration_method():
    with pytest.raises(ValueError, match="calibration_method"):
        CascadeConfig(calibration_method="marginal")


# --- Cascade policy branches ---


def test_rules_override_when_category_is_override_safe():
    signals = (Signal(phrase="python", category="coding"),)
    outcome = apply_cascade(
        signals,
        _prediction("math", 0.99),
        threshold=0.8,
        override_categories=frozenset({"coding"}),
    )
    assert outcome.category == "coding"
    assert outcome.source == SOURCE_RULES
    assert outcome.confidence is None


def test_classifier_wins_above_threshold():
    signals = (Signal(phrase="what", category="general_qa"),)
    outcome = apply_cascade(
        signals,
        _prediction("math", 0.91),
        threshold=0.8,
        override_categories=frozenset({"coding"}),
    )
    assert outcome.category == "math"
    assert outcome.source == SOURCE_CLASSIFIER
    assert outcome.confidence == pytest.approx(0.91)


def test_confidence_at_threshold_is_accepted():
    outcome = apply_cascade(
        (), _prediction("coding", 0.80), threshold=0.80, override_categories=frozenset()
    )
    assert outcome.source == SOURCE_CLASSIFIER


def test_low_confidence_falls_back_with_confidence_reported():
    outcome = apply_cascade(
        (), _prediction("coding", 0.79), threshold=0.80, override_categories=frozenset()
    )
    assert outcome.category == "unknown"
    assert outcome.source == SOURCE_FALLBACK
    assert outcome.confidence == pytest.approx(0.79)


def test_cascade_outcomes_require_aligned_inputs():
    with pytest.raises(ValueError, match="equal length"):
        cascade_outcomes(("coding",), (), threshold=0.8, override_categories=frozenset())


def test_cascade_outcomes_reject_out_of_range_threshold():
    with pytest.raises(ValueError, match="threshold"):
        cascade_outcomes((), (), threshold=1.5, override_categories=frozenset())


# --- Threshold selection ---


def test_threshold_selection_prefers_higher_tie_break():
    y_true = ("a", "a", "a")
    rule_cats = ("unknown", "unknown", "unknown")
    predictions = tuple(_prediction("a", 0.6) for _ in range(3))
    threshold, f1 = select_threshold(rule_cats, predictions, y_true, ("a", "b"), frozenset())
    assert threshold == 0.60
    assert f1 == pytest.approx(0.5)


def test_threshold_selection_requires_validation_data():
    with pytest.raises(ValueError, match="empty"):
        select_threshold((), (), (), ("a",), frozenset())


# --- Rule metrics ---


def test_rule_categories_use_only_keywords():
    categories = rule_categories(("write a python script", "translate to tamil"), SIGNAL_RULES)
    assert categories == ("coding", "translation")


def test_rule_precision_measured_from_predictions():
    precision = rule_precision_by_category(
        ("coding", "math"), ("coding", "math"), ("coding", "math")
    )
    assert precision["coding"] == 1.0
    assert precision["math"] == 1.0


def test_override_categories_exclude_unknown_and_low_precision():
    precision = {"coding": 1.0, "math": 0.5, "unknown": 1.0}
    assert select_override_categories(precision, 0.9) == frozenset({"coding"})


def test_select_override_categories_rejects_invalid_precision():
    with pytest.raises(ValueError, match="min_precision"):
        select_override_categories({"coding": 1.0}, 1.5)


# --- Cascade model ---


def test_fit_cascade_predicts_with_calibrated_confidence():
    model = fit_cascade(_dataset())
    assert set(model.classes) == {"coding", "math", "translation"}
    for text in model.classes:
        prediction = model.predict(f"sample {text} 3 prompt")
        assert prediction.category == text
        assert prediction.confidence is not None
        assert 0.0 <= prediction.confidence <= 1.0


def test_fit_cascade_predictions_are_deterministic():
    model = fit_cascade(_dataset())
    first = model.predict("sample math 5 prompt")
    second = model.predict("sample math 5 prompt")
    assert first.category == second.category
    assert first.confidence == second.confidence


def test_cascade_model_save_load_roundtrip(tmp_path):
    model = fit_cascade(_dataset())
    path = tmp_path / "cascade.joblib"
    save_cascade_model(model, path)
    loaded = load_cascade_model(path)
    assert isinstance(loaded, CascadeModel)
    original = model.predict("sample coding 0 prompt")
    reloaded = loaded.predict("sample coding 0 prompt")
    assert original.category == reloaded.category
    assert original.confidence == reloaded.confidence


def test_load_cascade_model_rejects_other_objects(tmp_path):
    path = tmp_path / "other.joblib"
    dump({"not": "cascade"}, path)
    with pytest.raises(ValueError, match="CascadeModel"):
        load_cascade_model(path)
