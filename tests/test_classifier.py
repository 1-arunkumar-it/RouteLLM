"""Tests for the baseline classifier: training, prediction, persistence."""

import pytest
from joblib import dump

from routellm.classification.classifier import (
    TrainedClassifier,
    load_classifier,
    save_classifier,
    train_classifier,
)
from routellm.classification.dataset import Dataset
from routellm.domain.categories import CATEGORIES


def _dataset(categories=("coding", "math", "translation"), per_class=10) -> Dataset:
    texts = []
    labels = []
    for category in categories:
        for index in range(per_class):
            texts.append(f"sample {category} {index} prompt")
            labels.append(category)
    return Dataset(texts=tuple(texts), categories=tuple(labels))


def test_train_classifier_roundtrip():
    classifier = train_classifier(_dataset())
    assert set(classifier.classes) == {"coding", "math", "translation"}
    assert all(category in CATEGORIES for category in classifier.classes)
    for text in classifier.classes:
        prediction = classifier.predict(f"sample {text} 3 prompt")
        assert prediction.category == text
        assert 0.0 <= prediction.confidence <= 1.0


def test_prediction_scores_are_sorted_and_match_confidence():
    classifier = train_classifier(_dataset())
    prediction = classifier.predict("sample coding 2 prompt")
    assert len(prediction.scores) == len(classifier.classes)
    assert prediction.scores[0][0] == prediction.category
    assert prediction.scores[0][1] == prediction.confidence
    scores = [score for _, score in prediction.scores]
    assert scores == sorted(scores, reverse=True)


def test_predict_batch_is_aligned():
    classifier = train_classifier(_dataset())
    texts = ("sample coding 0 prompt", "sample math 0 prompt")
    predictions = classifier.predict_batch(texts)
    assert [prediction.text for prediction in predictions] == list(texts)
    assert predictions[0].category == "coding"
    assert predictions[1].category == "math"


def test_save_and_load_roundtrip(tmp_path):
    classifier = train_classifier(_dataset())
    path = tmp_path / "model.joblib"
    save_classifier(classifier, path)
    assert path.exists()
    loaded = load_classifier(path)
    assert isinstance(loaded, TrainedClassifier)
    original = classifier.predict("sample coding 0 prompt")
    reloaded = loaded.predict("sample coding 0 prompt")
    assert original.category == reloaded.category
    assert original.confidence == reloaded.confidence


def test_load_rejects_unrelated_object(tmp_path):
    path = tmp_path / "not_a_model.joblib"
    dump({"not": "a classifier"}, path)
    with pytest.raises(ValueError, match="TrainedClassifier"):
        load_classifier(path)


def test_classes_are_plain_strings():
    classifier = train_classifier(_dataset())
    assert all(isinstance(category, str) for category in classifier.classes)


def test_baseline_confidence_is_a_float_probability():
    classifier = train_classifier(_dataset())
    prediction = classifier.predict("sample coding 0 prompt")
    assert prediction.confidence is not None
    assert isinstance(prediction.confidence, float)
    assert 0.0 <= prediction.confidence <= 1.0
