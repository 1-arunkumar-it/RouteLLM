"""Integration tests for the classifier service orchestration."""

from pathlib import Path

import pytest

from routellm.application.classifier_service import ClassifierService
from routellm.classification.dataset import Dataset, write_dataset


def _write_synthetic_dataset(path):
    texts = []
    categories = []
    for category in ("coding", "math", "translation"):
        for index in range(20):
            texts.append(f"sample {category} request {index}")
            categories.append(category)
    write_dataset(Dataset(texts=tuple(texts), categories=tuple(categories)), path)
    return path


def test_train_and_evaluate_integration(tmp_path):
    dataset_path = _write_synthetic_dataset(tmp_path / "prompts.csv")
    model_path = tmp_path / "models" / "classifier.joblib"
    service = ClassifierService()
    report = service.train(
        str(dataset_path),
        str(model_path),
        splits_dir=str(tmp_path / "splits"),
    )
    assert model_path.exists()
    assert report.n_train > report.n_validation > 0
    assert report.n_test > 0
    assert report.n_train + report.n_validation + report.n_test == 60
    assert 0.0 <= report.validation_metrics.accuracy <= 1.0
    assert (tmp_path / "splits" / "train.csv").exists()
    assert (tmp_path / "splits" / "provenance.json").exists()

    evaluation = service.evaluate(str(model_path))
    assert evaluation.n_prompts == report.n_test
    assert 0.0 <= evaluation.accuracy <= 1.0
    assert evaluation.classes == ("coding", "math", "translation")


def test_service_predict_returns_prediction(tmp_path):
    dataset_path = _write_synthetic_dataset(tmp_path / "prompts.csv")
    model_path = tmp_path / "classifier.joblib"
    service = ClassifierService()
    service.train(str(dataset_path), str(model_path), splits_dir=str(tmp_path / "splits"))
    prediction = service.predict(str(model_path), "sample math request 5")
    assert prediction.category == "math"
    assert prediction.confidence > 0.5


def test_evaluate_rejects_changed_dataset(tmp_path):
    dataset_path = _write_synthetic_dataset(tmp_path / "prompts.csv")
    model_path = tmp_path / "classifier.joblib"
    service = ClassifierService()
    service.train(str(dataset_path), str(model_path), splits_dir=str(tmp_path / "splits"))
    with dataset_path.open("a", encoding="utf-8") as handle:
        handle.write('"extra row","coding"\n')
    with pytest.raises(ValueError, match="differs from the one used"):
        service.evaluate(str(model_path))


def test_evaluate_missing_model_fails_clearly(tmp_path):
    with pytest.raises((ValueError, OSError), match="classifier"):
        ClassifierService().evaluate(str(tmp_path / "missing.joblib"))


def test_low_confidence_threshold_is_applied(tmp_path):
    dataset_path = _write_synthetic_dataset(tmp_path / "prompts.csv")
    model_path = tmp_path / "classifier.joblib"
    service = ClassifierService()
    service.train(str(dataset_path), str(model_path), splits_dir=str(tmp_path / "splits"))
    strict = service.evaluate(str(model_path), low_confidence_threshold=1.0)
    lenient = service.evaluate(str(model_path), low_confidence_threshold=0.0)
    assert strict.low_confidence_rate == pytest.approx(1.0)
    assert lenient.low_confidence_rate == pytest.approx(0.0)


def test_benchmark_writes_report_and_artifacts_under_tmp(tmp_path):
    dataset_path = _write_synthetic_dataset(tmp_path / "prompts.csv")
    out_dir = tmp_path / "bench"
    result = ClassifierService().benchmark(str(dataset_path), out_dir=str(out_dir))
    report_path = Path(result.report_path)
    assert report_path.parent == out_dir
    assert report_path.exists()
    assert (out_dir / "logistic_regression.joblib").exists()
    assert (out_dir / "linear_svm.joblib").exists()
    assert (out_dir / "multinomial_naive_bayes.joblib").exists()
    assert result.selected_name in {row.name for row in result.rows}
