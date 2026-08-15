"""Tests for classifier benchmarking: candidates, measurement, selection."""

from dataclasses import replace
from pathlib import Path

from joblib import load

from routellm.application.classifier_service import ClassifierService
from routellm.classification.algorithms import CANDIDATES, fit_candidate
from routellm.classification.dataset import Dataset, dataset_fingerprint, write_dataset
from routellm.evaluation.benchmark import (
    BenchmarkRow,
    run_benchmark,
    select_candidate,
)
from routellm.evaluation.report import compute_metrics


def _dataset(categories=("coding", "math", "translation"), per_class=20) -> Dataset:
    texts = []
    labels = []
    for category in categories:
        for index in range(per_class):
            texts.append(f"sample {category} {index} prompt")
            labels.append(category)
    return Dataset(texts=tuple(texts), categories=tuple(labels))


def _write_dataset(path):
    write_dataset(_dataset(), path)
    return path


def test_all_candidates_fit_and_predict_on_same_fixture():
    dataset = _dataset()
    for candidate in CANDIDATES:
        fitted = fit_candidate(candidate, dataset)
        assert set(fitted.classes) == {"coding", "math", "translation"}
        prediction = fitted.predict("sample coding 3 prompt")
        assert prediction.category == "coding"


def test_benchmark_rows_have_real_latency_and_size(tmp_path):
    result = run_benchmark(
        str(_write_dataset(tmp_path / "prompts.csv")),
        out_dir=str(tmp_path / "bench"),
    )
    assert [row.name for row in result.rows] == [candidate.name for candidate in CANDIDATES]
    for row in result.rows:
        assert row.size_bytes > 0
        assert row.mean_latency_ms > 0
        assert 0.0 <= row.validation_metrics.accuracy <= 1.0


def test_benchmark_candidates_share_split_and_feature_setup(tmp_path):
    dataset_path = _write_dataset(tmp_path / "prompts.csv")
    result = run_benchmark(str(dataset_path), out_dir=str(tmp_path / "bench"))
    assert result.fingerprint == dataset_fingerprint(_dataset())
    for row in result.rows:
        loaded = load(Path(result.report_path).parent / f"{row.name}.joblib")
        assert loaded.feature_config == result.feature_config
        assert loaded.split_config == result.split_config
        assert loaded.dataset_fingerprint == result.fingerprint
        assert loaded.classes == result.rows[0].validation_metrics.classes


def _base_metrics():
    return compute_metrics(("a", "b"), ("a", "b"), (0.9, 0.9), ("a", "b"))


def test_selection_follows_validation_f1():
    base = _base_metrics()
    rows = (
        BenchmarkRow("low_f1", replace(base, macro_f1=0.5), None, 10.0, 100),
        BenchmarkRow("high_f1", replace(base, macro_f1=0.9), None, 10.0, 100),
    )
    assert select_candidate(rows) == "high_f1"


def test_selection_ties_break_by_latency_then_size():
    base = _base_metrics()
    rows = (
        BenchmarkRow("slow", replace(base, macro_f1=0.7), None, 5.0, 100),
        BenchmarkRow("fast_big", replace(base, macro_f1=0.7), None, 1.0, 100),
        BenchmarkRow("fast_small", replace(base, macro_f1=0.7), None, 1.0, 50),
    )
    assert select_candidate(rows) == "fast_small"


def _binary_dataset(per_class=20) -> Dataset:
    texts = []
    labels = []
    for category in ("coding", "math"):
        for index in range(per_class):
            texts.append(f"sample {category} {index} prompt")
            labels.append(category)
    return Dataset(texts=tuple(texts), categories=tuple(labels))


def _linear_svm_candidate():
    return next(candidate for candidate in CANDIDATES if candidate.name == "linear_svm")


def test_binary_linear_svm_predict_and_batch():
    fitted = fit_candidate(_linear_svm_candidate(), _binary_dataset())
    assert set(fitted.classes) == {"coding", "math"}

    prediction = fitted.predict("sample coding 3 prompt")
    assert prediction.category == "coding"
    assert prediction.confidence is None
    assert len(prediction.scores) == 2
    assert sorted(name for name, _ in prediction.scores) == sorted(fitted.classes)
    assert prediction.scores[0][0] == prediction.category
    assert prediction.scores[0][1] >= prediction.scores[1][1]

    batch = fitted.predict_batch(("sample coding 3 prompt", "sample math 7 prompt"))
    assert [p.category for p in batch] == ["coding", "math"]
    for item in batch:
        assert len(item.scores) == len(fitted.classes)
        assert sorted(name for name, _ in item.scores) == sorted(fitted.classes)
        assert item.scores[0][0] == item.category
        assert item.confidence is None
        assert item.scores[0][1] >= item.scores[1][1]


def test_binary_benchmark_runs_all_candidates(tmp_path):
    dataset_path = tmp_path / "binary.csv"
    write_dataset(_binary_dataset(), dataset_path)
    result = run_benchmark(str(dataset_path), out_dir=str(tmp_path / "bench"))
    assert [row.name for row in result.rows] == [candidate.name for candidate in CANDIDATES]
    assert Path(result.report_path).exists()


def test_linear_svm_exposes_no_probability_confidence():
    svm = next(candidate for candidate in CANDIDATES if candidate.name == "linear_svm")
    assert not svm.probability_available
    fitted = fit_candidate(svm, _dataset())
    prediction = fitted.predict("sample math 2 prompt")
    assert prediction.category == "math"
    assert prediction.confidence is None
    metrics = compute_metrics(
        ("math",),
        (prediction.category,),
        (prediction.confidence,),
        fitted.classes,
    )
    assert metrics.low_confidence_rate is None


def test_probability_candidates_expose_float_confidence():
    for candidate in CANDIDATES:
        if not candidate.probability_available:
            continue
        fitted = fit_candidate(candidate, _dataset())
        prediction = fitted.predict("sample math 2 prompt")
        assert prediction.confidence is not None
        assert 0.0 <= prediction.confidence <= 1.0
        assert prediction.scores[0][1] == prediction.confidence


def test_benchmark_service_integration(tmp_path):
    dataset_path = _write_dataset(tmp_path / "prompts.csv")
    out_dir = tmp_path / "bench"
    result = ClassifierService().benchmark(str(dataset_path), out_dir=str(out_dir))
    report_path = Path(result.report_path)
    assert report_path.parent == out_dir
    assert report_path.exists()
    for row in result.rows:
        assert (out_dir / f"{row.name}.joblib").exists()
        if row.name == result.selected_name:
            assert row.test_metrics is not None
        else:
            assert row.test_metrics is None
    selected = next(row for row in result.rows if row.name == result.selected_name)
    assert selected.test_metrics.n_prompts == result.n_test


def test_benchmark_report_text_contains_required_fields(tmp_path):
    result = run_benchmark(
        str(_write_dataset(tmp_path / "prompts.csv")),
        out_dir=str(tmp_path / "bench"),
    )
    text = Path(result.report_path).read_text(encoding="utf-8")
    assert "Dataset fingerprint" in text
    assert "Feature configuration" in text
    assert "Split configuration" in text
    assert "Selection criterion" in text
    assert "Latency methodology" in text
    assert "Selected candidate" in text
    assert "held-out test split" in text
    assert "n/a" in text
