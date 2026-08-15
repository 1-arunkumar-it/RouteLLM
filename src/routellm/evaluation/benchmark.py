"""Benchmarking of interchangeable classifier candidates (Milestone 3).

Each candidate is trained on the same training split and compared on the same
validation split. The winner is chosen from validation metrics only, then
evaluated once on the held-out test split. All measurements are real; nothing
here is a hard-coded result.
"""

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from joblib import dump, load

from routellm.classification.algorithms import CANDIDATES, fit_candidate
from routellm.classification.dataset import (
    SplitConfig,
    check_no_leakage,
    dataset_fingerprint,
    load_dataset,
    stratified_split,
)
from routellm.classification.features import FeatureConfig
from routellm.evaluation.report import EvaluationReport, compute_metrics


@dataclass(frozen=True)
class BenchmarkRow:
    """Measured results for one candidate.

    ``test_metrics`` is populated only for the selected candidate, which is
    evaluated once on the held-out test split after selection.
    """

    name: str
    validation_metrics: EvaluationReport
    test_metrics: EvaluationReport | None
    mean_latency_ms: float
    size_bytes: int


@dataclass(frozen=True)
class BenchmarkResult:
    """The complete, reproducible benchmark run."""

    dataset_path: str
    fingerprint: str
    feature_config: FeatureConfig
    split_config: SplitConfig
    n_train: int
    n_validation: int
    n_test: int
    selected_name: str
    rows: tuple[BenchmarkRow, ...]
    report_path: str

    @property
    def selection_criterion(self) -> str:
        """Document the deterministic selection rule used."""
        return (
            "highest validation macro F1; ties break by lower warm inference "
            "latency, then smaller serialized model size"
        )


def select_candidate(rows: tuple[BenchmarkRow, ...]) -> str:
    """Select a candidate by validation macro F1 with deterministic ties.

    Selection uses only validation metrics. Ties break by lower warm inference
    latency, then smaller serialized model size. Fully equal rows resolve to
    the first in candidate order.
    """
    return min(
        rows,
        key=lambda row: (
            -row.validation_metrics.macro_f1,
            row.mean_latency_ms,
            row.size_bytes,
        ),
    ).name


def run_benchmark(
    dataset_path: str,
    out_dir: str = "models/benchmarks",
    feature_config: FeatureConfig = FeatureConfig(),
    split_config: SplitConfig = SplitConfig(),
) -> BenchmarkResult:
    """Fit every candidate, measure it on validation, select, and test it."""
    dataset = load_dataset(dataset_path)
    split = stratified_split(dataset, split_config)
    check_no_leakage(split)
    fingerprint = dataset_fingerprint(dataset)
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for candidate in CANDIDATES:
        fitted = fit_candidate(
            candidate,
            split.train,
            feature_config,
            split_config,
            dataset_path=str(Path(dataset_path)),
            dataset_fingerprint=fingerprint,
        )
        artifact_path = output_dir / f"{candidate.name}.joblib"
        dump(fitted, artifact_path)
        size_bytes = artifact_path.stat().st_size
        loaded = load(artifact_path)
        start = time.perf_counter()
        predictions = loaded.predict_batch(split.validation.texts)
        elapsed = time.perf_counter() - start
        validation_metrics = compute_metrics(
            y_true=split.validation.categories,
            y_pred=tuple(prediction.category for prediction in predictions),
            confidences=tuple(prediction.confidence for prediction in predictions),
            classes=loaded.classes,
            elapsed_seconds=elapsed,
        )
        rows.append(
            BenchmarkRow(
                name=candidate.name,
                validation_metrics=validation_metrics,
                test_metrics=None,
                mean_latency_ms=validation_metrics.mean_latency_ms,
                size_bytes=size_bytes,
            )
        )

    row_tuple = tuple(rows)
    selected_name = select_candidate(row_tuple)
    selected_path = output_dir / f"{selected_name}.joblib"
    loaded = load(selected_path)
    start = time.perf_counter()
    test_predictions = loaded.predict_batch(split.test.texts)
    test_elapsed = time.perf_counter() - start
    test_metrics = compute_metrics(
        y_true=split.test.categories,
        y_pred=tuple(prediction.category for prediction in test_predictions),
        confidences=tuple(prediction.confidence for prediction in test_predictions),
        classes=loaded.classes,
        elapsed_seconds=test_elapsed,
    )
    rows = tuple(
        BenchmarkRow(
            name=row.name,
            validation_metrics=row.validation_metrics,
            test_metrics=test_metrics if row.name == selected_name else None,
            mean_latency_ms=row.mean_latency_ms,
            size_bytes=row.size_bytes,
        )
        for row in rows
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    report_path = output_dir / f"benchmark_report_{timestamp}.txt"
    result = BenchmarkResult(
        dataset_path=str(Path(dataset_path)),
        fingerprint=fingerprint,
        feature_config=feature_config,
        split_config=split_config,
        n_train=len(split.train),
        n_validation=len(split.validation),
        n_test=len(split.test),
        selected_name=selected_name,
        rows=rows,
        report_path=str(report_path),
    )
    report_path.write_text(format_benchmark_report(result) + "\n", encoding="utf-8")
    return result


def _format_rate(rate: float | None) -> str:
    return "n/a" if rate is None else f"{rate:.3f}"


def format_benchmark_report(result: BenchmarkResult) -> str:
    """Render the full benchmark report as human-readable text."""
    config = result.feature_config
    split = result.split_config
    lines = [
        "RouteLLM classifier benchmark",
        "",
        f"Dataset: {result.dataset_path}",
        f"Dataset fingerprint: {result.fingerprint}",
        "Feature configuration:",
        f"  ngram_range={config.ngram_range} max_features={config.max_features} "
        f"min_df={config.min_df}",
        "Split configuration:",
        f"  test_fraction={split.test_fraction} validation_fraction={split.validation_fraction} "
        f"seed={split.seed}",
        f"Split sizes: train={result.n_train} validation={result.n_validation} "
        f"test={result.n_test}",
        "",
        "Selection criterion: " + result.selection_criterion,
        "  Selection uses the validation split only; the selected candidate is "
        "evaluated once on the held-out test split.",
        "",
        "Latency methodology: timing begins after model loading and uses a single "
        "batch prediction over the validation prompts, divided by the number of prompts.",
        "",
        "Candidate metrics (validation split):",
        f"{'candidate':<26}{'acc':>8}{'prec':>8}{'rec':>8}{'f1':>8}"
        f"{'low-conf':>9}{'latency_ms':>12}{'size_bytes':>12}",
    ]
    for row in result.rows:
        metrics = row.validation_metrics
        lines.append(
            f"{row.name:<26}{metrics.accuracy:>8.3f}{metrics.macro_precision:>8.3f}"
            f"{metrics.macro_recall:>8.3f}{metrics.macro_f1:>8.3f}"
            f"{_format_rate(metrics.low_confidence_rate):>9}"
            f"{row.mean_latency_ms:>12.3f}{row.size_bytes:>12d}"
        )
    selected = next(row for row in result.rows if row.name == result.selected_name)
    lines.extend(
        [
            "",
            f"Selected candidate: {result.selected_name} "
            f"(validation macro F1 {selected.validation_metrics.macro_f1:.3f})",
            "",
            "Selected candidate on the held-out test split:",
            f"  Accuracy: {selected.test_metrics.accuracy:.3f}",
            f"  Macro precision: {selected.test_metrics.macro_precision:.3f}",
            f"  Macro recall: {selected.test_metrics.macro_recall:.3f}",
            f"  Macro F1: {selected.test_metrics.macro_f1:.3f}",
        ]
    )
    return "\n".join(lines)
