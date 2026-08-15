"""Application layer orchestrating classifier training and evaluation."""

import time
from dataclasses import dataclass
from pathlib import Path

from routellm.classification.classifier import (
    load_classifier,
    save_classifier,
    train_classifier,
)
from routellm.classification.dataset import (
    SplitConfig,
    check_no_leakage,
    dataset_fingerprint,
    load_dataset,
    stratified_split,
    write_splits,
)
from routellm.classification.features import FeatureConfig
from routellm.domain.classifier_prediction import ClassifierPrediction
from routellm.evaluation.benchmark import BenchmarkResult, run_benchmark
from routellm.evaluation.report import EvaluationReport, compute_metrics

DEFAULT_SPLITS_DIR = "data/processed/splits"


@dataclass(frozen=True)
class TrainReport:
    """Summary of a completed training run."""

    dataset_path: str
    model_path: str
    fingerprint: str
    n_train: int
    n_validation: int
    n_test: int
    validation_metrics: EvaluationReport


class ClassifierService:
    """Coordinate dataset handling, training, prediction, and evaluation."""

    def train(
        self,
        dataset_path: str,
        model_path: str,
        feature_config: FeatureConfig | None = None,
        split_config: SplitConfig | None = None,
        splits_dir: str | None = None,
    ) -> TrainReport:
        """Load the dataset, split it, train a classifier, and persist it."""
        feature_config = feature_config or FeatureConfig()
        split_config = split_config or SplitConfig()
        dataset = load_dataset(dataset_path)
        split = stratified_split(dataset, split_config)
        check_no_leakage(split)
        fingerprint = dataset_fingerprint(dataset)
        classifier = train_classifier(
            dataset=split.train,
            feature_config=feature_config,
            split_config=split_config,
            dataset_path=str(Path(dataset_path)),
            dataset_fingerprint=fingerprint,
        )
        save_classifier(classifier, model_path)
        write_splits(
            split,
            splits_dir or DEFAULT_SPLITS_DIR,
            dataset_path=str(Path(dataset_path)),
            config=split_config,
        )
        predictions = classifier.predict_batch(split.validation.texts)
        validation_metrics = compute_metrics(
            y_true=split.validation.categories,
            y_pred=tuple(prediction.category for prediction in predictions),
            confidences=tuple(prediction.confidence for prediction in predictions),
            classes=classifier.classes,
        )
        return TrainReport(
            dataset_path=str(Path(dataset_path)),
            model_path=str(Path(model_path)),
            fingerprint=fingerprint,
            n_train=len(split.train),
            n_validation=len(split.validation),
            n_test=len(split.test),
            validation_metrics=validation_metrics,
        )

    def evaluate(
        self,
        model_path: str,
        dataset_path: str | None = None,
        low_confidence_threshold: float = 0.80,
    ) -> EvaluationReport:
        """Evaluate a trained classifier on the held-out test split."""
        classifier = load_classifier(model_path)
        dataset_path = dataset_path or classifier.dataset_path
        dataset = load_dataset(dataset_path)
        if dataset_fingerprint(dataset) != classifier.dataset_fingerprint:
            raise ValueError(
                "The dataset differs from the one used to train this model. Retrain to continue."
            )
        split = stratified_split(dataset, classifier.split_config)
        start = time.perf_counter()
        predictions = classifier.predict_batch(split.test.texts)
        elapsed = time.perf_counter() - start
        return compute_metrics(
            y_true=split.test.categories,
            y_pred=tuple(prediction.category for prediction in predictions),
            confidences=tuple(prediction.confidence for prediction in predictions),
            classes=classifier.classes,
            low_confidence_threshold=low_confidence_threshold,
            elapsed_seconds=elapsed,
        )

    def predict(self, model_path: str, text: str) -> ClassifierPrediction:
        """Classify a single prompt with a trained model."""
        return load_classifier(model_path).predict(text)

    def benchmark(
        self,
        dataset_path: str,
        out_dir: str = "models/benchmarks",
        feature_config: FeatureConfig | None = None,
        split_config: SplitConfig | None = None,
    ) -> BenchmarkResult:
        """Fit every candidate, select one on validation, and evaluate it on test."""
        return run_benchmark(
            dataset_path,
            out_dir=out_dir,
            feature_config=feature_config or FeatureConfig(),
            split_config=split_config or SplitConfig(),
        )
