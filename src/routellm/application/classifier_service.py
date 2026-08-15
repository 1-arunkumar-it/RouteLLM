"""Application layer orchestrating classifier training and evaluation."""

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from joblib import load as joblib_load

from routellm.classification.cascade_model import (
    CascadeModel,
    fit_cascade,
    save_cascade_model,
)
from routellm.classification.classifier import (
    TrainedClassifier,
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
from routellm.configuration.cascade import CascadeConfig
from routellm.domain.classifier_prediction import ClassifierPrediction
from routellm.evaluation.benchmark import BenchmarkResult, run_benchmark
from routellm.evaluation.cascade_report import (
    CascadeEvaluationReport,
    CascadeTrainReport,
    format_cascade_train_report,
)
from routellm.evaluation.report import EvaluationReport, compute_metrics
from routellm.evaluation.rule_metrics import (
    rule_categories,
    rule_precision_by_category,
    select_override_categories,
)
from routellm.evaluation.threshold_selection import select_threshold
from routellm.routing.cascade import cascade_outcomes

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


def load_model(path: str | Path) -> TrainedClassifier | CascadeModel:
    """Load a trained model, accepting either the baseline or cascade type."""
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained classifier or cascade model found at {model_path}."
        )
    model = joblib_load(model_path)
    if isinstance(model, (TrainedClassifier, CascadeModel)):
        return model
    raise ValueError(
        f"{path} does not contain a supported model (TrainedClassifier or CascadeModel)."
    )


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

    def train_cascade(
        self,
        dataset_path: str,
        model_path: str,
        feature_config: FeatureConfig | None = None,
        split_config: SplitConfig | None = None,
        cascade_config: CascadeConfig | None = None,
    ) -> CascadeTrainReport:
        """Train a calibrated cascade model and persist it with its settings.

        The classifier is calibrated on the training split only. Rule
        precision and the confidence threshold are then measured on the
        validation split, so nothing here leaks test information.
        """
        feature_config = feature_config or FeatureConfig()
        split_config = split_config or SplitConfig()
        cascade_config = cascade_config or CascadeConfig()
        dataset = load_dataset(dataset_path)
        split = stratified_split(dataset, split_config)
        check_no_leakage(split)
        fingerprint = dataset_fingerprint(dataset)
        model = fit_cascade(
            dataset=split.train,
            feature_config=feature_config,
            split_config=split_config,
            cascade_config=cascade_config,
            dataset_path=str(Path(dataset_path)),
            dataset_fingerprint=fingerprint,
        )
        rule_cats = rule_categories(split.validation.texts)
        precision_by_category = rule_precision_by_category(
            split.validation.categories, rule_cats, model.classes
        )
        override_categories = select_override_categories(
            precision_by_category, cascade_config.rule_override_min_precision
        )
        predictions = model.predict_batch(split.validation.texts)
        threshold, validation_macro_f1 = select_threshold(
            rule_cats,
            predictions,
            split.validation.categories,
            model.classes,
            override_categories,
        )
        model.threshold = threshold
        model.override_categories = override_categories
        model.rule_precision = precision_by_category
        model.validation_macro_f1 = validation_macro_f1
        save_cascade_model(model, model_path)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        report_path = Path(model_path).parent / f"cascade_report_{timestamp}.txt"
        report = CascadeTrainReport(
            dataset_path=str(Path(dataset_path)),
            model_path=str(Path(model_path)),
            report_path=str(report_path),
            fingerprint=fingerprint,
            n_train=len(split.train),
            n_validation=len(split.validation),
            n_test=len(split.test),
            threshold=threshold,
            validation_macro_f1=validation_macro_f1,
            override_categories=override_categories,
            rule_precision=precision_by_category,
            calibration_method=cascade_config.calibration_method,
            calibration_cv=cascade_config.calibration_cv,
        )
        report_path.write_text(format_cascade_train_report(report) + "\n", encoding="utf-8")
        return report

    def evaluate(
        self,
        model_path: str,
        dataset_path: str | None = None,
        low_confidence_threshold: float = 0.80,
    ) -> EvaluationReport | CascadeEvaluationReport:
        """Evaluate a trained model on the held-out test split.

        Baseline models produce an ``EvaluationReport``; cascade models produce
        a ``CascadeEvaluationReport`` using the model's validated threshold.
        """
        model = load_model(model_path)
        if isinstance(model, CascadeModel):
            return self._evaluate_cascade(model, dataset_path)
        dataset_path = dataset_path or model.dataset_path
        dataset = load_dataset(dataset_path)
        if dataset_fingerprint(dataset) != model.dataset_fingerprint:
            raise ValueError(
                "The dataset differs from the one used to train this model. Retrain to continue."
            )
        split = stratified_split(dataset, model.split_config)
        start = time.perf_counter()
        predictions = model.predict_batch(split.test.texts)
        elapsed = time.perf_counter() - start
        return compute_metrics(
            y_true=split.test.categories,
            y_pred=tuple(prediction.category for prediction in predictions),
            confidences=tuple(prediction.confidence for prediction in predictions),
            classes=model.classes,
            low_confidence_threshold=low_confidence_threshold,
            elapsed_seconds=elapsed,
        )

    def _evaluate_cascade(
        self, model: CascadeModel, dataset_path: str | None
    ) -> CascadeEvaluationReport:
        """Measure the full cascade (rules + confidence + fallback) on test."""
        dataset_path = dataset_path or model.dataset_path
        dataset = load_dataset(dataset_path)
        if dataset_fingerprint(dataset) != model.dataset_fingerprint:
            raise ValueError(
                "The dataset differs from the one used to train this model. Retrain to continue."
            )
        split = stratified_split(dataset, model.split_config)
        rule_cats = rule_categories(split.test.texts)
        start = time.perf_counter()
        predictions = model.predict_batch(split.test.texts)
        elapsed = time.perf_counter() - start
        outcomes = cascade_outcomes(
            rule_cats,
            predictions,
            threshold=model.threshold,
            override_categories=model.override_categories,
        )
        y_pred = tuple(outcome.category for outcome in outcomes)
        metrics = compute_metrics(
            y_true=split.test.categories,
            y_pred=y_pred,
            confidences=tuple(outcome.confidence for outcome in outcomes),
            classes=model.classes,
            low_confidence_threshold=model.threshold,
            elapsed_seconds=elapsed,
        )
        n_prompts = len(split.test)
        n_rules = sum(1 for outcome in outcomes if outcome.source == "rules")
        n_classifier = sum(1 for outcome in outcomes if outcome.source == "classifier")
        n_fallback = n_prompts - n_rules - n_classifier
        return CascadeEvaluationReport(
            n_prompts=n_prompts,
            accuracy=metrics.accuracy,
            macro_precision=metrics.macro_precision,
            macro_recall=metrics.macro_recall,
            macro_f1=metrics.macro_f1,
            per_class=metrics.per_class,
            classes=metrics.classes,
            confusion=metrics.confusion,
            mean_latency_ms=metrics.mean_latency_ms,
            threshold=model.threshold,
            n_rules=n_rules,
            n_classifier=n_classifier,
            n_fallback=n_fallback,
            override_rate=n_rules / n_prompts,
            fallback_rate=n_fallback / n_prompts,
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
