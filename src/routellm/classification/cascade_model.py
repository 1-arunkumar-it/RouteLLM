"""The calibrated cascade model (Milestone 4).

The benchmark selected Linear SVM, which exposes decision margins but not
probabilities. The cascade model wraps ``LinearSVC`` in
``CalibratedClassifierCV`` so every reported confidence is a calibrated
probability estimate rather than a raw margin (SPEC section 20).
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from joblib import dump, load
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from routellm.classification.dataset import Dataset, SplitConfig
from routellm.classification.features import FeatureConfig, build_vectorizer
from routellm.configuration.cascade import CascadeConfig
from routellm.domain.classifier_prediction import ClassifierPrediction


@dataclass
class CascadeModel:
    """A fitted vectorizer, calibrated Linear SVM, and validated cascade settings.

    ``threshold`` is chosen from held-out validation data. ``override_categories``
    lists categories whose rule decisions were precise enough on validation to
    bypass the classifier (SPEC section 48). ``confidence`` values produced by
    this model are calibrated probabilities.
    """

    vectorizer: TfidfVectorizer
    model: CalibratedClassifierCV
    classes: tuple[str, ...]
    threshold: float
    override_categories: frozenset[str]
    rule_precision: dict[str, float]
    validation_macro_f1: float
    feature_config: FeatureConfig
    split_config: SplitConfig
    cascade_config: CascadeConfig
    dataset_path: str
    dataset_fingerprint: str
    trained_at: str

    def predict(self, text: str) -> ClassifierPrediction:
        """Classify a single prompt."""
        return self.predict_batch((text,))[0]

    def predict_batch(self, texts: tuple[str, ...]) -> list[ClassifierPrediction]:
        """Classify many prompts in one vectorized pass with calibrated scores."""
        matrix = self.vectorizer.transform(texts)
        probabilities = self.model.predict_proba(matrix)
        predictions = []
        for text, row in zip(texts, probabilities):
            scores = tuple(
                sorted(
                    zip(self.classes, row.tolist()),
                    key=lambda item: item[1],
                    reverse=True,
                )
            )
            predictions.append(
                ClassifierPrediction(
                    text=text,
                    category=scores[0][0],
                    confidence=scores[0][1],
                    scores=scores,
                )
            )
        return predictions


def fit_cascade(
    dataset: Dataset,
    feature_config: FeatureConfig = FeatureConfig(),
    split_config: SplitConfig = SplitConfig(),
    cascade_config: CascadeConfig = CascadeConfig(),
    dataset_path: str = "",
    dataset_fingerprint: str = "",
) -> CascadeModel:
    """Fit the TF-IDF vectorizer and a calibrated Linear SVM on a dataset.

    Calibration uses internal cross-validation on the provided dataset only,
    so a separate validation split can later set the threshold and a test
    split can measure the result without leakage.
    """
    vectorizer = build_vectorizer(feature_config)
    matrix = vectorizer.fit_transform(dataset.texts)
    estimator = LinearSVC(random_state=split_config.seed)
    calibrated = CalibratedClassifierCV(
        estimator=estimator,
        method=cascade_config.calibration_method,
        cv=cascade_config.calibration_cv,
    )
    calibrated.fit(matrix, dataset.categories)
    return CascadeModel(
        vectorizer=vectorizer,
        model=calibrated,
        classes=tuple(str(class_name) for class_name in calibrated.classes_),
        threshold=0.0,
        override_categories=frozenset(),
        rule_precision={},
        validation_macro_f1=0.0,
        feature_config=feature_config,
        split_config=split_config,
        cascade_config=cascade_config,
        dataset_path=dataset_path,
        dataset_fingerprint=dataset_fingerprint,
        trained_at=datetime.now(timezone.utc).isoformat(),
    )


def save_cascade_model(model: CascadeModel, path: str | Path) -> None:
    """Persist a trained cascade model to a local joblib file."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dump(model, output)


def load_cascade_model(path: str | Path) -> CascadeModel:
    """Load a trained cascade model from a local joblib file."""
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"No cascade model found at {model_path}.")
    model = load(model_path)
    if not isinstance(model, CascadeModel):
        raise ValueError(f"{path} does not contain a CascadeModel.")
    return model
