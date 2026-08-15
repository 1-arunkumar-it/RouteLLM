"""The baseline statistical classifier.

Logistic Regression is the single traditional classifier introduced in
Milestone 2. It produces a probability estimate for every class, which is
reported as uncalibrated confidence (SPEC section 20). Benchmarking it
against Linear SVM and Naive Bayes is Milestone 3 work.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from routellm.classification.dataset import Dataset, SplitConfig
from routellm.classification.features import FeatureConfig, build_vectorizer
from routellm.domain.classifier_prediction import ClassifierPrediction


@dataclass
class TrainedClassifier:
    """A fitted TF-IDF vectorizer, classifier, and training metadata."""

    vectorizer: TfidfVectorizer
    model: LogisticRegression
    classes: tuple[str, ...]
    feature_config: FeatureConfig
    split_config: SplitConfig
    dataset_path: str
    dataset_fingerprint: str
    trained_at: str

    def predict(self, text: str) -> ClassifierPrediction:
        """Classify a single prompt."""
        return self.predict_batch((text,))[0]

    def predict_batch(self, texts: tuple[str, ...]) -> list[ClassifierPrediction]:
        """Classify many prompts in one vectorized pass."""
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


def train_classifier(
    dataset: Dataset,
    feature_config: FeatureConfig = FeatureConfig(),
    split_config: SplitConfig = SplitConfig(),
    dataset_path: str = "",
    dataset_fingerprint: str = "",
) -> TrainedClassifier:
    """Fit a TF-IDF vectorizer and Logistic Regression model on a dataset."""
    vectorizer = build_vectorizer(feature_config)
    matrix = vectorizer.fit_transform(dataset.texts)
    model = LogisticRegression(max_iter=1000, random_state=split_config.seed)
    model.fit(matrix, dataset.categories)
    return TrainedClassifier(
        vectorizer=vectorizer,
        model=model,
        classes=tuple(str(class_name) for class_name in model.classes_),
        feature_config=feature_config,
        split_config=split_config,
        dataset_path=dataset_path,
        dataset_fingerprint=dataset_fingerprint,
        trained_at=datetime.now(timezone.utc).isoformat(),
    )


def save_classifier(classifier: TrainedClassifier, path: str | Path) -> None:
    """Persist a trained classifier to a local joblib file."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dump(classifier, output)


def load_classifier(path: str | Path) -> TrainedClassifier:
    """Load a trained classifier from a local joblib file."""
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"No trained classifier found at {model_path}.")
    classifier = load(model_path)
    if not isinstance(classifier, TrainedClassifier):
        raise ValueError(f"{path} does not contain a TrainedClassifier.")
    return classifier
