"""Benchmark classifier candidates.

Milestone 3 compares Logistic Regression, Linear SVM, and Multinomial Naive
Bayes on the same TF-IDF features, deterministic split, and seed. Logistic
Regression remains the named baseline; this module exists only to measure
candidates and must not change routing policy or the persisted baseline model.
"""

from dataclasses import dataclass
from typing import Callable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from routellm.classification.dataset import Dataset, SplitConfig
from routellm.classification.features import FeatureConfig, build_vectorizer
from routellm.domain.classifier_prediction import ClassifierPrediction

Estimator = LogisticRegression | LinearSVC | MultinomialNB


@dataclass(frozen=True)
class Candidate:
    """A named benchmark candidate with an estimator factory.

    ``probability_available`` states whether the estimator exposes probability
    estimates via ``predict_proba``. Linear SVM exposes only decision margins,
    which must never be labelled or rendered as probabilities or confidence.
    """

    name: str
    probability_available: bool
    estimator_factory: Callable[[int], Estimator]


CANDIDATES = (
    Candidate(
        "logistic_regression",
        probability_available=True,
        estimator_factory=lambda seed: LogisticRegression(
            max_iter=1000, random_state=seed
        ),
    ),
    Candidate(
        "linear_svm",
        probability_available=False,
        estimator_factory=lambda seed: LinearSVC(random_state=seed),
    ),
    Candidate(
        "multinomial_naive_bayes",
        probability_available=True,
        estimator_factory=lambda seed: MultinomialNB(),
    ),
)


@dataclass
class CandidateClassifier:
    """A fitted candidate conforming to the classifier protocol."""

    name: str
    probability_available: bool
    vectorizer: TfidfVectorizer
    model: Estimator
    classes: tuple[str, ...]
    feature_config: FeatureConfig
    split_config: SplitConfig
    dataset_path: str
    dataset_fingerprint: str

    def predict(self, text: str) -> ClassifierPrediction:
        """Classify a single prompt."""
        return self.predict_batch((text,))[0]

    def predict_batch(self, texts: tuple[str, ...]) -> list[ClassifierPrediction]:
        """Classify many prompts in one vectorized pass."""
        matrix = self.vectorizer.transform(texts)
        if self.probability_available:
            return self._predict_with_probabilities(texts, matrix)
        return self._predict_with_margins(texts, matrix)

    def _predict_with_probabilities(
        self, texts: tuple[str, ...], matrix
    ) -> list[ClassifierPrediction]:
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

    def _predict_with_margins(
        self, texts: tuple[str, ...], matrix
    ) -> list[ClassifierPrediction]:
        predicted = self.model.predict(matrix)
        margins = self.model.decision_function(matrix)
        predictions = []
        if len(self.classes) == 2:
            # Binary LinearSVC returns one signed margin per sample. sklearn's
            # convention is that a positive margin supports classes_[1] and the
            # negated margin supports classes_[0]. Convert to one raw score per
            # class so the prediction contract is identical to the multiclass
            # case. Raw margins must never be treated as probabilities.
            for text, category, margin in zip(texts, predicted, margins):
                margin = float(margin)
                scores = tuple(
                    sorted(
                        ((self.classes[0], -margin), (self.classes[1], margin)),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                )
                predictions.append(
                    ClassifierPrediction(
                        text=text,
                        category=category,
                        confidence=None,
                        scores=scores,
                    )
                )
            return predictions
        for text, category, row in zip(texts, predicted, margins):
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
                    category=category,
                    confidence=None,
                    scores=scores,
                )
            )
        return predictions


def fit_candidate(
    candidate: Candidate,
    dataset: Dataset,
    feature_config: FeatureConfig = FeatureConfig(),
    split_config: SplitConfig = SplitConfig(),
    dataset_path: str = "",
    dataset_fingerprint: str = "",
) -> CandidateClassifier:
    """Fit one candidate on the shared TF-IDF features and training corpus."""
    vectorizer = build_vectorizer(feature_config)
    matrix = vectorizer.fit_transform(dataset.texts)
    model = candidate.estimator_factory(split_config.seed)
    model.fit(matrix, dataset.categories)
    return CandidateClassifier(
        name=candidate.name,
        probability_available=candidate.probability_available,
        vectorizer=vectorizer,
        model=model,
        classes=tuple(str(class_name) for class_name in model.classes_),
        feature_config=feature_config,
        split_config=split_config,
        dataset_path=dataset_path,
        dataset_fingerprint=dataset_fingerprint,
    )
