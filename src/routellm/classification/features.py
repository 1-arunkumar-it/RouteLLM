"""Configurable TF-IDF feature extraction."""

from dataclasses import dataclass
from typing import Callable

from sklearn.feature_extraction.text import TfidfVectorizer

from routellm.preprocessing import preprocessor

Tokenizer = Callable[[str], list[str]]


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for TF-IDF features.

    ``ngram_range`` selects word n-grams; the SPEC baseline is unigrams and
    bigrams. ``max_features`` caps the vocabulary size and ``min_df`` drops
    tokens seen in too few documents.
    """

    ngram_range: tuple[int, int] = (1, 2)
    max_features: int | None = None
    min_df: int | float = 1

    def __post_init__(self) -> None:
        ngram_min, ngram_max = self.ngram_range
        if ngram_min < 1:
            raise ValueError(f"ngram_range lower bound must be >= 1, got {ngram_min}.")
        if ngram_max < ngram_min:
            raise ValueError(
                f"ngram_range upper bound must be >= lower bound, got {self.ngram_range}."
            )
        if self.max_features is not None and self.max_features < 1:
            raise ValueError(f"max_features must be >= 1, got {self.max_features}.")
        if isinstance(self.min_df, int) and self.min_df < 1:
            raise ValueError(f"integer min_df must be >= 1, got {self.min_df}.")
        if isinstance(self.min_df, float) and not 0 < self.min_df <= 1:
            raise ValueError(f"fractional min_df must be in (0, 1], got {self.min_df}.")


def build_vectorizer(
    config: FeatureConfig = FeatureConfig(),
    tokenizer: Tokenizer = preprocessor.tokenize,
) -> TfidfVectorizer:
    """Build a TF-IDF vectorizer using RouteLLM preprocessing."""
    return TfidfVectorizer(
        ngram_range=config.ngram_range,
        max_features=config.max_features,
        min_df=config.min_df,
        lowercase=False,
        tokenizer=tokenizer,
        token_pattern=None,
    )
