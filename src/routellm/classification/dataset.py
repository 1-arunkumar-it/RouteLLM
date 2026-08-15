"""Labeled dataset loading, deterministic splitting, and leakage checking.

The dataset is a CSV file with ``text`` and ``category`` columns. It is split
into disjoint training, validation, and test sets with a fixed seed. Cross-split
leakage is checked with a Jaccard similarity measure over normalized tokens.
"""

import csv
import hashlib
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from sklearn.model_selection import train_test_split

from routellm.domain.categories import CATEGORIES
from routellm.preprocessing import preprocessor


@dataclass(frozen=True)
class Dataset:
    """A labeled collection of prompts."""

    texts: tuple[str, ...]
    categories: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.texts)


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for the deterministic stratified split."""

    test_fraction: float = 0.15
    validation_fraction: float = 0.15
    seed: int = 42

    def __post_init__(self) -> None:
        if not 0 < self.test_fraction < 1:
            raise ValueError(f"test_fraction must be between 0 and 1, got {self.test_fraction}.")
        if not 0 < self.validation_fraction < 1:
            raise ValueError(
                f"validation_fraction must be between 0 and 1, got {self.validation_fraction}."
            )
        if self.test_fraction + self.validation_fraction >= 1:
            raise ValueError("test_fraction and validation_fraction must sum to less than 1.")
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}.")


@dataclass(frozen=True)
class SplitResult:
    """Disjoint train, validation, and test datasets."""

    train: Dataset
    validation: Dataset
    test: Dataset


def load_dataset(path: str | Path) -> Dataset:
    """Load a labeled CSV dataset and validate its rows."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise ValueError(f"Dataset file not found: {dataset_path}.")
    rows = []
    with dataset_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Dataset file is empty: {dataset_path}.")
        if [column.strip().lower() for column in header] != ["text", "category"]:
            raise ValueError(f"Dataset header must be 'text,category', got {header!r}.")
        for line, row in enumerate(reader, start=2):
            if len(row) != 2:
                raise ValueError(
                    f"Dataset {dataset_path}:{line} has {len(row)} columns, expected 2."
                )
            text, category = row[0].strip(), row[1].strip()
            if not text:
                raise ValueError(f"Dataset {dataset_path}:{line} has an empty text.")
            if category not in CATEGORIES:
                raise ValueError(
                    f"Dataset {dataset_path}:{line} has unknown category {category!r}."
                )
            rows.append((text, category))
    if not rows:
        raise ValueError(f"Dataset contains no labeled rows: {dataset_path}.")
    return Dataset(
        texts=tuple(text for text, _ in rows),
        categories=tuple(category for _, category in rows),
    )


def stratified_split(
    dataset: Dataset,
    config: SplitConfig = SplitConfig(),
) -> SplitResult:
    """Split a dataset into train, validation, and test sets.

    The test set is carved off first, then the remainder is divided into train
    and validation. Both splits are stratified and deterministic for a seed.
    """
    train_texts, test_texts, train_categories, test_categories = train_test_split(
        dataset.texts,
        dataset.categories,
        test_size=config.test_fraction,
        stratify=dataset.categories,
        random_state=config.seed,
    )
    relative_validation = config.validation_fraction / (1 - config.test_fraction)
    train_texts, validation_texts, train_categories, validation_categories = train_test_split(
        train_texts,
        train_categories,
        test_size=relative_validation,
        stratify=train_categories,
        random_state=config.seed + 1,
    )
    return SplitResult(
        train=Dataset(texts=tuple(train_texts), categories=tuple(train_categories)),
        validation=Dataset(texts=tuple(validation_texts), categories=tuple(validation_categories)),
        test=Dataset(texts=tuple(test_texts), categories=tuple(test_categories)),
    )


def _token_set(text: str) -> set[str]:
    return set(preprocessor.normalize(text).split())


def _jaccard_similarity(first: str, second: str) -> float:
    first_tokens, second_tokens = _token_set(first), _token_set(second)
    union = first_tokens | second_tokens
    if not union:
        return 1.0
    return len(first_tokens & second_tokens) / len(union)


def check_no_leakage(split: SplitResult, threshold: float = 0.9) -> None:
    """Raise ``ValueError`` if similar prompts appear in different splits."""
    parts = (("train", split.train), ("validation", split.validation), ("test", split.test))
    for (first_name, first), (second_name, second) in combinations(parts, 2):
        for first_index, first_text in enumerate(first.texts):
            for second_index, second_text in enumerate(second.texts):
                score = _jaccard_similarity(first_text, second_text)
                if score >= threshold:
                    raise ValueError(
                        f"Leakage detected: {first_name}[{first_index}] and "
                        f"{second_name}[{second_index}] have similarity {score:.2f}: "
                        f"{first_text!r} / {second_text!r}."
                    )


def dataset_fingerprint(dataset: Dataset) -> str:
    """Return a stable content hash over the normalized dataset rows."""
    payload = "\n".join(
        f"{preprocessor.normalize(text)}\t{category}"
        for text, category in zip(dataset.texts, dataset.categories)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_dataset(dataset: Dataset, path: str | Path) -> None:
    """Write a dataset as a labeled CSV file."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["text", "category"])
        writer.writerows(zip(dataset.texts, dataset.categories))


def write_splits(
    split: SplitResult,
    out_dir: str | Path,
    dataset_path: str,
    config: SplitConfig,
) -> None:
    """Persist the split CSVs and provenance for inspection."""
    output_dir = Path(out_dir)
    for name, dataset in (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        write_dataset(dataset, output_dir / f"{name}.csv")
    provenance = {
        "dataset_path": str(dataset_path),
        "test_fraction": config.test_fraction,
        "validation_fraction": config.validation_fraction,
        "seed": config.seed,
        "n_train": len(split.train),
        "n_validation": len(split.validation),
        "n_test": len(split.test),
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
