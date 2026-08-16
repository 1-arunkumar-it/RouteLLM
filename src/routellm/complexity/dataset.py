"""Loading of the hand-labeled complexity evaluation dataset.

The labeled set (``data/datasets/complexity.csv``) holds ``text,complexity``
rows used only to measure the heuristic estimator. It is kept separate from
the category dataset so the Milestone 2 pipeline, splits, and fingerprints
stay untouched. It is an evaluation set, not a training set.
"""

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ComplexityDataset:
    """A labeled collection of prompts with expected complexity levels."""

    texts: tuple[str, ...]
    levels: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.texts)


def load_complexity_dataset(path: str | Path, valid_levels: tuple[str, ...]) -> ComplexityDataset:
    """Load a labeled CSV with a ``text,complexity`` header.

    Every level must be one of ``valid_levels``; empty texts and unknown
    levels raise ``ValueError``.
    """
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise ValueError(f"Dataset file not found: {dataset_path}.")
    rows = []
    with dataset_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Dataset file is empty: {dataset_path}.")
        if [column.strip().lower() for column in header] != ["text", "complexity"]:
            raise ValueError(f"Dataset header must be 'text,complexity', got {header!r}.")
        for line, row in enumerate(reader, start=2):
            if len(row) != 2:
                raise ValueError(
                    f"Dataset {dataset_path}:{line} has {len(row)} columns, expected 2."
                )
            text, level = row[0].strip(), row[1].strip()
            if not text:
                raise ValueError(f"Dataset {dataset_path}:{line} has an empty text.")
            if level not in valid_levels:
                raise ValueError(
                    f"Dataset {dataset_path}:{line} has unknown complexity level {level!r}."
                )
            rows.append((text, level))
    if not rows:
        raise ValueError(f"Dataset contains no labeled rows: {dataset_path}.")
    return ComplexityDataset(
        texts=tuple(text for text, _ in rows),
        levels=tuple(level for _, level in rows),
    )
