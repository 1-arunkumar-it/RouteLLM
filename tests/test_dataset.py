"""Tests for labeled dataset loading, splitting, and leakage checking."""

from pathlib import Path

import pytest

from routellm.classification.dataset import (
    Dataset,
    SplitConfig,
    SplitResult,
    check_no_leakage,
    dataset_fingerprint,
    load_dataset,
    stratified_split,
    write_dataset,
)
from routellm.domain.categories import CATEGORIES

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DATASET = REPO_ROOT / "data" / "datasets" / "prompts.csv"


def _dataset(categories=("coding", "math", "translation"), per_class=12) -> Dataset:
    texts = []
    labels = []
    for category in categories:
        for index in range(per_class):
            texts.append(f"sample {category} prompt {index}")
            labels.append(category)
    return Dataset(texts=tuple(texts), categories=tuple(labels))


def test_load_dataset_roundtrip(tmp_path):
    path = tmp_path / "prompts.csv"
    write_dataset(_dataset(per_class=3), path)
    loaded = load_dataset(path)
    assert len(loaded) == 9
    assert loaded.categories.count("coding") == 3


def test_load_dataset_rejects_unknown_category(tmp_path):
    path = tmp_path / "prompts.csv"
    path.write_text('text,category\n"hello","not_a_category"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="unknown category"):
        load_dataset(path)


def test_load_dataset_rejects_missing_file(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        load_dataset(tmp_path / "missing.csv")


def test_load_dataset_rejects_empty_file(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_dataset(path)


def test_load_dataset_rejects_wrong_header(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text('question,label\n"hello","coding"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="header"):
        load_dataset(path)


def test_load_dataset_rejects_bad_row_width(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text('text,category\n"hello","coding","extra"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="columns"):
        load_dataset(path)


def test_stratified_split_is_disjoint():
    split = stratified_split(_dataset())
    train_texts = set(split.train.texts)
    validation_texts = set(split.validation.texts)
    test_texts = set(split.test.texts)
    assert train_texts & validation_texts == set()
    assert train_texts & test_texts == set()
    assert validation_texts & test_texts == set()


def test_stratified_split_is_deterministic():
    first = stratified_split(_dataset())
    second = stratified_split(_dataset())
    assert first.train.texts == second.train.texts
    assert first.validation.texts == second.validation.texts
    assert first.test.texts == second.test.texts


def test_stratified_split_keeps_every_category_in_every_split():
    split = stratified_split(_dataset())
    for part in (split.train, split.validation, split.test):
        assert set(part.categories) == {"coding", "math", "translation"}


def test_split_config_validation():
    with pytest.raises(ValueError, match="test_fraction"):
        SplitConfig(test_fraction=0.0)
    with pytest.raises(ValueError, match="validation_fraction"):
        SplitConfig(validation_fraction=1.5)
    with pytest.raises(ValueError, match="sum"):
        SplitConfig(test_fraction=0.6, validation_fraction=0.6)
    with pytest.raises(ValueError, match="seed"):
        SplitConfig(seed=-1)


def test_check_no_leakage_passes_for_distinct_prompts():
    split = SplitResult(
        train=Dataset(texts=("write rest api",), categories=("coding",)),
        validation=Dataset(texts=("write java code",), categories=("coding",)),
        test=Dataset(texts=("read python file",), categories=("coding",)),
    )
    check_no_leakage(split)


def test_check_no_leakage_raises_on_similar_prompts_across_splits():
    split = SplitResult(
        train=Dataset(texts=("write rest api",), categories=("coding",)),
        validation=Dataset(texts=("write java code",), categories=("coding",)),
        test=Dataset(texts=("rest api write",), categories=("coding",)),
    )
    with pytest.raises(ValueError, match="Leakage detected"):
        check_no_leakage(split)


def test_dataset_fingerprint_is_stable_and_distinct():
    dataset = _dataset()
    assert dataset_fingerprint(dataset) == dataset_fingerprint(dataset)
    other = Dataset(texts=("different text",), categories=("coding",))
    assert dataset_fingerprint(dataset) != dataset_fingerprint(other)


def test_canonical_dataset_loads_splits_and_has_no_leakage():
    dataset = load_dataset(CANONICAL_DATASET)
    split = stratified_split(dataset)
    check_no_leakage(split)
    assert set(dataset.categories) == set(CATEGORIES)
    for part in (split.train, split.validation, split.test):
        assert len(part) > 0
        assert set(part.categories) == set(CATEGORIES)
