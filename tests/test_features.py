"""Tests for configurable TF-IDF feature extraction."""

import pytest

from routellm.classification.features import FeatureConfig, build_vectorizer


def _fit_features(texts, config):
    vectorizer = build_vectorizer(config)
    vectorizer.fit_transform(texts)
    return set(vectorizer.get_feature_names_out())


def test_vectorizer_includes_unigrams_and_bigrams():
    features = _fit_features(("write rest api", "write java code"), FeatureConfig())
    assert "rest" in features
    assert "rest api" in features
    assert "java code" in features


def test_vectorizer_preserves_technical_terms():
    features = _fit_features(("write a c++ program in c#",), FeatureConfig())
    assert "c++" in features
    assert "c#" in features


def test_vectorizer_uses_routellm_tokenizer():
    features = _fit_features(("Translate this into Tamil.",), FeatureConfig())
    assert "tamil" in features
    assert "tamil." not in features


def test_max_features_caps_vocabulary():
    config = FeatureConfig(max_features=5)
    features = _fit_features(("alpha beta gamma", "delta epsilon"), config)
    assert len(features) <= 5


def test_min_df_drops_rare_tokens():
    config = FeatureConfig(min_df=0.5, max_features=20)
    features = _fit_features(("alpha beta", "alpha gamma", "delta"), config)
    assert "alpha" in features
    assert "delta" not in features


def test_feature_config_validation():
    with pytest.raises(ValueError, match="ngram_range lower bound"):
        FeatureConfig(ngram_range=(0, 2))
    with pytest.raises(ValueError, match="upper bound"):
        FeatureConfig(ngram_range=(2, 1))
    with pytest.raises(ValueError, match="max_features"):
        FeatureConfig(max_features=0)
    with pytest.raises(ValueError, match="min_df"):
        FeatureConfig(min_df=0)
    with pytest.raises(ValueError, match="min_df"):
        FeatureConfig(min_df=1.5)
