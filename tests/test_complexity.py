"""Tests for the complexity estimator, config, and dataset (Milestone 5)."""

import pytest

from routellm.complexity.config import (
    DEFAULT_FEATURE_CAPS,
    DEFAULT_FEATURE_WEIGHTS,
    ComplexityConfig,
)
from routellm.complexity.dataset import load_complexity_dataset
from routellm.complexity.estimator import estimate
from routellm.domain.complexity import ComplexityEstimate


def test_config_defaults_are_valid():
    config = ComplexityConfig()
    assert config.levels == ("low", "medium", "high")
    assert config.low_threshold == 0.30
    assert config.high_threshold == 0.55
    assert dict(config.feature_weights) == {"length": 0.55, "signals": 0.45}
    assert dict(config.feature_caps) == {"length": 20, "signals": 6}
    assert dict(DEFAULT_FEATURE_WEIGHTS) == {"length": 0.55, "signals": 0.45}
    assert dict(DEFAULT_FEATURE_CAPS) == {"length": 20, "signals": 6}


def test_config_is_deeply_immutable():
    config = ComplexityConfig()
    with pytest.raises(TypeError):
        config.feature_weights["length"] = 0.9
    with pytest.raises(TypeError):
        config.feature_caps["length"] = 1
    with pytest.raises(TypeError):
        DEFAULT_FEATURE_WEIGHTS["length"] = 0.9


def test_config_passed_mappings_are_copied_not_shared():
    weights = {"length": 0.55, "signals": 0.45}
    config = ComplexityConfig(feature_weights=weights)
    weights["length"] = 0.9
    assert config.feature_weights["length"] == 0.55


@pytest.mark.parametrize(
    "levels",
    [
        ("low", "high"),
        ("low", "low", "high"),
        ("low", "medium", "critical"),
        ("simple", "medium", "hard"),
    ],
)
def test_config_rejects_invalid_levels(levels):
    with pytest.raises(ValueError, match="levels"):
        ComplexityConfig(levels=levels)


@pytest.mark.parametrize("low,high", [(0.0, 0.5), (0.3, 0.3), (0.6, 0.5), (0.5, 1.0)])
def test_config_rejects_invalid_thresholds(low, high):
    with pytest.raises(ValueError, match="thresholds"):
        ComplexityConfig(low_threshold=low, high_threshold=high)


def test_config_rejects_mismatched_weights_and_caps():
    with pytest.raises(ValueError, match="same features"):
        ComplexityConfig(feature_weights={"length": 1.0}, feature_caps={"length": 5, "signals": 6})


def test_config_rejects_weights_not_summing_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        ComplexityConfig(feature_weights={"length": 0.5, "signals": 0.2})


def test_config_rejects_non_positive_weights():
    with pytest.raises(ValueError, match="positive"):
        ComplexityConfig(feature_weights={"length": 0.0, "signals": 1.0})


def test_config_rejects_small_caps():
    with pytest.raises(ValueError, match=">= 1"):
        ComplexityConfig(feature_caps={"length": 0, "signals": 6})


def test_empty_prompt_is_low_complexity():
    estimate_value = estimate("")
    assert estimate_value.level == "low"
    assert estimate_value.score == 0.0


def test_short_simple_prompt_is_low_complexity():
    estimate_value = estimate("What is 2 plus 2")
    assert estimate_value.level == "low"
    assert 0.0 <= estimate_value.score <= 0.3


def test_multi_step_prompt_is_medium_complexity():
    estimate_value = estimate("Explain why the sky appears blue during the day")
    assert estimate_value.level == "medium"
    assert 0.3 <= estimate_value.score < 0.55


def test_reasoning_heavy_prompt_is_high_complexity():
    estimate_value = estimate(
        "Prove why the sum of an infinite geometric series converges "
        "only when the ratio is below one"
    )
    assert estimate_value.level == "high"
    assert 0.55 <= estimate_value.score <= 1.0


def test_estimate_is_deterministic():
    prompt = "Compare the advantages of SQL and NoSQL databases for analytics"
    first = estimate(prompt)
    second = estimate(prompt)
    assert first.level == second.level
    assert first.score == second.score
    assert first.signals == second.signals


def test_score_always_in_unit_interval():
    for text in (
        "hi",
        "Translate this into French and Spanish for a meeting tomorrow",
        "Analyze the failure modes of a distributed database under network partitions",
    ):
        score = estimate(text).score
        assert 0.0 <= score <= 1.0


def test_signals_are_truthful():
    estimate_value = estimate("Implement and test a rest api endpoint")
    joined = " ".join(estimate_value.signals)
    assert "rest api" in joined
    assert "implement" in joined
    assert isinstance(estimate_value, ComplexityEstimate)


def test_multi_token_indicator_requires_contiguous_sequence():
    low = estimate("set it up for later")
    assert low.level == "low"
    high = estimate("write down the steps for setting up a load balancer for a rest api")
    assert high.level != "low"


def test_custom_config_controls_thresholds():
    permissive = ComplexityConfig(low_threshold=0.01, high_threshold=0.02)
    assert estimate("What is 2 plus 2", permissive).level == "high"
    strict = ComplexityConfig(low_threshold=0.95, high_threshold=0.99)
    assert estimate(
        "Prove why the sum of an infinite geometric series converges "
        "only when the ratio is below one",
        strict,
    ).level == "low"


# --- Dataset loader ---


def _write_complexity_dataset(path, rows):
    path.write_text(
        "text,complexity\n" + "".join(f'"{text}","{level}"\n' for text, level in rows),
        encoding="utf-8",
    )


def test_load_complexity_dataset(tmp_path):
    path = tmp_path / "complexity.csv"
    _write_complexity_dataset(path, [("What is 2 plus 2", "low"), ("Prove a theorem", "high")])
    dataset = load_complexity_dataset(str(path), ("low", "medium", "high"))
    assert len(dataset) == 2
    assert dataset.levels == ("low", "high")


def test_load_complexity_dataset_rejects_unknown_level(tmp_path):
    path = tmp_path / "complexity.csv"
    _write_complexity_dataset(path, [("Anything", "extreme")])
    with pytest.raises(ValueError, match="unknown complexity level"):
        load_complexity_dataset(str(path), ("low", "medium", "high"))


def test_load_complexity_dataset_rejects_bad_header(tmp_path):
    path = tmp_path / "complexity.csv"
    path.write_text("text,category\n\"x\",\"low\"\n", encoding="utf-8")
    with pytest.raises(ValueError, match="header"):
        load_complexity_dataset(str(path), ("low", "medium", "high"))


def test_load_complexity_dataset_rejects_empty_text(tmp_path):
    path = tmp_path / "complexity.csv"
    _write_complexity_dataset(path, [("", "low")])
    with pytest.raises(ValueError, match="empty text"):
        load_complexity_dataset(str(path), ("low", "medium", "high"))


def test_load_complexity_dataset_missing_file():
    with pytest.raises(ValueError, match="not found"):
        load_complexity_dataset("/nonexistent/complexity.csv", ("low", "medium", "high"))
