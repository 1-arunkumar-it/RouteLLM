"""End-to-end tests for complexity-aware routing (Milestone 5)."""

import pytest

from routellm.application.route_service import RouteService
from routellm.classification.cascade_model import fit_cascade
from routellm.classification.dataset import Dataset
from routellm.cli import main as cli_main
from routellm.cli.render import format_decision
from routellm.complexity.config import ComplexityConfig
from routellm.evaluation.complexity import evaluate_complexity


def _write_synthetic_complexity(path):
    rows = [
        ("What is 2 plus 2", "low"),
        ("What is the capital of Japan", "low"),
        (
            "Prove why the sum of an infinite geometric series converges "
            "only when the ratio is below one",
            "high",
        ),
        (
            "Why is the sky blue during the day and why does it turn red at "
            "sunset because of atmospheric scattering and how do we measure "
            "that evidence",
            "high",
        ),
        (
            "Write a Python function that sorts a list of dictionaries by a key",
            "medium",
        ),
    ]
    path.write_text(
        "text,complexity\n"
        + "".join(f'"{text}","{level}"\n' for text, level in rows),
        encoding="utf-8",
    )


def test_complexity_subcommand_prints_report(capsys, tmp_path):
    dataset = tmp_path / "complexity.csv"
    _write_synthetic_complexity(dataset)
    exit_code = cli_main.main(["complexity", "--dataset", str(dataset)])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Complexity estimator evaluation" in output
    assert "Confusion matrix" in output
    assert "Route-policy behavior" in output
    assert "Re-routed at 'high' complexity:" in output


def test_complexity_subcommand_missing_dataset_returns_one(capsys, tmp_path):
    exit_code = cli_main.main(["complexity", "--dataset", str(tmp_path / "missing.csv")])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "complexity failed:" in captured.err


def test_route_service_attaches_complexity_to_decision():
    decision = RouteService().route("What is the capital of Japan")
    assert decision.complexity is not None
    assert decision.complexity.level == "low"
    assert decision.complexity.signals


def test_high_complexity_general_qa_routes_to_reasoning():
    decision = RouteService().route(
        "Why is the sky blue during the day and why does it turn red at sunset "
        "because of atmospheric scattering and how do we measure that evidence"
    )
    assert decision.complexity is not None
    assert decision.complexity.level == "high"
    assert decision.route == "reasoning"


def test_high_complexity_summarization_routes_to_reasoning():
    decision = RouteService().route(
        "Condense this long report into a summary because the original document "
        "spans many sections and we need to analyze the evidence and compare the "
        "findings and evaluate the implications and explain the methodology"
    )
    assert decision.complexity is not None
    assert decision.complexity.level == "high"
    assert decision.route == "reasoning"


def test_low_complexity_general_qa_stays_local():
    decision = RouteService().route("What is the capital of Japan")
    assert decision.complexity is not None
    assert decision.complexity.level == "low"
    assert decision.route == "general-local"


def test_high_complexity_coding_stays_coding_local():
    decision = RouteService().route("Write a Java Spring Boot REST API for user authentication")
    assert decision.complexity is not None
    assert decision.complexity.level == "high"
    assert decision.route == "coding-local"


def test_unknown_prompt_is_never_rerouted():
    decision = RouteService().route("The zebra florbled quizzically at dusk")
    assert decision.complexity is not None
    assert decision.category == "unknown"
    assert decision.route == "fallback"


def test_format_decision_includes_complexity():
    decision = RouteService().route("What is the capital of Japan")
    rendered = format_decision(decision)
    assert "Complexity    : low" in rendered
    assert "Complexity signals:" in rendered


def test_evaluate_complexity_reports_expected_fields(tmp_path):
    dataset = tmp_path / "complexity.csv"
    _write_synthetic_complexity(dataset)
    report = evaluate_complexity(str(dataset))
    assert report.n_prompts == 5
    assert 0.0 <= report.accuracy <= 1.0
    assert report.classes == ("low", "medium", "high")
    assert len(report.per_class) == 3
    assert isinstance(report.n_rerouted, int)
    assert report.routing_source == "rules"


def test_evaluate_complexity_can_run_under_cascade_routing(tmp_path):
    dataset = tmp_path / "complexity.csv"
    _write_synthetic_complexity(dataset)
    model = fit_cascade(_keyword_dataset())
    report = evaluate_complexity(str(dataset), model=model)
    assert report.n_prompts == 5
    assert 0.0 <= report.accuracy <= 1.0
    assert report.routing_source == "cascade"
    assert isinstance(report.n_rerouted, int)


def test_route_service_accepts_custom_complexity_config():
    config = ComplexityConfig(low_threshold=0.01, high_threshold=0.02)
    decision = RouteService(complexity_config=config).route("What is the capital of Japan")
    assert decision.complexity is not None
    assert decision.complexity.level == "high"
    assert decision.route == "reasoning"


def test_route_service_rejects_non_config_complexity_config():
    with pytest.raises(ValueError, match="ComplexityConfig"):
        RouteService(complexity_config={"levels": ("low", "medium", "high")})


def _keyword_dataset() -> Dataset:
    texts = []
    labels = []
    for category in ("coding", "math", "translation"):
        for index in range(12):
            texts.append(f"sample {category} prompt {index}")
            labels.append(category)
    return Dataset(texts=tuple(texts), categories=tuple(labels))
