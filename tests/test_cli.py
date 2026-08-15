"""Tests for the ``routellm`` command-line interface."""

import importlib.metadata
import runpy
import subprocess
import sys

import pytest

from routellm import __version__
from routellm.classification.dataset import Dataset, write_dataset
from routellm.cli import main as cli_main


def _run_module(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "routellm", *args],
        capture_output=True,
        text=True,
    )


def test_help_shows_usage_and_exits_zero(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli_main.main(["--help"])
    output = capsys.readouterr().out
    assert excinfo.value.code == 0
    assert "usage: routellm" in output


def test_version_prints_version_and_exits_zero(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli_main.main(["--version"])
    output = capsys.readouterr().out
    assert excinfo.value.code == 0
    assert output.strip() == f"routellm {__version__}"


def test_no_arguments_prints_help_and_returns_zero(capsys):
    exit_code = cli_main.main([])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "usage: routellm" in output


def test_invalid_argument_exits_with_code_two_and_uses_stderr(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli_main.main(["--definitely-not-an-option"])
    captured = capsys.readouterr()
    assert excinfo.value.code == 2
    assert "--definitely-not-an-option" in captured.err
    assert "usage: routellm" in captured.err


def test_python_m_routellm_help_exits_zero():
    result = _run_module("--help")
    assert result.returncode == 0
    assert "usage: routellm" in result.stdout


def test_python_m_routellm_version_exits_zero():
    result = _run_module("--version")
    assert result.returncode == 0
    assert result.stdout.strip() == f"routellm {__version__}"


def test_module_execution_propagates_return_code(monkeypatch):
    monkeypatch.setattr(cli_main, "main", lambda argv=None: 3)
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("routellm", run_name="__main__")
    assert excinfo.value.code == 3


def test_installed_distribution_version_matches_package():
    assert importlib.metadata.version("routellm") == __version__


def test_route_subcommand_prints_decision(capsys):
    exit_code = cli_main.main(["route", "write", "a", "python", "script"])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Category      : coding" in output
    assert "Route         : coding-local" in output
    assert "  python" in output


def test_route_subcommand_via_module():
    result = _run_module("route", "translate", "this", "into", "tamil")
    assert result.returncode == 0
    assert "Category      : translation" in result.stdout
    assert "Route         : translation" in result.stdout
    assert "  translate" in result.stdout
    assert "  tamil" in result.stdout


def test_unknown_prompt_routes_to_fallback(capsys):
    exit_code = cli_main.main(["route", "florb", "quizzically", "zzz"])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Category      : unknown" in output
    assert "Route         : fallback" in output


def test_route_requires_a_prompt():
    result = _run_module("route")
    assert result.returncode == 2
    assert "usage:" in result.stderr


def test_route_subcommand_help():
    result = _run_module("route", "--help")
    assert result.returncode == 0
    assert "usage: routellm route" in result.stdout


def test_route_subcommand_multiplication_expression(capsys):
    exit_code = cli_main.main(["route", "What is 25 × 48?"])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Category      : math" in output
    assert "Route         : calculator" in output
    assert "  25 × 48" in output


def _write_synthetic_dataset(path):
    texts = []
    categories = []
    for category in ("coding", "math", "translation"):
        for index in range(15):
            texts.append(f"sample {category} prompt {index}")
            categories.append(category)
    write_dataset(Dataset(texts=tuple(texts), categories=tuple(categories)), path)


def test_train_subcommand_writes_model(capsys, tmp_path):
    dataset_path = tmp_path / "prompts.csv"
    model_path = tmp_path / "classifier.joblib"
    _write_synthetic_dataset(dataset_path)
    exit_code = cli_main.main(["train", "--dataset", str(dataset_path), "--out", str(model_path)])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Training complete." in output
    assert "Validation metrics" in output
    assert model_path.exists()


def test_evaluate_subcommand_prints_report(capsys, tmp_path):
    dataset_path = tmp_path / "prompts.csv"
    model_path = tmp_path / "classifier.joblib"
    _write_synthetic_dataset(dataset_path)
    assert cli_main.main(["train", "--dataset", str(dataset_path), "--out", str(model_path)]) == 0
    capsys.readouterr()
    exit_code = cli_main.main(["evaluate", "--model", str(model_path)])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Evaluation on the held-out test split" in output
    assert "Confusion matrix" in output
    assert "Per-class metrics" in output


def test_evaluate_missing_model_returns_one(capsys, tmp_path):
    exit_code = cli_main.main(["evaluate", "--model", str(tmp_path / "missing.joblib")])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "evaluate failed:" in captured.err


def test_train_invalid_dataset_returns_one(capsys, tmp_path):
    missing = tmp_path / "missing.csv"
    exit_code = cli_main.main(["train", "--dataset", str(missing)])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "train failed:" in captured.err


def test_benchmark_subcommand_distinguishes_validation_and_test(capsys, tmp_path):
    dataset_path = tmp_path / "prompts.csv"
    _write_synthetic_dataset(dataset_path)
    exit_code = cli_main.main(
        ["benchmark", "--dataset", str(dataset_path), "--out", str(tmp_path / "bench")]
    )
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Benchmark complete." in output
    assert "Candidate metrics (validation split)" in output
    assert "linear_svm" in output
    assert "n/a" in output
    assert "Selected candidate on the held-out test split" in output
    assert (tmp_path / "bench" / "benchmark_report").exists() or any(
        (tmp_path / "bench").glob("benchmark_report_*.txt")
    )


def test_benchmark_invalid_dataset_returns_one(capsys, tmp_path):
    missing = tmp_path / "missing.csv"
    exit_code = cli_main.main(["benchmark", "--dataset", str(missing)])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "benchmark failed:" in captured.err
