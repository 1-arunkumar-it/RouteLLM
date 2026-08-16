"""Tests for the ``routellm`` command-line interface."""

import importlib.metadata
import runpy
import subprocess
import sys

import pytest

from routellm import __version__
from routellm.application.execution_service import ProviderStatusRow
from routellm.classification.dataset import Dataset, write_dataset
from routellm.cli import main as cli_main
from routellm.domain.provider import ProviderResponse


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


def _write_keyword_dataset(path):
    phrases = {
        "coding": (
            "write a python script",
            "fix the java bug",
            "build a rest api",
            "debug the docker container",
            "refactor this typescript class",
            "write a golang function",
            "query the sql database",
            "create a kubernetes deployment",
            "write a rust compiler",
            "implement binary search",
        ),
        "math": (
            "calculate the percentage",
            "what is the sum",
            "solve this equation",
            "compute the derivative",
            "divide the numbers",
            "multiply two values",
            "find the square root",
            "solve the algebra problem",
            "calculate the integral",
            "subtract the numbers",
        ),
        "translation": (
            "translate into tamil",
            "translate into hindi",
            "translate into telugu",
            "translate into spanish",
            "translate into german",
            "translate into japanese",
            "translate into chinese",
            "translate into korean",
            "translate into french",
            "translate this paragraph",
        ),
    }
    texts = []
    categories = []
    for category, category_phrases in phrases.items():
        for phrase in category_phrases:
            for index in range(2):
                texts.append(f"{phrase} {index}")
                categories.append(category)
    write_dataset(Dataset(texts=tuple(texts), categories=tuple(categories)), path)
    return path


def _write_complexity_dataset(path):
    rows = [
        ("What is 2 plus 2", "low"),
        ("What is the capital of Japan", "low"),
        ("Write a Python function that sorts a list of dictionaries by a key", "medium"),
    ]
    path.write_text(
        "text,complexity\n" + "".join(f'"{text}","{level}"\n' for text, level in rows),
        encoding="utf-8",
    )


def test_complexity_subcommand_with_cascade_model(capsys, tmp_path):
    dataset_path = _write_keyword_dataset(tmp_path / "prompts.csv")
    complexity_path = tmp_path / "complexity.csv"
    model_path = tmp_path / "cascade.joblib"
    _write_complexity_dataset(complexity_path)
    assert cli_main.main(["cascade", "--dataset", str(dataset_path), "--out", str(model_path)]) == 0
    capsys.readouterr()
    exit_code = cli_main.main(
        ["complexity", "--dataset", str(complexity_path), "--model", str(model_path)]
    )
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Route-policy behavior under the evaluated routing configuration" in output
    assert "cascade routing" in output


def test_complexity_subcommand_requires_cascade_model(capsys, tmp_path):
    dataset_path = _write_keyword_dataset(tmp_path / "prompts.csv")
    complexity_path = tmp_path / "complexity.csv"
    model_path = tmp_path / "classifier.joblib"
    _write_complexity_dataset(complexity_path)
    assert cli_main.main(
        ["train", "--dataset", str(dataset_path), "--out", str(model_path)]
    ) == 0
    capsys.readouterr()
    exit_code = cli_main.main(
        ["complexity", "--dataset", str(complexity_path), "--model", str(model_path)]
    )
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "complexity failed:" in captured.err
    assert "not a cascade model" in captured.err


def _ok_response(route="coding-local", model="qwen2.5-coder:3b", text="def f(): pass"):
    return ProviderResponse(
        requested_route=route,
        route=route,
        provider="ollama",
        model=model,
        status="ok",
        text=text,
        error="",
        latency_ms=12.5,
    )


def _unavailable_response(route="coding-local"):
    return ProviderResponse(
        requested_route=route,
        route=route,
        provider="ollama",
        model="qwen2.5-coder:3b",
        status="unavailable",
        text="",
        error="Provider for route 'coding-local' is unavailable and no fallback "
        "route is configured.",
        latency_ms=None,
    )


def _stub_execution_service(response, rows=()):
    class StubExecutionService:
        def __init__(self, config=None):
            self.config = config

        def execute(self, decision):
            return response

        def status_table(self):
            return rows

    return StubExecutionService


def test_run_subcommand_prints_model_output(monkeypatch, capsys):
    monkeypatch.setattr(cli_main, "ExecutionService", _stub_execution_service(_ok_response()))
    exit_code = cli_main.main(["run", "write", "a", "python", "script"])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Category      : coding" in output
    assert "Model output:" in output
    assert "def f(): pass" in output
    assert "Status        : ok" in output
    assert "Provider      : ollama" in output
    assert "Model         : qwen2.5-coder:3b" in output


def test_run_subcommand_reports_unavailable(monkeypatch, capsys):
    monkeypatch.setattr(
        cli_main, "ExecutionService", _stub_execution_service(_unavailable_response())
    )
    exit_code = cli_main.main(["run", "write", "a", "python", "script"])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Status        : unavailable" in output
    assert "Detail        : Provider for route 'coding-local' is unavailable" in output


def test_run_with_non_cascade_model_returns_one(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(cli_main, "ExecutionService", _stub_execution_service(_ok_response()))
    dataset_path = _write_keyword_dataset(tmp_path / "prompts.csv")
    model_path = tmp_path / "classifier.joblib"
    assert cli_main.main(
        ["train", "--dataset", str(dataset_path), "--out", str(model_path)]
    ) == 0
    capsys.readouterr()
    exit_code = cli_main.main(
        ["run", "--model", str(model_path), "write", "a", "python", "script"]
    )
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "run failed:" in captured.err
    assert "not a cascade model" in captured.err


def test_run_with_missing_config_returns_one(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(cli_main, "ExecutionService", _stub_execution_service(_ok_response()))
    exit_code = cli_main.main(
        ["run", "--config", str(tmp_path / "missing.toml"), "write", "a", "script"]
    )
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "run failed:" in captured.err


def test_providers_subcommand_prints_table(monkeypatch, capsys):
    rows = (
        ProviderStatusRow(
            route="coding-local", provider="ollama", model="qwen2.5-coder:3b", available=True
        ),
        ProviderStatusRow(route="calculator", provider=None, model=None, available=None),
    )
    monkeypatch.setattr(cli_main, "ExecutionService", _stub_execution_service(_ok_response(), rows))
    exit_code = cli_main.main(["providers"])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Provider configuration:" in output
    assert "coding-local" in output
    assert "yes" in output
    assert "calculator" in output
    assert "n/a" in output


def test_providers_with_invalid_config_returns_one(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(cli_main, "ExecutionService", _stub_execution_service(_ok_response()))
    bad = tmp_path / "bad.toml"
    bad.write_text("this is = not [valid toml", encoding="utf-8")
    exit_code = cli_main.main(["providers", "--config", str(bad)])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "providers failed:" in captured.err
