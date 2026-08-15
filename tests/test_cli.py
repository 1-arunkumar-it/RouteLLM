"""Tests for the ``routellm`` command-line interface."""

import importlib.metadata
import runpy
import subprocess
import sys

import pytest

from routellm import __version__
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
