# RouteLLM

RouteLLM is a Python, terminal-first learning project for building an explainable local routing layer for natural-language requests. Its baseline will use rules, NLP, TF-IDF/n-grams, and traditional machine learning before any LLM provider is considered.

## Status

In development. **Milestone 0: Python project foundation** is complete — the package installs and the `routellm` command works. Routing, categories, keyword rules, datasets, ML models, and Ollama integration are not implemented yet (they belong to later milestones).

## Planned technology stack

- Python 3.11+
- Hatchling packaging
- pytest, pytest-cov, and Ruff for development
- Later milestones: spaCy and scikit-learn

## Local setup

Create and activate a virtual environment, then install the project with its development dependencies after Milestone 0 creates the package:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Tests

Run the test suite with:

```bash
pytest
```

## CLI

The `routellm` console command is a skeleton implemented in Milestone 0. It currently supports only `--help` and `--version`; routing subcommands arrive in later milestones.

```bash
routellm --help
routellm --version
```

Running `routellm` with no arguments prints the help text. The package can also be run as a module:

```bash
python -m routellm --help
```

## Intended structure

```text
src/routellm/   Application package (created in Milestone 0)
tests/          Test suite (created in Milestone 0)
data/           Versioned datasets and local processed artifacts (Milestone 2)
models/         Generated local ML artifacts (not committed)
```

See [SPEC.md](SPEC.md), [ARCHITECTURE.md](ARCHITECTURE.md), and [ROADMAP.md](ROADMAP.md) for the project contract and milestone sequence.
