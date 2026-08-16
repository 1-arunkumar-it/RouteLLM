# RouteLLM

An explainable, local routing layer for natural-language requests that uses lightweight NLP and machine learning to route prompts to the right processor — before invoking an LLM.

## Highlights

- Routes natural-language prompts to the correct processing path using rule-based signals, TF-IDF classification, and calibrated confidence
- Achieves **97.3% accuracy** on a held-out test set with a cascaded routing policy
- Executes routed prompts through a local Ollama LLM with automatic fallback and provider health monitoring
- Fully explainable — every decision shows the exact signals, confidence score, and reasoning that produced it

## What it does

When a user sends a request like "Write a Python function to sort a list", RouteLLM:

1. Detects keyword signals (`python`, `function`)
2. Classifies the prompt as a **coding** request
3. Routes it to a local coding-optimized model
4. Executes the model and returns the response

For unfamiliar or ambiguous requests, it falls back gracefully rather than guessing. Routing itself never requires an LLM — the decision is made by lightweight NLP and statistical classification.

## Skills demonstrated

| Area | Technologies & Patterns |
|------|------------------------|
| **NLP** | TF-IDF, word n-grams, token preprocessing, keyword signal extraction |
| **Machine Learning** | Logistic Regression, Linear SVM, Naive Bayes, CalibratedClassifierCV |
| **Software Engineering** | Layered architecture, dependency injection, dataclass validation, immutable config |
| **Testing** | 310 tests, mocked HTTP seams, injectable adapters, deterministic splits |
| **CLI Design** | argparse, subcommands, structured output, TOML configuration |
| **LLM Integration** | Ollama REST API, provider fallback, health checks, cost/latency-aware routing |

## How it works

```
User Prompt
    |
    v
+-------------------+
|  Preprocessing     |  Normalize text, tokenize
+-------------------+
|  Signal Detection  |  Match keyword/phrase rules
+-------------------+
|  Classification    |  TF-IDF + calibrated ML model
+-------------------+
|  Cascade Policy    |  Rules -> Classifier -> Fallback
+-------------------+
|  Complexity        |  Heuristic: low / medium / high
+-------------------+
|  Route Selection   |  Map category -> logical route
+-------------------+
    |
    v
+-------------------+
|  Provider Layer    |  Resolve route -> Ollama model
+-------------------+
|  Execution         |  Call LLM, handle fallback
+-------------------+
    |
    v
  Response + Explanation
```

## Results

| Metric | Value |
|--------|-------|
| Test accuracy (cascade) | **97.3%** |
| Macro F1 (cascade) | **0.976** |
| Rule decision rate | 59.5% |
| Classifier decision rate | 40.5% |
| Fallback rate | 0.0% |
| Mean routing latency | 0.20 ms |
| Complexity accuracy | 86.7% |
| Test suite | 310 passing |

All metrics measured on held-out test data. No hard-coded results.

## Example

```bash
$ routellm route --model models/cascade.joblib "Write a Python function to sort a list"

Category      : coding
Route         : coding-local
Confidence    : n/a
Source        : rules
Complexity    : medium

Signals:
  python
  function

Complexity signals:
  length (8 tokens)
  operations (write)
  technical terms (python, function)

Reason: Rule signals (2 matched) selected category 'coding' with validated precision.
```

## Tech stack

- **Python 3.11+** — type hints, dataclasses, modern stdlib
- **scikit-learn** — TF-IDF, Logistic Regression, Linear SVM, Naive Bayes
- **Ollama** — local LLM execution (optional, not required for routing)
- **pytest** — 310 tests, fully offline with mocked providers
- **Ruff** — linting
- **Hatchling** — packaging

## Project structure

```
src/routellm/
  cli/              Terminal interface
  application/      Use-case orchestration
  domain/           Core data models
  preprocessing/    Text normalization
  signals/          Keyword/phrase detection
  classification/   ML training and prediction
  complexity/       Heuristic complexity estimation
  routing/          Cascade and constraint policy
  configuration/    Validated TOML config
  evaluation/       Metrics and benchmarking
  providers/        Ollama adapter and registry
tests/              310 tests
data/datasets/      Labeled evaluation data
```

## Running it

```bash
# Install
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Route a prompt
routellm route "Write a Python function to sort a list"

# Train the classifier
routellm train

# Run the cascade
routellm cascade
routellm route --model models/cascade.joblib "Explain photosynthesis"

# Run tests
pytest
```

## CLI commands

| Command | Description |
|---------|-------------|
| `routellm route` | Classify a prompt and select a route (rule-based) |
| `routellm route --model` | Route with the calibrated cascade classifier |
| `routellm run` | Route a prompt and execute it through the configured LLM |
| `routellm train` | Train the baseline classifier |
| `routellm evaluate` | Evaluate a classifier on the held-out test set |
| `routellm benchmark` | Compare multiple classifiers side by side |
| `routellm cascade` | Train the cascaded routing model |
| `routellm complexity` | Evaluate the complexity estimator |
| `routellm providers` | Show configured providers and availability |
| `routellm health` | Check provider health and response times |

## Configuration

Provider routes, fallbacks, cost profiles, and constraints are configured via TOML:

```toml
[ollama]
host = "http://localhost:11434"

[routes]
coding-local = "ollama:qwen2.5-coder:3b"
general-local = "ollama:qwen2.5-coder:3b"

[fallbacks]
coding-local = "general-local"

[profiles.coding-local]
cost_per_1k_tokens = 0.002
estimated_latency_ms = 150
capabilities = ["code", "reasoning"]

[constraints]
max_cost_per_prompt = 0.05
max_latency_ms = 500
```

## Documentation

- [SPEC.md](SPEC.md) — Full product specification
- [ARCHITECTURE.md](ARCHITECTURE.md) — Layered architecture design
- [ROADMAP.md](ROADMAP.md) — Milestone sequence and status
- [CHANGES.md](CHANGES.md) — Changelog

## License

MIT
