# RouteLLM

RouteLLM is a Python, terminal-first learning project for building an explainable local routing layer for natural-language requests. Its baseline will use rules, NLP, TF-IDF/n-grams, and traditional machine learning before any LLM provider is considered.

## Status

In development. **Milestone 1: rule-based routing**, **Milestone 2: dataset + TF-IDF classifier**, and **Milestone 3: classifier benchmarking** are complete. The `routellm route` command classifies obvious prompts into a small category taxonomy using configurable keyword rules and selects a logical route. A hand-authored labeled dataset (245 prompts, 7 categories) supports a Logistic Regression + TF-IDF baseline classifier (`routellm train` and `routellm evaluate`) and a benchmark comparing it fairly against Linear SVM and Naive Bayes (`routellm benchmark`). **Milestone 4** (cascaded routing) is not started and requires explicit human approval. spaCy and Ollama integration are not implemented yet.

## Planned technology stack

- Python 3.11+
- Hatchling packaging
- pytest, pytest-cov, and Ruff for development
- scikit-learn and joblib for statistical classification (Milestone 2)
- Later milestones: spaCy, after it is benchmarked against simpler preprocessing

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

The `routellm` command supports `--help` and `--version` (Milestone 0 skeleton) and a rule-based `route` subcommand (Milestone 1):

```bash
routellm route "Write a Java Spring Boot REST API"
routellm route "Translate this paragraph into Tamil"
```

The `route` command classifies a prompt into a small category taxonomy using keyword and phrase rules, selects a logical route, and prints the detected signals. Multiple words can also be passed unquoted:

```bash
routellm route write a python script
```

Running `routellm` with no arguments prints the help text. The package can also be run as a module:

```bash
python -m routellm route "Translate this into Tamil"
```

### Statistical classification (Milestone 2)

The `train` and `evaluate` commands train and measure the baseline classifier locally without any network access:

```bash
routellm train
routellm evaluate
```

- `routellm train` loads `data/datasets/prompts.csv`, splits it deterministically into train/validation/test sets, checks for leakage, fits a TF-IDF (word unigrams + bigrams) + Logistic Regression classifier, writes the model to `models/classifier.joblib`, and reports validation metrics. Options: `--dataset`, `--out`.
- `routellm evaluate` loads the trained model, runs the held-out test split, and reports accuracy, macro precision/recall/F1, a confusion matrix, mean inference latency, and the low-confidence rate. Options: `--model`, `--dataset`, `--threshold`.
- `routellm benchmark` fits Logistic Regression, Linear SVM, and Multinomial Naive Bayes on the same TF-IDF features and deterministic split, selects the candidate with the highest validation macro F1 (ties break by lower warm inference latency, then smaller serialized size), evaluates that candidate once on the held-out test split, and writes a full report under `models/benchmarks/`. Options: `--dataset`, `--out`. Selection uses only the validation split, and every candidate row shows measured metrics, latency, and model size. Linear SVM exposes decision margins, not probabilities, so its low-confidence rate is reported as `n/a`. Results are observations on the current dataset and methodology, not a claim that one model is generally best.

`routellm route` remains rule-based in this milestone; the classifier is built, persisted, and measured, but the cascade that merges rules, classifier scores, and confidence thresholds is a later milestone. Reported confidence is an uncalibrated probability estimate, not a trusted probability.

## Intended structure

```text
src/routellm/
├── cli/            Terminal interface: skeleton (M0), route subcommand (M1), train/evaluate (M2), benchmark (M3)
├── application/    Orchestrates the routing use case (M1) and classifier training/evaluation/benchmarking (M2/M3)
├── domain/         Stable concepts: categories, routes, signals, RouteDecision, ClassifierPrediction
├── preprocessing/  Prompt normalization and tokenization (M1)
├── signals/        Keyword/phrase rules and detection (M1)
├── classification/ Dataset, TF-IDF features, the baseline classifier, and benchmark candidates (M2/M3)
├── evaluation/     Classification metrics, reporting, and benchmarking (M2/M3)
└── routing/        Deterministic routing policy (M1)
tests/              Test suite
data/               Versioned datasets (data/datasets) and local processed artifacts (data/processed)
models/             Generated local ML artifacts (not committed)
```

See [SPEC.md](SPEC.md), [ARCHITECTURE.md](ARCHITECTURE.md), and [ROADMAP.md](ROADMAP.md) for the project contract and milestone sequence.
