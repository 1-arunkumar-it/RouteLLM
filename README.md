# RouteLLM

RouteLLM is a Python, terminal-first learning project for building an explainable local routing layer for natural-language requests. Its baseline will use rules, NLP, TF-IDF/n-grams, and traditional machine learning before any LLM provider is considered.

## Status

In development. **Milestone 1: rule-based routing**, **Milestone 2: dataset + TF-IDF classifier**, **Milestone 3: classifier benchmarking**, **Milestone 4: cascaded routing**, and **Milestone 5: complexity estimation** are complete. The `routellm route` command classifies obvious prompts into a small category taxonomy using configurable keyword rules and selects a logical route. A hand-authored labeled dataset (245 prompts, 7 categories) supports a Logistic Regression + TF-IDF baseline classifier (`routellm train` and `routellm evaluate`) and a benchmark comparing it fairly against Linear SVM and Naive Bayes (`routellm benchmark`). The cascaded router (`routellm cascade`) combines validated rule evidence with a calibrated Linear SVM and a confidence threshold chosen from validation data, then `routellm route --model` routes through it with a truthful decision source. The complexity estimator (`routellm complexity`) blends prompt length and matched indicator signals into a `low`/`medium`/`high` level and re-routes high-complexity `general_qa`/`summarization` prompts to a `reasoning` route. spaCy and Ollama integration are not implemented yet.

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

### Cascaded routing (Milestone 4)

The `cascade` subcommand trains the cascaded router and `route --model` uses it:

```bash
routellm cascade
routellm route --model models/cascade.joblib "Why does the sky appear blue?"
routellm evaluate --model models/cascade.joblib
```

- `routellm cascade` splits the dataset deterministically, calibrates a Linear SVM (the benchmark-selected model) with `CalibratedClassifierCV`, measures rule precision per category on the validation split, selects a confidence threshold that maximizes cascade macro F1 on validation, and writes `models/cascade.joblib` plus a report. Options: `--dataset`, `--out`.
- `routellm route --model` applies the cascade: rule signals win for categories validated as precise (default 0.90 precision on held-out data); otherwise a calibrated confidence at or above the validated threshold accepts the classifier's category; anything else falls back to `unknown`/`fallback`. Every decision reports its `source` (`rules`, `classifier`, or `fallback`) so the reason stays truthful. Without `--model`, `route` stays rule-only.
- `routellm evaluate --model models/cascade.joblib` measures the full cascade on the held-out test split: standard metrics plus override rate and fallback rate.

Confidence reported by the cascade is a calibrated probability; the threshold is justified by validation results, not assumed.

### Complexity estimation (Milestone 5)

The `complexity` subcommand measures the heuristic complexity estimator on a hand-labeled set, and `route` decisions now report a complexity level:

```bash
routellm complexity
routellm complexity --model models/cascade.joblib
routellm route "Why does the sky appear blue and how do we measure that evidence"
```

- `routellm complexity` loads `data/datasets/complexity.csv` (90 labeled prompts, 30 per level), routes each prompt through the same path as `routellm route`, and reports accuracy, macro precision/recall/F1, a confusion matrix, mean latency, and how many route decisions actually changed. Options: `--dataset`, `--model` (measure route-policy impact under cascade routing; default is rule-only routing, labeled in the report).
- Complexity is an ordinal `low`/`medium`/`high` level from two explainable heuristic signals: prompt length (capped) and the number of distinct matched indicators across reasoning, operation, technical, code, and clause vocabularies (capped). The blend weights, caps, and thresholds are validated, deeply immutable design constants in `ComplexityConfig`.
- The level only affects route selection for `general_qa` and `summarization` at high complexity, which re-route to a `reasoning` route (SPEC section 41). The category decision, `unknown`/`fallback`, and all other categories are never changed by complexity.

Measured on the labeled set under rule-only routing: accuracy 0.867, macro F1 0.862, mean latency 0.25 ms; 11 of 90 prompts re-routed to `reasoning` (6 under cascade routing). No LLM or external model is involved.

## Intended structure

```text
src/routellm/
├── cli/            Terminal interface: skeleton (M0), route subcommand (M1), train/evaluate (M2), benchmark (M3), cascade (M4), complexity (M5)
├── application/    Orchestrates the routing use case (M1/M4/M5) and classifier training/evaluation/benchmarking (M2/M3)
├── domain/         Stable concepts: categories, routes, signals, RouteDecision, ClassifierPrediction, ComplexityEstimate (M5)
├── preprocessing/  Prompt normalization and tokenization (M1)
├── signals/        Keyword/phrase rules and detection (M1)
├── classification/ Dataset, TF-IDF features, the baseline classifier, benchmark candidates, and the calibrated cascade model (M2/M3/M4)
├── complexity/     Heuristic complexity config, estimator, and labeled-set loading (M5)
├── evaluation/     Classification metrics, reporting, benchmarking, rule precision, threshold selection, and complexity evaluation (M2/M3/M4/M5)
├── routing/        Deterministic routing policy (M1) and the cascade policy (M4), complexity-aware route selection (M5)
└── configuration/  Validated configuration, including CascadeConfig (M4)
tests/              Test suite
data/               Versioned datasets (data/datasets) and local processed artifacts (data/processed)
models/             Generated local ML artifacts (not committed)
```

See [SPEC.md](SPEC.md), [ARCHITECTURE.md](ARCHITECTURE.md), and [ROADMAP.md](ROADMAP.md) for the project contract and milestone sequence.
