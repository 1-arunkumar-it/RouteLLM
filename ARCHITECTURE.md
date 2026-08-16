# RouteLLM Architecture

## Purpose

RouteLLM is a local, explainable routing layer. The baseline categorizes a prompt and selects a logical route without calling an LLM or network service. It is deliberately layered so that later model providers do not alter core routing behavior. Routing proceeds from cheapest reliable mechanism to most expensive: validated keyword rules, then a calibrated classifier, then a fallback.

## Layers and dependency direction

```text
CLI -> Application -> Preprocessing / Signals / Classification -> Routing
                  \-> Domain

Providers -> Domain + Configuration
```

The operational layers are CLI, Application, Domain, Preprocessing/Signals/Classification, Routing, and Providers. Dependencies point inward toward the domain: outer layers may depend on inner layers, while the domain must not depend on any outer layer or third-party ML/provider library.

| Layer | Responsibility | Allowed dependencies |
| --- | --- | --- |
| CLI | Parse commands and render results. | Application and standard presentation utilities only. |
| Application | Orchestrate one routing use case and return a structured result. | Domain, preprocessing, signals, classification, complexity, routing, configuration. |
| Domain | Define stable concepts such as `RouteDecision`, categories, routes, and signals. | Python standard library only. |
| Preprocessing | Normalize and prepare prompt text while preserving useful technical terms. | Domain and approved NLP libraries when introduced. |
| Signals | Extract keyword and phrase evidence for routing and explanation. | Domain, configuration, standard library. |
| Classification | Build features, train/predict with interchangeable traditional ML models, expose calibrated confidence metadata, and persist a `CascadeModel`. | Domain, configuration, approved ML/NLP libraries. |
| Complexity | Estimate an ordinal prompt-complexity level from length and indicator signals; load the labeled evaluation set. | Domain, configuration, preprocessing, standard library. |
| Routing | Apply the cascade policy to signal and classifier results and complexity-aware route selection: validated rules, then calibrated confidence against a validated threshold, then fallback; re-route `general_qa`/`summarization` to `reasoning` at high complexity. | Domain and configuration; never providers. |
| Providers | Map a logical route to executable local/remote services. Milestone 6 implements a config-driven `ProviderRegistry` and an Ollama adapter (standard-library `urllib` only), plus single-hop unavailability fallback. | Domain, configuration, provider-specific clients. |

The CLI must never directly perform preprocessing, classification, routing, or provider calls. It calls the application layer, which orchestrates the use case and returns a `RouteDecision` for rendering.

Classifier implementations must conform to a small interchangeable prediction interface. Logistic Regression, Linear SVM, and Naive Bayes can then be benchmarked and replaced without modifying routing policy.

Providers are isolated from routing policy. Routing selects a logical label such as `coding-local`; a future provider registry resolves that label to a provider and model.

## Intended repository structure

```text
src/routellm/
├── cli/
├── application/
├── domain/
├── preprocessing/
├── signals/
├── classification/
├── complexity/
├── routing/
├── configuration/
├── evaluation/
├── providers/
└── __main__.py
```

`cli/` and the package root are expected in Milestone 0. `application/` is deferred to Milestone 1, when a routing use case exists for it to orchestrate. Other directories are created only when their milestone needs them; do not create empty packages as architectural placeholders.

Top-level `data/` holds raw, processed, and versioned dataset files when Milestone 2 begins. Top-level `models/` is for generated local model artifacts and is ignored by Git. `tests/` mirrors only the behavior that exists.

## Data flow

```text
Prompt
  -> preprocessing
  -> signal extraction
  -> classification
  -> confidence evaluation
  -> complexity estimation
  -> routing policy
  -> RouteDecision
  -> CLI rendering
```

`RouteDecision` is the application boundary: it contains the selected category, confidence metadata, logical route, complexity estimate, and evidence needed for truthful explanation. The renderer does not infer or invent rationale.

### Cascade policy

`RouteDecision.source` records which mechanism made the decision, so the reason never overstates the evidence:

1. **Rules** — keyword signals decide immediately when the resulting category had at least `rule_override_min_precision` (default 0.90) precision on the validation split. Confidence is absent (`None`); a rule match is not presented as probability.
2. **Classifier** — otherwise the benchmark-selected Linear SVM, wrapped in `CalibratedClassifierCV`, predicts a category. The calibrated probability must be at or above a threshold chosen from the validation split (maximizing cascade macro F1 over a 0.05 grid, ties resolved toward the higher threshold).
3. **Fallback** — below threshold, the decision is `unknown`/`fallback`. The real confidence is still reported so uncertainty is never hidden.

Training calibrates on the train split only; rule precision and threshold selection use the validation split; the test split measures the final cascade (metrics, override rate, fallback rate). `RouteService` remains rule-only when no model is provided.

### Complexity policy

`RouteDecision.complexity` is a `ComplexityEstimate` (level `low`/`medium`/`high`, composite score in `[0, 1]`, and the exact signals that produced it). The estimator in `routellm.complexity` uses only the existing preprocessing tokenizer: no LLM, no spaCy, no model file. The composite score blends a capped token count with a capped count of distinct matched indicators across the reasoning, operation, technical, code, and clause vocabularies; `ComplexityConfig` validates every weight, cap, and threshold at construction and stores its mapping fields as read-only proxies so the invariants cannot be invalidated afterward.

Complexity changes routing only through `COMPLEXITY_REROUTES`: at `high` complexity, `general_qa` and `summarization` route to the `reasoning` label. It never changes the category decision, never re-routes `unknown` (always `fallback`), and leaves every other category on its base route. `RouteService` validates the reroute matrix at construction. Quality is measured on the hand-labeled `data/datasets/complexity.csv` evaluation set via `routellm complexity`, which routes each prompt through `RouteService` and reports how often the estimate actually changed a route under the evaluated configuration (rule-only or cascade).

Execution flow is separate from routing (Milestone 6):

```text
RouteDecision
  -> provider registry
  -> selected provider/model
  -> Ollama adapter
```

Routing always selects a logical label such as `coding-local`. Execution is an optional downstream step: `ExecutionService` resolves that label through `ProviderRegistry` against a validated `ProviderConfig`, calls the Ollama adapter for the configured model, and reports a `ProviderResponse` (`ok`, `not_configured`, `unavailable`, or `error`). When the primary provider is unavailable, a configured single-hop fallback route is attempted before reporting `unavailable` (SPEC section 38). `RouteService` never touches providers, so the routing engine remains fully usable with no Ollama installation; a logical route is a label, not an instruction to execute a model, until execution is explicitly requested.

## Architectural constraints

- Normal classification performs no network activity.
- Baseline routing requires no LLM.
- Imports must not download spaCy models, ML artifacts, or provider resources.
- The domain layer must remain independent of spaCy, scikit-learn, Ollama, provider clients, and CLI code.
- CLI rendering contains no business logic.
- Configuration must be validated and must not contain secrets; secrets are never committed or logged.
- Generated ML artifacts must not be committed.
- Confidence thresholds and rule overrides must be justified by held-out evaluation, not assumed certainty.
- Complexity levels and thresholds must be justified by measured evaluation on the labeled set, not assumed.
- spaCy must be benchmarked against simpler preprocessing before it is retained as a baseline dependency.
