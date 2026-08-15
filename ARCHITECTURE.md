# RouteLLM Architecture

## Purpose

RouteLLM is a local, explainable routing layer. The baseline categorizes a prompt and selects a logical route without calling an LLM or network service. It is deliberately layered so that later model providers do not alter core routing behavior. Routing proceeds from cheapest reliable mechanism to most expensive: validated keyword rules, then a calibrated classifier, then a fallback.

## Layers and dependency direction

```text
CLI -> Application -> Preprocessing / Signals / Classification -> Routing
                  \-> Domain

Providers (future) -> Domain + Configuration
```

The operational layers are CLI, Application, Domain, Preprocessing/Signals/Classification, Routing, and Providers. Dependencies point inward toward the domain: outer layers may depend on inner layers, while the domain must not depend on any outer layer or third-party ML/provider library.

| Layer | Responsibility | Allowed dependencies |
| --- | --- | --- |
| CLI | Parse commands and render results. | Application and standard presentation utilities only. |
| Application | Orchestrate one routing use case and return a structured result. | Domain, preprocessing, signals, classification, routing, configuration. |
| Domain | Define stable concepts such as `RouteDecision`, categories, routes, and signals. | Python standard library only. |
| Preprocessing | Normalize and prepare prompt text while preserving useful technical terms. | Domain and approved NLP libraries when introduced. |
| Signals | Extract keyword and phrase evidence for routing and explanation. | Domain, configuration, standard library. |
| Classification | Build features, train/predict with interchangeable traditional ML models, expose calibrated confidence metadata, and persist a `CascadeModel`. | Domain, configuration, approved ML/NLP libraries. |
| Routing | Apply the cascade policy to signal and classifier results: validated rules, then calibrated confidence against a validated threshold, then fallback. | Domain and configuration; never providers. |
| Providers | Map a logical route to executable local/remote services. Deferred until Milestone 6. | Domain, configuration, provider-specific clients. |

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
  -> routing policy
  -> RouteDecision
  -> CLI rendering
```

`RouteDecision` is the application boundary: it contains the selected category, confidence metadata, logical route, and evidence needed for truthful explanation. The renderer does not infer or invent rationale.

### Cascade policy

`RouteDecision.source` records which mechanism made the decision, so the reason never overstates the evidence:

1. **Rules** — keyword signals decide immediately when the resulting category had at least `rule_override_min_precision` (default 0.90) precision on the validation split. Confidence is absent (`None`); a rule match is not presented as probability.
2. **Classifier** — otherwise the benchmark-selected Linear SVM, wrapped in `CalibratedClassifierCV`, predicts a category. The calibrated probability must be at or above a threshold chosen from the validation split (maximizing cascade macro F1 over a 0.05 grid, ties resolved toward the higher threshold).
3. **Fallback** — below threshold, the decision is `unknown`/`fallback`. The real confidence is still reported so uncertainty is never hidden.

Training calibrates on the train split only; rule precision and threshold selection use the validation split; the test split measures the final cascade (metrics, override rate, fallback rate). `RouteService` remains rule-only when no model is provided.

Future execution flow is separate:

```text
RouteDecision
  -> provider registry
  -> selected provider/model
```

Before providers exist, a logical route is a label, not an instruction to execute a model.

## Architectural constraints

- Normal classification performs no network activity.
- Baseline routing requires no LLM.
- Imports must not download spaCy models, ML artifacts, or provider resources.
- The domain layer must remain independent of spaCy, scikit-learn, Ollama, provider clients, and CLI code.
- CLI rendering contains no business logic.
- Configuration must be validated and must not contain secrets; secrets are never committed or logged.
- Generated ML artifacts must not be committed.
- Confidence thresholds and rule overrides must be justified by held-out evaluation, not assumed certainty.
- spaCy must be benchmarked against simpler preprocessing before it is retained as a baseline dependency.
