# RouteLLM Roadmap

Milestone 3 is complete. Milestone 4 requires explicit human approval before implementation. Later milestones describe sequencing, not permission for speculative work.

## Milestone 0 — Python project foundation

**Objective:** Establish an installable, testable Python project foundation.

**Deliverables:** `pyproject.toml`; a minimal `src/routellm` package; console-entry-point-ready CLI skeleton; README; Git ignore rules; pytest structure.

**Prerequisites:** None.

**Acceptance criteria:** Project metadata targets Python 3.11+; `routellm --help` works after the CLI is implemented; tests are discoverable and pass; no routing functionality is claimed.

**Out of scope:** preprocessing, categories, keyword rules, datasets, ML, providers, interactive routing, and Ollama.

## Milestone 1 — Rule-based routing

**Objective:** Prove the end-to-end routing flow with deterministic, explainable rules.

**Deliverables:** prompt preprocessing, configurable keyword/phrase signals, a small taxonomy, deterministic policy, structured decisions, and terminal explanations.

**Prerequisites:** Milestone 0 complete.

**Acceptance criteria:** Supported obvious prompts produce a logical route and only actual detected signals are displayed; rule behavior has unit and integration tests.

**Out of scope:** supervised ML, trained confidence scores, subcategories beyond demonstrated need, complexity estimation, and provider execution.

## Milestone 2 — Dataset + TF-IDF classifier

**Objective:** Add a measurable supervised classification baseline.

**Deliverables:** labeled dataset policy and splits, TF-IDF word n-gram features, one traditional classifier, prediction persistence/loading, confidence metadata, and evaluation reporting.

**Prerequisites:** Milestone 1 behavior and taxonomy stabilized; dataset provenance and labeling guidance defined.

**Acceptance criteria:** Train/validation/test sets are disjoint and checked for leakage; metrics are generated from actual data; predictions work locally without an LLM.

**Out of scope:** classifier selection, rule/ML cascading, embedding methods, complexity, and providers.

## Milestone 3 — Classifier benchmarking

**Objective:** Select a baseline classifier from evidence.

**Deliverables:** comparable Logistic Regression, Linear SVM, and Naive Bayes experiments; accuracy, precision, recall, F1, latency, and model-size report; selection rationale.

**Prerequisites:** Milestone 2 dataset and repeatable evaluation harness.

**Acceptance criteria:** All candidates use equivalent splits/features where appropriate and results are measured, not hard-coded.

**Out of scope:** automatic production-style model selection, provider routing, embeddings, and complexity.

## Milestone 4 — Cascaded routing

**Objective:** Combine rule evidence and statistical prediction safely.

**Deliverables:** cascade policy, validated/calibrated confidence handling, configurable thresholds, unknown/fallback behavior, and explanation merging.

**Prerequisites:** Milestone 3 model choice and held-out evaluation of rules and confidence.

**Acceptance criteria:** Overrides and fallback decisions are measurable, configurable, and do not present uncertainty as certainty.

**Out of scope:** executable provider calls, cost optimization, embeddings, and complexity.

## Milestone 5 — Complexity estimation

**Objective:** Add a defined, lightweight complexity signal only after its purpose is specified.

**Deliverables:** documented output scale and routing use, feature/heuristic or model design, evaluation method, and tests.

**Prerequisites:** An approved complexity definition and a demonstrated routing need.

**Acceptance criteria:** Complexity changes behavior only according to an explicit policy and has measured evaluation.

**Out of scope:** LLM-based complexity estimation and unvalidated routing changes.

## Milestone 6 — Ollama integration

**Objective:** Resolve logical routes to configured local Ollama models.

**Deliverables:** provider registry, validated model configuration, Ollama adapter, availability/fallback behavior, and mocked provider tests.

**Prerequisites:** Milestone 4 routing decisions; explicit provider configuration design.

**Acceptance criteria:** The routing engine remains usable with no Ollama installation; normal tests use mocks; model/provider names come from configuration.

**Out of scope:** cloud-provider integration, provider-specific routing policy, and network-required baseline classification.

## Milestone 7 — Advanced routing

**Objective:** Evaluate only evidence-backed enhancements.

**Deliverables:** candidate capabilities such as embeddings, hybrid scoring, subcategories, capability profiles, cost-aware policy, latency-aware policy, or provider health checks.

**Prerequisites:** Measured baseline quality and an approved enhancement hypothesis.

**Acceptance criteria:** Each capability has a documented metric, comparison against baseline, and no regression in privacy or explainability.

**Out of scope:** unmeasured complexity, distributed orchestration, web UI, RAG, and broad framework adoption.
