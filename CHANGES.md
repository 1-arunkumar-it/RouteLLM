# Milestone 4 — Cascaded Router

## Status

Reviewed and accepted locally. The `routellm cascade` command trains a
calibrated cascade model, `routellm route --model` routes through the cascade,
and `routellm evaluate --model` measures cascade quality, overrides, and
fallbacks on the held-out test split. Milestone 5 remains pending explicit
human approval.

## Design

The cascade follows the "cheapest reliable mechanism" principle:

1. **Rules**: keyword signals win immediately for categories whose rule
   precision on the validation split is at or above a configurable minimum
   (default 0.90). Rule decisions report `confidence=None` and never claim
   probability (SPEC sections 20 and 48).
2. **Classifier**: the benchmark-selected Linear SVM is wrapped in
   `CalibratedClassifierCV` so confidence is a calibrated probability. A
   validated confidence threshold (chosen from the validation split, not
   assumed) accepts the classifier's category.
3. **Fallback**: anything below threshold routes to `unknown`/`fallback` and
   still reports the real confidence so uncertainty is never presented as
   certainty (SPEC section 38).

`RouteService` stays rule-only when no model is given.

## Cascade configuration review

`CascadeConfig` is an immutable, validated configuration boundary for the
cascade. It rejects rule-precision and threshold values outside `[0, 1]`,
calibration cross-validation values below `2`, and unsupported calibration
methods. The supported methods are `sigmoid` and `isotonic`.

The default cascade path remains evidence-based: training persists the
threshold selected from validation data. A `CascadeConfig.threshold` value can
override that threshold only for a programmatically constructed `RouteService`
instance; it is not persisted during training and the CLI intentionally does
not expose it. This prevents normal command-line routing from silently using
an unvalidated threshold.

## New commands

```text
routellm cascade                                # train models/cascade.joblib
routellm route --model models/cascade.joblib "..."   # cascade routing
routellm evaluate --model models/cascade.joblib      # cascade evaluation
```

## Measured results (current dataset, local run)

```text
Selected threshold: 0.35 (validation macro F1 0.925)
Rule override categories: coding, creative_writing, math, summarization, translation
Test split: accuracy 0.973, macro F1 0.976
Rule decisions 59.5%, classifier decisions 40.5%, fallback decisions 0%
Mean latency 0.31 ms
```

These are observations from this review run on the current dataset. Latency is
environment-dependent and is not a general performance claim.

## Tests

`tests/test_cascade.py` covers config validation, every cascade branch,
threshold selection, rule metrics, and cascade-model persistence.
`tests/test_cascade_service.py` covers cascade training/evaluation and the
CLI end to end.

## Verification

```text
.venv/bin/pytest                 138 passed
.venv/bin/ruff check .           All checks passed
.venv/bin/routellm cascade       completed; threshold selected from validation
.venv/bin/routellm evaluate --model models/cascade.joblib
                                 0.973 accuracy, 0.976 macro F1
```

Out of scope remains: providers, Ollama, embeddings, complexity estimation,
and cost optimization (Milestones 5-7).

# Milestone 3 Binary Linear SVM Correction

## Status

Completed and verified. Milestone 3 classifier benchmarking is complete.
Milestone 4 was subsequently implemented and is documented above.

## Change

`CandidateClassifier` now handles the binary `LinearSVC.decision_function()`
shape correctly. For a two-class classifier, its single signed margin is
converted into two raw class scores:

- the negated margin supports `classes_[0]`;
- the positive margin supports `classes_[1]`.

The scores remain sorted in descending order, the selected category matches
the highest raw score, and `confidence` remains `None`. Linear SVM margins are
not probabilities and are not used for low-confidence reporting.

## Tests Added

- Binary Linear SVM single-prompt prediction.
- Binary Linear SVM batch prediction.
- Binary benchmark execution across all three classifier candidates.

The existing multiclass Linear SVM and seven-category benchmark coverage is
retained.

## Verification

Completed in the project virtual environment:

```text
.venv/bin/pytest                 108 passed
.venv/bin/ruff check .           All checks passed
.venv/bin/routellm benchmark     completed; Linear SVM low-confidence rate is n/a
.venv/bin/routellm route "Write a Spring Boot REST API"
                                 coding -> coding-local
```

The correction itself was limited to the binary Linear SVM adapter and its
tests. It did not add cascade routing, calibration, threshold policy,
providers, spaCy, Ollama, or other Milestone 4 work.
