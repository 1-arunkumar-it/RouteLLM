# Milestone 3 Correction Handoff

## Review outcome

Milestone 3 correctly benchmarks the three required candidates on the current
seven-category dataset. The real benchmark command completed and selected
Linear SVM from validation results only, then evaluated it once on held-out
test data. The full suite currently reports 106 passing tests and Ruff passes.

However, the reusable Linear SVM adapter is not correct for binary-class
datasets. `LinearSVC.decision_function()` returns one signed margin per sample
for two classes, while `CandidateClassifier._predict_with_margins()` assumes a
row of one margin per class. Calling `predict()` then raises:

```text
TypeError: 'float' object is not iterable
```

This violates the interchangeable-classifier boundary required by
`ARCHITECTURE.md`. Do not start Milestone 4 until this correction is complete,
tested, and reviewed.

## Required implementation changes

### 1. Handle binary Linear SVM decision margins correctly

**Files to modify**

- `src/routellm/classification/algorithms.py`
- `tests/test_benchmark.py`

**Behavior required**

- Preserve the existing multiclass Linear SVM behavior.
- When a fitted Linear SVM has exactly two classes, convert its one-dimensional
  signed decision margin into one raw score for each class before constructing
  `ClassifierPrediction.scores`.
- Use scikit-learn's binary margin convention: the positive margin supports
  `classes_[1]`; the negated margin supports `classes_[0]`.
- Sort the resulting raw scores descending, as the existing prediction
  contract requires.
- The predicted category must equal the class with the highest raw margin.
- Keep `confidence=None`. Raw SVM margins are not probabilities and must never
  be presented as calibrated confidence or used for low-confidence routing.
- Do not change the Logistic Regression or Multinomial Naive Bayes probability
  behavior.

**Tests required**

- Add a two-category fixture dataset with enough examples to train a Linear
  SVM.
- Test `fit_candidate(linear_svm, binary_dataset).predict(...)` succeeds.
- Test `predict_batch(...)` succeeds for multiple binary prompts.
- Assert each binary SVM prediction has exactly one score for each fitted
  class, scores are descending, the top score's category equals
  `prediction.category`, and `prediction.confidence is None`.
- Add a binary-dataset `run_benchmark(...)` integration test to ensure the
  shared benchmark path creates a report and all three candidate rows without
  crashing.
- Keep the existing multiclass SVM and canonical-dataset benchmark tests.

### 2. Correct milestone-status documentation after the fix

**Files to modify**

- `README.md`
- `ROADMAP.md`

**Behavior required**

- Until the binary SVM correction passes review, state that Milestone 3 is
  under stabilization/review and that Milestone 4 is not authorized.
- After the correction and all verification pass, restore the accurate status:
  Milestone 3 complete; Milestone 4 pending explicit human approval.
- Do not alter milestone descriptions or claim cascade routing, calibration,
  thresholding, providers, spaCy, or Ollama have been implemented.

## Architectural constraints

- This is a correction to the Milestone 3 classifier interface only.
- Keep `routellm route` fully rule-based.
- Do not add routing thresholds, confidence calibration, fallback policy
  changes, or any Rule/ML cascade behavior.
- The domain layer remains independent of scikit-learn and CLI code.
- Do not add dependencies, providers, network activity, spaCy, embeddings,
  web/API work, or external datasets.
- Do not use SVM raw margins as probabilities anywhere in output, metrics, or
  selection logic.

## Verification and acceptance criteria

OpenCode must run and report:

```bash
pytest
ruff check .
routellm benchmark
routellm route "Write a Spring Boot REST API"
```

The correction is accepted only when:

- The binary Linear SVM prediction and benchmark tests pass.
- The full suite and Ruff pass.
- The real seven-category benchmark still completes and its report shows
  `n/a` for Linear SVM low-confidence rate.
- `routellm route` retains its existing rule-based behavior.
- `README.md` and `ROADMAP.md` give the same accurate milestone status.
- The diff contains only the binary-SVM correction, its tests, and status
  documentation updates.

## Explicitly out of scope

- All Milestone 4 work: cascaded routing, ML-backed route decisions,
  calibration, confidence thresholds, unknown/fallback policy changes, and
  explanation merging.
- Any changes to benchmark selection methodology or held-out test use.
- spaCy, embeddings, providers, Ollama, calculator execution, remote APIs,
  web/UI work, additional ML frameworks, or new dependencies.
