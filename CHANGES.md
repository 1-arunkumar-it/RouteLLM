# Milestone 3 Binary Linear SVM Correction

## Status

Completed and verified. Milestone 3 classifier benchmarking is complete.
Milestone 4 remains pending explicit human approval.

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

The correction is limited to the binary Linear SVM adapter and its tests.
No cascade routing, calibration, threshold policy, providers, spaCy, Ollama,
or other Milestone 4 work was added.
