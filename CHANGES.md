# Changelog

## Advanced routing

- Provider health checks with file persistence
- Model capability profiles with cost, latency, and capability metadata
- Cost-aware routing with constraint-based rerouting to cheaper alternatives
- Latency-aware routing using both estimated and measured data
- Extended benchmark reporting with cost/latency summaries

## Ollama integration

- Provider registry with validated TOML configuration
- Ollama adapter using only standard library (`urllib`)
- Availability fallback and `routellm run` / `routellm providers` commands
- Fully mocked provider tests; routing engine usable with no Ollama installation

## Cascaded routing

- Cascade policy combining rules, classifier, and fallback
- Calibrated confidence with validated thresholds
- Unknown/fallback behavior and explanation merging

## Classifier benchmarking

- Logistic Regression, Linear SVM, and Naive Bayes comparison
- Accuracy, precision, recall, F1, latency, and model size reporting

## Dataset + TF-IDF classifier

- Labeled dataset with stratified splits and leakage checking
- TF-IDF word n-gram features
- Prediction persistence and evaluation reporting

## Rule-based routing

- Preprocessing and tokenization
- Configurable keyword/phrase signals
- Deterministic policy and structured decisions
