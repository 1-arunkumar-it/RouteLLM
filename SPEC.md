# RouteLLM — Master Product Specification

**Project:** RouteLLM
**Version:** 1.1
**Status:** Pre-development
**Type:** Terminal-based Python application
**Primary Goal:** Build an explainable, lightweight intelligent routing system for LLM requests. Cost-aware routing is a later capability that requires measured cost and capability data.

---

# 1. Project Overview

RouteLLM is a lightweight intelligent routing system that analyzes an incoming natural-language prompt and determines the most appropriate processing route.

The system should determine:

1. What the user is asking for.
2. What category the request belongs to.
3. How complex the request appears to be.
4. Whether an LLM is necessary.
5. Which logical model/processing route should handle the request.
6. How confident the routing decision is.

The central idea is:

> **Do not use an expensive LLM to make a routing decision when a lightweight NLP/ML system can make that decision sufficiently well.**

The initial implementation will use:

* Python
* spaCy
* TF-IDF
* word and character n-grams
* traditional machine-learning classifiers
* rule-based keyword detection
* terminal interaction
* pytest

LLM integration through Ollama will be introduced only after the core routing engine works.

The project is primarily a learning and portfolio project. Its early milestones prioritize a small, understandable, measured baseline over feature breadth.

---

# 2. Project Scope

## 2.1 Initial Scope

RouteLLM will initially be a **terminal-only Python application**.

The entire first version must work without:

* a web browser
* a frontend
* a web server
* a database
* a cloud API
* an external LLM

The core classifier must be capable of running locally.

---

# 3. Explicit Technology Constraint

The project must use **Python only** for implementation.

Do not introduce:

* Java
* Spring Boot
* Node.js
* React
* Vue
* Angular
* TypeScript
* frontend frameworks
* backend frameworks

The initial application should be implemented as a normal Python project with a command-line interface.

---

# 4. User Interface Constraint

There will be **no graphical or web UI** during the initial development stages.

Interaction happens entirely through the terminal.

Example:

```text
$ routellm

RouteLLM
────────────────────────────

Enter a prompt:

> Write a Spring Boot REST API for managing users.

Category      : coding
Subcategory   : backend
Confidence    : 0.94
Route         : coding-local

Detected signals:
  spring boot
  REST API
  managing users

>
```

The terminal is the primary interface.

A web interface may be considered only as a future project extension after the routing engine is mature.

---

# 5. Problem Statement

Modern AI applications receive many different types of requests.

Examples:

```text
What is 25 × 48?

Summarize this document.

Translate this paragraph into Tamil.

Write a Python program that processes CSV files.

Debug this Java code.

Explain this insurance claim.

Research the differences between two databases.
```

Sending every request to the same large model can be inefficient.

RouteLLM should provide a lightweight decision layer before model execution.

Conceptually:

```text
User Prompt
     ↓
Preprocessing
     ↓
Feature Extraction
     ↓
Classification
     ↓
Confidence Evaluation
     ↓
Routing Decision
     ↓
Selected Route
```

---

# 6. Core Design Principle

RouteLLM follows a **cascaded routing architecture**.

The system should use the cheapest reliable mechanism capable of making the routing decision.

Priority:

```text
Simple rule
    ↓
Statistical ML
    ↓
Optional semantic method
    ↓
LLM fallback
```

An LLM must NOT be required merely to classify a prompt in the baseline system.

---

# 7. Primary Objectives

## Objective 1 — Lightweight Classification

Use:

* spaCy
* tokenization
* normalization
* lemmatization where useful
* keyword/phrase detection
* TF-IDF
* word n-grams
* optional character n-grams
* traditional ML

---

## Objective 2 — Intelligent Routing

Convert classification results into logical routes.

Example:

```text
math
    → calculator

translation
    → translation

coding
    → coding-local

simple_qa
    → general-local

complex_reasoning
    → reasoning

unknown
    → fallback
```

---

## Objective 3 — Cost Reduction

The routing layer should reduce unnecessary use of expensive LLMs.

The classifier itself should be lightweight enough to run locally without requiring an LLM.

---

## Objective 4 — Explainability

Every routing decision should be inspectable.

Example:

```text
Category:
coding

Confidence:
0.91

Detected signals:
java
spring boot
REST API

Selected route:
coding-local

Reason:
The classifier identified strong backend-development signals.
```

The explanation must reflect actual system signals and must not invent reasoning.

---

## Objective 5 — Model Agnosticism

The routing engine must not depend on a specific LLM provider.

The classifier should produce a logical route such as:

```text
coding-local
```

rather than directly hard-coding:

```text
qwen3-coder
```

Provider/model configuration belongs to a separate layer.

---

# 8. Non-Goals

The initial project must NOT attempt to build:

* A chatbot
* A web application
* A GUI
* A REST API
* A RAG system
* A vector database
* An autonomous agent
* A prompt optimizer
* A model-training platform
* A cloud inference platform
* A production distributed system
* A full LLM orchestration framework

These may be considered future extensions.

---

# 9. Target Users

The initial project is primarily intended for:

1. Developers building AI applications.
2. Students learning NLP and machine learning.
3. Developers using local LLMs.
4. Developers experimenting with multiple LLM models.
5. Applications where LLM cost and latency matter.

---

# 10. Initial Application

The first usable version should be a terminal application.

The primary command should eventually support:

```bash
routellm
```

and:

```bash
routellm route "Write a Spring Boot REST API"
```

Example:

```text
Category      : coding
Subcategory   : backend
Confidence    : 0.93
Route         : coding-local

Signals:
  spring boot
  REST API
```

---

# 11. Input

The primary input is free-form natural language.

Example:

```text
Medical billing, insurance claim, ₹18,500,
partially approved, summarize this.
```

The system should extract useful signals such as:

```text
medical
billing
insurance
claim
₹18,500
approved
summarize
```

The complete prompt must not be sent to an LLM simply to determine its category.

---

# 12. Preprocessing

spaCy should provide the initial NLP preprocessing pipeline.

Potential operations:

1. Normalize text.
2. Tokenize.
3. Identify useful entities.
4. Lemmatize where appropriate.
5. Preserve important technical terms.
6. Preserve useful numerical information.
7. Preserve domain-specific phrases.

The preprocessing pipeline must not destroy useful information.

For example:

```text
Spring Boot
REST API
TF-IDF
GPT-5
₹18,500
JavaScript
```

should remain meaningful to downstream classification.

---

# 13. Feature Extraction

The baseline numerical representation should use TF-IDF.

Initial features:

* word unigrams
* word bigrams
* optional word trigrams
* optional character n-grams

Initial configuration should favor simplicity.

Example:

```python
ngram_range=(1, 2)
```

The feature configuration must be configurable rather than hard-coded throughout the project.

---

# 14. Keyword Layer

A lightweight rule-based layer should operate before or alongside ML classification.

Example:

```text
coding:
    python
    java
    javascript
    spring boot
    docker
    api
    function
    class
    compiler

translation:
    translate
    translation
    tamil
    hindi

summarization:
    summarize
    summary
    tl;dr
```

The keyword layer should support phrases, not just individual words.

Its purposes are:

* fast obvious classification
* feature generation
* explainability
* signal extraction
* handling high-confidence cases

Keyword matching is NOT the final intelligence of the system.

---

# 15. Classification Taxonomy

The initial taxonomy should remain small.

Top-level categories:

```text
coding
math
translation
summarization
general_qa
creative_writing
data_analysis
research
reasoning
text_classification
unknown
```

The taxonomy must be configurable.

Do not create an unnecessarily large taxonomy during the first milestone.

---

# 16. Subcategories

Subcategories can be introduced after top-level classification works.

Example:

```text
coding
├── frontend
├── backend
├── database
├── devops
├── debugging
├── algorithms
└── general
```

Subcategories should not complicate the initial implementation unnecessarily.

---

# 17. Machine-Learning Classifier

The first ML classifier must use traditional supervised machine learning.

Candidate algorithms:

1. Logistic Regression
2. Linear SVM
3. Naive Bayes

At least two suitable classifiers should eventually be benchmarked.

Evaluation should consider:

* Accuracy
* Precision
* Recall
* F1 score
* Inference latency
* Model size
* Interpretability

The classifier must be replaceable without rewriting the routing engine.

---

# 18. Dataset

RouteLLM requires a labeled dataset.

Initial format:

```csv
text,category
"Write a Java REST API",coding
"Translate this into Tamil",translation
"Summarize this article",summarization
"What is 25 multiplied by 48?",math
```

The dataset should eventually contain:

```text
data/
├── raw/
├── processed/
└── datasets/
```

The dataset must eventually be divided into:

* training
* validation
* test

Avoid leakage and near-duplicate samples across splits.

---

# 19. Dataset Quality

The classifier must learn **intent**, not simply memorize exact phrases.

For example, coding should contain varied requests:

```text
Write Python code.
Debug this Java program.
How do I create a REST API?
Why does my Spring Boot application fail?
Implement binary search.
Explain this SQL query.
```

The dataset should contain multiple ways of expressing the same intent.

---

# 20. Confidence

The classifier should produce a confidence value where supported.

Example:

```json
{
  "category": "coding",
  "confidence": 0.94
}
```

Confidence thresholds must be configurable.

Example:

```text
HIGH_CONFIDENCE = 0.80
```

The threshold must ultimately be determined using validation results rather than arbitrarily assumed. A score must not be treated as a trusted probability until the chosen classifier's confidence has been calibrated or otherwise validated. This is especially important for classifiers that do not natively provide calibrated probabilities.

---

# 21. Routing Engine

Classification and routing are separate responsibilities.

Classification answers:

```text
What is the user asking?
```

Routing answers:

```text
What should handle it?
```

Example:

```text
Classification:
category = coding
subcategory = backend
complexity = medium

Routing:
route = coding-local
```

---

# 22. Logical Routes

Initial logical routes may include:

```text
calculator
translation
general-local
coding-local
reasoning
remote-large
fallback
```

These are logical identifiers.

They must not be tightly coupled to a particular provider or model.

Until executable providers are introduced, logical routes are labels only. Baseline routing may select a route, but it does not execute a model or external service.

---

# 23. Model Registry

A model registry should eventually map logical routes to providers/models.

Conceptual configuration:

```yaml
models:

  coding-local:
    provider: ollama
    model: configured-model

  general-local:
    provider: ollama
    model: configured-model

  reasoning:
    provider: remote
    model: configured-model
```

The exact configuration format should be determined during implementation.

---

# 24. Ollama Integration

Ollama integration belongs to a later milestone.

The initial classifier must work without Ollama.

Later:

```text
User Prompt
    ↓
RouteLLM classifier
    ↓
coding-local
    ↓
Ollama
    ↓
Configured coding model
```

Ollama should be treated as a provider rather than the core routing engine.

The provider name and model name must come from configuration.

---

# 25. Optional Embedding Layer

Embeddings are explicitly **not required for the first version**.

A future version may compare:

```text
TF-IDF classification
        +
Embedding similarity
```

and potentially combine the results.

This should only be introduced after the baseline classifier has been measured.

---

# 26. CLI Design

The terminal interface should eventually support commands such as:

```bash
routellm route "Write a Python script"
```

```bash
routellm train
```

```bash
routellm evaluate
```

```bash
routellm test
```

```bash
routellm inspect
```

```bash
routellm config
```

These commands should be introduced incrementally.

Do not implement all commands in the first milestone.

---

# 27. Interactive Terminal Mode

After the routing engine works:

```bash
routellm
```

should launch interactive mode.

Example:

```text
RouteLLM
────────────────────────────

> Write a Java Spring Boot API

Category      : coding
Subcategory   : backend
Confidence    : 0.94
Route         : coding-local

Signals:
  spring boot
  REST API

> Translate this paragraph into Tamil

Category      : translation
Confidence    : 0.97
Route         : translation

> exit
```

Interactive mode must be a thin interface over the routing engine.

Business logic must NOT be placed inside the CLI presentation layer.

---

# 28. Architecture

The application should maintain clear separation of concerns.

Conceptual architecture:

```text
                    CLI
                     │
                     ▼
              Routing Service
                     │
       ┌─────────────┼──────────────┐
       ▼             ▼              ▼
 Preprocessor   Keyword Engine   Classifier
       │             │              │
       └─────────────┼──────────────┘
                     ▼
              Confidence Layer
                     │
                     ▼
               Routing Policy
                     │
                     ▼
               Model Registry
                     │
                     ▼
                Providers
                     │
                 ┌───┴───┐
                 ▼       ▼
              Ollama   Remote
```

The initial project may implement only the components required for the current milestone.

Do not create empty abstractions solely for architectural appearance.

---

# 29. Recommended Python Structure

The final structure should be determined during planning.

A possible structure is:

```text
routeLLM/
│
├── src/
│   └── routellm/
│       ├── cli/
│       ├── preprocessing/
│       ├── features/
│       ├── classification/
│       ├── routing/
│       ├── providers/
│       ├── configuration/
│       └── evaluation/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── datasets/
│
├── models/
│
├── tests/
│
├── docs/
│
├── SPEC.md
├── README.md
├── pyproject.toml
└── .gitignore
```

This is an architectural proposal, not a requirement to create every directory immediately.

---

# 30. Technology Requirements

## Required

* Python
* spaCy
* scikit-learn
* TF-IDF
* n-grams
* traditional ML
* pytest
* Git

spaCy is part of the intended NLP baseline, but its contribution must be benchmarked against simpler preprocessing. It must not be assumed to improve quality or satisfy performance goals without measurement.

## Later

* Ollama
* embeddings
* NumPy
* Pandas
* additional ML tooling where justified

---

# 31. Explicitly Avoided Technologies

The initial project must NOT depend on:

* React
* Node.js
* TypeScript
* Java
* Spring Boot
* FastAPI
* Flask
* LangChain
* LlamaIndex
* vector databases
* cloud LLM APIs
* LLM-based classification

A dependency should only be introduced when there is a demonstrated requirement for it.

---

# 32. LLM Usage Policy

The baseline classifier must not require an LLM.

Correct:

```text
User Prompt
     ↓
spaCy
     ↓
TF-IDF
     ↓
Traditional ML
     ↓
Route
     ↓
Optional LLM
```

Incorrect:

```text
User Prompt
     ↓
LLM
     ↓
"What category is this?"
     ↓
Route
```

The second design undermines the project's primary purpose.

---

# 33. Privacy

Baseline classification should occur locally.

The user's prompt must not leave the machine merely to determine its category.

If a remote provider is eventually selected as the final processing route, the transfer should be explicit and configurable.

---

# 34. Security

The implementation must:

* never commit API keys
* never log secrets
* avoid exposing environment variables
* validate configuration
* avoid arbitrary command execution
* avoid transmitting prompts unexpectedly
* avoid logging sensitive prompt content by default

---

# 35. Logging

Support:

```text
DEBUG
INFO
WARNING
ERROR
```

Normal output should remain concise.

Debug mode may show:

```text
preprocessing
keywords
classifier scores
confidence
route
latency
```

Prompt contents should not be logged by default.

---

# 36. Testing

Testing is a core requirement.

## Unit Tests

Test:

* preprocessing
* keyword extraction
* feature extraction
* classifier
* confidence calculation
* routing policy
* configuration

## Integration Tests

Test:

```text
prompt
 ↓
preprocessing
 ↓
features
 ↓
classifier
 ↓
routing
```

## CLI Tests

Test:

* command parsing
* output
* invalid input
* interactive mode

## Provider Tests

Ollama integration must eventually have mocked tests.

The normal test suite must not require Ollama to be running.

---

# 37. Evaluation

Required metrics:

* accuracy
* precision
* recall
* F1
* confusion matrix
* inference latency
* low-confidence rate

Example benchmark format:

```text
Classifier            Accuracy    F1       Latency
---------------------------------------------------
Logistic Regression   measured    measured measured
Linear SVM             measured    measured measured
Naive Bayes            measured    measured measured
```

Example values must be generated from actual experiments.

Never hard-code expected performance.

---

# 38. Failure Handling

If confidence is low:

```text
Category:
unknown

Confidence:
0.41

Route:
fallback
```

The system must not present uncertain predictions as certain.

If an external provider is unavailable:

```text
Selected route:
coding-local

Status:
provider unavailable

Fallback:
general-local
```

Fallback behavior must be configurable.

---

# 39. Performance Goals

The baseline classifier should aim for:

* fast preprocessing
* low memory usage
* local execution
* millisecond-scale classification where practical
* no unnecessary network requests
* no LLM loading for classification

Exact targets must be measured on the actual development machine.

---

# 40. Development Philosophy

Priorities:

1. Correctness
2. Explainability
3. Simplicity
4. Measurable performance
5. Maintainability
6. Extensibility

Do not prematurely build a complex AI architecture.

A small measurable system is preferable to an elaborate system without evidence that it improves routing.

---

# 41. Milestones

## Milestone 0 — Python Project Foundation

Build:

* Python project
* package structure
* dependency management
* CLI skeleton
* README
* test structure
* Git configuration

Success:

```bash
routellm --help
```

works.

---

## Milestone 1 — Rule-Based Router

Build:

* preprocessing
* keyword dictionary
* phrase matching
* initial categories
* deterministic routing
* terminal explanations

Goal:

Understand the complete routing pipeline before introducing ML.

---

## Milestone 2 — Dataset + TF-IDF

Build:

* labeled dataset
* train/validation/test split
* TF-IDF
* word n-grams
* baseline classifier
* prediction
* confidence
* evaluation

Goal:

Move from manually defined rules to statistical classification.

---

## Milestone 3 — Classifier Benchmark

Compare:

* Logistic Regression
* Linear SVM
* Naive Bayes

Measure:

* accuracy
* precision
* recall
* F1
* latency

Select the best baseline model using measured results.

---

## Milestone 4 — Cascaded Router

Combine:

```text
Keyword signals
       +
ML classifier
       +
Confidence threshold
       +
Fallback
```

Goal:

Create the first complete intelligent RouteLLM routing engine.

---

## Milestone 5 — Complexity Estimation

Introduce lightweight estimation of request complexity.

Potential signals:

* prompt length
* number of clauses
* technical terminology
* requested operations
* reasoning indicators
* code complexity indicators

Do not use an LLM for this initially.

Complexity is explicitly deferred until its output scale, intended routing use, signals, and evaluation method are specified. It is not an implicit requirement of earlier milestones.

---

## Milestone 6 — Ollama Integration

Connect selected logical routes to Ollama.

Example:

```text
Prompt
 ↓
RouteLLM
 ↓
coding-local
 ↓
Ollama
 ↓
configured local model
```

The routing engine must remain usable without Ollama.

---

## Milestone 7 — Advanced Routing

Potential future capabilities:

* semantic embeddings
* hybrid TF-IDF + embeddings
* better subcategories
* model capability profiles
* cost-aware routing
* latency-aware routing
* provider health checks

Only introduce these after the baseline system has measurable performance.

---

# 42. Definition of Done

A milestone is complete only when:

* implementation exists
* tests exist
* tests pass
* behavior is documented
* edge cases are considered
* Git diff has been reviewed
* implementation matches SPEC.md
* the developer understands the implementation

The final requirement is important because RouteLLM is both a portfolio project and a learning project.

---

# 43. Learning Objective

By completing RouteLLM, the developer should understand:

```text
spaCy
  ↓
NLP preprocessing

TF-IDF
  ↓
Numerical text representation

n-grams
  ↓
Phrase/context features

Traditional ML
  ↓
Intent classification

Confidence
  ↓
Uncertainty estimation

Routing policy
  ↓
Processing/model selection

Ollama
  ↓
Local LLM execution
```

The developer should be able to explain every stage during a technical interview.

---

# 44. Portfolio Objective

RouteLLM should demonstrate practical knowledge of:

* NLP
* feature engineering
* TF-IDF
* n-grams
* supervised classification
* model evaluation
* confidence-based decisions
* routing architecture
* local LLM integration
* Python software engineering
* testing
* CLI application design

The project should be described as:

> **An explainable NLP routing layer that uses lightweight statistical classification to select an appropriate processing path before invoking an LLM.**

Cost-aware routing may be claimed only after it has explicit cost inputs and a measured routing policy.

It should NOT be described simply as:

> "An AI that chooses the best LLM."

---

# 45. Success Criteria

The system should eventually accept:

```text
Write a Spring Boot REST API for user authentication.
```

and produce something similar to:

```text
Category      : coding
Subcategory   : backend
Confidence    : 0.XX
Route         : coding-local

Signals:
  spring boot
  REST API
  authentication
```

For:

```text
Translate this paragraph into Tamil.
```

it should produce:

```text
Category      : translation
Confidence    : 0.XX
Route         : translation
```

For an unfamiliar request:

```text
Category      : unknown
Confidence    : 0.XX
Route         : fallback
```

The actual accuracy and confidence values must come from the trained system.

---

# 46. Long-Term Architecture

The eventual architecture may become:

```text
                         USER PROMPT
                              │
                              ▼
                       PREPROCESSING
                              │
              ┌───────────────┼────────────────┐
              │               │                │
              ▼               ▼                ▼
         Rules/Keywords    TF-IDF          Embeddings
              │               │                │
              └───────────────┼────────────────┘
                              ▼
                       ROUTING ENGINE
                              │
                ┌─────────────┼──────────────┐
                ▼             ▼              ▼
             Tool         Local Model    Remote Model
                │             │              │
                └─────────────┼──────────────┘
                              ▼
                           RESPONSE
```

The system should become progressively more capable without making the routing layer unnecessarily expensive.

---

# 47. Fundamental Rule

The defining principle of RouteLLM is:

> **Use the cheapest reliable mechanism capable of making the decision.**

If a rule is sufficient, use a rule.

If statistical classification is sufficient, use statistical classification.

If embeddings provide a measurable improvement, consider embeddings.

Only use an LLM when simpler mechanisms cannot reliably solve the problem.

That principle defines RouteLLM.

---

# 48. Measurement Requirements

Rule-based decisions are not inherently high confidence. Keyword and phrase rules must be evaluated on held-out examples for precision, recall, and failure modes before they bypass or override statistical classification.

The command for the Python test suite is `pytest`. A `routellm test` command is not part of the baseline and must not be added unless a later requirement justifies maintaining it.
