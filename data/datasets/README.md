# Labeled prompt dataset

`prompts.csv` is the curated, versioned labeled dataset for Milestone 2. It is
hand-authored to represent varied phrasings of each intent (SPEC section 19)
rather than exact template repetitions.

## Labeling policy

- Columns: `text` (the prompt) and `category` (one of the seven Milestone 1
  categories: `coding`, `math`, `translation`, `summarization`,
  `creative_writing`, `general_qa`, `unknown`).
- `unknown` rows are off-topic or vague prompts that should not be confidently
  classified into any concrete category.
- Every row is a distinct phrasing; duplicate or near-duplicate rows are
  avoided because they risk leakage across splits.
- Categories other than the seven above are rejected by `load_dataset`.

## Provenance

- Authored by the RouteLLM developer, August 2026.
- Source: hand-written prompts in English, informed by SPEC examples.
- Current size: 245 rows (coding 40, math 40, general_qa 40, translation 35,
  summarization 30, creative_writing 30, unknown 30).

## Splits

Training, validation, and test splits are produced deterministically by
`routellm.classification.dataset.stratified_split` (stratified, fixed seed,
70/15/15 by default). The train command writes the generated split CSVs and a
`provenance.json` under `data/processed/splits/` (Git-ignored) and records the
dataset fingerprint in the trained model. `routellm evaluate` recomputes the
same split from the model's recorded seed and refuses to run if the dataset
changed after training.

The split is checked for leakage: prompts in different splits must have token
set Jaccard similarity below 0.9, enforced by
`routellm.classification.dataset.check_no_leakage` during training.

# Complexity evaluation set

`complexity.csv` is the hand-labeled evaluation set for Milestone 5. It is
kept separate from `prompts.csv` so the Milestone 2 splits, fingerprints, and
pipeline stay untouched. It is an evaluation set for measuring the heuristic
estimator, not a training set.

## Labeling policy

- Columns: `text` (the prompt) and `complexity` (one of `low`, `medium`,
  `high`).
- `low`: short, single-step prompts (e.g. `What is 2 plus 2`).
- `medium`: prompts needing a couple of steps or moderate context.
- `high`: prompts that demand multi-step reasoning, analysis, or a deep
  technical response.
- Every row is a distinct phrasing; rows are written to be independent of the
  category dataset and are not drawn from it.
- Levels other than the three above are rejected by
  `load_complexity_dataset`.

## Provenance

- Authored by the RouteLLM developer, August 2026.
- Source: hand-written prompts in English, informed by SPEC examples.
- Current size: 90 rows (low 30, medium 30, high 30).
