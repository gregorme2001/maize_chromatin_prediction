# Instructions for this publication repository

## Scope
This repository supports the report on predicting maize cortex chromatin
accessibility from local DNA sequence. Its main benchmark is 9,666 ACRs,
137 eligible development genotypes and 495,900 observations. Earlier
reference-only CNNs used the 108,843-ACR catalog and are a separate comparison.

## Scientific constraints
- Preserve the response, motif-scoring definitions, genotype exclusions,
  selected observations, original fold assignments and penalty-selection rules.
  Request approval before changing these scientific choices.
- Preserve the original 169-genotype order and fixed 30-genotype holdout in
  `src/maize_regulatory/resources/`. Never regenerate the holdout on a reduced
  cohort or inspect holdout DTS. CML341 remains excluded; H95 does not meet the
  prediction requirement of 50 pooled cortex nuclei.
- Use shared development loaders and test explicit row/column alignment.
- All supervised selection, scaling, tuning and fitting belong inside training
  folds. Fixed sequence-only annotations may be precomputed; document them.
- ACRs in a genomic block stay together across every genotype. Evaluate saved
  predictions using the model that excluded that ACR's entire outer fold.
- Earlier CNN and matched CNN results have different training cohorts and
  architectures. Equal evaluation observations do not make their training
  procedures controlled comparisons.
- CPM is relative accessibility. Motif scores are sequence resemblance, not
  TF binding, occupancy or activity. Prediction does not establish causality.
- Keep ACR-average prediction and genotype differences within an ACR distinct.
  Library variance is descriptive and is not a technical/genetic decomposition.

## Workflow
- `src/maize_regulatory/` contains tested scientific logic; scripts orchestrate it.
- Preserve saved source snapshots and published result tables. Do not overwrite
  completed scientific outputs when experimenting; use a separate output path.
- Run `python -m pytest -q` after code changes. Report optional dependency skips.
  Do not download genomes, scan the full motif catalog or train CNNs in tests.
- Raw data, phenotype values, sequences, feature arrays, credentials and model
  binaries are excluded from Git. Do not copy the original exploratory history.
- The author report PDF is intentionally pending review. Do not add or publish
  it until the user supplies/approves it. Preserve the author's report wording.
