# Published result summaries

These are selected outputs from the completed research runs. No fitting is
triggered by opening them. Raw responses, genotype sequences, feature arrays,
checkpoint binaries and per-observation prediction arrays are intentionally
excluded. Source snapshots and receipts retain the history needed to distinguish
the earlier experiment from the matched benchmark.

| Directory | Purpose |
|---|---|
| [`sampled_linear_benchmark/`](sampled_linear_benchmark/metrics.csv) | 13 matched linear models; overall/subgroup/fold metrics, penalty choices and numerical validation |
| [`sampled_cnn_benchmark/`](sampled_cnn_benchmark/metrics.csv) | Four matched CNN configurations; metrics, learning curves, inner selections and fit receipts |
| [`cnn_saved_prediction_comparison/`](cnn_saved_prediction_comparison/main_comparison.csv) | Earlier CNNs rescored on the exact current observations; includes retrospective-comparison qualifications |
| [`cnn_saved_prediction_comparison/training_overlap_recheck.json`](cnn_saved_prediction_comparison/training_overlap_recheck.json) | Checkpoint/split audit: zero evaluation/training overlap in each corresponding outer fold |
| [`sequence_cnn_benchmark/`](sequence_cnn_benchmark/metrics.csv) | Earlier whole-catalog reference-only experiment; these original metrics use a different evaluation cohort |
| [`sampled_sequence_features/`](sampled_sequence_features/sample_receipt.json) | Frozen ACR selection, identity/configuration receipts and score validation |
| [`genomewide_sequence_census/`](genomewide_sequence_census/summary.json) | SNP-call completeness and sequence-diversity inventories |
| [`cortex_cpm_variance/`](cortex_cpm_variance/summary_statistics.csv) | Descriptive library deviations and between-genotype CPM variation |

Auxiliary frozen fold/feature definitions and the coordinate-only legacy example
panel are retained where downstream scripts depend on them. The small
`sampled_sequence_features/selected_acrs.parquet` is an ACR annotation manifest,
not an accessibility or genotype-call matrix. It is included so report figures
can be regenerated without downloading data.

For the headline comparison, use `cnn_saved_prediction_comparison/metrics.csv`
with `subgroup == all`. All models there use 495,900 observations/9,666 ACRs for
evaluation. **The earlier CNN did not use the same training cohort or architecture.**
Never replace its row with its original whole-catalog metric from
`sequence_cnn_benchmark/metrics.csv`.
