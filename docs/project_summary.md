# Predicting maize chromatin accessibility from DNA sequence

Gregory Orme | November 2026

## Question and data

Can local DNA predict accessibility at unseen genomic regions, and do genotype-specific SNPs help? Using Marand et al. (2025) single-nucleus ATAC-seq data, the matched cortex benchmark contains 9,666 accessible chromatin regions (ACRs), 137 development genotypes and 495,900 ACR–genotype observations. Libraries are pooled within genotype. The response is CPM: each region’s detections divided by all-genome ACR detections in the same genotype, multiplied by one million.

## Evaluation

Five outer and three inner cross-validation folds keep chromosome-specific 5-Mb blocks together. Supervised preprocessing and tuning use training observations only. All rows below evaluate the same observations using held-out-region predictions. This tests new regions within the studied genotypes, not new genotypes. The permanent 30-genotype holdout is unused.

## Headline comparison

| Model | Observation RMSE | ACR-average RMSE | ACR-average R² |
|---|---:|---:|---:|
| Covariates only | 7.7485 | 4.8881 | 0.0273 |
| Best matched motif ridge | 7.5558 | 4.5257 | 0.1662 |
| Best matched CNN | 7.7358 | 4.8581 | 0.0392 |
| Earlier padded CNN* | 7.3244 | 4.0912 | 0.3186 |

RMSE is in CPM; lower is better. ACR-average metrics weight ACRs equally.

*The earlier B73 CNN used a larger 108,843-ACR training catalog, 2,048-bp inputs, a different architecture and three-network averaging. Each evaluated ACR was excluded from its prediction networks’ training sets. This comparison fixes evaluation observations, not training cohorts or architectures. The cause of its advantage remains unresolved.

## What we learned

Motif/five-mer ridge summaries modestly predict differences in average accessibility between ACRs. SNP-substituted inputs provide essentially no overall improvement in the matched comparisons, and within-ACR genotype-deviation R² remains near zero. The four matched CNNs barely improve over covariates alone. The earlier padded CNN performs better than the best current ridge model in all five outer folds.

## Scope and limitations

CPM is relative accessibility. Complete homozygous SNP calls restrict eligibility; reconstructed sequences omit indels, structural variants and unlisted SNPs. Library variation does not separate genetic and measurement variance. Prediction does not establish TF binding or causality; weak genotype prediction does not contradict caQTL association.

https://github.com/gregorme2001/maize_chromatin_prediction

Source study: https://doi.org/10.1126/science.ads6601
