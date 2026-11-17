# Predicting maize chromatin accessibility from DNA sequence

Gregory Orme | November 2026

**[Read the full report (PDF)](Maize_ATAC_seq_summary.pdf)**

## Question and approach

How much of the variation in chromatin accessibility across genomic regions in maize can local sequence explain? Using Marand et al. (2025)'s single-nucleus ATAC-seq data, I designed summary statistics aggregating each region's resemblance to transcription factor motifs and generic five-base sequence patterns (5-mers). I used ridge regression with cross-validation to test whether these scores predicted accessibility at unseen genomic regions.

I compared reference sequence with SNP-derived genotype-specific sequences, and the core region's sequence with a larger window containing surrounding DNA. I also compared these predictions with convolutional neural networks  and a baseline using only GC content, region length, chromosome, and genetic ancestry.

## Evaluation

The matched comparison used approximately 9,700 genomic regions across 137 maize genotypes, restricted to cortex nuclei. I evaluated prediction on held-out genomic regions, keeping neighboring regions together and choosing model settings using training data only. This tests prediction at new regions within the genotypes studied.

## Results

| Model | Variation in average accessibility explained |
|---|---:|
| Baseline covariates only | 2.7% |
| Best motif-based ridge regression | 16.6% |
| Best CNN on the matched dataset | 3.9% |
| Earlier CNN using more training regions | 31.9% |

Values are held-out R² expressed as percentages, measuring prediction of accessibility averaged across genotypes at each region. All models were scored on the same held-out observations.

The earlier CNN used a larger catalog of 108,843 regions, a different architecture, and a different training procedure. Its stronger performance shows additional predictive information in sequence, but does not identify which changes explain its advantage. As a future experiment, I would like to test additional CNN architectures and regularization strengths using reference sequence only, and compare them with linear models that capture the spacing and co-occurrence of motif patterns.

## What we learned

- Motif and 5-mer summaries modestly predict average accessibility differences between regions. Wider sequence context did not improve the linear models, and the matched CNNs performed close to the baseline.
- SNP-based genotype-specific sequences provided essentially no overall improvement versus just using the reference. None of the tested models recovered appreciable accessibility differences between genotypes within a fixed region.
- The earlier CNN outperformed the best ridge model in all five held-out folds, supporting further investigation of reference-sequence prediction across more regions.

## Scope and limitations

The response measures relative accessibility in maize cortex nuclei. Incomplete genotype-specific sequences and measurement variation may limit genotype-level prediction. These observational results do not establish TF binding or causal regulatory effects.

https://github.com/gregorme2001/maize_chromatin_prediction

Source study: https://doi.org/10.1126/science.ads6601

[Code and reproduction instructions](docs/getting_started.md)
