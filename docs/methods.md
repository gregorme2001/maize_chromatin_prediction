# Methods at a glance

The [report source](research_report.tex) gives the full definitions. This file
summarizes the frozen analysis, rather than proposing new modeling choices.

## Measurements and response

For cortex nuclei of genotype `g` in library `l`, `Y_igl` counts nuclei detecting
ACR `i`. Each nucleus–ACR entry is binary. `T_gl = sum_j Y_jgl` sums detections
over **all 108,843 ACRs in those same nuclei**. The prediction response is
`C_ig = 1e6 * sum_l Y_igl / sum_l T_gl`. Libraries are pooled within genotype;
genotypes remain separate observations. Wider sequence inputs still predict the
original core ACR's CPM. There is no target log transform or prediction clipping.

## Eligibility and sequence

The frozen source order has 169 genotypes and a permanent unused holdout of 30.
CML341 is excluded from development; H95 has only 49 cortex nuclei and is excluded
from prediction, giving 137 eligible prediction genotypes. The library-variance
diagnostic instead uses all 138 development genotypes.

B73-v5 sequence is optionally modified at listed biallelic SNPs with complete
homozygous calls. Heterozygous/missing calls make an observation unavailable;
they are never replaced by a reference call. The full input is **2,000 bp total**,
centered at the ACR midpoint and containing the whole original core. Boundary-
crossing and ambiguous windows are excluded from the matched sample. Every
matched model uses this same full-window-complete eligibility mask.

Require at least 20 usable genotypes per ACR; sample up to 25 eligible ACRs per
occupied chromosome-specific 5-Mb block, seed 42, retaining every available
genotype at each sampled ACR. This yields 9,666 ACRs/495,900 observations. No MAF,
minimum-SNP-count, accessibility or phenotype screen is applied. The 1,755 windows
without listed SNPs do not establish absence of true genetic variation.

## Models and evaluation

The unpenalized baseline has an intercept, four fixed natural-cubic B73-core-GC
terms, core length, chromosome indicators and ten existing ancestry PCs. Each
linear sequence model adds globally shared, penalized coefficients for 927 TF
scores, 1,024 conceptual five-mer quasi-motif scores, or both. Five-mer scores
have 512 distinct reverse-complement pairs, with an algebraically corrected
penalty. Region scores are `log2(mean likelihood ratio)` over valid placements
and both orientations. Full-window scores do not distinguish core and flanks.

Five outer/three inner genomic-block folds keep all observations at an ACR and
all ACRs in a block together. Training-only preprocessing and inner tuning precede
fresh outer refits. The four matched CNNs share these observations and folds;
their architecture and training settings are in
[`configs/sampled_cnn_benchmark.json`](../configs/sampled_cnn_benchmark.json).

The earlier B73 CNNs use the larger 108,843-ACR catalog, 2,048-bp context, a core-
location channel, eight positional summaries and three independently trained
networks per outer fold. Saved predictions on the current observations come
only from the corresponding outer fold's excluded-region models. The audit
checks all 30 earlier core/context checkpoints and finds zero training overlap
for every evaluation fold. This comparison does not isolate architecture from
training-set size, eligibility, optimization or ensembling.

## Metrics

- **ACR–genotype RMSE:** square root of average squared CPM error over all actual
  observations, weighting each equally. More available genotypes give an ACR
  more observations in this metric.
- **ACR-average RMSE/R²:** average observed and predicted CPM over the same
  available genotypes at each ACR, then weight ACRs equally.
- **Within-ACR R²:** center observed and predicted values separately within each
  ACR; compare squared error of genotype deviations with predicting zero
  deviation. Centering is for evaluation only and does not adjust model output
  using a held-out response mean.

No significance test is attached to model-score differences. Results describe
unseen regions among called ACRs in these genotypes and cortex nuclei. They do
not establish binding, causality, a genetic/measurement variance decomposition,
or a demonstrated prediction ceiling.
