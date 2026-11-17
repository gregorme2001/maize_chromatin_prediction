# Predicting maize chromatin accessibility from DNA sequence

**Research question:** How much of chromatin accessibility at an unseen genomic
region can be predicted from local DNA sequence, and does adding genotype-specific
SNPs help? This repository accompanies a research report by Gregory Orme using
the maize single-nucleus ATAC-seq dataset of
[Marand et al. (2025)](https://doi.org/10.1126/science.ads6601).

The main benchmark contains **9,666 accessible chromatin regions (ACRs),
137 development genotypes, and 495,900 ACR–genotype observations**, restricted
to cortex nuclei. Libraries are pooled within genotype. The target is each
ACR's detections per million detections across all 108,843 source ACRs (CPM).
This measures relative accessibility, rather than the fraction of nuclei
detecting a region.

## Results

| Model | ACR–genotype RMSE ↓ | ACR-average RMSE ↓ | ACR-average R² ↑ |
|---|---:|---:|---:|
| GC/length/chromosome/ancestry baseline | 7.7485 | 4.8881 | 0.0273 |
| Best observed motif ridge: personalized core | 7.5558 | 4.5257 | 0.1662 |
| Best observed matched CNN: B73 2,000 bp | 7.7358 | 4.8581 | 0.0392 |
| Earlier CNN: B73 2,048 bp, larger training catalog | **7.3244** | **4.0912** | **0.3186** |

RMSE is in CPM. Every row evaluates the **same observations** using outer
held-out genomic-block predictions. The earlier CNN used a larger training
catalog, a different architecture and three-network averaging; its row is a
retrospective comparison, not a controlled architecture comparison. Its
evaluated ACRs were excluded from the corresponding prediction networks'
training sets, verified against saved checkpoints and fold records.

- Sequence summaries modestly predict differences in average accessibility
  **between ACRs**. The best ridge model reduces MSE by 4.91% versus baseline.
- SNP-substituted inputs provide essentially no overall improvement over B73
  for the matched models. **Within-ACR genotype-deviation R² stays near zero.**
- The earlier padded CNN improves on the best current ridge model by 6.03% in
  MSE. Why it outperforms the matched CNNs remains unresolved.

## Design and limits

The 13 linear models compare TF-motif scores, five-mer quasi-motif scores and
their combination, using B73 or SNP-substituted DNA and core or 2,000-bp inputs.
Four matched CNNs use the same sequence-source/extent combinations. Five outer
and three inner folds keep chromosome-specific 5-Mb blocks together; supervised
scaling, tuning and fitting use training observations only. This tests **new
regions within the studied genotypes**, not new genotypes. The permanent
30-genotype holdout is unused.

SNP-call completeness restricts eligibility; reconstructed sequences omit
indels, structural variants and unlisted SNPs. CPM is compositional, and the
observed library variation does not isolate genetic and technical variance.
Prediction does not establish TF binding or causality, and weak SNP-based
prediction does not contradict caQTL associations.

## Read and reproduce

- [Project summary](docs/project_summary.md)
- [Full report (PDF)](docs/Maize_ATAC_seq_summary.pdf)
- [Earlier report source draft](docs/research_report.tex): this source predates
  the supplied PDF; the PDF is the current report.
- [Methods and metric definitions](docs/methods.md)
- [Reproduction guide](docs/reproduction.md) and [data sources](docs/data_sources.md)
- [Published result tables and training-overlap audit](results/README.md)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,preparation,scoring]'
python -m pytest -q
python scripts/build_research_report_figures.py
python scripts/build_cnn_report_figures.py
```

Tests and figure generation need no raw data downloads or training. Neural tests
require the optional `neural` dependencies. Full numerical reproduction requires
external inputs, generated caches and additional computation; it is **not** a
single-command fresh-clone rebuild. Large data, sequence/feature arrays and model
checkpoints are excluded. The original exploratory workspace is preserved
separately; [the export inventory](docs/publication_inventory.json) records the
retained scientific source files and checksums.
