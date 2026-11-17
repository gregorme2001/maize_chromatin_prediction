# Reproduction and repository layout

There are two distinct levels of reproduction. **Tests and report-figure
generation work from this clone without raw datasets.** Repeating sequence scoring
or model training requires external data and the generated intermediate caches
listed below. Historical prerequisites have been preserved rather than silently
changing the completed experiment to manufacture a fresh-clone training command.

## 1. Read the results and reproduce presentation

Use Python 3.12 for the closest match to the recorded runs:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,preparation,scoring]'
python -m pytest -q
python scripts/build_research_report_figures.py
python scripts/build_cnn_report_figures.py
```

These commands do not train, download data or recompute motif scores. Tests use
synthetic inputs; R/Torch checks skip if their optional runtimes are unavailable.
The figure generators verify the published sample counts and result tables.

- `Maize_ATAC_seq_summary.pdf`: author-supplied report approved for inclusion.
- `docs/research_report.tex`: earlier author source draft; it may differ from the PDF.
- `README.md`: current author-edited project summary.
- `docs/getting_started.md`: setup and reproduction links.
- `docs/figures/`: exported report figures in PDF/PNG and the linear-results table.
- `results/`: selected small scientific outputs and frozen split/config records.
- `src/maize_regulatory/`: scientific implementations and necessary shared helpers.
- `scripts/`: retained orchestration, preprocessing and presentation commands.
- `tests/`: tests for the retained package, including optional neural checks.
- `docs/publication_inventory.json`: original file hashes and export scope.

The original exploratory notebooks, unrelated analyses, raw inputs, fitted model
binaries and old Git history were left in the research workspace. Some retained
helper modules retain historical names because the final workflows import their
tested functions. Source snapshots preserve the exact recorded implementations.

## 2. Reconstruct the research data locally

Read [data sources](data_sources.md) first. GNU Make >=4.3 and R with Matrix/edgeR
are needed for the original data branch; neural work requires a separate PyTorch
environment. No large data acquisition is part of testing.

```bash
cp .Renviron.example .Renviron
Rscript scripts/setup_r.R
python scripts/download_inputs.py --list
python scripts/prepare_data.py --check
# Supply the required original marker/identity-matching inputs under data/raw/.
# Preview before acquiring/processing the original five-parquet data branch:
make -n data
make data
```

`make data` rebuilds the original five products, including a historical phenotype
product; it is **not** the full report-training pipeline. `--overwrite` on the
download/preparation commands replaces generated files but never redefines the
frozen genotype identities. Do not download the obsolete normalized 58k matrix.

The report's dependency chain is:

1. Original sparse nucleus–ACR matrix, verified metadata, reference and motifs.
2. Development-only sparse aggregation and chromosome-10 variant checks:
   `scripts/inspect_panzea_chromosome.py` and
   `scripts/audit_cell_state_accessibility.py`. The latter creates the
   `results/cell_state_accessibility_audit/aggregation/` count/depth tables.
3. Whole-genome pooled cortex counts and reference comparisons:
   `scripts/analyze_reference_sequence.py`, then
   `scripts/analyze_sequence_summaries.py`. These retain/check the original
   `results/sequence_summary_comparison/fold_manifest.json`.
4. Earlier reference-only CNN data preparation and fitting:
   `scripts/analyze_sequence_cnn.py --stage prepare`, then its documented
   `--stage run --max-epochs 6 --batch-size 1024` experiment. It produces the
   ACR coordinates, genotype coverage and sequence caches used by later stages.
5. Listed variant files via `scripts/download_panzea_v5.py`; calibrated quasi-
   motif definition via `scripts/audit_personalized_chr10.py`; genome-wide
   sequence eligibility via `scripts/census_personalized_sequences.py`.
6. Frozen selection and sequence scores:
   `scripts/prepare_sampled_sequence_features.py --stage sample`, `sequences`,
   `score`, `verify`, `report`. The selected ACR coordinates/receipt are included;
   observation availability, DNA and score arrays must be reconstructed locally.
7. Thirteen matched ridge models: `scripts/analyze_sampled_linear.py`.
8. Matched CNN encoding/verification: `scripts/preflight_sampled_cnn.py`; training:
   `scripts/run_sampled_cnn.py`, with settings in
   [`configs/sampled_cnn_benchmark.json`](../configs/sampled_cnn_benchmark.json).
9. Descriptive library CPM variation: `scripts/analyze_cortex_variance.py`, then
   `scripts/analyze_cortex_cpm_variance.py`, using the preceding sparse aggregates.
10. Saved earlier/current prediction comparison: `scripts/compare_saved_cnn.py`;
    read-only checkpoint-overlap audit: `scripts/verify_cnn_training_overlap.py`.

The shipped summaries are evidence from completed runs, **not substitutes for
the excluded caches/checkpoints**. Stage runners check input/code hashes and
refuse incompatible resumptions. To rebuild scientifically identical outputs,
restore matching inputs and run the stages in a separate working copy; do not
overwrite published records or edit a receipt to bypass an identity check.
Some stages intentionally depend on earlier audit artifacts. Inspect `--help`
and the stage's file checks before attempting a full rebuild. This release does
not claim that an untested, single-command end-to-end reconstruction is available.

## Neural environment

`requirements-cnn-lock.txt` records the original native Apple-silicon Python 3.12
environment; `requirements-preparation.txt` constrains the separately tested
preparation/scoring environment. They are different environments, not two
constraint files to combine. The training run used MPS; the synthetic neural
tests can run on CPU. `run_sampled_cnn.py` intentionally does not silently fall
back to CPU for the overnight experiment. The older deadline in launcher examples
must be replaced by a future timezone-qualified deadline for any new run.

The released sources do not change model definitions, CV, response scales,
sample exclusions or fitted observation weights. Hardware/library changes may
affect stochastic CNN fitting even when identities and settings are unchanged.

## Report compilation

Use your LaTeX distribution to compile `docs/research_report.tex` from the root
or from `docs/`. The public copy only fixes the table-file lookup so both layouts
work; the author's paragraphs are unchanged. Upload the table and referenced
PDF/PNG figures when using Overleaf. The author-supplied PDF is already included as
`Maize_ATAC_seq_summary.pdf`. The retained source draft predates it and
should not be treated as an exact source for the supplied PDF.

The project summary is the author-edited root `README.md`. Edit it directly;
the obsolete generated summary exports and template generator have been removed.
