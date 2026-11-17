# Getting started

The [repository README](../README.md) contains the project summary.
The full report is [available as a PDF](../Maize_ATAC_seq_summary.pdf).
Run the commands below from the repository root.

## Read and reproduce

- [Project summary](../README.md)
- [Full report (PDF)](../Maize_ATAC_seq_summary.pdf)
- [Earlier report source draft](research_report.tex): this source predates
  the supplied PDF; the PDF is the current report.
- [Methods and metric definitions](methods.md)
- [Reproduction guide](reproduction.md) and [data sources](data_sources.md)
- [Published result tables and training-overlap audit](../results/README.md)

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
separately; [the export inventory](publication_inventory.json) records the
retained scientific source files and checksums.
