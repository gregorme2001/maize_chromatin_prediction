# Data sources and excluded inputs

No source dataset, supplied workbook, reconstructed DNA, SNP-call matrix or
phenotype values are distributed in this repository. Public source links below
identify upstream resources; their terms apply separately. Small derived result
summaries, identity lists, sample coordinates and scientific checksums accompany
the report.

| Resource | Source / local role |
|---|---|
| Marand et al., maize cell-type-specific cis regulation | [Science paper](https://doi.org/10.1126/science.ads6601), [GEO GSE273478](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE273478), [Dryad metadata](https://doi.org/10.5061/dryad.nk98sf82v) |
| Sparse nucleus–ACR detections | `data/raw/maize_282_scATAC_ACRs_x_barcodes.sparse.rds`; downloaded through `scripts/download_inputs.py` |
| Genotype, cell-state and library assignments | Upstream Dryad workbook, or verified supplied `data/raw/cell_assignments.csv`; nucleus identities must match sparse matrix order |
| B73-v5 reference | [MaizeGDB](https://download.maizegdb.org/Zm-B73-REFERENCE-NAM-5.0/), `Zm-B73-REFERENCE-NAM-5.0` FASTA |
| Plant TF probability-frequency matrices | JASPAR 2026 CORE Plants non-redundant collection, [resource paper](https://doi.org/10.1093/nar/gkaf1209); `jaspar_plants_2026.txt` |
| B73-v5, 282-panel SNP calls | [Panzea/MaizeGDB source folder](https://iastate.box.com/s/9edkm5f51q88ltzrb0hxaz4p36fdsr7q/folder/364949363629); exact files/checksums in [`configs/panzea_v5_sources.json`](../configs/panzea_v5_sources.json) |
| Ten ancestry PCs | Separate TASSEL tutorial `mdp_genotype.hmp.txt` panel; [source documentation](https://tassel.maizegenetics.net/user_manual/appendix/tasseltutorialdatasets/) |

## Historical identity dependency

The 169-line source order came from an earlier alignment with a flowering-time
table. The original preparation interface still expects that supplied table
(`maize282_blup_phenotypes.csv`, or its original workbook) as well as the supplied
TASSEL marker file when reconstructing the original five parquets. It is not a
new response used by this report. The accessibility analyses use only the frozen
development identities and cortex accessibility; the permanent holdout and DTS
are not analyzed. The matching table and marker file are not bundled here.

The frozen public identity lists and their extraction receipt are in
[`src/maize_regulatory/resources/`](../src/maize_regulatory/resources/README.md).
Do not regenerate them on a reduced genotype list.

## Local storage

Keep downloaded and supplied files in `data/raw/`, generated caches in
`data/intermediate/`, and original parquet products directly in `data/`.
These are Git-ignored. Never infer a missing SNP call is reference, silently
reorder nuclei/genotypes/ACRs, or replace the sparse matrix with the authors'
different normalized 58k-ACR matrix.

The public export contains code and small summaries, not a complete data archive.
Full reconstruction requires acquisition of these inputs and substantial disk
space. The published hashes let an authorized local reconstruction be checked
against the completed research run.
