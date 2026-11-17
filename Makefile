# GNU Make >=4.3 (grouped targets). Run from the repository root.
# Activate .venv first, or invoke .venv/bin/make directly.
.DEFAULT_GOAL := data
.NOTPARALLEL:

PYTHON ?= .venv/bin/python
DATA_DIR ?= data
RAW := $(DATA_DIR)/raw
INTER := $(DATA_DIR)/intermediate
CELLS := $(INTER)/cells
PKG := src/maize_regulatory
PREPARE = $(PYTHON) scripts/prepare_data.py --data-dir $(DATA_DIR)
COMMON := Makefile scripts/prepare_data.py $(PKG)/pipeline.py $(PKG)/downloads.py \
          requirements-preparation.txt pyproject.toml
IDENTITIES := $(wildcard $(PKG)/resources/*.txt $(PKG)/resources/*.json)
R_READY ?= .Rlib/.ready
R_PACKAGES := $(wildcard .Rlib/*/edgeR/DESCRIPTION .Rlib/*/limma/DESCRIPTION \
                        .Rlib/*/locfit/DESCRIPTION .Rlib/*/sessionInfo.txt)

# Preserve support for either legacy supplied CSVs or their source workbooks.
PHENOTYPES := $(or $(firstword $(wildcard $(RAW)/maize282_blup_phenotypes.csv \
                    $(RAW)/Peiffer2014Genetics_blupPhenos20150325.xlsx)),\
                    $(RAW)/maize282_blup_phenotypes.csv)
METADATA := $(or $(firstword $(wildcard $(RAW)/cell_assignments.csv $(RAW)/Dryad_Data_S2.xlsx)),\
                    $(RAW)/Dryad_Data_S2.xlsx)
GENOME := $(or $(firstword $(wildcard $(RAW)/Zm-B73-REFERENCE-NAM-5.0.fa)),\
                    $(RAW)/Zm-B73-REFERENCE-NAM-5.0.fa.gz)
RAW_CELLS := $(RAW)/maize_282_scATAC_ACRs_x_barcodes.sparse.rds
PFMS := $(RAW)/jaspar_plants_2026.txt
CELL_FILES := $(CELLS)/cell_by_acr.sparse.rds $(CELLS)/cell_metadata.csv \
              $(CELLS)/acr_index.csv $(CELLS)/matrix_summary.csv
CELL_OUTPUTS := $(CELL_FILES) $(CELLS)/sources.json
CPM_OUTPUTS := $(INTER)/maize_ACR_genotype_cpm.tsv $(CELLS)/genotype_by_acr_counts.rds \
               $(INTER)/cpm_sources.json
SAMPLE_OUTPUTS := $(DATA_DIR)/maize_accessibility_cpm.parquet \
                  $(DATA_DIR)/maize_flowering_dts.parquet $(DATA_DIR)/maize_ancestry_pcs.parquet \
                  $(INTER)/accessibility_provenance.json
SEQUENCE_OUTPUTS := $(DATA_DIR)/maize_acr_sequences.parquet $(INTER)/sequences_provenance.json
MOTIF_OUTPUTS := $(DATA_DIR)/maize_acr_motif_scores.parquet $(INTER)/motifs_provenance.json
PARQUETS := $(filter %.parquet,$(SAMPLE_OUTPUTS) $(SEQUENCE_OUTPUTS) $(MOTIF_OUTPUTS))

.PHONY: data downloads cells accessibility sequences motifs check test help
data: $(INTER)/validation.json
downloads: $(RAW_CELLS) $(METADATA) $(GENOME) $(PFMS)
cells: $(CELL_OUTPUTS)
accessibility: $(SAMPLE_OUTPUTS)
sequences: $(SEQUENCE_OUTPUTS)
motifs: $(MOTIF_OUTPUTS)

$(RAW_CELLS):
	$(PYTHON) scripts/download_inputs.py --raw-dir $(RAW) --only cells
$(RAW)/Dryad_Data_S2.xlsx:
	$(PYTHON) scripts/download_inputs.py --raw-dir $(RAW) --only metadata
$(RAW)/Zm-B73-REFERENCE-NAM-5.0.fa.gz:
	$(PYTHON) scripts/download_inputs.py --raw-dir $(RAW) --only genome
$(PFMS):
	$(PYTHON) scripts/download_inputs.py --raw-dir $(RAW) --only motifs

$(RAW)/maize282_blup_phenotypes.csv $(RAW)/mdp_genotype.hmp.txt:
	@echo "Supply the original $@ before building data."
	@exit 1

$(R_READY): scripts/setup_r.R .Renviron
	Rscript scripts/setup_r.R
	@touch $@

# The manifest records completion. Unchanged ACR CSVs retain their mtime, so a
# cell-label update does not unnecessarily rebuild reference sequence/motif data.
$(CELLS)/sources.json: $(RAW_CELLS) $(METADATA) scripts/preserve_cell_accessibility.R \
                       $(COMMON) $(PKG)/preparation.py $(R_READY)
	$(PREPARE) --stage cells --overwrite

$(CELL_FILES) &: | $(CELLS)/sources.json
	@if $(foreach file,$(CELL_FILES),test -f $(file) &&) true; then :; \
	else $(PREPARE) --stage cells --overwrite; fi

# One recipe builds each group, even if several products are missing.
$(CPM_OUTPUTS) &: $(CELL_OUTPUTS) scripts/reconstruct_cpm.R $(COMMON) $(R_PACKAGES)
	$(PREPARE) --stage cpm --reuse-intermediates --overwrite

$(SAMPLE_OUTPUTS) &: $(CPM_OUTPUTS) $(PHENOTYPES) $(RAW)/mdp_genotype.hmp.txt \
                      $(IDENTITIES) $(COMMON) $(PKG)/preparation.py $(PKG)/sample_identity.py \
                      $(PKG)/evaluation.py $(PKG)/models.py
	$(PREPARE) --stage accessibility --reuse-intermediates --overwrite

# Sequence and motif nodes depend on the ACR index, independently of DTS/PCs/CPM.
$(SEQUENCE_OUTPUTS) &: $(CELLS)/acr_index.csv $(GENOME) $(COMMON) $(PKG)/spatial.py
	$(PREPARE) --stage sequences --overwrite

$(MOTIF_OUTPUTS) &: $(DATA_DIR)/maize_acr_sequences.parquet $(PFMS) $(COMMON) $(PKG)/scoring.py
	$(PREPARE) --stage motifs --overwrite

$(INTER)/validation.json: $(SAMPLE_OUTPUTS) $(SEQUENCE_OUTPUTS) $(MOTIF_OUTPUTS) $(COMMON) $(PKG)/data.py \
                           $(PKG)/sample_identity.py $(IDENTITIES)
	$(PREPARE) --stage validate

check:
	$(PREPARE) --check
test:
	$(PYTHON) -m pytest -q
help:
	@echo "make data          Build/validate all five parquets; unchanged nodes are reused."
	@echo "make -n data       Show the commands that would run."
	@echo "make downloads     Fetch only missing public inputs."
	@echo "make accessibility / sequences / motifs / cells   Build a selected branch."
	@echo "make check / test  Check source files / run synthetic tests."
	@echo "Requires GNU Make >=4.3. On this Mac: .venv/bin/python scripts/setup_make.py"
