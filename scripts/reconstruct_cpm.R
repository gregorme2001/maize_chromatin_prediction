# Parameterized extraction of the legacy counts -> edgeR CPM path.
# Original scripts/process_accessibility.R is retained unchanged.
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2L)
library(Matrix)
library(edgeR)

cell_by_acr <- readRDS(file.path(args[1], "cell_by_acr.sparse.rds"))
cells <- read.csv(file.path(args[1], "cell_metadata.csv"))
stopifnot(inherits(cell_by_acr, "sparseMatrix"))
stopifnot(all(c("cellID", "genotype") %in% names(cells)))
stopifnot(nrow(cell_by_acr) == nrow(cells))
stopifnot(!anyDuplicated(colnames(cell_by_acr)))
stopifnot(!anyDuplicated(rownames(cell_by_acr)))
stopifnot(!anyNA(cells$cellID), !anyNA(cells$genotype))
stopifnot(all(rownames(cell_by_acr) == cells$cellID))

cell_genotype <- sparse.model.matrix(~ 0 + genotype, data = cells)
stopifnot(nrow(cell_genotype) == nrow(cell_by_acr))
acr_genotype_counts <- t(cell_by_acr) %*% cell_genotype
colnames(acr_genotype_counts) <- sub("^genotype", "", colnames(acr_genotype_counts))
counts_path <- file.path(args[1], "genotype_by_acr_counts.rds")
saveRDS(t(acr_genotype_counts), paste0(counts_path, ".part"))
stopifnot(file.rename(paste0(counts_path, ".part"), counts_path))
acr_genotype_cpm <- cpm(acr_genotype_counts)
write.table(acr_genotype_cpm, args[2], sep = "\t", quote = FALSE, col.names = NA)
