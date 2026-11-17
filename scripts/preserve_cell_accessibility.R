# Preserve the original binary accessibility values as sparse cells x ACRs.
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 3L)
library(Matrix)

cell_accessibility <- readRDS(args[1])
stopifnot(inherits(cell_accessibility, "sparseMatrix"))
if (args[2] == "-") {
    cells <- data.frame(cellID = colnames(cell_accessibility))
} else {
    cells <- read.csv(args[2], stringsAsFactors = FALSE)
    stopifnot(all(c("cellID", "genotype") %in% names(cells)))
    stopifnot(!anyNA(cells$genotype))
}
stopifnot(ncol(cell_accessibility) == nrow(cells))
stopifnot(!anyDuplicated(colnames(cell_accessibility)))
stopifnot(!anyDuplicated(rownames(cell_accessibility)))
stopifnot(!anyNA(cells$cellID))
stopifnot(all(colnames(cell_accessibility) == cells$cellID))

cell_by_acr <- t(cell_accessibility)
stopifnot(inherits(cell_by_acr, "sparseMatrix"))
stopifnot(identical(rownames(cell_by_acr), as.character(cells$cellID)))
dir.create(args[3], recursive = TRUE, showWarnings = FALSE)
atomic_rds <- function(object, filename) {
    path <- file.path(args[3], filename)
    saveRDS(object, paste0(path, ".part"))
    stopifnot(file.rename(paste0(path, ".part"), path))
}
atomic_csv <- function(object, filename) {
    path <- file.path(args[3], filename)
    write.csv(object, paste0(path, ".part"), row.names = FALSE)
    # Keep identical ACR indexes unchanged so metadata edits do not rescore motifs.
    if (file.exists(path) && identical(unname(tools::md5sum(path)),
                                      unname(tools::md5sum(paste0(path, ".part"))))) {
        unlink(paste0(path, ".part"))
    } else {
        stopifnot(file.rename(paste0(path, ".part"), path))
    }
}
atomic_rds(cell_by_acr, "cell_by_acr.sparse.rds")
cells$matrix_row <- seq_len(nrow(cells))
atomic_csv(cells, "cell_metadata.csv")
atomic_csv(data.frame(matrix_column = seq_len(ncol(cell_by_acr)),
                      ACR = colnames(cell_by_acr)), "acr_index.csv")
atomic_csv(data.frame(n_cells = nrow(cell_by_acr), n_acrs = ncol(cell_by_acr),
                      n_nonzero = nnzero(cell_by_acr),
                      genotype_labels_available = "genotype" %in% names(cells),
                      sparse_bytes_in_memory = as.numeric(object.size(cell_by_acr)),
                      rds_bytes = file.info(file.path(args[3], "cell_by_acr.sparse.rds"))$size),
           "matrix_summary.csv")
