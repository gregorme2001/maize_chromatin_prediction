# Minimal R/Matrix bridge: stream sparse library-level detections to Python.
# Only development rows are selected before examining matrix values.
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 8L)
suppressPackageStartupMessages(library(Matrix))
meta <- read.csv(args[2], stringsAsFactors = FALSE)
selected <- read.csv(args[3], stringsAsFactors = FALSE)
acrs <- read.csv(args[4], stringsAsFactors = FALSE)
chunk_size <- as.integer(args[6])
ngroups <- as.integer(args[7])
stopifnot(chunk_size > 0, ngroups > 0, nrow(selected) > 0, nrow(acrs) > 0)
message("Reading sparse RDS; no dense cell-by-ACR allocation")
x <- readRDS(args[1])
stopifnot(inherits(x, "dgCMatrix"), nrow(x) == nrow(meta),
          identical(rownames(x), meta$cellID), !anyDuplicated(colnames(x)),
          identical(meta$matrix_row, seq_len(nrow(meta))),
          !anyDuplicated(selected$matrix_row),
          all(selected$matrix_row >= 1 & selected$matrix_row <= nrow(x)),
          all(selected$library_group >= 0 & selected$library_group < ngroups),
          identical(colnames(x)[acrs$matrix_column], acrs$ACR))
indicator <- sparseMatrix(i = selected$library_group + 1L,
                          j = seq_len(nrow(selected)), x = 1,
                          dims = c(ngroups, nrow(selected)))
depth <- numeric(nrow(selected))
source_nonbinary <- 0
# A FIFO inside the writable output directory avoids restricted /dev/stdout.
con <- file(args[8], open = "wb")
cat("READY\n")
flush(stdout())
writeBin(charToRaw("MAIZEAUDIT1\n"), con)
put <- function(v) writeBin(as.integer(v), con, size = 4L, endian = "little")
put(c(ngroups, nrow(acrs)))
for (start in seq.int(1L, nrow(acrs), by = chunk_size)) {
    at <- start:min(start + chunk_size - 1L, nrow(acrs))
    block <- x[selected$matrix_row, acrs$matrix_column[at], drop = FALSE]
    stopifnot(all(is.finite(block@x)), all(block@x >= 0))
    source_nonbinary <- source_nonbinary + sum(block@x != 0 & block@x != 1)
    block <- drop0(block)
    # Y counts nuclei with a positive event, even if a future source stores counts.
    block@x[] <- 1
    depth <- depth + rowSums(block)
    counts <- as(indicator %*% block, "dgCMatrix")
    stopifnot(all(counts@x == floor(counts@x)))
    put(c(start - 1L, length(at), length(counts@x)))
    put(counts@p)
    put(counts@i)
    put(counts@x)
    flush(con)
    if (start == 1L || start %% (chunk_size * 20L) == 1L)
        message("Streamed ", max(at), "/", nrow(acrs), " ACRs")
}
put(c(-1L, 0L, 0L))
close(con)
qc <- data.frame(matrix_row = selected$matrix_row, detected_acrs = depth)
write.csv(qc, gzfile(file.path(args[5], "cell_detected_acrs.csv.gz")), row.names = FALSE)
write.csv(data.frame(nonbinary_positive_source_entries = source_nonbinary,
                     n_selected_cells = nrow(selected), n_acrs = nrow(acrs),
                     source_sparse_bytes = as.numeric(object.size(x))),
          file.path(args[5], "stream_validation.csv"), row.names = FALSE)
message("Sparse stream complete")
