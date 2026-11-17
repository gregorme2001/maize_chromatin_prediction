# Run from the repository root: Rscript scripts/setup_r.R
# Install dependencies locally; never update the system R library.
script <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])
root <- dirname(dirname(normalizePath(script)))
r_version <- paste(R.version$major, strsplit(R.version$minor, ".", fixed = TRUE)[[1]][1], sep = ".")
project_library <- file.path(root, ".Rlib", r_version)
dir.create(project_library, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(project_library, .libPaths()))
options(timeout = 120)

if (Sys.info()[["sysname"]] == "Darwin" && r_version == "4.0") {
    # Bioconductor 3.12 targets R 4.0. These official Intel Mac binaries avoid
    # requiring Xcode/compiler tools on this machine. Versions are intentional.
    urls <- c(
        Rcpp = "https://cran.r-project.org/bin/macosx/contrib/4.0/Rcpp_1.0.6.tgz",
        locfit = "https://cran.r-project.org/bin/macosx/contrib/4.0/locfit_1.5-9.4.tgz",
        limma = "https://bioconductor.org/packages/3.12/bioc/bin/macosx/contrib/4.0/limma_3.46.0.tgz",
        edgeR = "https://bioconductor.org/packages/3.12/bioc/bin/macosx/contrib/4.0/edgeR_3.32.1.tgz"
    )
    for (package in names(urls)) {
        if (!requireNamespace(package, quietly = TRUE)) {
            install.packages(urls[[package]], lib = project_library,
                             repos = NULL, type = "mac.binary")
        }
        if (!requireNamespace(package, quietly = TRUE)) {
            stop("Installation failed for ", package)
        }
    }
} else {
    if (!requireNamespace("BiocManager", quietly = TRUE)) {
        install.packages("BiocManager", lib = project_library, repos = "https://cloud.r-project.org")
    }
    if (!requireNamespace("edgeR", quietly = TRUE)) {
        BiocManager::install("edgeR", lib = project_library, ask = FALSE, update = FALSE)
    }
}

library(Matrix)
library(edgeR)
counts <- matrix(c(0, 2, 1, 0, 1, 1), nrow = 3)
stopifnot(isTRUE(all.equal(cpm(counts), sweep(counts, 2, colSums(counts), "/") * 1e6)))
cat("R:", as.character(getRversion()), "\n")
cat("edgeR:", as.character(packageVersion("edgeR")), "\n")
cat("edgeR library:", find.package("edgeR"), "\n")
cat("Counts-to-CPM check passed.\n")
writeLines(capture.output(sessionInfo()), file.path(project_library, "sessionInfo.txt"))
