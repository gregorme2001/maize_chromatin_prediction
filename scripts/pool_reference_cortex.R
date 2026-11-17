# Pool ONLY selected development cortex nuclei, in bounded ACR chunks.
# The RDS is sparse; no dense cell-by-ACR matrix is created.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==6L)
suppressPackageStartupMessages(library(Matrix))
metadata <- read.csv(args[2],stringsAsFactors=FALSE)
selected <- read.csv(args[3],stringsAsFactors=FALSE)
acrs <- read.csv(args[4],stringsAsFactors=FALSE)
ng <- as.integer(args[6])
stopifnot(ng>0L,nrow(selected)>0L,!anyDuplicated(selected$matrix_row),
          all(selected$genotype_index>=0L & selected$genotype_index<ng),
          all(metadata$celltype_full[selected$matrix_row]=="cortex"))
message("Reading cached sparse matrix for development-only cortex pooling")
x <- readRDS(args[1])
stopifnot(inherits(x,"dgCMatrix"),nrow(x)==nrow(metadata),
          identical(rownames(x),metadata$cellID),
          identical(colnames(x),acrs$ACR),
          identical(acrs$matrix_column,seq_len(ncol(x))),
          all(selected$matrix_row>=1L & selected$matrix_row<=nrow(x)))
indicator <- sparseMatrix(i=selected$genotype_index+1L,j=seq_len(nrow(selected)),
                          x=1,dims=c(ng,nrow(selected)))
con <- file(paste0(args[5],".part"),open="wb")
for(start in seq.int(1L,ncol(x),by=256L)) {
    at <- start:min(start+255L,ncol(x))
    block <- drop0(x[selected$matrix_row,at,drop=FALSE])
    stopifnot(all(is.finite(block@x)),all(block@x>=0))
    block@x[] <- 1
    counts <- as.matrix(indicator %*% block)
    stopifnot(all(counts==floor(counts)),all(counts>=0))
    # R column-major: all genotypes at ACR 1, then all at ACR 2, etc.
    writeBin(as.integer(counts),con,size=4L,endian="little")
    if(start==1L || start%%(256L*40L)==1L)
        message("Pooled ",max(at),"/",ncol(x)," ACRs")
}
close(con)
stopifnot(file.rename(paste0(args[5],".part"),args[5]))
message("Pooled cortex counts complete")
