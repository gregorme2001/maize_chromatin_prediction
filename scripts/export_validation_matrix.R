#load reconstructed inverse-normalized ACR x genotype matrix
acr_genotype_int <- readRDS("maize_ACR_genotype_INT.rds")

#export final reconstructed matrix for validation in Python
write.table(
    acr_genotype_int,
    "maize_ACR_genotype_INT.tsv",
    sep = "\t",
    quote = FALSE,
    col.names = NA
)