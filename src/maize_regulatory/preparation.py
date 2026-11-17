"""Legacy genotype matching and ancestry preprocessing, without model evaluation."""

import re

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def normalize_genotype(name) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(name).upper())


def normalize_reconstructed_genotype(name) -> str:
    return normalize_genotype(re.sub(r"^282set_", "", str(name)).replace("Goodman-Buckler", ""))


def require_unique(index: pd.Index, label: str) -> None:
    if not index.is_unique or index.hasnans:
        raise ValueError(f"Missing or duplicate {label} identifiers")


def match_phenotypes(published_names: pd.Index, pheno: pd.DataFrame):
    """Keep unique phenotype matches in published order, exactly as legacy cells 4–6."""
    required = {"Family_Inbred_Name", "DTS"}
    if not required.issubset(pheno.columns):
        raise ValueError(f"Phenotype file requires {sorted(required)}")
    published = pd.Index([normalize_genotype(n) for n in published_names], name="genotype")
    require_unique(published, "normalized published genotype")
    names = pheno["Family_Inbred_Name"].dropna().unique()
    require_unique(pd.Index([normalize_genotype(n) for n in names]), "normalized phenotype name")
    normalized = pheno["Family_Inbred_Name"].map(normalize_genotype)
    counts = normalized.value_counts()
    keep = published[published.isin(counts.index[counts == 1])]
    matched = pheno.loc[normalized.isin(keep), ["DTS"]].copy()
    matched.index = pd.Index(normalized.loc[normalized.isin(keep)], name="genotype")
    # Explicit lookup restores published order after phenotype-table selection.
    y = matched.loc[keep].astype(float)
    if not np.isfinite(y.to_numpy()).all():
        raise ValueError("Missing/nonfinite DTS; refusing to change sample exclusions")
    excluded = published[~published.isin(keep)].tolist()
    return y, excluded


def ancestry_pcs(snp_data: pd.DataFrame, n_components: int = 10) -> pd.DataFrame:
    """Legacy two-character dosage parsing, mean imputation and full-panel PCA."""
    names = snp_data.columns[11:]
    normalized = pd.Index([normalize_genotype(n) for n in names], name="genotype")
    require_unique(normalized, "SNP genotype")
    G = np.full((len(names), len(snp_data)), np.nan)
    for snp_index, (_, row) in enumerate(snp_data.iterrows()):
        alleles = str(row["alleles"]).split("/")
        if len(alleles) != 2:
            continue
        first, second = alleles
        for genotype_index, call in enumerate(row.iloc[11:]):
            if pd.notna(call) and len(call) == 2 and all(a in {first, second} for a in call):
                G[genotype_index, snp_index] = call.count(second)
    G = G[:, ~np.all(np.isnan(G), axis=0)]
    if not G.shape[1]:
        raise ValueError("No usable SNPs under legacy two-character dosage parsing")
    means = np.nanmean(G, axis=0)
    missing_rows, missing_cols = np.where(np.isnan(G))
    G[missing_rows, missing_cols] = means[missing_cols]
    G = G[:, G.std(axis=0) > 0]
    if min(G.shape) < n_components:
        raise ValueError("Not enough variable SNPs/genotypes for requested ancestry PCs")
    PC = PCA(n_components=n_components, svd_solver="full").fit_transform(
        StandardScaler().fit_transform(G)
    )
    return pd.DataFrame(PC, index=normalized,
                        columns=[f"genetic_PC{i}" for i in range(1, n_components + 1)])


def align_cpm(cpm_acr_by_genotype: pd.DataFrame, genotype_order: pd.Index) -> pd.DataFrame:
    """Explicit legacy subset/ordering; no new sample exclusions or ACR normalization."""
    cpm = cpm_acr_by_genotype.copy()
    cpm.columns = pd.Index([
        normalize_reconstructed_genotype(n)
        for n in cpm.columns
    ])
    require_unique(cpm.columns, "CPM genotype")
    require_unique(cpm.index, "ACR")
    missing = genotype_order.difference(cpm.columns)
    if len(missing):
        raise ValueError(f"Missing required CPM genotypes: {missing.tolist()}")
    X = cpm.T.loc[genotype_order].copy()
    if not X.index.equals(genotype_order):
        raise ValueError("CPM genotype alignment failed")
    if not np.isfinite(X.to_numpy()).all() or (X.to_numpy() < 0).any():
        raise ValueError("Invalid CPM values")
    if not np.allclose(X.sum(axis=1), 1e6, rtol=1e-4, atol=1):
        raise ValueError("CPM rows must sum to 1,000,000 over the original ACR universe")
    return X
