"""Load precomputed parquet inputs, refusing to silently misalign axes."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DataBundle:
    accessibility: pd.DataFrame
    phenotype: pd.DataFrame
    ancestry_pcs: pd.DataFrame
    motif_scores: pd.DataFrame
    acr_sequences: pd.DataFrame


def validate_data(bundle: DataBundle) -> None:
    x, y, pcs, m, sequences = (
        bundle.accessibility,
        bundle.phenotype,
        bundle.ancestry_pcs,
        bundle.motif_scores,
        bundle.acr_sequences,
    )
    if not x.index.is_unique or not x.columns.is_unique:
        raise ValueError("Duplicate genotype or ACR identifiers in accessibility matrix")
    if not m.index.is_unique or not m.columns.is_unique:
        raise ValueError("Duplicate ACR or motif identifiers")
    if not x.index.equals(y.index) or not x.index.equals(pcs.index):
        raise ValueError("Genotype row order differs across accessibility, DTS, and ancestry PCs")
    if not x.columns.equals(m.index) or not x.columns.equals(sequences.index):
        raise ValueError("ACR order differs across accessibility, motifs, and reference sequences")
    if list(y.columns) != ["DTS"]:
        raise ValueError("Phenotype file must have exactly one column named DTS")
    if "sequence" not in sequences.columns:
        raise ValueError("Sequence dataframe requires a sequence column")
    if not all(np.isfinite(df.to_numpy(dtype=float)).all() for df in [x, y, pcs, m]):
        raise ValueError("Numeric inputs contain missing or non-finite values")
    if (x.to_numpy(dtype=float) < 0).any():
        raise ValueError("CPM accessibility inputs must be nonnegative")
    total = x.sum(axis=1).to_numpy(dtype=float)
    if not np.allclose(total, 1_000_000, rtol=1e-4, atol=1):
        raise ValueError("Accessibility rows are not CPM normalized to total 1,000,000")


def load_data(data_dir: str | Path = "data") -> DataBundle:
    """Load all source rows for pipeline validation; use load_development_data for analysis."""
    directory = Path(data_dir)
    files = {
        "accessibility": "maize_accessibility_cpm.parquet",
        "phenotype": "maize_flowering_dts.parquet",
        "ancestry_pcs": "maize_ancestry_pcs.parquet",
        "motif_scores": "maize_acr_motif_scores.parquet",
        "acr_sequences": "maize_acr_sequences.parquet",
    }
    absent = [name for name in files.values() if not (directory / name).exists()]
    if absent:
        raise FileNotFoundError(f"Missing input parquet files in {directory}: {absent}")
    bundle = DataBundle(**{
        key: pd.read_parquet(directory / name) for key, name in files.items()
    })
    validate_data(bundle)
    # The real-data loader, unlike the generic synthetic-data validator, must
    # preserve the permanent sample identities underlying the final holdout.
    from .sample_identity import validate_analysis_order

    validate_analysis_order(bundle.accessibility.index)
    return bundle


def load_development_data(data_dir: str | Path = "data") -> DataBundle:
    """Analysis default: eligible development values only, with explicit QC exclusions.

    Raw parquets retain the frozen 169 identities. Neither held-out nor excluded
    genotype values enter this returned bundle, including the DTS column.
    """
    from .cohort import read_development_parquet

    directory = Path(data_dir)
    bundle = DataBundle(
        accessibility=read_development_parquet(directory / 'maize_accessibility_cpm.parquet'),
        phenotype=read_development_parquet(directory / 'maize_flowering_dts.parquet', columns=['DTS']),
        ancestry_pcs=read_development_parquet(directory / 'maize_ancestry_pcs.parquet'),
        motif_scores=pd.read_parquet(directory / 'maize_acr_motif_scores.parquet'),
        acr_sequences=pd.read_parquet(directory / 'maize_acr_sequences.parquet'),
    )
    validate_data(bundle)
    return bundle
