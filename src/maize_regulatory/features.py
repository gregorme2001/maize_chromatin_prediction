"""ACR sequence-composition corrections and genotype-level projections."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import DataBundle, validate_data


@dataclass(frozen=True)
class MotifFeatures:
    sequence_features: pd.DataFrame
    motif_raw: np.ndarray
    motif_residual: np.ndarray
    composition: np.ndarray
    nuisance_coefficients: np.ndarray
    composition_design: np.ndarray


def sequence_features(sequences: pd.DataFrame) -> pd.DataFrame:
    if "sequence" not in sequences.columns:
        raise ValueError("Missing sequence column")
    seq = sequences["sequence"].str.upper()
    lengths = seq.str.len()
    if lengths.isna().any() or (lengths <= 0).any():
        raise ValueError("Missing or empty ACR sequence")
    return pd.DataFrame({
        "GC": (seq.str.count("G") + seq.str.count("C")) / lengths,
        "length": lengths,
    }, index=sequences.index)


def _motif_decomposition(sf: pd.DataFrame, M: np.ndarray):
    """The existing, phenotype-independent B73 sequence nuisance definition."""
    B = np.column_stack([
        np.ones(len(sf)),
        sf["GC"].to_numpy(dtype=float),
        np.log(sf["length"].to_numpy(dtype=float)),
    ])
    # This regression uses only fixed sequence properties, not the phenotype.
    A = np.linalg.lstsq(B, M, rcond=None)[0]
    R = M - B @ A
    return B, A, R


def residualize_motif_scores(
    motif_scores: pd.DataFrame, sequences: pd.DataFrame,
) -> pd.DataFrame:
    """Global per-motif residuals on [1, GC, log(length)]; no genotype inputs.

    This fixed reference-sequence transform is independent of DTS. Require exact
    ACR identity/order rather than silently aligning or dropping rows.
    """
    if not motif_scores.index.equals(sequences.index):
        raise ValueError("ACR order differs between motifs and sequences")
    if not motif_scores.index.is_unique or not motif_scores.columns.is_unique:
        raise ValueError("Duplicate ACR or motif identifiers")
    M = motif_scores.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(M).all():
        raise ValueError("Motif scores contain non-finite values")
    _, _, R = _motif_decomposition(sequence_features(sequences), M)
    return pd.DataFrame(R, index=motif_scores.index, columns=motif_scores.columns)


def calculate_motif_features(bundle: DataBundle) -> MotifFeatures:
    validate_data(bundle)
    sf = sequence_features(bundle.acr_sequences)
    M = bundle.motif_scores.to_numpy(dtype=np.float64, copy=False)
    B, A, R = _motif_decomposition(sf, M)
    P = bundle.accessibility.to_numpy(dtype=np.float64, copy=False) / 1_000_000
    raw = P @ M
    residual = P @ R
    comp = P @ B[:, 1:]
    np.testing.assert_allclose(raw, (P @ B) @ A + residual, atol=1e-7, rtol=1e-7)
    return MotifFeatures(
        sequence_features=sf,
        motif_raw=raw,
        motif_residual=residual,
        composition=comp,
        nuisance_coefficients=A,
        composition_design=B,
    )


def model_matrices(bundle: DataBundle, features: MotifFeatures) -> dict[str, np.ndarray]:
    covariates = bundle.ancestry_pcs.to_numpy(dtype=np.float64)
    return {
        "ancestry": covariates,
        "ancestry + composition": np.column_stack([covariates, features.composition]),
        "ancestry + raw motifs": np.column_stack([covariates, features.motif_raw]),
        "ancestry + residual motifs": np.column_stack([
            covariates, features.composition, features.motif_residual
        ]),
    }
