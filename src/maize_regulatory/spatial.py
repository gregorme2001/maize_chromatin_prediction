"""Descriptive genomic positions, neighbor covariance, and TF-specific local ACR tracks."""

import re

import numpy as np
import pandas as pd

ACR_ID = re.compile(r"^chr(?P<chrom>\d+)_(?P<start>\d+)_(?P<end>\d+)$")


def parse_acr_coordinates(acr_ids) -> pd.DataFrame:
    records = []
    for acr in acr_ids:
        match = ACR_ID.fullmatch(str(acr))
        if not match:
            raise ValueError(f"Invalid ACR ID: {acr!r}")
        chrom, start, end = map(int, match.groups())
        if end <= start:
            raise ValueError(f"ACR has nonpositive length: {acr!r}")
        records.append((chrom, start, end, (start + end) / 2))
    return pd.DataFrame(records, columns=["chrom", "start", "end", "midpoint"], index=pd.Index(acr_ids, name="ACR"))


def local_motif_track(
    accessibility: pd.DataFrame,
    motif_scores: pd.DataFrame,
    *,
    genotype: str,
    motif_id: str,
    chromosome: int,
    start: int | None = None,
    end: int | None = None,
) -> pd.DataFrame:
    """One ACR per row: position, CPM, signed motif score, and their product.

    A signed log-likelihood-ratio score is NOT binding intensity. This track is
    descriptive only; in particular negative weighted scores are not evidence
    for negative physical occupancy. No phenotype is used.
    """
    if not accessibility.columns.equals(motif_scores.index):
        raise ValueError("ACR order must match before multiplying accessibility and motif scores")
    coords = parse_acr_coordinates(motif_scores.index)
    mask = coords["chrom"].eq(chromosome)
    if start is not None:
        mask &= coords["end"].ge(start)
    if end is not None:
        mask &= coords["start"].le(end)
    ids = coords.index[mask]
    subset = coords.loc[ids].copy()
    subset["accessibility_cpm"] = accessibility.loc[genotype, ids].to_numpy(dtype=float)
    subset["motif_score"] = motif_scores.loc[ids, motif_id].to_numpy(dtype=float)
    subset["weighted_score"] = subset["accessibility_cpm"] / 1_000_000 * subset["motif_score"]
    return subset.sort_values("midpoint")


def nearby_acr_correlations(accessibility: pd.DataFrame, n_neighbors: int = 10) -> pd.DataFrame:
    """Pearson correlations across genotypes for nearby ACR pairs, without n_ACR^2 allocation."""
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be >= 1")
    coords = parse_acr_coordinates(accessibility.columns)
    chrom = coords["chrom"].to_numpy()
    mid = coords["midpoint"].to_numpy()
    X = accessibility.to_numpy(dtype=float)
    centered = X - X.mean(axis=0)
    norms = np.linalg.norm(centered, axis=0)
    order = np.lexsort((mid, chrom))
    parts = []
    for offset in range(1, n_neighbors + 1):
        a = order[:-offset]
        b = order[offset:]
        valid = (chrom[a] == chrom[b]) & (norms[a] > 0) & (norms[b] > 0)
        a, b = a[valid], b[valid]
        if len(a) == 0:
            continue
        corr = np.sum(centered[:, a] * centered[:, b], axis=0) / (norms[a] * norms[b])
        parts.append(pd.DataFrame({
            "acr_a": accessibility.columns[a].values,
            "acr_b": accessibility.columns[b].values,
            "distance_bp": np.abs(mid[b] - mid[a]),
            "correlation": np.clip(corr, -1, 1),
        }))
    if not parts:
        return pd.DataFrame(columns=["acr_a", "acr_b", "distance_bp", "correlation"])
    return pd.concat(parts, ignore_index=True)
