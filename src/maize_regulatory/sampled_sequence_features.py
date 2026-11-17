"""Frozen spatial sampling and exact sequence-summary helpers; no responses."""

from functools import lru_cache

import numpy as np

from .kmer_pilot import count_kmers, kmer_dictionary


def spatial_sample(census, *, min_genotypes=20, quota=25, seed=42):
    """Uniform without replacement within blocks, invariant to input row order."""
    required = ['global_acr_index', 'block', 'context_n_available']
    if (not census.index.is_unique or not census.global_acr_index.is_unique
            or census[required].isna().any().any() or quota < 1 or min_genotypes < 1):
        raise ValueError('Invalid census identities or sampling settings')
    eligible = census.loc[census.context_n_available >= min_genotypes].sort_values('global_acr_index')
    if eligible.empty:
        raise ValueError('No eligible ACRs')
    rng = np.random.default_rng(seed)
    parts = []
    for _, part in eligible.groupby('block', sort=True):
        n = min(quota, len(part))
        selected = part.iloc[rng.choice(len(part), n, replace=False)].copy()
        selected['eligible_ACRs_in_block'] = len(part)
        selected['selected_ACRs_in_block'] = n
        selected['selection_probability'] = n/len(part)
        parts.append(selected)
    import pandas as pd
    result = pd.concat(parts).sort_values('global_acr_index')
    result['sample_acr_index'] = np.arange(len(result), dtype=np.int32)
    return result


@lru_cache(maxsize=1)
def _canonical_mapping():
    return kmer_dictionary(5)[3]


def canonical_word_counts(fragments):
    """Sum counts over separate fragments without creating cross-fragment sites."""
    mapping = _canonical_mapping()
    result = np.zeros(512, dtype=np.int32)
    for fragment in fragments:
        if len(fragment) >= 5:
            result += np.bincount(mapping, weights=count_kmers(fragment), minlength=512).astype(np.int32)
    if result.sum() == 0:
        raise ValueError('No valid five-mer placements')
    return result


def pool_log_mean_lr(scores, fragment_lengths, motif_lengths):
    """Pool mean LRs with motif-specific placement weights, then take log2.

    Each score already averages both strands within one fragment. Disconnected
    flanks are never concatenated and log scores are never averaged directly.
    """
    scores = np.asarray(scores, float)
    lengths, widths = np.asarray(fragment_lengths), np.asarray(motif_lengths)
    weights = np.maximum(lengths[:, None]-widths[None, :]+1, 0)
    if (scores.shape != weights.shape or not np.isfinite(scores).all()
            or np.any(weights <= 0)):
        raise ValueError('Every scored fragment must have positive valid placements')
    return np.logaddexp2.reduce(scores+np.log2(weights), axis=0)-np.log2(weights.sum(axis=0))
