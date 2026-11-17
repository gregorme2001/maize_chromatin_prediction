"""Empirical binary-accessibility variation at three nested sampling levels.

No hypothesis tests, latent variance estimates, or independence assumptions.
Genotypes receive equal weight; cells receive equal weight within genotype.
"""

import numpy as np
import pandas as pd


VARIANCE_COLUMNS = [
    'within_genotype_library', 'between_libraries_within_genotype',
    'between_genotype_pooled_means',
]


def binary_nested_variance(counts, nuclei, genotype_indices):
    """Summarize library-by-ACR positive-nucleus counts without expanding cells.

    Each row is one observed (genotype, library) group. Population variances
    (denominator N, not N-1) describe the observed empirical distribution.
    Its sampling scheme chooses genotype uniformly, then one of its cells
    uniformly. Library weights within genotype are therefore N_gl / N_g.
    """
    y = np.asarray(counts, dtype=float)
    n = np.asarray(nuclei, dtype=float)
    g = np.asarray(genotype_indices)
    if (y.ndim != 2 or n.shape != (len(y),) or g.shape != (len(y),)
            or len(y) == 0 or y.shape[1] == 0
            or not np.isfinite(y).all() or not np.isfinite(n).all()
            or np.any(n <= 0) or np.any(n != np.floor(n))
            or np.any(y < 0) or np.any(y > n[:, None])
            or np.any(y != np.floor(y))):
        raise ValueError('Require integer detection counts between zero and positive nucleus counts')
    if g.dtype.kind not in 'iu' or np.any(g < 0):
        raise ValueError('Genotype indices must be nonnegative integers')
    groups = np.unique(g)
    if not np.array_equal(groups, np.arange(len(groups))):
        raise ValueError('Genotype indices must be contiguous and each genotype observed')
    p = y / n[:, None]
    means, within, library = [], [], []
    # Explicit centered sums avoid cancellation in almost constant groups.
    for genotype in groups:
        selected = g == genotype
        w = n[selected] / n[selected].sum()
        local = p[selected]
        mean = w @ local
        means.append(mean)
        within.append(w @ (local * (1 - local)))
        library.append(w @ (local - mean) ** 2)
    means = np.asarray(means)
    grand_mean = means.mean(axis=0)
    result = pd.DataFrame({
        'mean_detection_fraction': grand_mean,
        'within_genotype_library': np.mean(within, axis=0),
        'between_libraries_within_genotype': np.mean(library, axis=0),
        'between_genotype_pooled_means': np.mean((means - grand_mean) ** 2, axis=0),
        'total_empirical_variance': grand_mean * (1 - grand_mean),
    })
    result['decomposition_error'] = (
        result[VARIANCE_COLUMNS].sum(axis=1) - result.total_empirical_variance)
    if not np.allclose(result.decomposition_error, 0, rtol=0, atol=1e-12):
        raise AssertionError('Nested variances do not reconstruct total binary variance')
    return result, means


def summarize_components(frame):
    """Summarize ACRs equally; sqrt(mean variance) is an RMS SD, not mean SD."""
    records = []
    for scope, part in frame.groupby('scope', sort=False):
        for name in VARIANCE_COLUMNS:
            values = part[name].to_numpy()
            records.append(dict(
                scope=scope, quantity=name, n_acrs=len(part),
                mean_variance=float(values.mean()),
                median_variance=float(np.median(values)),
                variance_q10=float(np.quantile(values, .1)),
                variance_q90=float(np.quantile(values, .9)),
                rms_sd_percentage_points=float(100 * np.sqrt(values.mean())),
                median_sd_percentage_points=float(100 * np.median(np.sqrt(values))),
            ))
    return pd.DataFrame(records)
