"""One-motif regressions of bulk accessibility, with composition adjustment.

These are across-ACR associations, not genotype prediction or TF activity tests.
Batchwise Frisch-Waugh-Lovell fits avoid an ACR-by-ACR covariance matrix.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.stats import false_discovery_control, t

from .features import sequence_features
from .spatial import parse_acr_coordinates


def gc_spline_basis(gc, knots):
    """Four non-intercept natural cubic terms from five fixed GC knots.

Cardinal spline values span the natural spline space on [min GC, max GC].
The five cardinal terms sum to one, so omit the first when using an intercept.
Knots are min, quartiles and max; no extrapolation or response-based tuning.
"""
    gc, knots = np.asarray(gc, dtype=float), np.asarray(knots, dtype=float)
    if (knots.shape != (5,) or not np.isfinite(knots).all()
            or np.any(np.diff(knots) <= 0)):
        raise ValueError('Require five finite, distinct, increasing GC knots')
    if not np.isfinite(gc).all() or np.any((gc < knots[0]) | (gc > knots[-1])):
        raise ValueError('GC must be finite and inside the fixed boundary knots')
    return CubicSpline(knots, np.eye(5), bc_type='natural', extrapolate=False)(gc)[:, 1:]


def composition_design(sequences, config):
    if not sequences.index.is_unique:
        raise ValueError('Duplicate ACR identities')
    annotation = parse_acr_coordinates(sequences.index)
    sf = sequence_features(sequences)
    if not np.array_equal(sf.length.to_numpy(), (annotation.end - annotation.start).to_numpy()):
        raise ValueError('Sequence lengths disagree with ACR coordinates')
    if config['gc_spline_df'] != 4 or config['gc_knot_quantiles'] != [0, .25, .5, .75, 1]:
        raise ValueError('This analysis prespecifies a four-df natural GC spline')
    knots = np.quantile(sf.GC, config['gc_knot_quantiles'])
    spline = gc_spline_basis(sf.GC, knots)
    base = pd.DataFrame({'intercept': np.ones(len(sf))}, index=sf.index)
    for k in range(4):
        base[f'GC_spline_{k+1}'] = spline[:, k]
    base['log_length'] = np.log(sf.length)
    for chrom in sorted(annotation.chrom.unique())[1:]:
        base[f'chrom_{chrom}'] = annotation.chrom.eq(chrom).astype(float)
    annotation['GC'], annotation['length'] = sf.GC, sf.length
    return base, annotation, knots


def bulk_responses(accessibility, subsets):
    """Equal genotype weights; explicitly average CPM before taking natural log."""
    if not accessibility.index.equals(subsets.index):
        raise ValueError('Genotype subset identity/order mismatch')
    if (not accessibility.index.is_unique or not accessibility.columns.is_unique
            or set(subsets) != {'A', 'B'}):
        raise ValueError('Require unique identities and both genotype subsets')
    values = accessibility.to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('Require finite, nonnegative CPM')
    if not np.allclose(values.sum(axis=1), 1e6, rtol=1e-4, atol=1):
        raise ValueError('Expected CPM totals of one million')
    result = pd.DataFrame(index=accessibility.columns)
    result['mean_cpm'] = values.mean(axis=0)
    result['all'] = np.log1p(result.mean_cpm)
    for label in ['A', 'B']:
        result[label] = np.log1p(values[subsets.eq(label).to_numpy()].mean(axis=0))
    return result


def block_membership(annotation, width):
    """Disjoint, chromosome-origin grids; each ACR midpoint belongs to one block."""
    if width <= 0:
        raise ValueError('Block width must be positive')
    keys = pd.MultiIndex.from_arrays([annotation.chrom, (annotation.midpoint // width).astype(int)])
    codes, unique = pd.factorize(keys, sort=True)
    blocks = pd.DataFrame(unique.tolist(), columns=['chrom', 'block_number'])
    blocks['start_bp'] = blocks.block_number * width
    blocks['end_bp'] = blocks.start_bp + width
    blocks['n_acrs'] = np.bincount(codes)
    return codes, blocks


def _projector(design):
    z = np.asarray(design, dtype=float)
    if (z.ndim != 2 or not np.isfinite(z).all()
            or np.linalg.matrix_rank(z) != z.shape[1]):
        raise ValueError('Baseline design must be finite and full rank')
    if len(z) <= z.shape[1] + 2:
        raise ValueError('Insufficient ACR residual degrees of freedom')
    return np.linalg.qr(z, mode='reduced')[0]


def _moments(x, y):
    xx = np.einsum('ij,ij->j', x, x)
    yy = np.einsum('ij,ij->j', y, y)
    if np.any(xx <= 1e-18) or np.any(yy <= 1e-18):
        raise ValueError('Constant response or motif after baseline adjustment; no silent dropping')
    xy = x.T @ y
    beta = xy / xx[:, None]
    partial_r = xy / np.sqrt(xx[:, None] * yy[None, :])
    return xx, yy, beta, np.clip(partial_r, -1, 1)


def clustered_standard_errors(x, y, beta, xx, codes, n_parameters):
    """CR1 variance of the motif coefficient after FWL residualization.

The finite-sample correction counts ALL parameters in the expanded model,
not just the residualized motif. Group scores use expanded-model residuals.
"""
    n, m = x.shape
    g = int(codes.max()) + 1
    if g < 2 or n <= n_parameters:
        raise ValueError('Need multiple blocks and positive model residual df')
    scores = np.zeros((g, m, y.shape[1]))
    for response in range(y.shape[1]):
        error = y[:, response, None] - x * beta[:, response]
        np.add.at(scores[:, :, response], codes, x * error)
    correction = g / (g - 1) * (n - 1) / (n - n_parameters)
    return np.sqrt(correction * np.sum(scores ** 2, axis=0) / xx[:, None] ** 2)


def selected_motif_similarity(motifs, design, selected):
    """Descriptive residual-score correlations; related hits are not independent signals."""
    if not motifs.index.equals(design.index):
        raise ValueError('ACR identity/order mismatch in selected motif similarity')
    q = _projector(design)
    values = motifs[selected].to_numpy(dtype=float)
    residual = values - q @ (q.T @ values)
    return pd.DataFrame(np.corrcoef(residual, rowvar=False), index=selected, columns=selected)


def fit_associations(motifs, responses, design, annotation, config, progress=None):
    """Fit all motifs separately; primary plus prespecified sensitivity analyses.

Raw M is unchanged. Both M and responses are projected off the SAME baseline.
Raw and partial correlations are distinct; partial R² is squared partial r.
"""
    for frame in [responses, design, annotation]:
        if not motifs.index.equals(frame.index):
            raise ValueError('ACR identity/order mismatch; no silent alignment')
    if not motifs.index.is_unique or not motifs.columns.is_unique:
        raise ValueError('Duplicate motif/ACR identifiers')
    labels = ['all', 'A', 'B']
    y = responses[labels].to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError('Nonfinite bulk accessibility response')
    z = design.to_numpy(dtype=float)
    q = _projector(z)
    yr = y - q @ (q.T @ y)
    yc = y - y.mean(axis=0)
    yss = np.sum(yc ** 2, axis=0)
    if np.any(yss <= 1e-18):
        raise ValueError('Constant bulk accessibility response')
    baseline = pd.DataFrame({'response': labels, 'n_acrs': len(y), 'n_parameters': z.shape[1],
                             'response_mean': y.mean(axis=0), 'response_sd': y.std(axis=0),
                             'sse': np.sum(yr ** 2, axis=0),
                             'r_squared': 1 - np.sum(yr ** 2, axis=0) / yss})
    widths = [config['primary_block_bp'], config['sensitivity_block_bp']]
    groups = {width: block_membership(annotation, width) for width in widths}
    contexts = {}
    for chrom in sorted(annotation.chrom.unique()):
        rows = np.flatnonzero(annotation.chrom.to_numpy() == chrom)
        # Chromosome indicators are constant locally; retain intercept, GC and length.
        local_q = _projector(design.iloc[rows][list(design.columns[:6])])
        local_y = y[rows] - local_q @ (local_q.T @ y[rows])
        contexts[chrom] = (rows, local_q, local_y)
    effects, inference, chromosome_effects = [], [], []
    batch_size = config['motif_batch_size']
    for start in range(0, motifs.shape[1], batch_size):
        names = motifs.columns[start:start+batch_size]
        m = motifs.iloc[:, start:start+batch_size].to_numpy(dtype=float)
        if not np.isfinite(m).all():
            raise ValueError('Nonfinite motif scores')
        mr = m - q @ (q.T @ m)
        xx, yy, beta, partial_r = _moments(mr, yr)
        mc = m - m.mean(axis=0)
        _, _, _, raw_r = _moments(mc, yc)
        for j, label in enumerate(labels):
            effects.append(pd.DataFrame({
                'motif': names, 'response': label, 'beta': beta[:, j],
                'raw_r': raw_r[:, j], 'partial_r': partial_r[:, j],
                'partial_r_squared': partial_r[:, j] ** 2,
                'delta_r_squared': partial_r[:, j] ** 2 * yy[j] / yss[j],
                'motif_sd': m.std(axis=0), 'residual_motif_sd': np.sqrt(xx / len(m)),
                'baseline_sse': yy[j],
                'expanded_sse': yy[j] * (1 - partial_r[:, j] ** 2),
            }))
        for width, (codes, blocks) in groups.items():
            # Formal tests only for the full response. Halves remain consistency checks.
            se = clustered_standard_errors(mr, yr[:, :1], beta[:, :1], xx, codes,
                                           n_parameters=z.shape[1]+1)[:, 0]
            if np.any(se <= 0) or not np.isfinite(se).all():
                raise ValueError('Degenerate clustered standard error')
            df = len(blocks) - 1
            statistic = beta[:, 0] / se
            critical = t.ppf(.975, df)
            inference.append(pd.DataFrame({
                'motif': names, 'block_bp': width, 'n_blocks': len(blocks),
                'df': df, 'cluster_se': se, 't_statistic': statistic,
                'p_value': 2 * t.sf(np.abs(statistic), df),
                'beta_ci_low': beta[:, 0] - critical * se,
                'beta_ci_high': beta[:, 0] + critical * se,
            }))
        for chrom, (rows, local_q, local_y) in contexts.items():
            local_m = m[rows] - local_q @ (local_q.T @ m[rows])
            _, _, local_beta, local_r = _moments(local_m, local_y)
            chromosome_effects.append(pd.DataFrame({
                'motif': names, 'chrom': chrom, 'n_acrs': len(rows),
                'beta': local_beta[:, 0], 'partial_r': local_r[:, 0],
                'partial_r_squared': local_r[:, 0] ** 2,
            }))
        if progress:
            progress(f'Fitted {min(start+batch_size, motifs.shape[1])}/{motifs.shape[1]} motifs')
    inference = pd.concat(inference, ignore_index=True)
    inference['p_adjusted_by'] = inference.groupby('block_bp').p_value.transform(
        lambda p: false_discovery_control(p.to_numpy(), method='by'))
    inference['reject_fdr'] = inference.p_adjusted_by.le(config['fdr_alpha'])
    effects = pd.concat(effects, ignore_index=True)
    main = effects.loc[effects.response.eq('all')].drop(columns='response').merge(
        inference.loc[inference.block_bp.eq(widths[0])], on='motif', validate='one_to_one')
    return main, effects, inference, pd.concat(chromosome_effects, ignore_index=True), baseline
