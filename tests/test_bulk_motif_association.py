"""Synthetic checks against full-model OLS and direct clustered sandwich variance."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import t

from maize_regulatory.bulk_motif_association import (
    block_membership,
    bulk_responses,
    composition_design,
    fit_associations,
    gc_spline_basis,
)


def synthetic_inputs():
    rng = np.random.default_rng(82)
    n = 360
    length = rng.integers(150, 450, n)
    gc_count = (length * rng.uniform(.15, .8, n)).astype(int)
    ids = pd.Index([f'chr{1+i//180}_{i%180*1000000}_{i%180*1000000+length[i]}'
                    for i in range(n)], name='ACR')
    seq = pd.DataFrame({'sequence': ['G'*g + 'A'*(size-g)
                                    for size, g in zip(length, gc_count)]}, index=ids)
    config = json.loads(Path('configs/bulk_motif_association.json').read_text())
    config['motif_batch_size'] = 2
    design, annotation, knots = composition_design(seq, config)
    motifs = pd.DataFrame(rng.normal(size=(n, 5)), index=ids,
                          columns=[f'motif_{i}' for i in range(5)])
    motifs['motif_1'] += 7 * annotation.GC
    motifs['motif_2'] = -motifs['motif_0'] + rng.normal(0, .2, n)
    base_y = design.to_numpy() @ rng.normal(size=design.shape[1])
    y = base_y + .8 * motifs.motif_0 + rng.normal(0, .4, n)
    responses = pd.DataFrame({'all': y, 'A': y+rng.normal(0, .2, n),
                              'B': y+rng.normal(0, .2, n)}, index=ids)
    return motifs, responses, design, annotation, knots, config


def test_bulk_response_averages_before_log_and_validates_identity():
    x = pd.DataFrame([[0., 1e6], [1e6, 0.], [2e5, 8e5], [4e5, 6e5]],
                     index=['g1', 'g2', 'g3', 'g4'], columns=['a1', 'a2'])
    subsets = pd.Series(['A', 'A', 'B', 'B'], index=x.index)
    result = bulk_responses(x, subsets)
    np.testing.assert_allclose(result['all'], np.log1p(x.mean(axis=0)))
    assert not np.allclose(result['all'], np.log1p(x).mean(axis=0))
    np.testing.assert_allclose(result.A, np.log1p(x.iloc[:2].mean(axis=0)))
    with pytest.raises(ValueError, match='identity/order'):
        bulk_responses(x, subsets.iloc[::-1])


def test_natural_gc_basis_has_four_df_beyond_intercept_and_contains_linear_trend():
    gc = np.linspace(.1, .9, 301)
    knots = np.quantile(gc, [0, .25, .5, .75, 1])
    basis = np.column_stack([np.ones(len(gc)), gc_spline_basis(gc, knots)])
    assert np.linalg.matrix_rank(basis) == 5
    np.testing.assert_allclose(basis @ np.linalg.lstsq(basis, gc, rcond=None)[0], gc, atol=1e-14)
    # One-sided second finite differences approach zero at natural boundaries.
    eps = 1e-5
    left = gc_spline_basis(knots[0] + np.arange(3)*eps, knots)
    right = gc_spline_basis(knots[-1] - np.arange(3)*eps, knots)
    assert np.max(np.abs((left[0]-2*left[1]+left[2])/eps**2)) < .01
    assert np.max(np.abs((right[0]-2*right[1]+right[2])/eps**2)) < .01
    with pytest.raises(ValueError, match='inside'):
        gc_spline_basis([1.], knots)


def test_batched_fwl_fit_matches_full_ols_and_independent_cluster_sandwich():
    motifs, responses, design, annotation, _, config = synthetic_inputs()
    main, effects, inference, chromosomes, baseline = fit_associations(
        motifs, responses, design, annotation, config)
    assert len(main) == 5 and len(effects) == 15 and len(chromosomes) == 10
    y = responses['all'].to_numpy()
    z = design.to_numpy()
    baseline_error = y - z @ np.linalg.lstsq(z, y, rcond=None)[0]
    assert baseline.set_index('response').loc['all', 'sse'] == pytest.approx(baseline_error @ baseline_error)
    codes, blocks = block_membership(annotation, config['primary_block_bp'])
    for name in motifs.columns:
        full = np.column_stack([z, motifs[name]])
        coef = np.linalg.lstsq(full, y, rcond=None)[0]
        error = y - full @ coef
        row = main.set_index('motif').loc[name]
        assert row.beta == pytest.approx(coef[-1])
        expected_r2 = 1 - error @ error / (baseline_error @ baseline_error)
        assert row.partial_r_squared == pytest.approx(expected_r2)
        assert row.raw_r == pytest.approx(np.corrcoef(motifs[name], y)[0, 1])
        # Independent full design sandwich, rather than residualized score formula.
        scores = np.array([full[codes == g].T @ error[codes == g]
                           for g in range(len(blocks))])
        bread = np.linalg.inv(full.T @ full)
        covariance = bread @ (scores.T @ scores) @ bread
        n, k = full.shape
        g = len(blocks)
        covariance *= g/(g-1) * (n-1)/(n-k)
        se = np.sqrt(covariance[-1, -1])
        assert row.cluster_se == pytest.approx(se, rel=1e-9)
        assert row.p_value == pytest.approx(2*t.sf(abs(coef[-1]/se), g-1), rel=1e-8)
    assert main.set_index('motif').loc['motif_0', 'partial_r_squared'] > .5
    assert main.set_index('motif').loc['motif_2', 'beta'] < 0
    assert main.p_adjusted_by.ge(main.p_value).all()
    assert inference.groupby('block_bp').size().eq(5).all()
    for chrom in [1, 2]:
        rows = annotation.chrom.eq(chrom)
        local_z = design.loc[rows].iloc[:, :6].to_numpy()
        full = np.column_stack([local_z, motifs.loc[rows, 'motif_0']])
        local_coef = np.linalg.lstsq(full, y[rows], rcond=None)[0][-1]
        actual = chromosomes.loc[chromosomes.chrom.eq(chrom)
                                 & chromosomes.motif.eq('motif_0'), 'beta'].iloc[0]
        assert actual == pytest.approx(local_coef)


def test_blocks_keep_chromosomes_distinct_and_fits_reject_reordered_acrs():
    motifs, responses, design, annotation, _, config = synthetic_inputs()
    codes, blocks = block_membership(annotation, 5000000)
    assert codes[0] != codes[180]
    assert blocks.n_acrs.sum() == len(annotation)
    with pytest.raises(ValueError, match='identity/order'):
        fit_associations(motifs, responses.iloc[::-1], design, annotation, config)
    bad = motifs.copy()
    bad['motif_0'] = 1.
    with pytest.raises(ValueError, match='Constant'):
        fit_associations(bad, responses, design, annotation, config)


def test_batch_size_does_not_change_fits_or_fdr_results():
    motifs, responses, design, annotation, _, config = synthetic_inputs()
    first = fit_associations(motifs, responses, design, annotation, config)
    second = fit_associations(motifs, responses, design, annotation, {**config, 'motif_batch_size': 5})
    for a, b in zip(first, second):
        keys = [key for key in ['motif', 'response', 'chrom', 'block_bp'] if key in a]
        if keys:
            a, b = a.sort_values(keys).reset_index(drop=True), b.sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b, rtol=1e-8, atol=1e-10)
