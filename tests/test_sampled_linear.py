import numpy as np

from maize_regulatory.pooled_sequence import Moments, moment_mse, pattern_moments, ridge_path
from maize_regulatory.sampled_linear import (
    aggregate_block_moments, partial_ridge_path, prediction_metrics,
)


def moments(a, x, y):
    z = np.column_stack([a, x])
    return Moments(len(y), z.T@z, z.T@y, float(y@y))


def test_mixed_penalty_matches_full_duplicate_word_representation_and_old_solver():
    rng = np.random.default_rng(61)
    a = np.column_stack([np.ones(100), rng.normal(size=(100, 3))])
    x = rng.normal(size=(100, 7))+a[:, 1, None]*np.arange(7)[None, :]/3
    y = x@rng.normal(size=7)+a@np.arange(4)+rng.normal(size=100)
    alphas = [None, .001, .1, 10.]
    expanded = np.column_stack([x[:, :3], np.repeat(x[:, 3:], 2, axis=1)])
    full = ridge_path(moments(a, expanded, y), 4, alphas, penalty_multiplier=1.)
    compressed = partial_ridge_path(moments(a, x, y), 4, alphas, [1]*3+[.5]*4)
    same = partial_ridge_path(moments(a, x, y), 4, alphas, 1.)
    old = ridge_path(moments(a, x, y), 4, alphas, penalty_multiplier=1.)
    for alpha in alphas:
        np.testing.assert_allclose(np.c_[a, x]@compressed['coefficients'][alpha],
                                   np.c_[a, expanded]@full['coefficients'][alpha], atol=2e-11)
        np.testing.assert_allclose(same['coefficients'][alpha], old['coefficients'][alpha], atol=2e-11)
    # Feature-free nuisance-only fit equals explicit OLS, including the intercept.
    baseline = partial_ridge_path(moments(a, np.empty((100, 0)), y), 4, [None], np.empty(0))
    np.testing.assert_allclose(a@baseline['coefficients'][None], a@np.linalg.lstsq(a, y, rcond=None)[0], atol=1e-11)


def test_unbalanced_pattern_block_moments_and_training_only_fit_match_explicit_pairs():
    rng = np.random.default_rng(62)
    ids = rng.integers(0, 12, 89)
    x = rng.normal(size=(12, 6))
    a = np.column_stack([np.ones(89), ids/12+rng.normal(size=89)])
    y = 2*a[:, 1]+x[ids, 0]+rng.normal(size=89)
    blocks = np.arange(89) % 4
    partitions = dict(train={0, 1, 2}, validation={3})
    grouped = aggregate_block_moments(a, ids, x, y, blocks, partitions)
    train = blocks != 3
    explicit = moments(a[train], x[ids[train]], y[train])
    np.testing.assert_allclose(grouped['train'].xx, explicit.xx, atol=1e-12)
    np.testing.assert_allclose(grouped['train'].xy, explicit.xy, atol=1e-12)
    fitted = partial_ridge_path(grouped['train'], 2, [.1], [.5]*6)
    reference = partial_ridge_path(explicit, 2, [.1], [.5]*6)
    np.testing.assert_allclose(fitted['coefficients'][.1], reference['coefficients'][.1], atol=1e-12)
    prediction = np.c_[a[~train], x[ids[~train]]]@fitted['coefficients'][.1]
    assert np.isclose(np.mean((prediction-y[~train])**2), moment_mse(grouped['validation'], fitted['coefficients'][.1]))
    changed = y.copy(); changed[~train] += 1e6
    changed_m = aggregate_block_moments(a, ids, x, changed, blocks, partitions)
    again = partial_ridge_path(changed_m['train'], 2, [.1], [.5]*6)
    np.testing.assert_array_equal(again['coefficients'][.1], fitted['coefficients'][.1])


def test_within_acr_metrics_remove_region_level_offsets_on_actual_available_pairs():
    acr = np.array([0, 0, 1, 1, 1])
    y = np.array([1., 3., 7., 8., 9.])
    p = y+np.array([10., 10., -3., -3., -3.])
    scores = prediction_metrics(y, p, acr)
    assert scores['within_ACR_rmse'] == 0 and scores['within_ACR_r2'] == 1
    assert scores['ACR_mean_rmse'] > 0 and scores['pair_rmse'] > 0
