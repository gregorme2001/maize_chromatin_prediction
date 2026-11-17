import numpy as np
import pytest

from maize_regulatory.cortex_cpm_variance import cpm_variances


def test_event_weighted_library_variance_matches_expanded_observations():
    counts = np.array([[1, 0], [3, 1], [2, 1]])
    totals = np.array([2, 6, 4])
    g = np.array([0, 0, 1])
    result, means = cpm_variances(counts, totals, g)
    r = 1e6*counts/totals[:, None]
    expected_means = np.array([.25*r[0]+.75*r[1], r[2]])
    assert np.allclose(means, expected_means)
    assert np.allclose(means[0], 1e6*counts[:2].sum(axis=0)/totals[:2].sum())
    expected_library = (.25*(r[0]-means[0])**2+.75*(r[1]-means[0])**2)/2
    assert np.allclose(result.weighted_library_deviations, expected_library)
    assert np.allclose(result.variance_of_genotype_means, means.var(axis=0, ddof=0))
    # Repeat library values by integer event weights, with equal genotype weights.
    expanded = [np.repeat(r[:2], totals[:2], axis=0), np.repeat(r[2:], totals[2:], axis=0)]
    center = means.mean(axis=0)
    total = np.mean([((x-center)**2).mean(axis=0) for x in expanded], axis=0)
    assert np.allclose(result.total_cpm_variance, total)
    assert np.allclose(result.weighted_library_deviations+result.variance_of_genotype_means,total)


def test_pure_between_genotype_and_all_zero_regions():
    result, means = cpm_variances([[1, 0], [9, 0], [20, 0]], [100, 900, 1000], np.array([0,0,1]))
    assert np.allclose(means, [[10000,0],[20000,0]])
    assert np.allclose(result.weighted_library_deviations, [0,0])
    assert np.allclose(result.variance_of_genotype_means, [25000000,0])


def test_invalid_event_totals_counts_and_genotype_indices():
    for y,t,g in [([[1]],[0],[0]), ([[2]],[1],[0]), ([[.5]],[2],[0]),
                  ([[1]],[2.5],[0]), ([[1]],[2],[2]), ([[1]],[2],[.5])]:
        with pytest.raises(ValueError):
            cpm_variances(y,t,np.array(g))
