import numpy as np
import pytest

from maize_regulatory.cortex_variance import binary_nested_variance, VARIANCE_COLUMNS


def test_components_match_explicit_cells_with_unequal_group_sizes():
    # Two genotypes; genotype 0 has unequal library sizes; genotype 1 has one.
    cells = [np.array([[1, 0], [0, 1]]),
             np.array([[1, 1], [1, 0], [0, 0]]),
             np.array([[0, 1], [0, 1], [0, 1], [1, 0]])]
    counts = np.array([x.sum(axis=0) for x in cells])
    n = np.array([len(x) for x in cells])
    actual, means = binary_nested_variance(counts, n, np.array([0, 0, 1]))
    genotype_cells = [np.concatenate(cells[:2]), cells[2]]
    expected_means = np.array([x.mean(axis=0) for x in genotype_cells])
    within, library = [], []
    for indices, all_cells in [([0, 1], genotype_cells[0]), ([2], genotype_cells[1])]:
        within.append(sum(((cells[j] - cells[j].mean(axis=0)) ** 2).sum(axis=0)
                          for j in indices) / len(all_cells))
        library.append(sum(len(cells[j]) * (cells[j].mean(axis=0)
                           - all_cells.mean(axis=0)) ** 2 for j in indices) / len(all_cells))
    assert np.allclose(means, expected_means)
    assert np.allclose(actual.within_genotype_library, np.mean(within, axis=0))
    assert np.allclose(actual.between_libraries_within_genotype, np.mean(library, axis=0))
    assert np.allclose(actual.between_genotype_pooled_means,
                       expected_means.var(axis=0, ddof=0))
    # Compute total variance by expanding actual cells with the stated weights.
    grand = expected_means.mean(axis=0)
    explicit = np.mean([((x-grand)**2).mean(axis=0) for x in genotype_cells], axis=0)
    assert np.allclose(actual[VARIANCE_COLUMNS].sum(axis=1), explicit)


def test_singletons_zeros_and_complete_genotype_separation_are_observed_not_missing():
    actual, means = binary_nested_variance([[0, 1, 0], [0, 0, 1]], [1, 1], np.array([0, 1]))
    assert np.array_equal(actual.within_genotype_library, [0, 0, 0])
    assert np.array_equal(actual.between_libraries_within_genotype, [0, 0, 0])
    assert np.array_equal(actual.between_genotype_pooled_means, [0, .25, .25])
    assert np.array_equal(means, [[0, 1, 0], [0, 0, 1]])


def test_invalid_counts_and_genotype_indices_fail_explicitly():
    for counts, n, g in [([[2]], [1], [0]), ([[.5]], [1], [0]),
                         ([[0]], [0], [0]), ([[0]], [1], [1]),
                         ([[0]], [1], [.5]), ([[np.nan]], [1], [0])]:
        with pytest.raises(ValueError):
            binary_nested_variance(counts, n, np.array(g))
