import numpy as np
import pytest

from maize_regulatory.kmer_pilot import count_kmers, kmer_dictionary
from maize_regulatory.quasi_motifs import (
    calibrated_word_definition, quasi_word_kernel, quasi_word_lme,
    read_pfm_catalog, word_probability_matrices,
)


def test_definition_normalizes_each_motif_before_equal_weighting_and_rc_pooling(tmp_path):
    path = tmp_path/'motifs.txt'
    path.write_text('>M1\\tfirst\nA [9 8 7 6 5]\nC [1 2 3 4 5]\n'
                    'G [0 0 0 0 0]\nT [0 0 0 0 0]\n'
                    '>M2 second\nA [10 10 10 10 10]\nC [0 0 0 0 0]\n'
                    'G [0 0 0 0 0]\nT [0 0 0 0 0]\n'
                    '>LONG\nA [1 1 1 1 1 1]\nC [1 1 1 1 1 1]\n'
                    'G [1 1 1 1 1 1]\nT [1 1 1 1 1 1]\n')
    definition = calibrated_word_definition(read_pfm_catalog(path))
    assert definition['source_motif_ids'] == ['M1', 'M2']
    expected = .99*(np.array([.9, .8, .7, .6, .5])+1)/2+.0025
    np.testing.assert_allclose(definition['original_position_means'], expected)
    np.testing.assert_allclose(definition['preferred_probabilities'], (expected+expected[::-1])/2)


def test_exact_canonical_scores_match_independent_both_strand_position_scan():
    p = np.array([.82, .91, .76, .91, .82])
    matrices, words = word_probability_matrices(p)
    np.testing.assert_allclose(matrices.sum(axis=1), 1)
    kernel, names = quasi_word_kernel(p)
    _, reverse, canonical, mapping = kmer_dictionary(5)
    sequence = 'AACGTACCGTTTTACGT'
    counts = np.bincount(mapping, weights=count_kmers(sequence), minlength=512)
    result = quasi_word_lme(counts[None], kernel)[0]
    for column in [0, 13, 101, 299, 511]:
        q = canonical[column]
        values = []
        for start in range(len(sequence)-4):
            digits = ['ACGT'.index(b) for b in sequence[start:start+5]]
            for target in [q, reverse[q]]:
                values.append(np.prod(matrices[target, digits, np.arange(5)]/.25))
        assert np.isclose(result[column], np.log2(np.mean(values)), atol=1e-12)
        np.testing.assert_allclose(matrices[reverse[q]], matrices[q][[3, 2, 1, 0], ::-1])
        assert names[column] == words[q]
    rc = sequence.translate(str.maketrans('ACGT', 'TGCA'))[::-1]
    rc_counts = np.bincount(mapping, weights=count_kmers(rc), minlength=512)
    np.testing.assert_array_equal(counts, rc_counts)
    with pytest.raises(ValueError, match='reverse-symmetric'):
        quasi_word_kernel([.7, .8, .9, .8, .6])
