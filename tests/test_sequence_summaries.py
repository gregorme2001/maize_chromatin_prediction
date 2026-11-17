import numpy as np
import pandas as pd

from maize_regulatory.kmer_pilot import count_kmers, kmer_dictionary
from maize_regulatory.reference_sequence import reference_features
from maize_regulatory.sequence_summaries import soft_word_kernel, soft_word_lme


def canonical_counts(sequence):
    _, _, canonical, mapping = kmer_dictionary(5)
    return np.bincount(mapping, weights=count_kmers(sequence), minlength=len(canonical))


def test_soft_words_equal_independent_position_pwm_scan_on_both_strands():
    sequences = ['AAAAACCCCCGGGGGTTTTT', 'ACGTACGTACGTACGT', 'TTTTTTTTTTTTTTTT']
    counts = np.array([canonical_counts(s) for s in sequences])
    actual = soft_word_lme(counts, np.array(list(map(len, sequences))), chunk_size=1)
    _, names = soft_word_kernel()
    complement = str.maketrans('ACGT', 'TGCA')
    for i, sequence in enumerate(sequences):
        for j in [0, 1, 23, 129, 270, 511]:
            word = names[j]
            reverse = word.translate(complement)[::-1]
            values = []
            for start in range(len(sequence)-4):
                window = sequence[start:start+5]
                for target in [word, reverse]:
                    probability = np.prod([.99*float(a == b)+.01*.25 for a, b in zip(window, target)])
                    values.append(probability/(.25**5))
            expected = np.log2(np.mean(values))
            assert np.isclose(actual[i, j], expected, rtol=1e-13, atol=1e-13)
    assert np.isfinite(actual).all()
    # All-absent exact words still have finite scores from the PWM smoothing.
    assert np.any(actual[-1] < 0) and np.any(actual[-1] > 0)


def test_reverse_complements_and_repeated_window_distributions_match():
    sequence = 'ACGTTTTGGGACCTAT'
    reverse = sequence.translate(str.maketrans('ACGT','TGCA'))[::-1]
    counts = canonical_counts(sequence)
    assert np.array_equal(counts, canonical_counts(reverse))
    first = soft_word_lme(counts[None], np.array([len(sequence)]))
    # A proportional count table with a proportional number of windows has the
    # identical region summary. This verifies mean-LR rather than summed-LR.
    second = soft_word_lme(3*counts[None], np.array([3*(len(sequence)-4)+4]))
    assert np.allclose(first, second, atol=1e-13)
    kernel, _ = soft_word_kernel()
    assert np.allclose(kernel, kernel.T)


def test_frequency_and_lme_use_same_canonical_counts_without_changing_counts():
    sequences = pd.DataFrame({'sequence':['ACGTACGTACGT','AAAAAACCCCCC']}, index=['a','b'])
    data = dict(acrs=sequences.index, sequences=sequences,
        coordinates=pd.DataFrame({'length':[12,12]}))
    counts, names, multiplier = reference_features(None, data, 'kmer')
    saved = counts.copy()
    frequency, f_names, f_multiplier = reference_features(None, data, 'frequency')
    soft, s_names, s_multiplier = reference_features(None, data, 'softlme')
    assert np.allclose(frequency.sum(axis=1), 1)
    assert np.array_equal(frequency, counts/8)
    assert names == f_names == s_names and multiplier == f_multiplier == s_multiplier == .5
    assert np.array_equal(counts, saved)
    assert np.isfinite(soft).all()
