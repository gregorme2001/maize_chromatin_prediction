"""Five-mer PWM log2-mean-LR summaries with the legacy TF smoothing/background."""

import numpy as np

from .kmer_pilot import kmer_dictionary


def soft_word_kernel(k=5, alpha=.99):
    """LR averaged over a word and its reverse complement, for each target word.

    Rows/columns use the existing canonical dictionary. With one-hot word
    profiles, preferred-base LR is 1+3*alpha; every other base has LR 1-alpha.
    Odd k gives no self-reverse-complement words, so every group has two words.
    """
    if k < 1 or k % 2 == 0 or not 0 < alpha < 1:
        raise ValueError('Require positive odd k and 0 < alpha < 1')
    words, reverse, canonical, _ = kmer_dictionary(k)
    powers = 4**np.arange(k-1, -1, -1)
    digits = (canonical[:, None]//powers[None, :]) % 4
    rc = (reverse[canonical, None]//powers[None, :]) % 4
    match = (digits[:, None, :] == digits[None, :, :]).sum(axis=2)
    reverse_match = (rc[:, None, :] == digits[None, :, :]).sum(axis=2)
    plus = (1+3*alpha)**match*(1-alpha)**(k-match)
    minus = (1+3*alpha)**reverse_match*(1-alpha)**(k-reverse_match)
    return .5*(plus+minus), [words[i] for i in canonical]


def soft_word_lme(canonical_counts, lengths, *, alpha=.99, chunk_size=4096):
    """Exact strand-averaged word-PWM scores from sufficient window counts.

    S_iq = log2[sum_r C_ir * kernel_rq / (L_i-4)]. C_ir is the raw sum of
    forward-strand counts of word r and its reverse complement. This exactly
    equals legacy TF log2(mean LR) over positions and both strands. No clipping,
    added count pseudocount, frequency threshold or learned transformation.
    """
    counts, lengths = np.asarray(canonical_counts), np.asarray(lengths)
    if (counts.ndim != 2 or counts.shape != (len(lengths), 512)
            or not np.isfinite(counts).all() or np.any(counts < 0)
            or np.any(lengths < 5) or chunk_size < 1
            or not np.array_equal(counts.sum(axis=1), lengths-4)):
        raise ValueError('Five-mer counts and sequence lengths disagree')
    kernel, _ = soft_word_kernel(5, alpha)
    result = np.empty(counts.shape, dtype=float)
    for start in range(0, len(counts), chunk_size):
        stop = min(start+chunk_size, len(counts))
        mean_lr = counts[start:stop]@kernel/(lengths[start:stop, None]-4)
        if np.any(mean_lr <= 0):
            raise ValueError('Smoothed word likelihood ratios must be positive')
        result[start:stop] = np.log2(mean_lr)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite soft word scores')
    return result
