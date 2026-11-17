"""Fixed JASPAR-calibrated word PWMs; independent of accessibility outcomes."""

from pathlib import Path
import re

import numpy as np

from .kmer_pilot import kmer_dictionary


def read_pfm_catalog(path):
    """Read four-row JASPAR count matrices, including literal backslash-t headers."""
    result = {}
    current, rows = None, {}

    def finish():
        if current is None:
            return
        if set(rows) != set('ACGT') or len({len(v) for v in rows.values()}) != 1:
            raise ValueError(f'Incomplete/ragged PFM: {current}')
        counts = np.array([rows[b] for b in 'ACGT'], dtype=float)
        if (counts.shape[1] == 0 or not np.isfinite(counts).all()
                or np.any(counts < 0) or np.any(counts.sum(axis=0) <= 0)):
            raise ValueError(f'Invalid PFM: {current}')
        if current in result:
            raise ValueError(f'Duplicate motif ID: {current}')
        result[current] = counts

    for line in Path(path).read_text().splitlines():
        if line.startswith('>'):
            finish()
            current = line[1:].replace('\\t', '\t').split()[0]
            rows = {}
        elif line.strip():
            match = re.fullmatch(r'([ACGT])\s*\[([^]]+)\]\s*', line)
            if current is None or match is None or match[1] in rows:
                raise ValueError('Unexpected JASPAR row')
            rows[match[1]] = [float(v) for v in match[2].split()]
    finish()
    if not result:
        raise ValueError('Empty PFM catalog')
    return result


def calibrated_word_definition(catalog, alpha=.99):
    """Equal-motif mean of column maxima, after TF smoothing, then RC pooling."""
    if not 0 < alpha < 1:
        raise ValueError('Require 0 < alpha < 1')
    selected = {key: value for key, value in catalog.items() if value.shape == (4, 5)}
    if not selected:
        raise ValueError('No exactly five-base PFMs')
    probabilities = np.stack([
        alpha * c / c.sum(axis=0, keepdims=True) + (1-alpha)/4
        for c in selected.values()])
    maxima = probabilities.max(axis=1)
    original = maxima.mean(axis=0)
    symmetric = (original + original[::-1])/2
    return dict(k=5, alpha=alpha, background=[.25]*4,
                source_motif_ids=list(selected), n_source_motifs=len(selected),
                per_motif_preferred_probabilities=maxima.tolist(),
                original_position_means=original.tolist(),
                preferred_probabilities=symmetric.tolist(),
                other_base_probabilities=((1-symmetric)/3).tolist(),
                conceptual_columns=1024, canonical_columns=512,
                source_rule='Exactly length-five PFMs; equal motif weights; '
                            '99% normalized counts + 1% uniform; mean of column maxima; '
                            'pool reverse-complement positions; no second smoothing',
                score='log2(mean likelihood ratio across valid positions and both strands)')


def word_probability_matrices(preferred):
    """All 1,024 five-mer profiles; columns are positions and rows are A,C,G,T."""
    preferred = np.asarray(preferred, float)
    if preferred.shape != (5,) or np.any(preferred <= .25) or np.any(preferred >= 1):
        raise ValueError('Need five preferred probabilities strictly between .25 and 1')
    words, _, _, _ = kmer_dictionary(5)
    powers = 4**np.arange(4, -1, -1)
    digits = (np.arange(1024)[:, None]//powers) % 4
    matrices = np.broadcast_to((1-preferred)[None, None, :]/3, (1024, 4, 5)).copy()
    matrices[np.arange(1024)[:, None], digits, np.arange(5)] = preferred
    return matrices, words


def quasi_word_kernel(preferred):
    """Exact canonical word-count to mean-LR kernel; requires symmetric profile."""
    preferred = np.asarray(preferred, float)
    if not np.allclose(preferred, preferred[::-1], rtol=0, atol=1e-15):
        raise ValueError('Canonical compression requires a reverse-symmetric profile')
    matrices, words = word_probability_matrices(preferred)
    _, reverse, canonical, _ = kmer_dictionary(5)
    powers = 4**np.arange(4, -1, -1)
    digits = (canonical[:, None]//powers) % 4
    rc = (reverse[canonical, None]//powers) % 4
    kernel = np.zeros((512, 512))
    for source in [digits, rc]:
        terms = np.ones((512, 512))
        for j in range(5):
            terms *= matrices[canonical[None, :], source[:, None, j], j]/.25
        kernel += terms/2
    return kernel, [words[i] for i in canonical]


def quasi_word_lme(counts, kernel):
    """Score rows of canonical counts, excluding ambiguous windows beforehand."""
    counts, kernel = np.asarray(counts), np.asarray(kernel)
    if (counts.ndim != 2 or counts.shape[1] != 512 or kernel.shape != (512, 512)
            or not np.isfinite(counts).all() or np.any(counts < 0)
            or not np.isfinite(kernel).all() or np.any(kernel <= 0)
            or np.any(counts.sum(axis=1) <= 0)):
        raise ValueError('Need finite nonnegative counts with positive window totals/kernel')
    return np.log2((counts @ kernel)/counts.sum(axis=1, keepdims=True))
