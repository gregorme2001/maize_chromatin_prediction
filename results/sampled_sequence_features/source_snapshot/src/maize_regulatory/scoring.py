"""Legacy block-lookup scanner extracted from archive cells 17, 19, 20, 22.

The numerical kernels below are unchanged. MotifScorer replaces notebook globals
with explicit state. Scores remain signed log2 mean likelihood ratios, not binding.
"""

import numpy as np
import pandas as pd
from numba import njit
from numba.typed import List

BASE_CODES = np.full(256, 255, dtype=np.uint8)
BASE_CODES[np.frombuffer(b"ACGT", dtype=np.uint8)] = np.arange(4)

def build_block_lookup(pwm_stack, start, width):
    """Log2 likelihood-ratio scores for all DNA words of a given width."""
    n_motifs = pwm_stack.shape[0]
    n_words = 4 ** width

    words = np.arange(n_words, dtype=np.int64)
    table = np.zeros((n_words, n_motifs), dtype=np.float32)

    for pos in range(width):
        digit = (words // (4 ** (width - pos - 1))) % 4
        table += pwm_stack[:, digit, start + pos].T

    return table


def build_motif_lookups(pwms, block_size=8):
    """Group motifs by length and construct forward/reverse block tables."""
    groups = {}

    for length in sorted({pwm.shape[1] for pwm in pwms.values()}):
        ids = [
            motif_id for motif_id, pwm in pwms.items()
            if pwm.shape[1] == length
        ]

        forward = np.stack(
            [pwms[motif_id] for motif_id in ids]
        ).astype(np.float32)

        reverse = forward[:, [3, 2, 1, 0], ::-1].copy()

        blocks = []

        for start in range(0, length, block_size):
            width = min(block_size, length - start)

            blocks.append({
                "start": start,
                "width": width,
                "forward": build_block_lookup(forward, start, width),
                "reverse": build_block_lookup(reverse, start, width),
            })

        groups[length] = {
            "ids": ids,
            "blocks": blocks,
        }

    return groups


def encode_sequence(sequence):
    encoded = BASE_CODES[
        np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    ]
    if not np.all(encoded < 4):
        raise ValueError("Sequence contains non-ACGT characters.")
    return encoded.astype(np.int64)


@njit
def rolling_indices(sequence, k):
    """Base-4 indices for consecutive k-mers."""
    n = len(sequence) - k + 1
    indices = np.empty(n, dtype=np.int64)

    value = 0
    for j in range(k):
        value = 4 * value + sequence[j]

    indices[0] = value
    leading = 4 ** (k - 1)

    for i in range(1, n):
        value = 4 * (value - sequence[i - 1] * leading)
        value += sequence[i + k - 1]
        indices[i] = value

    return indices


@njit
def score_short_group(encoded, length, lr_table):
    indices = rolling_indices(encoded, length)
    n_motifs = lr_table.shape[1]
    totals = np.zeros(n_motifs, dtype=np.float64)

    for i in range(len(indices)):
        idx = indices[i]
        for m in range(n_motifs):
            totals[m] += lr_table[idx, m]

    return np.log2(totals / len(indices))


@njit
def score_two_blocks(
    sequence_length, motif_length,
    forward1, forward2, reverse1, reverse2,
    indices1, indices2, second_start
):
    n_windows = sequence_length - motif_length + 1
    n_motifs = forward1.shape[1]
    totals = np.zeros(n_motifs, dtype=np.float64)

    for j in range(n_windows):
        idx1 = indices1[j]
        idx2 = indices2[j + second_start]

        for m in range(n_motifs):
            plus = forward1[idx1, m] * forward2[idx2, m]
            minus = reverse1[idx1, m] * reverse2[idx2, m]
            totals[m] += 0.5 * (plus + minus)

    return np.log2(totals / n_windows)


@njit
def score_group_lr(
    sequence_length, motif_length,
    starts, widths, forward_tables, reverse_tables,
    indices_by_width
):
    n_windows = sequence_length - motif_length + 1
    n_motifs = forward_tables[0].shape[1]
    totals = np.zeros(n_motifs, dtype=np.float64)

    for j in range(n_windows):
        for m in range(n_motifs):
            plus = 1.0
            minus = 1.0

            for b in range(len(starts)):
                idx = indices_by_width[widths[b] - 1][j + starts[b]]
                plus *= forward_tables[b][idx, m]
                minus *= reverse_tables[b][idx, m]

            totals[m] += 0.5 * (plus + minus)

    return np.log2(totals / n_windows)


def direct_motif_score(sequence, pwm):
    """Independent full-PWM reference implementation."""
    encoded = encode_sequence(sequence)
    reverse_pwm = pwm[[3, 2, 1, 0], ::-1]

    length = pwm.shape[1]
    n_windows = len(encoded) - length + 1
    positions = np.arange(length)

    total = 0.0

    for j in range(n_windows):
        window = encoded[j:j + length]

        plus = pwm[window, positions].sum()
        minus = reverse_pwm[window, positions].sum()

        total += 0.5 * (2.0**plus + 2.0**minus)

    return np.log2(total / n_windows)

def motif_to_pwm(motif):
    """Original alpha=.99 convex mixing with a uniform background."""
    counts = np.array([motif.counts[base] for base in "ACGT"], dtype=np.float64)
    pfm_probs = counts / counts.sum(axis=0, keepdims=True)
    background = np.full(4, 0.25)
    alpha = 0.99
    probabilities = alpha * pfm_probs + (1 - alpha) * background[:, None]
    pwm = np.log2(probabilities / background[:, None])
    if not np.isfinite(pwm).all():
        raise ValueError("Invalid motif counts")
    return pwm


class MotifScorer:
    def __init__(self, pwms):
        if not pwms or any(p.ndim != 2 or p.shape[0] != 4 or not np.isfinite(p).all()
                           for p in pwms.values()):
            raise ValueError("Need finite four-row PWMs")
        self.pwms = pwms
        self.max_length = max(p.shape[1] for p in pwms.values())
        lookups = build_motif_lookups(pwms, block_size=8)
        self.short_lr_tables = {}
        self.lr_groups = {}
        for length, group in lookups.items():
            blocks = group["blocks"]
            self.lr_groups[length] = {
                "ids": group["ids"],
                "starts": np.array([b["start"] for b in blocks], dtype=np.int64),
                "widths": np.array([b["width"] for b in blocks], dtype=np.int64),
                "forward": List([
                    np.exp2(b["forward"].astype(np.float64)).astype(np.float32)
                    for b in blocks
                ]),
                "reverse": List([
                    np.exp2(b["reverse"].astype(np.float64)).astype(np.float32)
                    for b in blocks
                ]),
            }
            if length <= 8:
                block = blocks[0]
                self.short_lr_tables[length] = (0.5 * (
                    np.exp2(block["forward"].astype(np.float64))
                    + np.exp2(block["reverse"].astype(np.float64))
                )).astype(np.float32)
        self.motif_ids = [mid for group in self.lr_groups.values() for mid in group["ids"]]

    def score_acr(self, sequence):
        if len(sequence) < max(8, self.max_length):
            raise ValueError("ACR shorter than a motif or the legacy 8-base lookup requirement")
        encoded = encode_sequence(sequence)
        indices_by_width = List([rolling_indices(encoded, k) for k in range(1, 9)])
        parts = []
        for length, group in self.lr_groups.items():
            if length <= 8:
                scores = score_short_group(encoded, length, self.short_lr_tables[length])
            elif length <= 16:
                scores = score_two_blocks(
                    len(encoded), length,
                    group["forward"][0], group["forward"][1],
                    group["reverse"][0], group["reverse"][1],
                    indices_by_width[group["widths"][0] - 1],
                    indices_by_width[group["widths"][1] - 1], group["starts"][1],
                )
            else:
                scores = score_group_lr(
                    len(encoded), length, group["starts"], group["widths"],
                    group["forward"], group["reverse"], indices_by_width,
                )
            parts.append(scores)
        return np.concatenate(parts)

    def verify(self, sequences):
        """Legacy independent comparison: shortest/median/longest ACR, each motif length."""
        values = sequences["sequence"].tolist()
        lengths = np.array([len(s) for s in values])
        picks = [int(np.argmin(lengths)), int(np.argsort(lengths)[len(lengths) // 2]),
                 int(np.argmax(lengths))]
        positions = {mid: i for i, mid in enumerate(self.motif_ids)}
        for idx in picks:
            optimized = self.score_acr(values[idx])
            for group in self.lr_groups.values():
                mid = group["ids"][0]
                direct = direct_motif_score(values[idx], self.pwms[mid])
                np.testing.assert_allclose(optimized[positions[mid]], direct, rtol=1e-5, atol=1e-5)

    def score_sequences(self, sequences):
        self.verify(sequences)
        scores = np.empty((len(sequences), len(self.motif_ids)), dtype=np.float32)
        for i, sequence in enumerate(sequences["sequence"]):
            scores[i] = self.score_acr(sequence)
            if (i + 1) % 10_000 == 0:
                print(f"Scored {i + 1:,}/{len(sequences):,} ACRs", flush=True)
        if not np.isfinite(scores).all():
            raise ValueError("Nonfinite motif scores")
        return pd.DataFrame(scores, index=sequences.index, columns=self.motif_ids)
