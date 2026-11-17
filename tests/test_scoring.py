import numpy as np
import pandas as pd
import pytest

pytest.importorskip("numba")

from maize_regulatory.scoring import MotifScorer, direct_motif_score


def test_legacy_scanner_matches_direct_formula_across_block_boundaries():
    rng = np.random.default_rng(72)
    # Deliberately unsorted lengths, including short/two-block/multi-block paths.
    pwms = {}
    for length in [17, 4, 9, 8, 16]:
        counts = rng.integers(1, 20, size=(4, length)).astype(float)
        probabilities = .99 * counts / counts.sum(axis=0) + (1 - .99) * .25
        pwms[f"M{length}"] = np.log2(probabilities / .25)
    sequences = pd.DataFrame({"sequence": ["ACGT" * 10, "T" * 35, "ATGC" * 12]},
                             index=pd.Index(["acr3", "acr1", "acr2"], name="ACR"))
    scanner = MotifScorer(pwms)
    scores = scanner.score_sequences(sequences)
    assert scores.columns.tolist() == ["M4", "M8", "M9", "M16", "M17"]
    assert scores.index.equals(sequences.index)
    assert scores.dtypes.eq(np.float32).all()
    for acr, sequence in sequences["sequence"].items():
        for mid, pwm in pwms.items():
            np.testing.assert_allclose(scores.loc[acr, mid], direct_motif_score(sequence, pwm),
                                       rtol=1e-5, atol=1e-5)
    with pytest.raises(ValueError, match="non-ACGT"):
        scanner.score_acr("N" * 40)


def test_uniform_pwm_zero_and_unfavorable_sequence_negative():
    pwm = np.log2(np.tile(np.array([[.97], [.01], [.01], [.01]]) / .25, (1, 4)))
    scanner = MotifScorer({"uniform": np.zeros((4, 4)), "A": pwm})
    scores = scanner.score_acr("C" * 20)
    assert scores[0] == 0
    # Both strand placements are unfavorable: negative scores are preserved.
    np.testing.assert_allclose(scores[1], 4 * np.log2(.01 / .25), atol=1e-5)
