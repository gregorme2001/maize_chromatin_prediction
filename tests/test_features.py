import numpy as np
import pandas as pd
import pytest

from maize_regulatory.data import DataBundle, validate_data
from maize_regulatory.features import (
    calculate_motif_features,
    model_matrices,
    residualize_motif_scores,
)


def sample_bundle():
    ac = ["chr1_0_10", "chr1_20_30", "chr2_0_12", "chr2_40_55"]
    g = ["G1", "G2", "G3", "G4", "G5"]
    x = pd.DataFrame([
        [250000, 250000, 250000, 250000],
        [400000, 100000, 200000, 300000],
        [300000, 400000, 200000, 100000],
        [200000, 500000, 100000, 200000],
        [300000, 200000, 400000, 100000],
    ], index=g, columns=ac)
    y = pd.DataFrame({"DTS": [60., 68., 58., 65., 70.]}, index=g)
    pcs = pd.DataFrame({"PC1": [.3, -.1, .5, -.2, -.5]}, index=g)
    motif = pd.DataFrame({"MA": [1.5, 0., -2., .8], "MB": [-.5, .8, 2., -1.]}, index=ac)
    seq = pd.DataFrame({"sequence": ["ACGTACGTAC", "AAAAAAAAAA", "GGGGAAAACCCC", "TTTTAAAAACCCCCC"]}, index=ac)
    return DataBundle(x, y, pcs, motif, seq)


def test_dimensions_and_exact_decomposition():
    bundle = sample_bundle()
    validate_data(bundle)
    f = calculate_motif_features(bundle)
    matrices = model_matrices(bundle, f)
    assert f.motif_raw.shape == (5, 2)
    assert f.motif_residual.shape == (5, 2)
    assert f.composition.shape == (5, 2)
    assert matrices["ancestry + residual motifs"].shape == (5, 5)
    np.testing.assert_allclose(
        f.motif_raw,
        (bundle.accessibility.to_numpy() / 1e6 @ f.composition_design)
        @ f.nuisance_coefficients + f.motif_residual,
    )


def test_incorrect_alignment_raises():
    bundle = sample_bundle()
    swapped = DataBundle(
        bundle.accessibility,
        bundle.phenotype,
        bundle.ancestry_pcs,
        bundle.motif_scores.iloc[::-1],
        bundle.acr_sequences,
    )
    with pytest.raises(ValueError, match="ACR order"):
        validate_data(swapped)


def test_cpm_rows_required():
    bundle = sample_bundle()
    wrong = DataBundle(bundle.accessibility / 2, bundle.phenotype, bundle.ancestry_pcs,
                       bundle.motif_scores, bundle.acr_sequences)
    with pytest.raises(ValueError, match="CPM"):
        validate_data(wrong)


def test_sequence_only_residuals_preserve_existing_decomposition():
    bundle = sample_bundle()
    existing = calculate_motif_features(bundle)
    residuals = residualize_motif_scores(bundle.motif_scores, bundle.acr_sequences)
    B = existing.composition_design
    # Direct check against the pre-existing definition, including signed scores.
    M = bundle.motif_scores.to_numpy(dtype=float)
    expected = M - B @ np.linalg.lstsq(B, M, rcond=None)[0]
    np.testing.assert_array_equal(residuals.to_numpy(), expected)
    np.testing.assert_allclose(B.T @ residuals, 0, atol=1e-12)
    np.testing.assert_allclose(bundle.accessibility.to_numpy() / 1e6 @ residuals,
                               existing.motif_residual)
    # Fitting only selected response columns must give the same residuals.
    single = residualize_motif_scores(bundle.motif_scores[['MB']], bundle.acr_sequences)
    np.testing.assert_allclose(single['MB'], residuals['MB'], atol=1e-14)
    with pytest.raises(ValueError, match='ACR order'):
        residualize_motif_scores(bundle.motif_scores, bundle.acr_sequences.iloc[::-1])
