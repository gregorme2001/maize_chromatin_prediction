import numpy as np
import pandas as pd
import pytest

from maize_regulatory.spatial import (local_motif_track, nearby_acr_correlations,
                                       parse_acr_coordinates)


def test_parse_and_chr_boundary():
    ids = ["chr1_10_20", "chr1_22_30", "chr2_10_20"]
    c = parse_acr_coordinates(ids)
    assert c.loc[ids[0], "midpoint"] == 15
    assert c.loc[ids[2], "chrom"] == 2
    with pytest.raises(ValueError):
        parse_acr_coordinates(["bad_name"])


def test_local_track_product_and_alignment():
    ids = ["chr1_10_20", "chr1_22_30", "chr2_10_20"]
    x = pd.DataFrame([[200000, 300000, 500000]], columns=ids, index=["G1"])
    m = pd.DataFrame({"M1": [2.0, -1.0, 3.0]}, index=ids)
    track = local_motif_track(x, m, genotype="G1", motif_id="M1", chromosome=1)
    assert len(track) == 2
    np.testing.assert_allclose(track["weighted_score"], [0.4, -0.3])
    with pytest.raises(ValueError):
        local_motif_track(x, m.iloc[::-1], genotype="G1", motif_id="M1", chromosome=1)


def test_neighbor_pairs_stay_on_same_chromosome():
    ids = ["chr1_10_20", "chr1_22_30", "chr2_10_20", "chr2_25_35"]
    X = pd.DataFrame([
        [1, 2, 5, 10], [3, 4, 3, 6], [5, 6, 1, 2]
    ], columns=ids)
    pairs = nearby_acr_correlations(X, n_neighbors=2)
    assert len(pairs) == 2
    assert pairs["correlation"].abs().le(1).all()
