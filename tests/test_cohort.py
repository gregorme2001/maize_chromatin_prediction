"""QC exclusions must never redefine the original holdout or silently misalign rows."""

import numpy as np
import pandas as pd
import pytest

from maize_regulatory.cohort import (
    analysis_exclusions,
    development_cv_indices,
    development_genotypes,
    read_development_parquet,
)
from maize_regulatory.data import load_development_data
from maize_regulatory.evaluation import make_holdout_indices
from maize_regulatory.sample_identity import load_identities


def test_exclusion_preserves_holdout_and_original_development_cv_order():
    _, order, holdout, original_dev, original_test = load_identities()
    dev = development_genotypes()
    assert len(dev) == 138 and 'CML341' not in dev
    assert [entry['genotype'] for entry in analysis_exclusions()] == ['CML341']
    assert dev.equals(order[~order.isin(holdout) & (order != 'CML341')])
    assert holdout.equals(order[original_test]) and len(holdout) == 30
    _, untouched_test = make_holdout_indices(169)
    np.testing.assert_array_equal(original_test, untouched_test)
    positions = development_cv_indices(dev)
    expected = order[original_dev]
    assert dev[positions].equals(expected[expected != 'CML341'])
    with pytest.raises(ValueError, match='validated order'):
        development_cv_indices(dev[::-1])


def test_all_analysis_inputs_exclude_cml341_and_holdout_before_value_validation(tmp_path):
    _, order, holdout, _, _ = load_identities()
    excluded = holdout.union(pd.Index(['CML341']))
    acrs = pd.Index(['chr1_0_150', 'chr1_200_350'], name='ACR')
    x = pd.DataFrame(500000., index=order, columns=acrs)
    y = pd.DataFrame({'DTS': np.arange(len(order)) + 50.}, index=order)
    pcs = pd.DataFrame({'gPC1': np.arange(len(order))}, index=order, dtype=float)
    for frame, filename in [(x, 'maize_accessibility_cpm.parquet'),
                            (y, 'maize_flowering_dts.parquet'),
                            (pcs, 'maize_ancestry_pcs.parquet')]:
        frame.loc[excluded] = np.nan
        frame.to_parquet(tmp_path / filename)
    pd.DataFrame({'M1': [1., 2.]}, index=acrs).to_parquet(tmp_path / 'maize_acr_motif_scores.parquet')
    pd.DataFrame({'sequence': ['A' * 150, 'C' * 150]}, index=acrs).to_parquet(
        tmp_path / 'maize_acr_sequences.parquet')
    bundle = load_development_data(tmp_path)
    for frame in [bundle.accessibility, bundle.phenotype, bundle.ancestry_pcs]:
        assert frame.index.equals(development_genotypes())
        assert np.isfinite(frame.to_numpy()).all()
    assert bundle.accessibility.columns.equals(bundle.motif_scores.index)
    assert bundle.accessibility.columns.equals(bundle.acr_sequences.index)
    x.iloc[::-1].to_parquet(tmp_path / 'reordered.parquet')
    with pytest.raises(ValueError, match='permanent'):
        read_development_parquet(tmp_path / 'reordered.parquet')
