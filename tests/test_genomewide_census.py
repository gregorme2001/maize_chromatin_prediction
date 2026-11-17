import numpy as np
import pandas as pd
import pytest

from maize_regulatory.genomewide_census import (
    census_from_variants, coverage_diversity_grid, sequence_pattern_statistics,
    spatial_block_summary,
)
from maize_regulatory.personalized_context_audit import context_windows


def test_sequence_classes_are_local_and_count_reference_rare_and_common_patterns():
    positions = np.array([10, 12, 40])
    calls = np.array([[0, 0, 2, 2, -1], [0, 0, 0, 2, -1], [2, 2, 2, 2, -1]], np.int8)
    available = np.array([[1, 1, 1, 1, 0]]*3, bool)
    stats, ids = sequence_pattern_statistics(positions, calls, [10, 30, 50], [15, 45, 60], available)
    assert stats.n_sequences.tolist() == [3, 1, 1]
    assert stats.largest_sequence_n.tolist() == [2, 4, 4]
    assert stats.second_sequence_n.tolist() == [1, 0, 0]
    assert stats.B73_sequence_n.tolist() == [2, 0, 4]
    assert stats.non_B73_pairs.tolist() == [2, 4, 0]
    assert ids[:, -1].tolist() == [-1, -1, -1]
    assert ids[0, 0] == ids[0, 1]
    with pytest.raises(ValueError, match='missing/heterozygous'):
        sequence_pattern_statistics(positions, calls, [10], [15], np.ones((1, 5), bool))


def test_census_distinguishes_core_variation_from_flank_variation_and_ambiguity():
    reference = bytearray(b'A'*100)
    reference[71] = ord('N')
    coordinates = pd.DataFrame(dict(chrom=[1]*3, start=[20, 40, 70], end=[25, 45, 75],
        midpoint=[22.5, 42.5, 72.5], block=[0]*3, length=[5]*3), index=['a', 'b', 'c'])
    windows = context_windows(coordinates, len(reference), 20)
    variants = pd.DataFrame(dict(pos_0based=[15, 22, 42], REF=['A']*3, ALT=['C']*3))
    calls = np.array([[0, 2, -1], [0, 0, 2], [0, 2, 1]], np.int8)
    frame, per_genotype, arrays, summary = census_from_variants(windows, variants, calls, reference)
    assert frame.core_n_available.tolist() == [3, 2, 0]
    assert frame.context_n_available.tolist() == [2, 2, 0]
    assert frame.core_n_sequences.tolist() == [2, 2, 0]
    assert frame.context_n_sequences.tolist() == [2, 2, 0]
    # A's core variant is lost by the context-complete mask, leaving only flank variation.
    assert frame.common_core_n_sequences.tolist() == [1, 2, 0]
    assert summary['common_pairs'] == 4
    assert summary['core']['pairs_lost_to_ambiguous_or_padded_DNA'] == 3
    assert not arrays['context_available'][2].any()
    assert per_genotype.context_available_ACRs.tolist() == [2, 2, 0]
    grid = coverage_diversity_grid(frame)
    selected = grid.loc[(grid.sequence_scope == 'context') & (grid.min_available_genotypes == 1)
                        & (grid.min_second_sequence_carriers == 0)].iloc[0]
    assert selected.n_ACRs == 2 and selected.available_pairs == 4
    blocks = spatial_block_summary(frame, 3)
    assert blocks.context_available_pairs.iloc[0] == 4
    assert blocks.context_variable_ACRs.iloc[0] == 2
    assert blocks.core_variable_in_common_pairs_ACRs.iloc[0] == 1


def test_no_available_genotypes_has_zero_classes_not_a_reference_sequence():
    frame, ids = sequence_pattern_statistics(np.array([2]), np.array([[0, 2]], np.int8),
                                              [1], [3], np.zeros((1, 2), bool))
    assert frame.n_sequences.iloc[0] == 0
    assert frame.B73_sequence_n.iloc[0] == 0
    assert (ids == -1).all()
