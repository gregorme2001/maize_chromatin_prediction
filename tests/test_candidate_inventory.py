import numpy as np
import pandas as pd
import pytest

from maize_regulatory.candidate_inventory import (
    candidate_grid, prepare_variant_support, support_at_cell_threshold,
)


def fixture_support():
    genotypes = pd.Index(['g1', 'g2', 'g3', 'g4', 'g5', 'g6'], name='genotype')
    variants = pd.DataFrame({'ACR': ['a', 'a', 'a', 'b'],
                             'ref_matches_b73': [True] * 4},
                            index=['common1', 'common2', 'rare', 'b1'])
    calls = pd.DataFrame([[0, 0, 0, 2, 2, 2], [0, 0, 0, 2, 2, 2],
                          [0, -1, 1, 0, 0, 0], [0, 0, 0, 2, 2, 2]],
                         index=variants.index, columns=genotypes)
    return variants, calls, genotypes, pd.Index(['a', 'b'])


def test_all_snps_matter_for_completeness_and_heterozygosity():
    v, d, g, a = fixture_support()
    support = prepare_variant_support(v, d, g, a, call_rate=.9)
    result = support_at_cell_threshold(support, [50] * 6, 50, .2)
    assert result['n_global_snps'].tolist() == [2, 1]
    # Rare, low-call-rate SNP still excludes missing g2 and heterozygous g3.
    assert result['usable'][0].tolist() == [True, False, False, True, True, True]
    assert result['n_complete'].tolist() == [5, 6]
    assert result['n_usable'].tolist() == [4, 6]
    assert result['n_usable_common_snps'].tolist() == [2, 1]


def test_maf_rechecked_on_actual_genotypes_with_cells():
    v, d, g, a = fixture_support()
    support = prepare_variant_support(v, d, g, a)
    result = support_at_cell_threshold(support, [50, 50, 50, 0, 0, 0], 50, .1)
    assert result['n_global_snps'].tolist() == [2, 1]
    assert result['n_usable_common_snps'].tolist() == [0, 0]
    assert result['n_cell_eligible'] == 3


def test_maf_called_diploid_denominator_and_identity_checks():
    v, d, g, a = fixture_support()
    support = prepare_variant_support(v, d, g, a)
    assert support['maf'][2] == pytest.approx(1 / 10)  # one ALT / 5 called diploids
    with pytest.raises(ValueError, match='identities/order'):
        prepare_variant_support(v, d[g[::-1]], g, a)
    with pytest.raises(ValueError, match='genotype axis'):
        support_at_cell_threshold(support, [50] * 5, 50, .1)
    assert not np.any(support_at_cell_threshold(support, [0] * 6, 50, .1)['usable'])


def test_grid_keeps_measured_zeros_and_excludes_only_unusable_sequences():
    from scipy import sparse

    g = pd.Index([f'g{i}' for i in range(60)], name='genotype')
    v = pd.DataFrame({'ACR': ['a'] * 3, 'ref_matches_b73': [True] * 3},
                     index=['s1', 's2', 'rare'])
    common = [0] * 30 + [2] * 30
    d = pd.DataFrame([common, common, [-1, 1] + [0] * 58],
                     index=v.index, columns=g)
    support = prepare_variant_support(v, d, g, pd.Index(['a', 'b']))
    groups = pd.DataFrame({'genotype': g.tolist() * 2,
                           'cell_state': ['sampled'] * 60 + ['thin'] * 60,
                           'n_nuclei': [50] * 60 + [49] * 60})
    counts = sparse.csr_matrix((120, 2), dtype=int)
    _, candidates, obs, stages = candidate_grid(support, counts, groups)
    assert candidates[['ACR', 'cell_state']].values.tolist() == [['a', 'sampled']]
    assert candidates.n_distinct_panel_sequences.iloc[0] == 2
    assert len(obs) == 58 and obs.n_detected.eq(0).all()
    assert obs.genotype.tolist() == g[2:].tolist()
    assert stages.set_index('cell_state').loc['sampled', 'n_complete_observations'] == 59
    with pytest.raises(ValueError, match='genotype order'):
        candidate_grid(support, counts, groups.iloc[::-1])
