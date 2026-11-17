import numpy as np
import pandas as pd
import pytest

from maize_regulatory.quasi_motifs import quasi_word_kernel, quasi_word_lme
from maize_regulatory.sampled_sequence_features import (
    canonical_word_counts, pool_log_mean_lr, spatial_sample,
)


def test_sampling_has_exact_quotas_probabilities_and_is_independent_of_row_order():
    frame = pd.DataFrame(dict(global_acr_index=np.arange(12), block=[0]*7+[1]*3+[2]*2,
        context_n_available=[30]*6+[19]+[20]*3+[0]*2), index=[f'a{i}' for i in range(12)])
    first = spatial_sample(frame, quota=3)
    shuffled = spatial_sample(frame.sample(frac=1, random_state=5), quota=3)
    pd.testing.assert_frame_equal(first, shuffled)
    assert first.groupby('block').size().to_dict() == {0: 3, 1: 3}
    assert np.all(first.loc[first.block == 0, 'selection_probability'] == .5)
    assert np.all(first.loc[first.block == 1, 'selection_probability'] == 1)
    assert np.isclose((1/first.selection_probability).sum(), 9)
    assert first.context_n_available.min() >= 20
    assert first.global_acr_index.is_monotonic_increasing


def test_flank_pooling_matches_direct_placement_mean_not_mean_logs_or_concatenation():
    fragments = ['AAAAAAA', 'CCCCCCCCCCC']
    counts = canonical_word_counts(fragments)
    assert counts.sum() == 10  # 3 + 7; not 14 windows from concatenation.
    kernel, names = quasi_word_kernel([.7, .8, .9, .8, .7])
    scores = np.stack([quasi_word_lme(canonical_word_counts([f])[None], kernel)[0] for f in fragments])
    pooled = pool_log_mean_lr(scores, list(map(len, fragments)), np.full(512, 5))
    np.testing.assert_allclose(pooled, quasi_word_lme(counts[None], kernel)[0], atol=1e-12)
    assert not np.allclose(pooled, scores.mean(axis=0))
    assert not np.array_equal(counts, canonical_word_counts([''.join(fragments)]))
    # Stable for very unfavorable scores and motif-dependent placement counts.
    stable = pool_log_mean_lr(np.array([[-1000., 0.], [-1000., 0.]]), [10, 20], [5, 8])
    np.testing.assert_allclose(stable, [-1000., 0.])
    with pytest.raises(ValueError, match='positive'):
        pool_log_mean_lr(np.zeros((2, 1)), [3, 10], [5])
