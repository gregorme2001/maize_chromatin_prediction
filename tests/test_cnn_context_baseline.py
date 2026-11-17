import numpy as np
import pytest

pytest.importorskip('torch')
from maize_regulatory.cnn_context_baseline import word_frequencies
from maize_regulatory.kmer_pilot import count_kmers,kmer_dictionary


def test_context_frequencies_match_direct_words_and_ignore_padding_windows():
    seq='ACGTAGCTACGATCGT'
    encoded=np.array([list(map('ACGT'.index,seq))+[4]*5],dtype=np.uint8)
    _,_,canonical,mapping=kmer_dictionary(5)
    expected=np.bincount(mapping,weights=count_kmers(seq),minlength=len(canonical))
    expected=expected/expected.sum()
    actual=word_frequencies(encoded)
    assert np.allclose(actual[0],expected,atol=1e-8)
    assert np.isclose(actual.sum(),1)
