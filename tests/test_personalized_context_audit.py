import gzip

import numpy as np
import pandas as pd
import pytest

from maize_regulatory.personalized_context_audit import (
    availability_summary, collect_context_variants, context_windows,
    interval_call_counts, reconstruct_patterns,
    width_sensitivity,
)


def test_context_partitions_preserve_cores_and_prevent_cross_block_overlap():
    c = pd.DataFrame(dict(chrom=[10, 10, 10], start=[10, 24, 60],
                          end=[16, 30, 66], block=[0, 1, 1]), index=['a', 'b', 'c'])
    w = context_windows(c, 100, 30)
    assert w.context_end.iloc[0] == 20
    assert w.context_start.iloc[1] == 20
    assert np.all(w.context_start <= w.start) and np.all(w.context_end >= w.end)
    assert w.context_clipped.tolist() == [True, True, False]


def test_overlapping_windows_missing_heterozygotes_boundaries_and_reconstruction(tmp_path):
    reference = b'A'*60
    windows = pd.DataFrame(dict(chrom=[10, 10], start=[10, 25], end=[15, 30],
                                block=[0, 0]), index=['a', 'b'])
    windows = context_windows(windows, 60, 20)
    vcf = tmp_path/'calls.gz'
    with gzip.open(vcf, 'wt') as handle:
        handle.write('##fileformat=VCFv4.2\n##reference=B73_v5\n'
                     '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tG1\tIGNORED\tG0\n')
        for pos, first, last in [(3, '1/1', '0/0'), (11, '1/1', '0/0'),
                                 (21, './.', '0/0'), (26, '0/1', '1/1'),
                                 (38, '0/0', '0/0')]:
            handle.write(f'10\t{pos}\t.\tA\tC\t.\t.\t.\tGT\t{first}\tBAD\t{last}\n')
    genotypes = pd.Index(['G0', 'G1'], name='genotype')
    variants, calls, mapping, info = collect_context_variants(
        vcf, windows, genotypes, reference, progress=None)
    assert mapping.source_sample_column.tolist() == [2, 0]
    assert variants.pos_0based.tolist() == [2, 10, 20, 25]
    core = interval_call_counts(variants.pos_0based, calls, windows.start, windows.end)
    context = interval_call_counts(variants.pos_0based, calls,
                                   windows.context_start, windows.context_end)
    np.testing.assert_array_equal(core['complete_homozygous'], [[True, True], [True, False]])
    np.testing.assert_array_equal(context['complete_homozygous'], [[True, False], [True, False]])
    assert context['missing'][0, 1] == 1  # SNP at 0-based 20 is in first context.
    patterns, ids, refs, unambiguous, _ = reconstruct_patterns(
        windows, reference, variants, calls, context, 'context', width=20)
    assert ids[:, 1].tolist() == [-1, -1]
    assert patterns.iloc[ids[1, 0]].sequence[8] == 'C'
    assert np.all(unambiguous)
    summary = availability_summary(context, unambiguous, ids, patterns)
    assert summary['available_pairs'] == 2
    assert summary['pairs_with_both'] == 1
    assert summary['pairs_with_sequence_different_from_B73'] == 1
    with gzip.open(vcf, 'rt') as handle:
        text = handle.read()
    with gzip.open(vcf, 'wt') as handle:
        handle.write(text.replace('10\t11\t.\tA\tC', '10\t11\t.\tG\tC'))
    with pytest.raises(ValueError, match='REF disagreement'):
        collect_context_variants(vcf, windows, genotypes, reference, progress=None)


def test_ambiguous_or_padded_DNA_is_separate_from_complete_SNP_calls():
    windows = context_windows(pd.DataFrame(dict(chrom=[10], start=[1], end=[6],
                                                 block=[0]), index=['a']), 20, 10)
    variants = pd.DataFrame(dict(pos_0based=[2], REF=['A'], ALT=['C']))
    calls = np.array([[0, 2]], dtype=np.int8)
    support = interval_call_counts(variants.pos_0based, calls,
                                   windows.context_start, windows.context_end)
    # Include a second, unpadded window so the function has a pattern to retain.
    windows = pd.concat([windows, context_windows(pd.DataFrame(dict(
        chrom=[10], start=[10], end=[15], block=[0]), index=['b']), 20, 10)])
    support = interval_call_counts(variants.pos_0based, calls,
                                   windows.context_start, windows.context_end)
    patterns, ids, refs, unambiguous, _ = reconstruct_patterns(
        windows, b'A'*20, variants, calls, support, 'context', width=10)
    assert unambiguous.tolist() == [False, True]
    assert (ids[0] == -1).all() and refs[0] == -1
    summary = availability_summary(support, unambiguous, ids, patterns)
    assert summary['complete_homozygous_pairs'] == 4
    assert summary['complete_pairs_lost_to_ambiguous_or_padded_DNA'] == 2


def test_width_sensitivity_checks_added_flank_calls_and_never_truncates_cores():
    windows = context_windows(pd.DataFrame(dict(chrom=[10, 10], start=[10, 40],
        end=[15, 50], block=[0, 0]), index=['a', 'b']), 100, 20)
    variants = pd.DataFrame(dict(pos_0based=[3, 44], REF=['A', 'A'], ALT=['C', 'C']))
    calls = np.array([[-1, 0], [0, 2]], dtype=np.int8)
    result = width_sensitivity(windows, variants, calls, b'A'*100, widths=[6, 20])
    assert result.available_pairs.tolist() == [2, 3]
    assert result.acrs_longer_than_window.tolist() == [1, 0]
