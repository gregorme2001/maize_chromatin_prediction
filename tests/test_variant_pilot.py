"""Tiny synthetic VCFs verify coordinates, identity, missingness and REF audit."""

import gzip

import numpy as np
import pandas as pd
import pytest

from maize_regulatory.variant_pilot import audit_chromosome, sample_mapping


def fixture_vcf(tmp_path, reference='B73_v5'):
    path = tmp_path/'small.vcf.gz'
    header = f'''##fileformat=VCFv4.2
##reference={reference}
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tG1\tUNSELECTED\tG0
'''
    # UNSELECTED sentinel calls are deliberately malformed: never interpret them.
    rows = [
        'chr10\t100\t.\tA\tC\t.\t.\tMQ=.\tGT\t1/1\tDO_NOT_PARSE\t0/0',
        'chr10\t101\t.\tA\tC\t.\t.\tMQ=50;CVP=90\tGT\t1/1\tDO_NOT_PARSE\t0/0',
        'chr10\t110\t.\tC\tT\t.\t.\tMQ=.\tGT\t0/1\tDO_NOT_PARSE\t0/0',
        'chr10\t111\t.\tA\tG\t.\t.\tMQ=.\tGT\t1/1\tDO_NOT_PARSE\t0/0',
        'chr10\t121\t.\tG\tA\t.\t.\tMQ=60\tGT:DP\t0|1:10\tDO_NOT_PARSE\t./.:0',
        'chr10\t122\t.\tAA\tA\t.\t.\tMQ=50\tGT\t1/1\tDO_NOT_PARSE\t0/0',
    ]
    with gzip.open(path, 'wt') as f:
        f.write(header+'\n'.join(rows)+'\n')
    seq = pd.DataFrame({'sequence': ['ACGTACGTAA', 'GGGGGGGGGG']},
                       index=pd.Index(['chr10_100_110', 'chr10_120_130'], name='ACR'))
    genotypes = pd.Index(['G0', 'G1', 'GMISSING'], name='genotype')
    return path, seq, genotypes


def test_pilot_handles_boundaries_ref_mismatches_and_never_parses_unselected_calls(tmp_path):
    path, seq, genotypes = fixture_vcf(tmp_path)
    result = audit_chromosome(path, seq, genotypes)
    assert result.summary['vcf_records'] == 6
    assert result.summary['biallelic_snp_records'] == 5
    assert result.summary['acr_overlapping_biallelic_snp_records'] == 3
    assert result.summary['reference_matching_acr_snp_records'] == 2
    assert result.summary['reference_mismatching_acr_snp_records'] == 1
    assert result.summary['reference_matching_variable_snp_records'] == 1
    assert result.summary['n_acrs_with_reference_matching_variable_snp'] == 1
    assert result.summary['unmatched_development_genotypes'] == ['GMISSING']
    assert result.dosage.columns.equals(genotypes)
    np.testing.assert_array_equal(result.dosage.to_numpy(), [[0, 2, -1], [0, 1, -1], [-1, 1, -1]])
    assert result.variants.pos_1based.tolist() == [101, 110, 121]
    assert result.variants.ref_matches_b73.tolist() == [True, False, True]
    assert result.genotype_qc.loc['G0', 'call_rate'] == .5
    assert result.genotype_qc.loc['G1', 'call_rate'] == 1
    assert result.genotype_qc.loc['GMISSING', 'call_rate'] == 0
    assert result.summary['observed_FORMAT_fields'] == ['DP', 'GT']
    assert result.summary['per_genotype_depth_field_available']
    assert not result.summary['per_genotype_quality_field_available']
    assert result.summary['acr_snp_records_with_mapping_quality'] == 2


def test_mapping_never_guesses_aliases_or_collapses_duplicate_normalized_names():
    names = pd.Index(['282set_A441-5', 'B73'])
    genotypes = pd.Index(['A4415', 'B73HTRHM'], name='genotype')
    mapping = sample_mapping(names, genotypes)
    assert mapping.index.equals(genotypes)
    assert mapping.source_sample_column.tolist() == [0, -1]
    with pytest.raises(ValueError, match='Ambiguous'):
        sample_mapping(pd.Index(['A441-5', 'A4415']), genotypes)


def test_pilot_rejects_wrong_assembly_overlapping_acrs_and_bad_selected_gt(tmp_path):
    path, seq, genotypes = fixture_vcf(tmp_path, reference='AGPv4')
    with pytest.raises(ValueError, match='B73_v5'):
        audit_chromosome(path, seq, genotypes)
    path, seq, genotypes = fixture_vcf(tmp_path)
    overlapping = pd.concat([seq, pd.DataFrame({'sequence': ['A'*10]}, index=['chr10_105_115'])])
    with pytest.raises(ValueError, match='nonoverlapping'):
        audit_chromosome(path, overlapping, genotypes)
    with gzip.open(path, 'rt') as f:
        contents = f.read()
    with gzip.open(path, 'wt') as f:
        f.write(contents.replace('1/1\tDO_NOT_PARSE\t0/0', '0/2\tDO_NOT_PARSE\t0/0'))
    with pytest.raises(ValueError, match='Unsupported GT'):
        audit_chromosome(path, seq, genotypes)
