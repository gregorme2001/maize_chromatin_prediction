"""Sparse aggregation and joint screens are verified independently on tiny data."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from maize_regulatory.cell_state_audit import (
    aggregate_detections, cell_groups, detection_summary, joint_feasibility,
    stream_library_counts, variant_inventory,
)
from maize_regulatory.cell_state_audit_report import candidate_panel


def small_cells():
    metadata = pd.DataFrame({
        'cellID': ['c0', 'c1', 'c2', 'c3', 'excluded'],
        'genotype': ['G0', 'G0', 'G1', 'G1', 'HOLDOUT'],
        'normalized_genotype': ['G0', 'G0', 'G1', 'G1', 'HOLDOUT'],
        'celltype_full': ['S1', 'S1', 'S1', 'S2', 'S1'],
        'library': ['A', 'B', 'A', 'A', 'A'], 'matrix_row': [1, 2, 3, 4, 5]})
    # 2 is counted as ONE detected nucleus. An invalid excluded value is never examined.
    matrix = sparse.csr_matrix([[2, 0, 1], [0, 0, 1], [1, 0, 0], [0, 1, 0], [-999, 0, 0]])
    return metadata, matrix, pd.Index(['G0', 'G1'], name='genotype')


def test_sparse_aggregation_preserves_implicit_zeros_and_absent_groups():
    metadata, matrix, genotypes = small_cells()
    pooled, lib, groups, libraries = aggregate_detections(matrix, metadata.cellID, metadata, genotypes)
    assert sparse.issparse(pooled) and sparse.issparse(lib)
    assert list(zip(groups.genotype, groups.cell_state)) == [('G0', 'S1'), ('G0', 'S2'), ('G1', 'S1'), ('G1', 'S2')]
    assert groups.n_nuclei.tolist() == [2, 0, 1, 1]
    np.testing.assert_array_equal(pooled.toarray(), [[1, 0, 2], [0, 0, 0], [1, 0, 0], [0, 1, 0]])
    assert libraries.n_nuclei.sum() == 4
    measured = pooled[[0, 2]].toarray()
    metrics, bins, hist = detection_summary(measured, np.array([2, 1]))
    np.testing.assert_allclose(metrics['mean_detection_frequency'], [.75, 0, .5])
    assert bins.tolist() == [3, 2, 1, 0, 0] and hist.sum() == 3
    # A group with no nuclei must not be inserted as an observed zero probability.
    with pytest.raises(ValueError, match='positive denominators'):
        detection_summary(pooled.toarray(), groups.n_nuclei.to_numpy())
    with pytest.raises(ValueError, match='identity'):
        aggregate_detections(matrix, metadata.cellID[::-1], metadata, genotypes)
    with pytest.raises(ValueError, match='identity'):
        cell_groups(metadata.iloc[::-1], genotypes)
    with pytest.raises(ValueError, match='<= nuclei'):
        detection_summary(np.array([[3]]), np.array([2]))


def test_variant_audit_counts_alleles_not_lines_and_never_calls_missing_reference():
    genotypes = pd.Index(['G0', 'G1', 'G2', 'G3'], name='genotype')
    ids = pd.Index(['v0', 'v1', 'v2'], name='variant_id')
    variants = pd.DataFrame({'ACR': ['chr10_0_10'] * 2 + ['chr10_20_30'],
        'chrom': [10] * 3, 'pos_0based': [1, 2, 21], 'pos_1based': [2, 3, 22],
        'REF': ['A'] * 3, 'ALT': ['C'] * 3, 'ref_matches_b73': [True] * 3}, index=ids)
    calls = pd.DataFrame([[0, 2, -1, -1], [0, 1, 2, -1], [1, 1, 1, 1]], index=ids, columns=genotypes)
    acrs = pd.Index(['chr10_0_10', 'chr10_20_30', 'chr10_40_50'], name='ACR')
    seq = pd.DataFrame({'sequence': ['A' * 10] * 3}, index=acrs)
    annotated, _, inventory = variant_inventory(variants, calls, seq, genotypes, acrs)
    assert annotated.minor_allele_count.tolist() == [2, 3, 4]
    np.testing.assert_allclose(annotated.minor_allele_frequency, [.5, .5, .5])
    assert annotated.call_rate_audit.tolist() == [.5, .75, 1]
    assert annotated.dosage_variable.tolist() == [True, True, False]
    assert inventory.loc['chr10_0_10', 'n_complete_genotypes_all_panel_snps'] == 2
    assert inventory.loc['chr10_0_10', 'n_distinct_complete_dosage_patterns'] == 2
    assert inventory.loc['chr10_40_50', 'n_panel_snps'] == 0
    assert pd.isna(inventory.loc['chr10_40_50', 'missing_call_fraction'])
    with pytest.raises(ValueError, match='identity'):
        variant_inventory(variants, calls.iloc[:, ::-1], seq, genotypes, acrs)
    bad = variants.copy()
    bad.loc['v0', 'REF'] = 'G'
    with pytest.raises(ValueError, match='alignment'):
        variant_inventory(bad, calls, seq, genotypes, acrs)


def test_joint_support_must_come_from_the_same_snp_and_correct_genotypes():
    genotypes = pd.Index(['G0', 'G1', 'G2', 'G3', 'G4'], name='genotype')
    groups = pd.DataFrame({'genotype': genotypes, 'cell_state': ['S'] * 5,
                           'n_nuclei': [50, 50, 50, 50, 0]})
    acrs = pd.Index(['chr10_0_10', 'chr10_20_30'], name='ACR')
    ids = pd.Index(['v0', 'v1', 'v2'], name='variant_id')
    variants = pd.DataFrame({'ACR': [acrs[0], acrs[0], acrs[1]], 'dosage_variable': True,
        'minor_allele_count': 4, 'call_rate_audit': .8,
        'REF': 'A', 'ALT': 'C', 'pos_1based': [2, 3, 22]}, index=ids)
    calls = pd.DataFrame([[0, 0, 2, 2, -1], [2, 2, 0, 0, -1], [0, 0, 2, 2, -1]], index=ids, columns=genotypes)
    y = sparse.csr_matrix([[5, 5], [5, 5], [0, 5], [0, 5], [0, 0]])
    scenarios = [dict(name='benchmark', min_nuclei=20, min_mac=2, min_call_rate=.8,
                      min_genotypes_per_allele=2, min_events=3, min_detected_genotypes_per_allele=2)]
    summary, best = joint_feasibility(y, groups, acrs, variants, calls, genotypes, scenarios)
    assert summary.iloc[0].n_allele_represented_acrs == 2
    assert summary.iloc[0].n_joint_acrs == 1
    assert best.ACR.tolist() == [acrs[1]]
    assert best.iloc[0].n_ref_nuclei == best.iloc[0].n_alt_nuclei == 100
    panel = candidate_panel(pd.concat([best, best.assign(cell_state='T')]), 200)
    assert len(panel) == 1 and panel.iloc[0].n_benchmark_supported_states == 2
    with pytest.raises(ValueError, match='order'):
        joint_feasibility(y, groups.iloc[::-1].reset_index(drop=True), acrs,
                          variants, calls, genotypes, scenarios)


def test_r_stream_matches_independent_sparse_aggregation(tmp_path):
    rscript = shutil.which('Rscript')
    if not rscript:
        pytest.skip('Rscript unavailable')
    check = subprocess.run([rscript, '-e', 'quit(status=if(requireNamespace("Matrix",quietly=TRUE))0 else 1)'], capture_output=True)
    if check.returncode:
        pytest.skip('R Matrix unavailable')
    metadata, matrix, genotypes = small_cells()
    _, selected, groups, libraries, projection, _ = cell_groups(metadata, genotypes)
    metadata.to_csv(tmp_path / 'meta.csv', index=False)
    selected.to_csv(tmp_path / 'selected.csv', index=False)
    acrs = pd.DataFrame({'matrix_column': [1, 2, 3], 'ACR': ['a0', 'a1', 'a2']})
    acrs.to_csv(tmp_path / 'acrs.csv', index=False)
    code = '''args<-commandArgs(TRUE);library(Matrix);
    x<-as(Matrix(matrix(c(2,0,1,0,0,1,1,0,0,0,1,0,-999,0,0),nrow=5,byrow=TRUE),sparse=TRUE),"dgCMatrix");
    rownames(x)<-c("c0","c1","c2","c3","excluded");colnames(x)<-c("a0","a1","a2");saveRDS(x,args[1])'''
    subprocess.run([rscript, '-e', code, str(tmp_path / 'x.rds')], check=True, capture_output=True)
    bridge = Path(__file__).resolve().parents[1] / 'scripts/stream_accessibility_groups.R'
    command = [rscript, str(bridge), str(tmp_path / 'x.rds'), str(tmp_path / 'meta.csv'),
               str(tmp_path / 'selected.csv'), str(tmp_path / 'acrs.csv'), str(tmp_path), '2', str(len(libraries))]
    chunks = list(stream_library_counts(command, len(libraries), 3, tmp_path / 'stream.log'))
    assert [start for start, _ in chunks] == [0, 2]
    actual = sparse.hstack([m for _, m in chunks], format='csr')
    expected, expected_lib, _, _ = aggregate_detections(matrix, metadata.cellID, metadata, genotypes)
    np.testing.assert_array_equal(actual.toarray(), expected_lib.toarray())
    np.testing.assert_array_equal((projection @ actual).toarray(), expected.toarray())
    qc = pd.read_csv(tmp_path / 'cell_detected_acrs.csv.gz')
    assert qc.detected_acrs.tolist() == [2, 1, 1, 1]
    assert pd.read_csv(tmp_path / 'stream_validation.csv').iloc[0].nonbinary_positive_source_entries == 1
