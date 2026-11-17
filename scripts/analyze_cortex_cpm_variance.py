#!/usr/bin/env python
"""Compare observed genotype-mean CPM variance with weighted library deviations."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from maize_regulatory.cohort import cohort_metadata, development_genotypes
from maize_regulatory.cortex_cpm_variance import cpm_variances, summarize, write_report


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=Path('results/cortex_variance_descriptive'))
    parser.add_argument('--output-dir', type=Path, default=Path('results/cortex_cpm_variance'))
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source, out = args.source_dir.resolve(), args.output_dir.resolve()
    if out == source or out in source.parents or source in out.parents or root not in out.parents:
        raise ValueError('Use a separate non-nested output directory inside the repository')
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise FileExistsError('Use a new output directory or --overwrite')
    genotypes = development_genotypes()
    receipt_path = source/'aggregation/receipt.json'
    receipt = json.loads(receipt_path.read_text())
    prior = json.loads((source/'summary.json').read_text())
    if (receipt['genotype_order'] != genotypes.tolist() or receipt['state'] != 'cortex'
            or prior['genotype_order'] != genotypes.tolist()
            or not prior['no_DTS_read'] or not prior['no_holdout_values_read']):
        raise ValueError('Source cohort or provenance differs from current development policy')
    print('Verifying existing cortex count-cache inputs', flush=True)
    for path, expected in receipt['source_sha256'].items():
        if digest(root/path) != expected:
            raise ValueError(f'Source input changed: {path}')
    groups_path = source/'library_groups.csv'
    groups = pd.read_csv(groups_path, dtype={'genotype':str})
    depth_path = root/'results/cell_state_accessibility_audit/aggregation/genotype_cell_state_library_depth.csv'
    depth = pd.read_csv(depth_path, dtype={'genotype':str})
    expected = depth.loc[depth.cell_state.eq('cortex')].reset_index(drop=True)
    keys = ['library_group','genotype','library','n_nuclei','total_detected_events']
    if not groups[keys].equals(expected[keys]) or groups.duplicated(['genotype','library']).any():
        raise ValueError('Cortex genotype/library identities or denominators are misaligned')
    g = genotypes.get_indexer(groups.genotype)
    if np.any(g < 0) or not np.array_equal(np.unique(g),np.arange(len(genotypes))):
        raise ValueError('Development genotype identity/order mismatch')
    acr_path = source/'aggregation/acrs.csv'
    acrs = pd.read_csv(acr_path)
    if not acrs.equals(pd.read_csv(root/'data/intermediate/cells/acr_index.csv')):
        raise ValueError('ACR identities/order disagree with source')
    shape = (len(acrs),len(groups))
    cache_path = source/'aggregation/library_counts.int32'
    if (receipt['shape'] != list(shape) or receipt['dtype'] != '<i4'
            or cache_path.stat().st_size != np.prod(shape)*4):
        raise ValueError('Count cache shape, dtype or size mismatch')
    cache = np.memmap(cache_path,dtype='<i4',mode='r',shape=shape)
    totals = groups.total_detected_events.to_numpy()
    if not np.array_equal(cache.sum(axis=0,dtype=np.int64),totals):
        raise ValueError('Whole-genome group totals do not match count cache')
    # Compare with independently aggregated, genome-wide genotype counts.
    reference_dir = root/'data/intermediate/reference_cortex'
    reference_receipt = json.loads((reference_dir/'receipt.json').read_text())
    if reference_receipt['genotypes'] != genotypes.tolist() or reference_receipt['n_acrs'] != len(acrs):
        raise ValueError('Independent pooled-count axes differ')
    for path in ['data/intermediate/cells/cell_by_acr.sparse.rds',
                 'data/intermediate/cells/cell_metadata.csv','data/intermediate/cells/acr_index.csv']:
        if reference_receipt['inputs'][path] != receipt['source_sha256'][path]:
            raise ValueError('Independent cache source differs')
    reference_path = reference_dir/'counts.int32'
    if digest(reference_path) != reference_receipt['counts_sha256']:
        raise ValueError('Independent pooled count cache hash mismatch')
    reference = np.memmap(reference_path,dtype='<i4',mode='r',shape=(len(acrs),len(genotypes)))
    for genotype in range(len(genotypes)):
        if not np.array_equal(cache[:,g==genotype].sum(axis=1,dtype=np.int64),reference[:,genotype]):
            raise ValueError('Library counts do not reconstruct independent genotype counts')
    library_numbers = np.bincount(g,minlength=len(genotypes))
    scopes = [('all_development',np.ones(len(groups),dtype=bool)),
              ('multiple_libraries',library_numbers[g]>=2)]
    frames, support = [], []
    for scope,mask in scopes:
        selected_g = np.unique(g[mask])
        local_g = np.searchsorted(selected_g,g[mask])
        support.append(dict(scope=scope,n_genotypes=len(selected_g),n_library_groups=int(mask.sum()),
                            n_nuclei=int(groups.n_nuclei[mask].sum())))
        for first in range(0,len(acrs),512):
            counts = cache[first:first+512,mask].T
            if np.any(counts < 0) or np.any(counts > groups.n_nuclei.to_numpy()[mask,None]):
                raise ValueError('Cache detection counts outside nucleus bounds')
            part,_ = cpm_variances(counts,totals[mask],local_g)
            part.insert(0,'ACR',acrs.ACR.iloc[first:first+512].to_numpy())
            part.insert(0,'scope',scope)
            frames.append(part)
    result = pd.concat(frames,ignore_index=True)
    stats = summarize(result)
    out.mkdir(parents=True,exist_ok=True)
    result.to_parquet(out/'acr_variances.parquet',index=False)
    stats.to_csv(out/'summary_statistics.csv',index=False)
    groups.to_csv(out/'library_groups.csv',index=False)
    comparison = []
    for scope,part in result.groupby('scope',sort=False):
        comparison.append(dict(scope=scope,
            fraction_acrs_library_variance_larger=float((part.weighted_library_deviations
                                                       > part.variance_of_genotype_means).mean()),
            max_abs_decomposition_error=float(part.decomposition_error.abs().max())))
    paths = [receipt_path,source/'summary.json',groups_path,acr_path,cache_path,
             reference_dir/'receipt.json',reference_path,Path(__file__),
             root/'src/maize_regulatory/cortex_cpm_variance.py']
    summary = dict(completed_utc=datetime.now(timezone.utc).isoformat(),cohort=cohort_metadata(),
        genotype_order=genotypes.tolist(),n_acrs=len(acrs),scope_support=support,
        comparisons=comparison,pooled_count_values_verified=int(np.prod(reference.shape)),
        source_input_sha256=receipt['source_sha256'],
        input_and_code_sha256={str(p.relative_to(root)):digest(p) for p in paths},
        weighting='Equal genotypes; total-detection T_gl weights for libraries within genotype; equal ACRs',
        response='Cortex CPM: 1e6*Y_igl/T_gl, pooled as 1e6*sum_l(Y_igl)/sum_l(T_gl)',
        variance_denominators='Observed population variances; divide by G, not G-1',
        no_DTS_read=True,no_holdout_values_read=True,no_statistical_tests=True,
        no_cell_level_variance=True,no_noise_subtraction=True,no_SNP_or_outcome_screen=True,
        new_sample_exclusions=False,predictive_CV_changed=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    write_report(out)
    print(stats.to_string(index=False),flush=True)
    print(json.dumps(comparison,indent=2),flush=True)
    print(f'Open {out/"report.html"}',flush=True)


if __name__ == '__main__':
    main()
