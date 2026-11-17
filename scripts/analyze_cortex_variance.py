#!/usr/bin/env python
"""Describe cortex binary-accessibility variance within and between nested groups."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from maize_regulatory.cell_state_audit import stream_library_counts
from maize_regulatory.cohort import cohort_metadata, development_genotypes
from maize_regulatory.cortex_variance import binary_nested_variance, summarize_components
from maize_regulatory.sample_identity import RESOURCE_DIR


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('results/cortex_variance_descriptive'))
    parser.add_argument('--resume', action='store_true', help='Reuse a completed, verified count cache')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError('Use a new output directory or --resume for identical cached inputs')
    if root not in out.parents or out == root:
        raise ValueError('Output must be a separate subdirectory within this repository')
    aggregation = out/'aggregation'
    aggregation.mkdir(parents=True, exist_ok=True)
    genotypes = development_genotypes()
    matrix = root/'data/intermediate/cells/cell_by_acr.sparse.rds'
    metadata_path = root/'data/intermediate/cells/cell_metadata.csv'
    acr_path = root/'data/intermediate/cells/acr_index.csv'
    audit = root/'results/cell_state_accessibility_audit/aggregation'
    depth_path = audit/'genotype_cell_state_library_depth.csv'
    source_receipt = json.loads((audit/'aggregation.json').read_text())
    if not source_receipt['completed'] or source_receipt['genotype_order'] != genotypes.tolist():
        raise ValueError('Existing audit development identities do not match current policy')
    metadata = pd.read_csv(metadata_path, dtype={'genotype': str, 'normalized_genotype': str})
    if (not metadata.cellID.is_unique
            or not np.array_equal(metadata.matrix_row, np.arange(1, len(metadata)+1))):
        raise ValueError('Source cell identity/order mismatch')
    cells = metadata.loc[metadata.celltype_full.eq('cortex')
                         & metadata.normalized_genotype.isin(genotypes)].copy()
    depth = pd.read_csv(depth_path, dtype={'genotype': str})
    groups = depth.loc[depth.cell_state.eq('cortex')].copy().reset_index(drop=True)
    if (not groups.genotype.isin(genotypes).all()
            or groups.duplicated(['genotype', 'library']).any()):
        raise ValueError('Unexpected or repeated cortex genotype/library identities')
    group_axis = pd.MultiIndex.from_frame(groups[['genotype', 'library']])
    cell_axis = pd.MultiIndex.from_arrays([cells.normalized_genotype, cells.library])
    assignments = group_axis.get_indexer(cell_axis)
    if np.any(assignments < 0):
        raise ValueError('Cortex cell missing an audited genotype/library identity')
    if not np.array_equal(np.bincount(assignments, minlength=len(groups)), groups.n_nuclei):
        raise ValueError('Cortex group nucleus counts disagree with independent audit')
    g = genotypes.get_indexer(groups.genotype)
    if np.any(g < 0) or not np.array_equal(np.unique(g), np.arange(len(genotypes))):
        raise ValueError('Development cortex genotype identity/order mismatch')
    acrs = pd.read_csv(acr_path)
    if (not acrs.ACR.is_unique
            or not np.array_equal(acrs.matrix_column, np.arange(1, len(acrs)+1))):
        raise ValueError('Source ACR identity/order mismatch')
    shape = (len(acrs), len(groups))
    sources = [matrix, metadata_path, acr_path, depth_path,
               root/'scripts/stream_accessibility_groups.R']
    sources += [RESOURCE_DIR/name for name in ['analysis_genotypes.txt', 'holdout_genotypes.txt',
                'analysis_exclusions.json', 'sample_identity.json']]
    print('Checking source hashes and development-only cell identities', flush=True)
    hashes = {str(p.relative_to(root)): digest(p) for p in sources}
    expected_receipt = dict(source_sha256=hashes, shape=list(shape), dtype='<i4',
                            genotype_order=genotypes.tolist(), state='cortex')
    receipt_path = aggregation/'receipt.json'
    cache_path = aggregation/'library_counts.int32'
    if args.resume and receipt_path.exists():
        if json.loads(receipt_path.read_text()) != expected_receipt:
            raise ValueError('Cached counts differ from current inputs/identities')
        if cache_path.stat().st_size != np.prod(shape)*4:
            raise ValueError('Cached cortex library counts have wrong size')
    else:
        if cache_path.exists():
            raise ValueError('Unreceipted cache exists; use a new output directory')
        executable = shutil.which('Rscript')
        if executable is None:
            raise RuntimeError('Rscript with Matrix is required for the initial sparse read')
        selected_path = aggregation/'selected_cells.csv'
        pd.DataFrame({'matrix_row': cells.matrix_row, 'library_group': assignments}).to_csv(
            selected_path, index=False)
        selected_acrs = aggregation/'acrs.csv'
        acrs.to_csv(selected_acrs, index=False)
        temporary = aggregation/'library_counts.int32.part'
        cache = np.memmap(temporary, dtype='<i4', mode='w+', shape=shape)
        command = [executable, str(root/'scripts/stream_accessibility_groups.R'), str(matrix),
                   str(metadata_path), str(selected_path), str(selected_acrs),
                   str(aggregation), '256', str(len(groups))]
        print(f'Streaming {len(cells):,} cortex nuclei into {len(groups)} groups; '
              f'{len(acrs):,} ACRs', flush=True)
        for start, block in stream_library_counts(command, len(groups), len(acrs),
                                                  aggregation/'stream.log'):
            values = block.toarray()
            if np.any(values > groups.n_nuclei.to_numpy()[:, None]):
                raise ValueError('Positive detection count exceeds nucleus count')
            cache[start:start+block.shape[1]] = values.T
            if start == 0 or start % (256*40) == 0:
                print(f'Aggregated {start+block.shape[1]:,}/{len(acrs):,} ACRs', flush=True)
        cache.flush()
        del cache
        temporary.rename(cache_path)
        receipt_path.write_text(json.dumps(expected_receipt, indent=2)+'\n')
    cache = np.memmap(cache_path, dtype='<i4', mode='r', shape=shape)
    if not np.array_equal(cache.sum(axis=0, dtype=np.int64), groups.total_detected_events):
        raise ValueError('Genome-wide cortex detections disagree with previous audit')
    # Independent old chromosome-10 library counts, including all implicit zeros.
    from scipy import sparse
    old_acrs = pd.read_csv(audit/'chr10_acrs.csv')
    positions = pd.Index(acrs.ACR).get_indexer(old_acrs.ACR)
    if np.any(positions < 0):
        raise ValueError('Old chromosome-10 ACR identity missing from source')
    start = 0
    old_rows = groups.library_group.to_numpy()
    for filename in json.loads((audit/'chr10_library_counts/files.json').read_text()):
        chunk = sparse.load_npz(audit/'chr10_library_counts'/filename)
        width = chunk.shape[1]
        if not np.array_equal(cache[positions[start:start+width]], chunk[old_rows].toarray().T):
            raise ValueError('New cortex counts disagree with cached chromosome-10 counts')
        start += width
    if start != len(old_acrs):
        raise ValueError('Incomplete chromosome-10 cache validation')
    # All genotypes are retained. A second explicitly labelled view omits only
    # genotypes with no observed between-library comparison; no policy changes.
    library_counts = np.bincount(g, minlength=len(genotypes))
    scopes = [('all_development', np.ones(len(groups), dtype=bool)),
              ('multiple_libraries', library_counts[g] >= 2)]
    output_frames, scope_support = [], []
    for scope, mask in scopes:
        present = np.unique(g[mask])
        local_g = np.searchsorted(present, g[mask])
        scope_support.append(dict(scope=scope, n_genotypes=len(present),
            n_library_groups=int(mask.sum()), n_nuclei=int(groups.n_nuclei[mask].sum())))
        for first in range(0, len(acrs), 512):
            result, _ = binary_nested_variance(cache[first:first+512, mask].T,
                                               groups.n_nuclei.to_numpy()[mask], local_g)
            result.insert(0, 'ACR', acrs.ACR.iloc[first:first+512].to_numpy())
            result.insert(0, 'scope', scope)
            output_frames.append(result)
    results = pd.concat(output_frames, ignore_index=True)
    results.to_parquet(out/'acr_variances.parquet', index=False)
    stats = summarize_components(results)
    stats.to_csv(out/'summary_statistics.csv', index=False)
    group_details = groups.copy()
    group_details['n_libraries_for_genotype'] = library_counts[g]
    group_details.to_csv(out/'library_groups.csv', index=False)
    genotype_support = pd.DataFrame({'genotype': genotypes,
        'n_libraries': library_counts,
        'n_nuclei': np.bincount(g, weights=groups.n_nuclei, minlength=len(genotypes)).astype(int)})
    genotype_support.to_csv(out/'genotype_support.csv', index=False)
    example_panel = pd.read_csv(root/'results/library_repeatability/acr_panel.csv')
    example_ids = example_panel.ACR.iloc[[0, len(example_panel)//2, len(example_panel)-1]]
    examples = []
    for acr in example_ids:
        i = pd.Index(acrs.ACR).get_loc(acr)
        example = groups[['genotype', 'library', 'n_nuclei']].copy()
        example['ACR'] = acr
        example['n_detected'] = cache[i]
        example['detection_fraction'] = cache[i]/groups.n_nuclei.to_numpy()
        pooled = np.bincount(g, weights=cache[i], minlength=len(genotypes))/genotype_support.n_nuclei
        example['pooled_genotype_fraction'] = pooled.to_numpy()[g]
        examples.append(example)
    pd.concat(examples, ignore_index=True).to_csv(out/'example_measurements.csv', index=False)
    code_paths = [Path(__file__), root/'src/maize_regulatory/cortex_variance.py',
                  root/'src/maize_regulatory/cortex_variance_report.py']
    summary = dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        cohort=cohort_metadata(), genotype_order=genotypes.tolist(),
        n_acrs=len(acrs), scope_support=scope_support,
        single_nucleus_groups=int(groups.n_nuclei.eq(1).sum()),
        single_library_genotypes=int(np.sum(library_counts == 1)),
        min_median_max_nuclei_per_group=[int(groups.n_nuclei.min()),
                                       float(groups.n_nuclei.median()), int(groups.n_nuclei.max())],
        max_absolute_decomposition_error=float(results.decomposition_error.abs().max()),
        chr10_library_count_values_verified=int(len(old_acrs)*len(groups)),
        no_DTS_read=True, no_holdout_values_read=True, no_statistical_tests=True,
        no_SNP_screen=True, no_outcome_screen=True, new_sample_exclusions=False,
        predictive_CV_changed=False, scale='Binary per-nucleus ACR detection (0/1)',
        weighting='Equal genotypes; nucleus-count-weighted libraries within genotype; '
                  'equal nuclei within genotype/library; equal ACRs in summaries',
        variance_denominators='Empirical population variances (N, not N-1); no noise subtraction',
        source_sha256=hashes,
        code_sha256={str(p.relative_to(root)): digest(p) for p in code_paths})
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    from maize_regulatory.cortex_variance_report import write_report
    write_report(out)
    print(stats.to_string(index=False), flush=True)
    print(f'Open {out/"report.html"}', flush=True)


if __name__ == '__main__':
    main()
