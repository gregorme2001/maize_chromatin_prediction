"""Audit sparse cell-state detections and existing chromosome-10 calls, without inference."""

import argparse
import hashlib
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from maize_regulatory.cell_state_audit import (
    joint_feasibility, joint_scenarios, run_aggregation, variant_inventory,
)
from maize_regulatory.cell_state_audit_report import (
    accessibility_tables, candidate_panel, library_diagnostics, plots, write_report,
)
from maize_regulatory.cohort import EXCLUSIONS_PATH, cohort_metadata, development_genotypes
from maize_regulatory.sample_identity import RESOURCE_DIR
from maize_regulatory.spatial import parse_acr_coordinates


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--config', type=Path, default=Path('configs/cell_state_accessibility_audit.json'))
    parser.add_argument('--output-dir', type=Path, default=Path('results/cell_state_accessibility_audit'))
    parser.add_argument('--aggregation-dir', type=Path, help='Reuse a completed aggregation; output must still be new')
    parser.add_argument('--rscript', default='Rscript')
    parser.add_argument('--chromosome', type=int, help='Bounded accessibility subset; genetic audit remains chr10')
    parser.add_argument('--max-acrs', type=int, help='First N selected ACRs for a real-data smoke run')
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError('Output directory is not empty; choose a new directory to preserve results')
    config = json.loads(args.config.read_text())
    root = args.data_dir
    cell_dir = root / 'intermediate/cells'
    metadata_path = cell_dir / 'cell_metadata.csv'
    matrix_path = cell_dir / 'cell_by_acr.sparse.rds'
    variant_path = root / 'intermediate/panzea_v5_pilot/chr10_variants.parquet'
    dosage_path = root / 'intermediate/panzea_v5_pilot/chr10_alt_dosage.parquet'
    sequence_path = root / 'maize_acr_sequences.parquet'
    genotypes = development_genotypes()
    acrs = pd.read_csv(cell_dir / 'acr_index.csv')
    if not acrs.ACR.is_unique or not np.array_equal(acrs.matrix_column, np.arange(1, len(acrs) + 1)):
        raise ValueError('ACR column identity/order mismatch')
    if args.chromosome:
        acrs = acrs.loc[parse_acr_coordinates(acrs.ACR).chrom.to_numpy() == args.chromosome]
    if args.max_acrs:
        if args.max_acrs <= 0:
            raise ValueError('max-acrs must be positive')
        acrs = acrs.head(args.max_acrs)
    if not len(acrs) or not acrs.ACR.str.startswith('chr10_').any():
        raise ValueError('This joint audit needs some chromosome-10 ACRs; use --chromosome 10 for a smoke run')
    acrs = acrs.reset_index(drop=True)
    sequences = pd.read_parquet(sequence_path)
    if args.chromosome is None and args.max_acrs is None and not sequences.index.equals(pd.Index(acrs.ACR, name='ACR')):
        raise ValueError('Reference-sequence/ACR axis mismatch')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    aggregation = args.aggregation_dir or args.output_dir / 'aggregation'
    inputs = [matrix_path, metadata_path, cell_dir / 'acr_index.csv', sequence_path,
              variant_path, dosage_path, EXCLUSIONS_PATH, args.config] + [
                  RESOURCE_DIR / name for name in ['analysis_genotypes.txt', 'holdout_genotypes.txt', 'sample_identity.json']]
    print('Hashing local inputs; no downloads or phenotype files', flush=True)
    fingerprints = {str(p.resolve()): sha256(p) for p in inputs}
    if args.aggregation_dir:
        manifest = json.loads((aggregation / 'aggregation.json').read_text())
        cached = json.loads((aggregation / 'input_fingerprints.json').read_text())
        if (not manifest['completed'] or manifest['genotype_order'] != genotypes.tolist()
                or manifest['config'] != config or cached != fingerprints
                or not pd.read_csv(aggregation / 'audited_acrs.csv').equals(acrs)):
            raise ValueError('Aggregation cache inputs/configuration/axes differ')
    else:
        print(f'Streaming {len(acrs):,} ACRs, {len(genotypes)} development genotypes', flush=True)
        manifest = run_aggregation(pd.read_csv(metadata_path), genotypes, acrs, matrix_path,
                                   config, aggregation, args.rscript)
        (aggregation / 'input_fingerprints.json').write_text(json.dumps(fingerprints, indent=2) + '\n')
    print('Reducing accessibility summaries and checking library/depth metadata', flush=True)
    groups, libraries, sampling, categories = accessibility_tables(aggregation, args.output_dir)
    depth, pairs = library_diagnostics(aggregation, args.output_dir)
    variants = pd.read_parquet(variant_path)
    dosage = pd.read_parquet(dosage_path)
    print('Rechecking chromosome-10 REF, coordinates, genotype axes and allele counts', flush=True)
    variants, dosage, acr_variants = variant_inventory(variants, dosage, sequences,
                                                       genotypes, pd.Index(acrs.ACR, name='ACR'))
    variants.to_parquet(args.output_dir / 'chr10_snp_audit.parquet')
    acr_variants.to_csv(args.output_dir / 'chr10_acr_variant_summary.csv')
    chr10_acrs = pd.Index(pd.read_csv(aggregation / 'chr10_acrs.csv').ACR, name='ACR')
    if not chr10_acrs.equals(acr_variants.index):
        raise ValueError('Chromosome-10 aggregation/reference order mismatch')
    counts = sparse.load_npz(aggregation / 'chr10_pooled_counts.npz')
    joint, best = joint_feasibility(counts, groups, chr10_acrs, variants, dosage,
                                    genotypes, joint_scenarios(config))
    joint.to_csv(args.output_dir / 'joint_feasibility.csv', index=False)
    best.to_csv(args.output_dir / 'benchmark_acr_state_support.csv', index=False)
    candidates = candidate_panel(best, config['candidate_count'])
    if candidates.empty:
        candidates = pd.DataFrame(columns=['candidate_rank', 'ACR', 'cell_state'])
    candidates.to_csv(args.output_dir / 'candidate_pilot_acrs.csv', index=False)
    summary = {'n_chr10_acrs': len(acr_variants), 'n_overlapping_snps': len(variants),
               'acrs_no_panel_snp': int((acr_variants.n_panel_snps == 0).sum()),
               'acrs_variable': int((acr_variants.n_dosage_variable_snps > 0).sum()),
               'acrs_mac_ge_5': int((acr_variants.n_mac_ge_5 > 0).sum()),
               'acrs_mac_ge_10': int((acr_variants.n_mac_ge_10 > 0).sum()),
               'acrs_common_called': int((acr_variants.n_mac_ge_10_callrate_ge_90pct > 0).sum()),
               'acrs_two_common_called_snps': int((acr_variants.n_mac_ge_10_callrate_ge_90pct >= 2).sum()),
               'n_unique_benchmark_acrs': int(best.ACR.nunique()) if len(best) else 0,
               'n_candidate_acrs': len(candidates)}
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    plots(args.output_dir, groups, sampling, categories, acr_variants, joint, depth)
    write_report(args.output_dir, aggregation, manifest, groups, sampling, categories,
                 acr_variants, joint, candidates, pairs, summary)
    repo = Path(__file__).resolve().parents[1]
    code = [Path(__file__).resolve(), repo / 'scripts/stream_accessibility_groups.R'] + [
        repo / 'src/maize_regulatory' / name for name in
        ['cell_state_audit.py', 'cell_state_audit_report.py', 'cohort.py', 'sample_identity.py']]
    run = {'created_utc': datetime.now(UTC).isoformat(), 'config': config, 'cohort': cohort_metadata(),
           'aggregation_dir': str(aggregation.resolve()), 'aggregation_reused': bool(args.aggregation_dir),
           'input_sha256': fingerprints, 'code_sha256': {str(p.relative_to(repo)): sha256(p) for p in code},
           'n_cells': manifest['n_cells'], 'n_acrs': len(acrs), 'n_cell_states': manifest['n_states'],
           'n_genotypes': len(genotypes), 'phenotype_loaded': False, 'holdout_genotypes_analyzed': 0,
           'association_tests': 0, 'sequences_reconstructed': 0, 'downloads': 0,
           'genetic_scope': 'chromosome 10 only', 'summary': summary,
           'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__}}
    (args.output_dir / 'run.json').write_text(json.dumps(run, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)
    print(f'Done: {args.output_dir / "report.html"}', flush=True)


if __name__ == '__main__':
    main()
