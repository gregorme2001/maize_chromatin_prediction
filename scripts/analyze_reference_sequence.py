#!/usr/bin/env python
"""Matched genome-wide B73 five-mer and motif prediction of pooled cortex CPM."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from maize_regulatory.cohort import cohort_metadata
from maize_regulatory.reference_sequence import digest, prepare_reference
from maize_regulatory.reference_sequence_evaluation import evaluate_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/reference_sequence_comparison.json'))
    parser.add_argument('--cache-dir', type=Path, default=Path('data/intermediate/reference_cortex'))
    parser.add_argument('--output-dir', type=Path, default=Path('results/reference_sequence_comparison'))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stage', choices=['all', 'report'], default='all')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.output_dir.resolve()
    if args.stage == 'report':
        from maize_regulatory.reference_sequence_report import write_report
        write_report(out)
        print(f'Open {out/"report.html"}')
        return
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError('Use a new output directory or --resume for identical inputs/code')
    config = json.loads(args.config.read_text())
    if (config['cell_state'] != 'cortex' or not config['reference_only']
            or config['response'] != 'pooled_raw_CPM' or config['min_pooled_nuclei'] != 50
            or config['fixed_gc_knots'] != [0,.25,.5,.75,1]):
        raise ValueError('Unsupported reference comparison definition')
    before = time.monotonic()
    print('Verifying development-only whole-genome cortex pool', flush=True)
    data = prepare_reference(root, args.cache_dir.resolve(), config)
    out.mkdir(parents=True, exist_ok=True)
    paths = [root/p for p in ['data/maize_acr_sequences.parquet', 'data/maize_acr_motif_scores.parquet',
        'data/maize_ancestry_pcs.parquet', 'scripts/pool_reference_cortex.R',
        'src/maize_regulatory/reference_sequence.py', 'src/maize_regulatory/reference_sequence_evaluation.py',
        'src/maize_regulatory/pooled_sequence.py', 'src/maize_regulatory/bulk_motif_association.py',
        'src/maize_regulatory/cohort.py', 'src/maize_regulatory/spatial.py']]
    paths += [args.config.resolve(), args.cache_dir.resolve()/'receipt.json', Path(__file__)]
    manifest = dict(config=config, genotype_order=data['genotypes'].tolist(),
        acr_order_sha256=digest(root/'data/intermediate/cells/acr_index.csv'),
        inputs_and_code_sha256={str(p.relative_to(root)):digest(p) for p in paths})
    manifest_path = out/'input_manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Cached experiment differs in code/inputs/config; use a new output directory')
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    data['coordinates'].to_csv(out/'acr_coordinates.csv')
    data['blocks'].to_csv(out/'genomic_blocks.csv',index=False)
    data['cohort'].to_csv(out/'genotype_coverage.csv',index=False)
    (out/'covariates.json').write_text(json.dumps(dict(names=data['covariate_names'],
        bases={k:v.tolist() for k,v in data['bases'].items()}),indent=2)+'\n')
    print(f'{len(data["acrs"]):,} ACRs, {len(data["genotypes"])} eligible genotypes, '
        f'{data["y"].size:,} matched observations, {len(data["blocks"])} genomic blocks',flush=True)
    metrics = evaluate_reference(data,config,root,out)
    summary = dict(completed_utc=datetime.now(timezone.utc).isoformat(),cohort=cohort_metadata(),
        n_acrs=len(data['acrs']), n_genotypes=len(data['genotypes']), n_pairs=int(data['y'].size),
        chromosomes=sorted(int(v) for v in data['coordinates'].chrom.unique()),
        n_blocks=len(data['blocks']), evaluation_seconds=time.monotonic()-before,
        metrics=metrics.to_dict('records'), genotype_order=data['genotypes'].tolist(),
        no_DTS_read=True,no_holdout_values_used=True,old_results_changed=False,
        previous_phenotype_CV_changed=False,reference_only=True,
        response='Raw pooled cortex CPM; all-genome event denominators',
        validation='Five outer and three inner folds of chromosome-specific 5 Mb blocks; same folds for all models',
        coefficient_levels='Global intercept, GC/length/chromosome/ancestry and penalized sequence coefficients; no ACR-specific coefficients',
        gc_control='Primary linear GC; prespecified 4-df natural cubic GC sensitivity with fixed knots 0,.25,.5,.75,1',
        observations='Every reference ACR crossed with every development genotype having >=50 cortex nuclei; no SNP-call/MAF/F screening',
        prediction_storage='Exact factorization: ACR predicted means plus fold/model/genotype centered ancestry contributions',
        report_code_sha256=digest(root/'src/maize_regulatory/reference_sequence_report.py'))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    from maize_regulatory.reference_sequence_report import write_report
    write_report(out)
    print(metrics[['model','rmse','r2','mse_reduction_vs_matched_baseline']].to_string(index=False))
    print(f'Open {out/"report.html"}')


if __name__=='__main__':
    main()
