#!/usr/bin/env python
"""Matched raw count, frequency, word-PWM LME and TF-PWM LME comparison."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from maize_regulatory.cohort import cohort_metadata
from maize_regulatory.pooled_sequence import blocked_manifest
from maize_regulatory.reference_sequence import digest, prepare_reference
from maize_regulatory.reference_sequence_evaluation import evaluate_reference
from maize_regulatory.sequence_summaries import soft_word_kernel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/sequence_summary_comparison.json'))
    parser.add_argument('--cache-dir', type=Path, default=Path('data/intermediate/reference_cortex'))
    parser.add_argument('--output-dir', type=Path, default=Path('results/sequence_summary_comparison'))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stage', choices=['all','report'], default='all')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.output_dir.resolve()
    if args.stage == 'report':
        from maize_regulatory.sequence_summary_report import write_report
        write_report(out)
        print(f'Open {out/"report.html"}')
        return
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError('Use a new output directory or --resume with identical inputs/code')
    config = json.loads(args.config.read_text())
    prior = root/'results/reference_sequence_comparison'
    old_config = json.loads((prior/'config.json').read_text())
    if (any(config[k] != v for k,v in old_config.items())
            or config['feature_kinds'] != ['kmer','frequency','softlme','motif']
            or config['word_pwm_alpha'] != .99):
        raise ValueError('Expected exactly the previous experiment plus the two approved feature summaries')
    previous_manifest = json.loads((prior/'input_manifest.json').read_text())
    for name, expected in previous_manifest['inputs_and_code_sha256'].items():
        if name.startswith('data/') and digest(root/name) != expected:
            raise ValueError(f'Original data changed: {name}')
    before = time.monotonic()
    data = prepare_reference(root,args.cache_dir.resolve(),config)
    old_summary = json.loads((prior/'summary.json').read_text())
    if (data['genotypes'].tolist() != old_summary['genotype_order']
            or data['y'].size != old_summary['n_pairs']):
        raise ValueError('Previous observation identities changed')
    folds = blocked_manifest(data['coordinates'].block,config['outer_splits'],config['inner_splits'],config['seed'])
    if folds != json.loads((prior/'fold_manifest.json').read_text()):
        raise ValueError('Genomic-block folds differ from the frozen previous experiment')
    out.mkdir(parents=True,exist_ok=True)
    paths = [root/p for p in ['data/maize_acr_sequences.parquet','data/maize_acr_motif_scores.parquet',
        'data/maize_ancestry_pcs.parquet','src/maize_regulatory/sequence_summaries.py',
        'src/maize_regulatory/reference_sequence.py','src/maize_regulatory/reference_sequence_evaluation.py',
        'src/maize_regulatory/pooled_sequence.py','src/maize_regulatory/bulk_motif_association.py',
        'src/maize_regulatory/cohort.py','src/maize_regulatory/kmer_pilot.py','src/maize_regulatory/spatial.py']]
    paths += [args.cache_dir.resolve()/'receipt.json', args.config.resolve(), Path(__file__),
        prior/'fold_manifest.json', prior/'metrics.csv', prior/'acr_predictions.parquet']
    manifest = dict(config=config,genotype_order=data['genotypes'].tolist(),
        inputs_and_code_sha256={str(p.relative_to(root)):digest(p) for p in paths})
    manifest_path = out/'input_manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Checkpoint input/code/config differs; use a new output directory')
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    (out/'covariates.json').write_text(json.dumps(dict(names=data['covariate_names'],
        bases={k:v.tolist() for k,v in data['bases'].items()}),indent=2)+'\n')
    data['coordinates'].to_csv(out/'acr_coordinates.csv')
    data['blocks'].to_csv(out/'genomic_blocks.csv',index=False)
    data['cohort'].to_csv(out/'genotype_coverage.csv',index=False)
    kernel,names = soft_word_kernel()
    np.save(out/'word_pwm_kernel.npy',kernel)
    (out/'word_pwm_definition.json').write_text(json.dumps(dict(k=5,alpha=.99,background=.25,
        preferred_base_probability=.9925,other_base_probability=.0025,
        summary='log2(mean likelihood ratio), positions and both strands',
        canonical_words=names,no_count_pseudocount=True),indent=2)+'\n')
    print(f'{len(data["acrs"]):,} ACRs × {len(data["genotypes"])} genotypes; frozen folds reproduced',flush=True)
    metrics = evaluate_reference(data,config,root,out)
    old = pd.read_csv(prior/'metrics.csv').set_index('model')
    current = metrics.set_index('model')
    for column in ['mse','rmse','r2','mean_acr_rmse','mean_acr_r2']:
        if not np.allclose(current.loc[old.index,column],old[column],rtol=1e-11,atol=1e-11):
            raise AssertionError(f'Previously fitted result changed: {column}')
    old_predictions = pd.read_parquet(prior/'acr_predictions.parquet')
    new_predictions = pd.read_parquet(out/'acr_predictions.parquet')
    if not old_predictions.ACR.equals(new_predictions.ACR):
        raise AssertionError('Previous ACR identity/order changed')
    for column in old_predictions.filter(regex='_CPM$').columns:
        if not np.allclose(old_predictions[column],new_predictions[column],rtol=1e-11,atol=1e-11):
            raise AssertionError(f'Previous predictions changed: {column}')
    summary = dict(completed_utc=datetime.now(timezone.utc).isoformat(),cohort=cohort_metadata(),
        n_acrs=len(data['acrs']),n_genotypes=len(data['genotypes']),n_pairs=int(data['y'].size),
        n_blocks=len(data['blocks']),genotype_order=data['genotypes'].tolist(),metrics=metrics.to_dict('records'),
        evaluation_seconds=time.monotonic()-before,previous_models_reproduced=True,previous_folds_reproduced=True,
        no_DTS_read=True,no_holdout_values_used=True,old_results_changed=False,previous_phenotype_CV_changed=False,
        report_code_sha256=digest(root/'src/maize_regulatory/sequence_summary_report.py'))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    from maize_regulatory.sequence_summary_report import write_report
    write_report(out)
    print(metrics[['model','rmse','r2','mean_acr_rmse','mean_acr_r2']].to_string(index=False))
    print(f'Open {out/"report.html"}')


if __name__=='__main__':
    main()
