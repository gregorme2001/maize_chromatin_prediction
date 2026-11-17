#!/usr/bin/env python3
"""Four frozen matched CNNs, with resumable nested CV and a wall-clock stop."""

import argparse
from datetime import datetime
import fcntl
import gc
import json
from pathlib import Path
import signal
import shutil
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch

from maize_regulatory.bulk_motif_association import gc_spline_basis
from maize_regulatory.cohort import development_genotypes, read_development_parquet
from maize_regulatory.sampled_cnn import SequenceBank
from maize_regulatory.sampled_cnn_run import (PauseRequested, RunControl, choose_candidate,
    digest, outer_predictions, run_fit, save_json, summarize)
from maize_regulatory.sampled_linear import prediction_metrics


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/sampled_cnn_benchmark'
CACHE = ROOT/'data/intermediate/sampled_cnn_preflight'
CODE = ['scripts/run_sampled_cnn.py', 'scripts/launch_sampled_cnn.py',
    'src/maize_regulatory/sampled_cnn.py', 'src/maize_regulatory/sampled_cnn_precision.py',
    'src/maize_regulatory/sampled_cnn_run.py', 'src/maize_regulatory/sampled_linear.py',
    'src/maize_regulatory/cohort.py', 'src/maize_regulatory/sample_identity.py']


def prepare(config):
    OUT.mkdir(parents=True, exist_ok=True); (OUT/'outer').mkdir(exist_ok=True)
    frozen = ROOT/'results/sampled_sequence_features'
    prior = json.loads((ROOT/'results/sampled_cnn_preflight/input_state.json').read_text())
    inputs = dict(prior['inputs'])
    for relative, checksum in inputs.items():
        if digest(ROOT/relative) != checksum: raise ValueError('Frozen input changed: '+relative)
    for receipt_name in ['sample_receipt.json', 'sequence_receipt.json']:
        for relative, checksum in json.loads((frozen/receipt_name).read_text())['outputs'].items():
            if digest(ROOT/relative) != checksum: raise ValueError('Frozen input changed: '+relative)
    sample = pd.read_parquet(frozen/'selected_acrs.parquet')
    pairs = pd.read_parquet(frozen/'feature_lookup.parquet')
    original = pd.read_parquet(frozen/'pair_manifest.parquet')
    pd.testing.assert_frame_equal(pairs[original.columns], original)
    observations = pd.read_parquet(ROOT/'results/sampled_linear_benchmark/observations.parquet')
    pd.testing.assert_frame_equal(observations[original.columns], original)
    data = dict(np.load(ROOT/'data/intermediate/sampled_linear_benchmark/observations.npz'))
    ids = dict(np.load(ROOT/'data/intermediate/sampled_linear_benchmark/pattern_ids.npz'))
    a, g = data['sample_acr_index'], data['genotype_index']
    np.testing.assert_array_equal(data['y'], observations.observed_CPM)
    np.testing.assert_array_equal(sample.index.to_numpy()[a], pairs.ACR)
    np.testing.assert_array_equal(sample.block.to_numpy()[a], data['blocks'])
    for key in ['sample_acr_index', 'genotype_index', 'outer_fold']:
        np.testing.assert_array_equal(data[key], pairs[key])
    development = development_genotypes()
    coverage = pd.read_csv(ROOT/'results/sequence_cnn_benchmark/genotype_coverage.csv')
    genotypes = pd.read_csv(frozen/'genotypes.csv').genotype.tolist()
    assert coverage.genotype.tolist() == development.tolist()
    assert coverage.loc[coverage.eligible, 'genotype'].tolist() == genotypes
    assert len(sample) == 9666 and len(pairs) == 495900 and len(genotypes) == 137
    np.testing.assert_array_equal(np.asarray(genotypes)[g], pairs.genotype)
    positions = development.get_indexer(genotypes)
    if np.any(positions < 0): raise ValueError('Unknown development genotype')
    pcs = read_development_parquet(ROOT/'data/maize_ancestry_pcs.parquet')
    region = np.column_stack([np.ones(len(sample)), gc_spline_basis(sample.GC, [0,.25,.5,.75,1]),
        sample.length, *[sample.chrom.eq(c).to_numpy(float) for c in range(2, 11)]])
    np.testing.assert_allclose(data['a'], np.c_[region[a], pcs.to_numpy()[positions[g]]], atol=1e-12, rtol=1e-12)
    counts = np.memmap(ROOT/'data/intermediate/reference_cortex/counts.int32', mode='r',
                       dtype=np.int32, shape=(108843, len(development)))
    expected = counts[sample.global_acr_index.to_numpy()[a], positions[g]]*1e6/coverage.total_events.to_numpy()[positions[g]]
    np.testing.assert_array_equal(data['y'], expected)
    for panel in config['panels']:
        source, extent = panel.split('_')
        column = f'{extent}_{"B73" if source == "B73" else "personalized"}_pattern_id'
        np.testing.assert_array_equal(ids[panel], pairs[column])
    # Recheck every encoded row against the frozen DNA, including padding.
    for extent in ['core', 'context']:
        path = CACHE/f'{extent}_codes.npy'; codes = np.load(path, mmap_mode='r')
        patterns = pd.read_parquet(ROOT/f'data/intermediate/sampled_sequence_features/{extent}_patterns.parquet')
        np.testing.assert_array_equal(patterns.pattern_id, np.arange(len(patterns)))
        assert len(codes) == len(patterns)
        lookup = np.full(256, 255, np.uint8); lookup[np.frombuffer(b'ACGT', np.uint8)] = np.arange(4)
        for i, sequence in enumerate(patterns.sequence):
            encoded = lookup[np.frombuffer(sequence.encode('ascii'), np.uint8)]
            np.testing.assert_array_equal(encoded, codes[i, :len(encoded)])
            if not np.all(codes[i, len(encoded):] == 4): raise ValueError('Padding differs')
        inputs[str(path.relative_to(ROOT))] = digest(path)
        del codes, patterns
    folds = json.loads((frozen/'fold_manifest.json').read_text())
    assert folds == json.loads((ROOT/'results/sampled_linear_benchmark/fold_manifest.json').read_text())
    assert len(folds) == config['outer_folds'] and all(len(f['inner']) == config['inner_folds'] for f in folds)
    for path in ['data/maize_ancestry_pcs.parquet', 'data/intermediate/reference_cortex/counts.int32',
                 'results/sequence_cnn_benchmark/genotype_coverage.csv']:
        inputs[path] = digest(ROOT/path)
    state = dict(config=config, inputs=inputs, code={p:digest(ROOT/p) for p in CODE},
        torch=torch.__version__, numpy=np.__version__, device='mps', n_ACRs=len(sample),
        n_observations=len(pairs), n_genotypes=len(genotypes), verified_raw_CPM_values=len(expected),
        cohort_or_response_changed=False, holdout_or_DTS_read=False)
    manifest = OUT/'run_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != state:
        raise ValueError('Frozen run inputs/config/code differ; refusing to mix experiments')
    save_json(manifest, state)
    for relative in CODE:
        destination = OUT/'source_snapshot'/relative
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT/relative, destination)
    observations.to_parquet(OUT/'observations.parquet', index=False)
    sample.to_parquet(OUT/'selected_acrs.parquet')
    save_json(OUT/'fold_manifest.json', folds); save_json(OUT/'run_config.json', config)
    path = OUT/'predictions.npy'
    shape = (len(pairs), len(config['panels']))
    if not path.exists():
        values = np.lib.format.open_memmap(path, mode='w+', dtype=np.float64, shape=shape)
        values[:] = np.nan; values.flush(); del values
    if np.load(path, mmap_mode='r').shape != shape: raise ValueError('Prediction axes changed')
    print(f'Verified {len(pairs):,} observations / {len(sample):,} ACRs; configuration frozen', flush=True)
    return data, ids, sample, observations, folds, state


def run(config, data, ids, sample, observations, folds, state, control):
    if not torch.backends.mps.is_available(): raise RuntimeError('MPS unavailable; no CPU fallback')
    predictions = np.load(OUT/'predictions.npy', mmap_mode='r+')
    banks = {}
    control.emit(phase='running', active_fit=None, force=True)
    for spec in folds:
        outer = spec['number']
        train = np.flatnonzero(np.isin(data['blocks'], spec['training_blocks']))
        validation = np.flatnonzero(np.isin(data['blocks'], spec['validation_blocks']))
        assert not np.intersect1d(train, validation).size
        assert np.all(data['outer_fold'][validation] == outer)
        for column, panel in enumerate(config['panels']):
            control.tick()
            for relative, expected in state['code'].items():
                if digest(ROOT/relative) != expected:
                    raise PauseRequested('Training source changed; checkpoint retained')
            complete = OUT/'outer'/f'{panel}_outer_{outer}.json'
            if complete.exists():
                if not np.isfinite(predictions[validation, column]).all():
                    raise ValueError('Completed fold has missing predictions')
                continue
            extent = panel.split('_')[1]
            if extent not in banks:
                banks[extent] = SequenceBank(np.load(CACHE/f'{extent}_codes.npy', mmap_mode='r'), 'mps')
            bank = banks[extent]; candidates = []
            for candidate, l2 in enumerate(config['candidate_l2']):
                records = []
                setting = dict(config['architecture'], learning_rate=config['learning_rate'], l2=l2)
                for j, inner in enumerate(spec['inner']):
                    tr = np.flatnonzero(np.isin(data['blocks'], inner['training_blocks']))
                    va = np.flatnonzero(np.isin(data['blocks'], inner['validation_blocks']))
                    assert not np.intersect1d(tr, va).size
                    assert not np.intersect1d(tr, validation).size and not np.intersect1d(va, validation).size
                    assert set(np.r_[tr, va]) == set(train)
                    fit_id = f'{panel}_outer{outer}_candidate{candidate}_inner{j}'
                    records.append(run_fit(OUT, fit_id, bank, data['a'], data['y'], ids[panel],
                        tr, va, setting, config, control, seed=config['inner_seed_base']+100*outer+j))
                candidates.append(records)
            chosen, summaries = choose_candidate(candidates, config['candidate_l2'])
            epochs = max(1, int(np.median(chosen['best_epochs'])))
            selection = dict(panel=panel, outer_fold=outer, selected_l2=chosen['l2'],
                             refit_epochs=epochs, candidates=summaries)
            save_json(OUT/'selection'/f'{panel}_outer_{outer}.json', selection)
            setting = dict(config['architecture'], learning_rate=config['learning_rate'], l2=chosen['l2'])
            refit = run_fit(OUT, f'{panel}_outer{outer}_refit', bank, data['a'], data['y'],
                ids[panel], train, None, setting, config, control,
                seed=config['outer_seed_base']+outer, fixed_epochs=epochs)
            control.emit(phase='outer_prediction', active_fit=refit['fit_id'], force=True)
            value = outer_predictions(bank, data['a'], ids[panel], validation, refit, OUT,
                                      control, config['batch_size'])
            predictions[validation, column] = value; predictions.flush()
            metric = prediction_metrics(data['y'][validation], value, data['sample_acr_index'][validation])
            save_json(complete, dict(selection, **metric, checkpoint=refit['checkpoint']))
            progress = summarize(OUT, config, observations, sample)
            control.emit(phase='running', active_fit=None, **progress, force=True)
            print(f'Completed {panel}, outer fold {outer+1}/5', flush=True)
            gc.collect(); torch.mps.empty_cache()
    progress = summarize(OUT, config, observations, sample)
    control.emit(phase='complete', active_fit=None, **progress, force=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--stop-at', default='2026-11-16T05:00:00-05:00')
    parser.add_argument('--gentle-until', default='2026-11-15T21:00:00-05:00')
    parser.add_argument('--batch-pause', type=float, default=.05)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT/'.worker.lock').open('w')
    try: fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise SystemExit('Another matched CNN worker is already running')
    torch.set_num_threads(2)
    config = json.loads((ROOT/'configs/sampled_cnn_benchmark.json').read_text())
    control = RunControl(OUT, datetime.fromisoformat(args.stop_at).timestamp(),
                         datetime.fromisoformat(args.gentle_until).timestamp(), args.batch_pause)
    for sig in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, lambda signum, frame: setattr(control, 'stop_reason', f'Signal {signum}; checkpointing'))
    try:
        values = prepare(config)
        if args.prepare_only:
            control.emit(phase='prepared', force=True); return 0
        run(config, *values, control)
        return 0
    except PauseRequested as error:
        if 'values' in locals():
            _, _, sample, observations, _, _ = values
            summarize(OUT, config, observations, sample)
        control.emit(phase='paused', reason=str(error), force=True)
        print('Paused safely:',error,flush=True)
        return 0
    except Exception as error:
        control.emit(phase='failed', reason=str(error), force=True)
        traceback.print_exc()
        return 1


if __name__ == '__main__': sys.exit(main())
