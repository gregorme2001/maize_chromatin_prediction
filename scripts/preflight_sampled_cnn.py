#!/usr/bin/env python3
"""Training-only feasibility checks for four matched CNNs; never runs outer CV."""

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import pandas as pd
import torch

from maize_regulatory.bulk_motif_association import gc_spline_basis
from maize_regulatory.cohort import development_genotypes, read_development_parquet
from maize_regulatory.sampled_cnn import (
    GroupedLoss, SequenceBank, Trainer, apply_transform, evaluate, group_observations,
    predict_groups, synchronize, training_transform,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/sampled_cnn_preflight'
CACHE = ROOT/'data/intermediate/sampled_cnn_preflight'
SETTING = dict(width=64, depth=3, dropout=.1, learning_rate=.001, l2=1e-5)
PANELS = ['B73_core', 'personalized_core', 'B73_context', 'personalized_context']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


def save_json(path, value):
    def convert(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x).__name__)
    temporary = Path(str(path)+'.part')
    temporary.write_text(json.dumps(value, indent=2, default=convert, allow_nan=False)+'\n')
    temporary.replace(path)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
    frozen, sequence_dir = ROOT/'results/sampled_sequence_features', ROOT/'data/intermediate/sampled_sequence_features'
    for receipt_name in ['sample_receipt.json', 'sequence_receipt.json']:
        receipt = json.loads((frozen/receipt_name).read_text())
        for path, expected in receipt['outputs'].items():
            if digest(ROOT/path) != expected: raise ValueError('Frozen input digest mismatch: '+path)
    sample = pd.read_parquet(frozen/'selected_acrs.parquet')
    pairs = pd.read_parquet(frozen/'feature_lookup.parquet')
    original = pd.read_parquet(frozen/'pair_manifest.parquet')
    pd.testing.assert_frame_equal(pairs[original.columns], original)
    observations = pd.read_parquet(ROOT/'results/sampled_linear_benchmark/observations.parquet')
    pd.testing.assert_frame_equal(observations[original.columns], original)
    data = dict(np.load(ROOT/'data/intermediate/sampled_linear_benchmark/observations.npz'))
    pattern_ids = dict(np.load(ROOT/'data/intermediate/sampled_linear_benchmark/pattern_ids.npz'))
    np.testing.assert_array_equal(data['y'], observations.observed_CPM)
    for key in ['sample_acr_index', 'genotype_index', 'outer_fold']:
        np.testing.assert_array_equal(data[key], pairs[key])
    a, g = data['sample_acr_index'], data['genotype_index']
    genotypes = pd.read_csv(frozen/'genotypes.csv').genotype.tolist()
    development = development_genotypes()
    coverage = pd.read_csv(ROOT/'results/sequence_cnn_benchmark/genotype_coverage.csv')
    assert coverage.genotype.tolist() == development.tolist()
    assert coverage.loc[coverage.eligible, 'genotype'].tolist() == genotypes
    assert len(sample) == 9666 and len(pairs) == 495900 and len(genotypes) == 137
    np.testing.assert_array_equal(sample.index.to_numpy()[a], pairs.ACR)
    np.testing.assert_array_equal(np.asarray(genotypes)[g], pairs.genotype)
    np.testing.assert_array_equal(sample.block.to_numpy()[a], data['blocks'])
    pcs = read_development_parquet(ROOT/'data/maize_ancestry_pcs.parquet')
    positions = development.get_indexer(genotypes)
    if np.any(positions < 0): raise ValueError('Unknown prediction genotype')
    region = np.column_stack([np.ones(len(sample)),
        gc_spline_basis(sample.GC, [0, .25, .5, .75, 1]), sample.length,
        *[sample.chrom.eq(chrom).to_numpy(float) for chrom in range(2, 11)]])
    nuisance = np.c_[region[a], pcs.to_numpy()[positions[g]]]
    np.testing.assert_allclose(nuisance, data['a'], atol=1e-12, rtol=1e-12)
    # Verify the unchanged response directly from original-core counts and
    # whole-genome totals; never open a phenotype table.
    counts = np.memmap(ROOT/'data/intermediate/reference_cortex/counts.int32',
        mode='r', dtype=np.int32, shape=(108843, len(development)))
    global_rows = sample.global_acr_index.to_numpy()[a]
    expected_y = counts[global_rows, positions[g]]*1e6/coverage.total_events.to_numpy()[positions[g]]
    np.testing.assert_array_equal(data['y'], expected_y)
    for panel in PANELS:
        source, extent = panel.split('_')
        name = f'{extent}_{"B73" if source == "B73" else "personalized"}_pattern_id'
        np.testing.assert_array_equal(pattern_ids[panel], pairs[name])
    folds = json.loads((frozen/'fold_manifest.json').read_text())
    assert folds == json.loads((ROOT/'results/sampled_linear_benchmark/fold_manifest.json').read_text())
    first = folds[0]; inner = first['inner'][0]
    train = np.flatnonzero(np.isin(data['blocks'], inner['training_blocks']))
    validation = np.flatnonzero(np.isin(data['blocks'], inner['validation_blocks']))
    outer = np.flatnonzero(np.isin(data['blocks'], first['validation_blocks']))
    assert not np.intersect1d(train, validation).size
    assert not np.intersect1d(train, outer).size and not np.intersect1d(validation, outer).size
    transform = training_transform(data['a'], data['y'], train)
    z, yy = apply_transform(data['a'], data['y'], transform)
    save_json(OUT/'transform.json', transform)
    source_paths = [frozen/p for p in ['selected_acrs.parquet', 'feature_lookup.parquet',
        'genotypes.csv', 'sample_receipt.json', 'sequence_receipt.json', 'fold_manifest.json']]
    source_paths += [ROOT/'data/intermediate/sampled_linear_benchmark'/p
                     for p in ['observations.npz', 'pattern_ids.npz']]
    source_paths += [sequence_dir/f'{scope}_patterns.parquet' for scope in ['core', 'context']]
    code_paths = [Path(__file__), ROOT/'src/maize_regulatory/sampled_cnn.py']
    state = dict(inputs={str(p.relative_to(ROOT)): digest(p) for p in source_paths},
        code={str(p.relative_to(ROOT)): digest(p) for p in code_paths},
        n_ACRs=len(sample), n_pairs=len(pairs), n_genotypes=len(genotypes), n_covariates=data['a'].shape[1],
        inner_training_pairs=len(train), inner_validation_pairs=len(validation),
        outer_validation_pairs_not_scored=len(outer), setting=SETTING,
        torch=torch.__version__, numpy=np.__version__, platform=platform.platform(),
        dtype='float32', original_core_response_verified=True, holdout_or_DTS_read=False,
        full_outer_CV_started=False, learned_core_mask=False,
        semantics='Equal actual ACR/genotype observations; exact DNA-grouped loss with nuisance cross-terms')
    existing = OUT/'input_state.json'
    if existing.exists() and json.loads(existing.read_text()) != state:
        raise ValueError('Preflight sources/config changed; choose a new output directory')
    save_json(existing, state)
    for p in code_paths:
        destination = OUT/'source_snapshot'/p.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, destination)
    for scope in ['core', 'context']:
        path = CACHE/f'{scope}_codes.npy'
        if path.exists(): continue
        frame = pd.read_parquet(sequence_dir/f'{scope}_patterns.parquet')
        np.testing.assert_array_equal(frame.pattern_id, np.arange(len(frame)))
        length = frame.sequence.str.len().to_numpy()
        width = 2000 if scope == 'context' else int(np.ceil(length.max()/32)*32)
        codes = np.full((len(frame), width), 4, np.uint8)
        lookup = np.full(256, 255, np.uint8)
        lookup[np.frombuffer(b'ACGT', np.uint8)] = np.arange(4, dtype=np.uint8)
        for j, sequence in enumerate(frame.sequence):
            encoded = lookup[np.frombuffer(sequence.encode('ascii'), np.uint8)]
            if np.any(encoded > 3): raise ValueError('Selected DNA is ambiguous')
            codes[j, :len(encoded)] = encoded
        np.save(path, codes)
    print(f'Verified {len(pairs):,} observations; timing only inner split 0/0: '
          f'{len(train):,} train, {len(validation):,} validation', flush=True)
    return data, pattern_ids, folds, train, validation, z, yy, transform


def allocator():
    if torch.backends.mps.is_available():
        return dict(current_bytes=torch.mps.current_allocated_memory(),
                    driver_bytes=torch.mps.driver_allocated_memory())
    return {}


def release():
    gc.collect()
    if torch.backends.mps.is_available(): torch.mps.empty_cache()


def runtime_projection(ids, data, folds, rows, epochs):
    fits, seconds = [], 0.
    for row in rows:
        panel = row['panel']; measurements = row['epoch_timings']
        # Slowest full-epoch measurements allow for warm-up/thermal variance.
        training_cost = max(v['seconds'] for v in measurements)/row['training_patterns']
        validation_cost = max(v['validation_seconds'] for v in measurements)/row['validation_patterns']
        for spec in folds:
            for candidate in range(2):
                for j, inner in enumerate(spec['inner']):
                    ntrain = len(np.unique(ids[panel][np.isin(data['blocks'], inner['training_blocks'])]))
                    nval = len(np.unique(ids[panel][np.isin(data['blocks'], inner['validation_blocks'])]))
                    cost = epochs*(ntrain*training_cost+nval*validation_cost)
                    fits.append(dict(panel=panel, role='inner', outer=spec['number'],
                        candidate=candidate, inner=j, seconds=cost)); seconds += cost
            ntrain = len(np.unique(ids[panel][np.isin(data['blocks'], spec['training_blocks'])]))
            nval = len(np.unique(ids[panel][np.isin(data['blocks'], spec['validation_blocks'])]))
            cost = epochs*ntrain*training_cost+nval*validation_cost
            fits.append(dict(panel=panel, role='outer_refit', outer=spec['number'], seconds=cost)); seconds += cost
    return dict(epoch_cap=epochs, no_early_stopping_hours=seconds/3600,
        with_50_percent_runtime_margin_hours=1.5*seconds/3600,
        counted_inner_fits=sum(f['role'] == 'inner' for f in fits),
        counted_outer_refits=sum(f['role'] == 'outer_refit' for f in fits), fits=fits,
        qualification='Extrapolation, not a runtime guarantee; assumes measured throughput persists. '
                      'Uses all epochs, no benefit from early stopping, and includes epoch validation.')


def run(device, batches):
    if device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS GPU is hidden/unavailable; run the preflight with GPU access')
    data, ids, folds, train, validation, z, yy, transform = prepare()
    timing_rows = []; preflight_started = time.monotonic()
    for panel in PANELS:
        source, extent = panel.split('_')
        train_groups = group_observations(z[train], yy[train], ids[panel][train])
        val_groups = group_observations(z[validation], yy[validation], ids[panel][validation])
        bank = SequenceBank(np.load(CACHE/f'{extent}_codes.npy', mmap_mode='r'), device)
        results = []
        print(f'{panel}: {train_groups["n_patterns"]:,} training DNA patterns', flush=True)
        for batch in batches:
            trainer = Trainer(bank, train_groups, SETTING, seed=42)
            # Isolate throughput from stochastic order and use real inner-train groups.
            order = np.random.default_rng(42).permutation(train_groups['n_patterns'])
            for step in range(4): trainer.step(order[step*batch:(step+1)*batch])
            synchronize(device); started = time.monotonic()
            memory = allocator()
            for step in range(8):
                at = np.resize(order, 8*batch).reshape(8, batch)[step]
                trainer.step(at)
                observed = allocator()
                for key, value in observed.items(): memory[key] = max(memory.get(key, 0), value)
            synchronize(device)
            duration = time.monotonic()-started
            result = dict(batch_size=batch, measured_steps=8, seconds=duration,
                          seconds_per_1000_patterns=duration/(8*batch)*1000,
                          sampled_allocator_bytes=memory,
                          n_parameters=sum(p.numel() for p in trainer.model.parameters()))
            results.append(result); print('  batch timing', result, flush=True)
            del trainer; release()
        selected = min(results, key=lambda v: v['seconds_per_1000_patterns'])['batch_size']
        trainer = Trainer(bank, train_groups, SETTING, seed=142)
        initial_train = evaluate(trainer.model, bank, train_groups, transform['target_sd'], selected)
        initial_val = evaluate(trainer.model, bank, val_groups, transform['target_sd'], selected)
        epoch_timings = []
        n_epochs = 5 if panel == 'personalized_context' else 3
        for epoch in range(n_epochs):
            info = trainer.train_epoch(selected)
            score = evaluate(trainer.model, bank, val_groups, transform['target_sd'], selected)
            info.update(validation_seconds=score['seconds'], validation_pair_mse_CPM2=score['pair_mse_CPM2'])
            trainer.save(OUT/'checkpoints'/f'{panel}.pt', {'panel': panel, 'transform': transform})
            epoch_timings.append(info)
            print('  full inner epoch', info, flush=True)
            save_json(OUT/f'{panel}_progress.json', epoch_timings)
        final_train = evaluate(trainer.model, bank, train_groups, transform['target_sd'], selected)
        # Independent explicit-pair validation of deterministic grouped loss.
        sequence_values = predict_groups(trainer.model, bank, val_groups['pattern_ids'], selected)
        gamma = trainer.model.linear.weight.detach().cpu().numpy()[0]
        intercept = float(trainer.model.linear.bias.detach().cpu()[0])
        mapping = np.searchsorted(val_groups['pattern_ids'], ids[panel][validation])
        explicit_prediction = sequence_values[mapping]+z[validation]@gamma+intercept
        explicit_mse = float(np.mean((explicit_prediction-yy[validation])**2))*transform['target_sd']**2
        grouped_mse = evaluate(trainer.model, bank, val_groups, transform['target_sd'], selected)['pair_mse_CPM2']
        if not np.isclose(explicit_mse, grouped_mse, rtol=2e-5, atol=2e-4):
            raise AssertionError('Real-data explicit and grouped validation losses differ')
        # Verify optimizer, orientation/dropout RNG, and weights survive reload.
        checkpoint = OUT/'checkpoints'/f'{panel}.pt'
        next_rows = np.arange(min(selected, train_groups['n_patterns']))
        trainer.step(next_rows); synchronize(device)
        expected_weights = {k: v.detach().cpu().clone() for k, v in trainer.model.state_dict().items()}
        restored = Trainer(bank, train_groups, SETTING, seed=999)
        restored.restore(checkpoint); restored.step(next_rows); synchronize(device)
        max_diff = max(float((value.detach().cpu()-expected_weights[name]).abs().max())
                       for name, value in restored.model.state_dict().items())
        if max_diff > 2e-5: raise AssertionError(f'GPU resume mismatch: {max_diff}')
        result = dict(panel=panel, batch_timings=results, selected_batch_size=selected,
            training_patterns=train_groups['n_patterns'], validation_patterns=val_groups['n_patterns'],
            training_observations=len(train), validation_observations=len(validation),
            initial_training_pair_mse_CPM2=initial_train['pair_mse_CPM2'],
            initial_validation_pair_mse_CPM2=initial_val['pair_mse_CPM2'],
            final_training_pair_mse_CPM2=final_train['pair_mse_CPM2'],
            epoch_timings=epoch_timings, explicit_validation_pair_mse_CPM2=explicit_mse,
            grouped_validation_pair_mse_CPM2=grouped_mse, checkpoint_resume_max_parameter_difference=max_diff,
            checkpoint_bytes=checkpoint.stat().st_size)
        timing_rows.append(result); save_json(OUT/'timings.json', timing_rows)
        del trainer, restored, bank; release()
    projections = [runtime_projection(ids, data, folds, timing_rows, cap) for cap in [15, 30, 40]]
    save_json(OUT/'runtime_projections.json', projections)
    summary = dict(completed_utc=datetime.now(timezone.utc).isoformat(), device=device,
        torch=torch.__version__, seconds=time.monotonic()-preflight_started,
        n_parameters=timing_rows[0]['batch_timings'][0]['n_parameters'],
        rows=timing_rows, projections=[{k:v for k,v in p.items() if k != 'fits'} for p in projections],
        free_disk_bytes=shutil.disk_usage(ROOT).free, outer_validation_scored=False,
        full_nested_CV_started=False, holdout_or_DTS_read=False,
        note='Short trials diagnose feasibility, not reportable performance or outer-model selection.')
    save_json(OUT/'summary.json', summary)
    text = ['# Four matched CNNs: preflight results', '',
        'Only the first inner training/validation split was used. No outer performance was calculated.', '',
        '| Input | Batch | Train seconds/epoch | Validation seconds/epoch | Train MSE before → after |',
        '|---|---:|---:|---:|---:|']
    for row in timing_rows:
        text.append(f'| {row["panel"]} | {row["selected_batch_size"]} | '
            f'{np.mean([e["seconds"] for e in row["epoch_timings"]]):.2f} | '
            f'{np.mean([e["validation_seconds"] for e in row["epoch_timings"]]):.2f} | '
            f'{row["initial_training_pair_mse_CPM2"]:.4f} → {row["final_training_pair_mse_CPM2"]:.4f} |')
    text += ['', '## Full-run projections', '',
        '120 inner fits + 20 outer refits; one architecture, two L2 candidates, one seed per fit.', '']
    for p in summary['projections']:
        text.append(f'- {p["epoch_cap"]} epochs: {p["no_early_stopping_hours"]:.2f} hours; '
                    f'{p["with_50_percent_runtime_margin_hours"]:.2f} hours with a 50% runtime margin.')
    text += ['', 'These projections include validation every inner epoch, assume no early stopping, '
        'and are not runtime guarantees. Checkpoints, exact objective checks, and source hashes are retained.',
        '', 'The network has four nucleotide channels, 64 filters, a stride-one 15-bp stem, '
        'pooling by four, three two-convolution residual blocks (5-bp kernels; dilations 1/2/4) '
        'with pooling by two, masked mean/max aggregation, a 128-unit dense layer, and a scalar '
        'sequence output. A separately parameterized nuisance branch is jointly trained, unpenalized. '
        'The mask only excludes artificial padding; it does not mark the ACR within a full window.', '']
    (OUT/'report.md').write_text('\n'.join(text))
    print('\n'.join(text), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='mps', choices=['mps', 'cpu'])
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[64, 128, 256])
    args = parser.parse_args()
    torch.set_num_threads(4)
    run(args.device, args.batch_sizes)
