#!/usr/bin/env python
"""Fit the 13 approved linear models on frozen matched pairs; never trains a CNN."""

import argparse
import gc
import json
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import pandas as pd

from maize_regulatory.bulk_motif_association import gc_spline_basis
from maize_regulatory.cohort import development_genotypes, read_development_parquet
from maize_regulatory.pooled_sequence import Moments, add_moments, choose_alpha, moment_mse
from maize_regulatory.reference_sequence import digest
from maize_regulatory.sampled_linear import (
    aggregate_block_moments, partial_ridge_path, prediction_metrics, subset_moments,
)


def save_json(path, value):
    tmp = Path(str(path)+'.part')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); tmp.replace(path)


def names(config):
    return ['baseline']+[f'{source}_{extent}_{feature}' for source in config['sequence_sources']
                         for extent in config['sequence_extents'] for feature in config['feature_sets']]


def prepare(root, out, cache, config):
    out.mkdir(parents=True, exist_ok=True); cache.mkdir(parents=True, exist_ok=True)
    frozen = root/'results/sampled_sequence_features'
    features = root/'data/intermediate/sampled_sequence_features'
    sample = pd.read_parquet(frozen/'selected_acrs.parquet')
    pairs = pd.read_parquet(frozen/'feature_lookup.parquet')
    original_pairs = pd.read_parquet(frozen/'pair_manifest.parquet')
    pd.testing.assert_frame_equal(pairs[original_pairs.columns], original_pairs)
    expected = pd.read_csv(frozen/'genotypes.csv').genotype.tolist()
    coverage = pd.read_csv(root/'results/sequence_cnn_benchmark/genotype_coverage.csv')
    development = development_genotypes()
    if (coverage.genotype.tolist() != development.tolist()
            or coverage.loc[coverage.eligible, 'genotype'].tolist() != expected):
        raise ValueError('Frozen genotype order/cohort mismatch')
    a, g = pairs.sample_acr_index.to_numpy(), pairs.genotype_index.to_numpy()
    if (not np.array_equal(sample.sample_acr_index, np.arange(len(sample)))
            or not np.array_equal(pairs.pair_index, np.arange(len(pairs)))
            or not np.array_equal(sample.index.to_numpy()[a], pairs.ACR)
            or not np.array_equal(np.array(expected)[g], pairs.genotype)):
        raise ValueError('Frozen pair/ACR/genotype axes differ')
    sample_receipt = json.loads((frozen/'sample_receipt.json').read_text())
    if any(digest(root/p) != checksum for p, checksum in sample_receipt['outputs'].items()):
        raise ValueError('Frozen sample file digest mismatch')
    sequence_receipt = json.loads((frozen/'sequence_receipt.json').read_text())
    if any(digest(root/p) != checksum for p, checksum in sequence_receipt['outputs'].items()):
        raise ValueError('Frozen sequence/lookup digest mismatch')
    score_receipt = json.loads((frozen/'scoring_receipt.json').read_text())
    inputs = [frozen/p for p in ['selected_acrs.parquet', 'feature_lookup.parquet',
        'genotypes.csv', 'sample_receipt.json', 'sequence_receipt.json', 'scoring_receipt.json',
        'feature_columns.json', 'fold_manifest.json']]
    for extent in config['sequence_extents']:
        entry = next(v for v in score_receipt['timings'] if v['kind'] == extent)
        for feature in ['TF', 'word']:
            path = features/f'features/{extent}_{feature}.npy'
            if digest(path) != entry['feature_files'][feature]['sha256']:
                raise ValueError('Feature score digest mismatch')
            inputs.append(path)
    folds = json.loads((frozen/'fold_manifest.json').read_text())
    if folds != json.loads((root/'results/sequence_summary_comparison/fold_manifest.json').read_text()):
        raise ValueError('Original genomic CV assignments changed')
    global_acrs = pd.read_csv(root/'data/intermediate/cells/acr_index.csv')
    if (not np.array_equal(global_acrs.matrix_column, np.arange(1, len(global_acrs)+1))
            or not np.array_equal(global_acrs.ACR.to_numpy()[sample.global_acr_index], sample.index)):
        raise ValueError('Raw count ACR axis differs from frozen sample')
    counts_cache = root/'data/intermediate/reference_cortex'
    receipt = json.loads((counts_cache/'receipt.json').read_text())
    if (receipt['genotypes'] != development.tolist()
            or pd.read_csv(counts_cache/'genotypes.csv').genotype.tolist() != development.tolist()
            or digest(counts_cache/'counts.int32') != receipt['counts_sha256']):
        raise ValueError('Source pooled counts/axes changed')
    positions = development.get_indexer(expected)
    if np.any(positions < 0):
        raise ValueError('Unknown development genotype')
    counts = np.memmap(counts_cache/'counts.int32', dtype='int32', mode='r',
                       shape=(len(global_acrs), len(development)))
    source_rows = sample.global_acr_index.to_numpy()[a]
    total_events = coverage.total_events.to_numpy()[positions]
    y = np.asarray(counts[source_rows, positions[g]], float)*1e6/total_events[g]
    old_response = np.load(root/'data/intermediate/sequence_cnn/response.npy', mmap_mode='r')
    np.testing.assert_array_equal(y, old_response[source_rows, g])
    pc_frame = read_development_parquet(root/'data/maize_ancestry_pcs.parquet')
    if not pc_frame.index.equals(development):
        raise ValueError('Ancestry source genotype order differs')
    pcs = pc_frame.to_numpy()[positions]
    region = np.column_stack([np.ones(len(sample)),
        gc_spline_basis(sample.GC, config['fixed_gc_knots']), sample.length,
        *[sample.chrom.eq(chrom).to_numpy(float) for chrom in range(2, 11)]])
    old_region = np.load(root/'data/intermediate/sequence_cnn/covariates.npy', mmap_mode='r')
    np.testing.assert_allclose(region, old_region[sample.global_acr_index], atol=1e-12, rtol=1e-12)
    np.testing.assert_array_equal(pcs, np.load(root/'data/intermediate/sequence_cnn/pcs.npy'))
    nuisance = np.column_stack([region[a], pcs[g]])
    covariate_names = ['intercept']+[f'B73_core_GC_spline_{j+1}' for j in range(4)]+['core_length']+[
        f'chromosome_{c}' for c in range(2, 11)]+[f'ancestry_PC{j+1}' for j in range(10)]
    if nuisance.shape != (len(pairs), len(covariate_names)):
        raise ValueError('Nuisance basis dimensions disagree')
    inputs += [root/p for p in ['data/intermediate/reference_cortex/counts.int32',
        'data/intermediate/reference_cortex/receipt.json', 'data/intermediate/cells/acr_index.csv',
        'data/maize_ancestry_pcs.parquet', 'results/sequence_cnn_benchmark/genotype_coverage.csv']]
    code = ['scripts/analyze_sampled_linear.py', 'src/maize_regulatory/sampled_linear.py',
            'src/maize_regulatory/pooled_sequence.py', 'src/maize_regulatory/sampled_linear_report.py']
    state = dict(config=config, inputs={str(p.relative_to(root)): digest(p) for p in inputs},
        code={p: digest(root/p) for p in code}, numpy=np.__version__, platform=platform.platform(),
        n_pairs=len(pairs), n_ACRs=len(sample), n_genotypes=len(expected), n_covariates=nuisance.shape[1],
        raw_CPM_values_independently_verified=len(y), nuisance_matches_existing_basis=True,
        holdout_or_DTS_read=False, model_names=names(config))
    path = out/'run_config.json'
    if path.exists() and json.loads(path.read_text()) != state:
        raise ValueError('Run inputs/config/code changed; choose new output/cache directories')
    save_json(path, state)
    for relative in code:
        dest = out/'source_snapshot'/relative
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(root/relative, dest)
    np.savez(cache/'observations.npz', a=nuisance, y=y, sample_acr_index=a, genotype_index=g,
             blocks=sample.block.to_numpy()[a], outer_fold=pairs.outer_fold.to_numpy())
    np.savez(cache/'pattern_ids.npz', **{f'{source}_{extent}': pairs[f'{extent}_{"B73" if source == "B73" else "personalized"}_pattern_id'].to_numpy()
        for source in config['sequence_sources'] for extent in config['sequence_extents']})
    pairs[['pair_index', 'sample_acr_index', 'genotype_index', 'ACR', 'genotype', 'outer_fold']].assign(
        observed_CPM=y).to_parquet(out/'observations.parquet', index=False)
    sample.to_parquet(out/'selected_acrs.parquet')
    shutil.copyfile(frozen/'fold_manifest.json', out/'fold_manifest.json')
    save_json(out/'covariate_names.json', covariate_names)
    print(f'Prepared {len(pairs):,} exact raw-CPM pairs; {nuisance.shape[1]} fixed nuisance terms', flush=True)


def moments_for_panel(data, ids, x, folds, directory):
    partitions = {}
    for spec in folds:
        f = spec['number']
        partitions[f'outer_{f}'] = set(spec['validation_blocks'])
        for j, inner in enumerate(spec['inner']):
            partitions[f'outer_{f}_inner_{j}'] = set(inner['validation_blocks'])
    directory.mkdir(parents=True, exist_ok=True)
    complete = directory/'complete.json'
    if complete.exists():
        print('Using completed partition cross-products', flush=True); return
    result = aggregate_block_moments(data['a'], ids, x, data['y'], data['blocks'], partitions,
        progress=lambda n, total: print(f'  cross-products: {n}/{total} genomic blocks', flush=True))
    for name, value in result.items():
        np.savez(directory/f'{name}.npz', n=value.n, xx=value.xx, xy=value.xy, yy=value.yy)
    save_json(complete, dict(partitions=list(partitions), n_pairs=len(data['y']), shape=list(next(iter(result.values())).xx.shape)))


def read_moments(path):
    with np.load(path) as z:
        return Moments(int(z['n']), z['xx'], z['xy'], float(z['yy']))


def run(root, out, cache, config):
    prepare(root, out, cache, config)
    data = dict(np.load(cache/'observations.npz'))
    ids_all = dict(np.load(cache/'pattern_ids.npz'))
    folds = json.loads((out/'fold_manifest.json').read_text())
    feature_dir = root/'data/intermediate/sampled_sequence_features/features'
    model_names = names(config); n_cov = data['a'].shape[1]
    prediction_path = out/'predictions.npy'
    if prediction_path.exists():
        predictions = np.load(prediction_path, mmap_mode='r+')
        if predictions.shape != (len(data['y']), len(model_names)):
            raise ValueError('Existing prediction shape mismatch')
    else:
        predictions = np.lib.format.open_memmap(prediction_path, mode='w+', dtype='float64',
            shape=(len(data['y']), len(model_names))); predictions[:] = np.nan; predictions.flush()
    (out/'fits').mkdir(exist_ok=True); (out/'inner').mkdir(exist_ok=True)
    alpha_grid = config['alphas']+[None]
    started = time.monotonic()
    for source in config['sequence_sources']:
        for extent in config['sequence_extents']:
            panel = f'{source}_{extent}'
            raw_ids = ids_all[panel]
            used, ids = np.unique(raw_ids, return_inverse=True)
            if used.min() < 0:
                raise ValueError('Unavailable feature row in frozen pair set')
            x = np.column_stack([np.load(feature_dir/f'{extent}_TF.npy', mmap_mode='r')[used],
                                 np.load(feature_dir/f'{extent}_word.npy', mmap_mode='r')[used]]).astype(float)
            print(f'Panel {panel}: {len(used):,} sequence patterns, {x.shape[1]} distinct features', flush=True)
            directory = cache/'moments'/panel
            moments_for_panel(data, ids, x, folds, directory)
            outer_parts = [read_moments(directory/f'outer_{spec["number"]}.npz') for spec in folds]
            for spec in folds:
                f = spec['number']; train_full = add_moments([m for j, m in enumerate(outer_parts) if j != f])
                val_rows = np.flatnonzero(np.isin(data['blocks'], spec['validation_blocks']))
                if not np.all(data['outer_fold'][val_rows] == f):
                    raise ValueError('Prediction validation rows differ from frozen outer fold')
                base_fit = partial_ridge_path(subset_moments(train_full, np.arange(n_cov)), n_cov, [None], [])
                baseline_prediction = data['a'][val_rows]@base_fit['coefficients'][None]
                if np.isfinite(predictions[val_rows, 0]).all():
                    np.testing.assert_allclose(predictions[val_rows, 0], baseline_prediction, atol=1e-8, rtol=1e-10)
                else:
                    predictions[val_rows, 0] = baseline_prediction
                    np.savez(out/'fits'/f'baseline_outer_{f}.npz', coefficients=base_fit['coefficients'][None],
                        mean=base_fit['mean'], scale=base_fit['scale'])
                inner_parts = [read_moments(directory/f'outer_{f}_inner_{j}.npz') for j in range(3)]
                for feature in config['feature_sets']:
                    model = f'{panel}_{feature}'; column = model_names.index(model)
                    done = out/'fits'/f'{model}_outer_{f}.json'
                    if done.exists():
                        if not np.isfinite(predictions[val_rows, column]).all():
                            raise ValueError('Completed fit has missing predictions')
                        print(f'Keeping {model}, outer fold {f+1}', flush=True); continue
                    selected = (np.arange(927) if feature == 'TF' else np.arange(927, 1439)
                                if feature == 'word' else np.arange(1439))
                    indices = np.r_[np.arange(n_cov), n_cov+selected]
                    penalty = np.where(selected < 927, 1., .5)
                    scores = []
                    for j, inner_val in enumerate(inner_parts):
                        train = subset_moments(add_moments([part for k, part in enumerate(inner_parts) if k != j]), indices)
                        validation = subset_moments(inner_val, indices)
                        fitted = partial_ridge_path(train, n_cov, alpha_grid, penalty)
                        for alpha in alpha_grid:
                            loss = moment_mse(validation, fitted['coefficients'][alpha])
                            scores.append(dict(alpha=alpha, sse=loss*validation.n, n=validation.n, inner_fold=j))
                    alpha, losses = choose_alpha(scores, alpha_grid)
                    train = subset_moments(train_full, indices)
                    fitted = partial_ridge_path(train, n_cov, [alpha], penalty)
                    coefficient = fitted['coefficients'][alpha]
                    full_beta = np.zeros(x.shape[1]); full_beta[selected] = coefficient[n_cov:]
                    sequence_effect = x@full_beta
                    predicted = data['a'][val_rows]@coefficient[:n_cov]+sequence_effect[ids[val_rows]]
                    expected_loss = moment_mse(subset_moments(outer_parts[f], indices), coefficient)
                    direct_loss = float(np.mean((predicted-data['y'][val_rows])**2))
                    if not np.isclose(direct_loss, expected_loss, rtol=1e-8, atol=1e-8):
                        raise AssertionError('Explicit pair prediction loss differs from moment loss')
                    predictions[val_rows, column] = predicted; predictions.flush()
                    np.savez(out/'fits'/f'{model}_outer_{f}.npz', coefficients=coefficient,
                        mean=fitted['mean'], scale=fitted['scale'], constant=fitted['constant'],
                        feature_indices=selected, penalty=penalty,
                        selected_alpha=np.nan if alpha is None else alpha)
                    save_json(out/'inner'/f'{model}_outer_{f}.json', dict(scores=scores,
                        pooled_MSE_by_alpha=[dict(alpha=a, mse=losses[a]) for a in alpha_grid]))
                    save_json(done, dict(model=model, outer_fold=f, selected_alpha=alpha,
                        validation_pairs=len(val_rows), training_pairs=train.n,
                        moment_MSE=expected_loss, explicit_pair_MSE=direct_loss,
                        nuisance_rank=fitted['nuisance_rank']))
                    print(f'{model}, outer {f+1}/5: alpha={alpha}, RMSE={np.sqrt(direct_loss):.6f}', flush=True)
                del inner_parts; gc.collect()
            del x, outer_parts; gc.collect()
    predictions.flush()
    if not np.isfinite(predictions).all():
        raise AssertionError('Missing or nonfinite out-of-fold predictions')
    save_json(out/'fit_completion.json', dict(complete=True, models=model_names,
        seconds_this_invocation=time.monotonic()-started, n_pairs=len(data['y']),
        prediction_sha256=digest(prediction_path), no_CNN=True, holdout_or_DTS_read=False))
    summarize(out, config)


def summarize(out, config):
    observations = pd.read_parquet(out/'observations.parquet')
    sample = pd.read_parquet(out/'selected_acrs.parquet')
    predictions = np.load(out/'predictions.npy', mmap_mode='r')
    y = observations.observed_CPM.to_numpy(); acr = observations.sample_acr_index.to_numpy()
    rows, fold_rows = [], []
    for j, model in enumerate(names(config)):
        for group in config['subgroups']:
            mask = np.ones(len(y), bool) if group == 'all' else sample[group].to_numpy()[acr]
            rows.append(dict(model=model, subgroup=group, **prediction_metrics(y[mask], predictions[mask, j], acr[mask])))
        for fold in range(5):
            mask = observations.outer_fold.to_numpy() == fold
            fold_rows.append(dict(model=model, outer_fold=fold,
                **prediction_metrics(y[mask], predictions[mask, j], acr[mask])))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out/'metrics.csv', index=False)
    pd.DataFrame(fold_rows).to_csv(out/'fold_metrics.csv', index=False)
    saved = []
    for path in sorted((out/'fits').glob('*_outer_*.json')):
        saved.append(json.loads(path.read_text()))
    pd.DataFrame(saved).to_csv(out/'selected_penalties.csv', index=False)
    save_json(out/'summary.json', dict(n_ACRs=len(sample), n_pairs=len(y), n_genotypes=observations.genotype.nunique(),
        model_count=len(names(config)), models=names(config), subgroups=config['subgroups'],
        raw_CPM_response=True, fitted_on_actual_unbalanced_pairs=True,
        full_context_has_no_core_flank_coefficient_split=True, CNN_fitted=False, holdout_or_DTS_read=False))


def verify(root, out, cache, config):
    """Independent raw-feature prediction reconstruction and joint normal solve."""
    from scipy import linalg
    data = dict(np.load(cache/'observations.npz'))
    lookup = pd.read_parquet(root/'results/sampled_sequence_features/feature_lookup.parquet')
    folds = json.loads((out/'fold_manifest.json').read_text())
    predictions = np.load(out/'predictions.npy', mmap_mode='r')
    n_cov = data['a'].shape[1]
    rng = np.random.default_rng(410)
    maximum_prediction_difference = 0.; checked_pairs = 0
    for column, model in enumerate(names(config)):
        for spec in folds:
            f = spec['number']
            rows = np.flatnonzero(data['outer_fold'] == f)
            rows = rng.choice(rows, min(32, len(rows)), replace=False)
            fit = np.load(out/'fits'/f'{model}_outer_{f}.npz')
            coefficient = fit['coefficients']
            predicted = data['a'][rows]@coefficient[:n_cov]
            if model != 'baseline':
                source, extent, feature = model.split('_')
                ids = lookup[f'{extent}_{"B73" if source == "B73" else "personalized"}_pattern_id'].to_numpy()[rows]
                x = np.column_stack([np.load(root/f'data/intermediate/sampled_sequence_features/features/{extent}_{kind}.npy',
                    mmap_mode='r')[ids] for kind in ['TF', 'word']]).astype(float)
                predicted += x[:, fit['feature_indices']]@coefficient[n_cov:]
            difference = float(np.max(np.abs(predicted-predictions[rows, column])))
            maximum_prediction_difference = max(maximum_prediction_difference, difference)
            np.testing.assert_allclose(predicted, predictions[rows, column], atol=1e-8, rtol=1e-9)
            checked_pairs += len(rows)
    # Check the largest feature systems against a direct joint penalized normal solve,
    # without FWL residualization or the fitting solver's eigendecomposition.
    normal_checks = []
    for source in config['sequence_sources']:
        for extent in config['sequence_extents']:
            panel = f'{source}_{extent}'; model = panel+'_both'; fold = 0
            fit = np.load(out/'fits'/f'{model}_outer_{fold}.npz')
            alpha = float(fit['selected_alpha'])
            directory = cache/'moments'/panel
            training = add_moments([read_moments(directory/f'outer_{j}.npz') for j in range(1, 5)])
            mean, scale = fit['mean'], fit['scale']
            covariance = (training.xx[1:, 1:]/training.n-np.outer(mean, mean))/np.outer(scale, scale)
            target = (training.xy[1:]/training.n-mean*training.xy[0]/training.n)/scale
            if np.isnan(alpha):
                standardized = np.r_[linalg.lstsq(covariance[:n_cov-1, :n_cov-1], target[:n_cov-1],
                                                  cond=1e-12)[0], np.zeros(1439)]
            else:
                diagonal = np.r_[np.zeros(n_cov-1), fit['penalty']]*alpha
                joint = covariance+np.diag(diagonal)
                standardized = linalg.lstsq(joint, target, cond=1e-12)[0]
            slopes = standardized/scale
            direct = np.r_[training.xy[0]/training.n-mean@slopes, slopes]
            difference = float(np.max(np.abs(direct-fit['coefficients'])))
            np.testing.assert_allclose(direct, fit['coefficients'], atol=2e-6, rtol=2e-6)
            normal_checks.append(dict(model=model, outer_fold=fold, max_raw_coefficient_difference=difference))
    completion = json.loads((out/'fit_completion.json').read_text())
    if digest(out/'predictions.npy') != completion['prediction_sha256']:
        raise ValueError('Final prediction checksum changed')
    value = dict(all_13_models_have_complete_finite_out_of_fold_predictions=bool(np.isfinite(predictions).all()),
        independently_reconstructed_pair_predictions=checked_pairs,
        maximum_prediction_difference=maximum_prediction_difference,
        independent_joint_normal_equation_checks=normal_checks,
        all_60_sequence_refits_direct_pair_MSE_matches_moment_MSE=True,
        all_raw_CPM_pairs_reconstructed_and_verified=len(data['y']),
        permanent_holdout_or_DTS_used=False, CNN_trained=False)
    save_json(out/'validation.json', value)
    print(json.dumps(value, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'run', 'verify', 'report'], required=True)
    parser.add_argument('--output-dir', default='results/sampled_linear_benchmark')
    parser.add_argument('--cache-dir', default='data/intermediate/sampled_linear_benchmark')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out, cache = root/args.output_dir, root/args.cache_dir
    config = json.loads((root/'configs/sampled_linear_benchmark.json').read_text())
    if args.stage == 'prepare':
        prepare(root, out, cache, config)
    elif args.stage == 'run':
        run(root, out, cache, config)
    elif args.stage == 'verify':
        verify(root, out, cache, config)
    else:
        from maize_regulatory.sampled_linear_report import write_report
        write_report(out)


if __name__ == '__main__':
    main()
