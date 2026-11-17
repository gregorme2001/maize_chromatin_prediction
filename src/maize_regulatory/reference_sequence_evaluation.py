"""Matched B73 k-mer/motif ridge, exact pair-level loss, nested genomic blocks."""

import json

import numpy as np
import pandas as pd

from .pooled_sequence import Moments, add_moments, blocked_manifest, choose_alpha, moment_mse, ridge_path
from .reference_sequence import reference_features, reference_moments, subset_moments


def saved_moments(path, build):
    if path.exists():
        z = np.load(path)
        return Moments(int(z['n']), z['xx'], z['xy'], float(z['yy']))
    m = build()
    np.savez_compressed(path, n=m.n, xx=m.xx, xy=m.xy, yy=m.yy)
    return m


def score_arrays(y, predicted):
    error = y-predicted
    return dict(n=int(y.size), sse=float(np.sum(error**2)), sae=float(np.sum(np.abs(error))),
        y_sum=float(y.sum()), yy=float(np.sum(y*y)), prediction_sum=float(predicted.sum()),
        prediction_squared_sum=float(np.sum(predicted*predicted)),
        y_prediction_sum=float(np.sum(y*predicted)), n_negative=int(np.sum(predicted < 0)))


def summarize_totals(totals):
    n = totals['n']
    vy = totals['yy']-totals['y_sum']**2/n
    vp = totals['prediction_squared_sum']-totals['prediction_sum']**2/n
    cov = totals['y_prediction_sum']-totals['y_sum']*totals['prediction_sum']/n
    return dict(n_pairs=n, rmse=float(np.sqrt(totals['sse']/n)), mae=totals['sae']/n,
        mse=totals['sse']/n, r2=1-totals['sse']/vy,
        pearson_r=float(cov/np.sqrt(vy*vp)) if vp > 0 else np.nan,
        fraction_negative_predictions=totals['n_negative']/n)


def evaluate_reference(data, config, root, out):
    region, p, y = data['region'], data['pcs'], data['y']
    blocks = data['coordinates'].block.to_numpy()
    manifest = blocked_manifest(blocks, config['outer_splits'], config['inner_splits'], config['seed'])
    (out/'fold_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    n_covariates = region.shape[1]+p.shape[1]
    means = pd.DataFrame(dict(ACR=data['acrs'], observed_CPM=y.mean(axis=1),
        block=blocks, chromosome=data['coordinates'].chrom.to_numpy(),
        outer_fold=np.full(len(y), -1)))
    totals, genotype_sse, folds, inner_scores, coefficients, genotype_contributions = {}, {}, [], [], [], []
    alpha_grid = config['alphas']+[None]
    (out/'moments').mkdir(exist_ok=True)
    (out/'models').mkdir(exist_ok=True)
    feature_kinds = config.get('feature_kinds', ['kmer', 'motif'])
    if not feature_kinds or len(set(feature_kinds)) != len(feature_kinds):
        raise ValueError('Feature kinds must be nonempty and unique')
    for kind in feature_kinds:
        print(f'Preparing fixed B73 {kind} features', flush=True)
        x, feature_names, multiplier = reference_features(root, data, kind)
        (out/f'{kind}_features.json').write_text(json.dumps(feature_names, indent=2)+'\n')
        outer_parts = []
        for spec in manifest:
            rows = np.flatnonzero(np.isin(blocks, spec['validation_blocks']))
            outer_parts.append(saved_moments(out/'moments'/f'{kind}_outer_{spec["number"]}.npz',
                lambda rows=rows: reference_moments(region[rows], p, x[rows], y[rows])))
        for spec in manifest:
            fold = spec['number']
            train_full = add_moments([m for j, m in enumerate(outer_parts) if j != fold])
            inner_parts = []
            for j, inner in enumerate(spec['inner']):
                rows = np.flatnonzero(np.isin(blocks, inner['validation_blocks']))
                inner_parts.append(saved_moments(out/'moments'/f'{kind}_outer_{fold}_inner_{j}.npz',
                    lambda rows=rows: reference_moments(region[rows], p, x[rows], y[rows])))
            rows = np.flatnonzero(np.isin(blocks, spec['validation_blocks']))
            means.loc[rows, 'outer_fold'] = fold
            for basis_name, cov_ix in data['bases'].items():
                scores = []
                for j, part in enumerate(inner_parts):
                    train = subset_moments(add_moments([v for k, v in enumerate(inner_parts) if k != j]), cov_ix, n_covariates)
                    val = subset_moments(part, cov_ix, n_covariates)
                    path = ridge_path(train, len(cov_ix), alpha_grid, penalty_multiplier=multiplier)
                    for alpha in alpha_grid:
                        loss = moment_mse(val, path['coefficients'][alpha])
                        scores.append(dict(alpha=alpha, sse=loss*val.n, n=val.n))
                        inner_scores.append(dict(feature=kind, adjustment=basis_name, outer_fold=fold,
                            inner_fold=j, alpha=alpha, exact_baseline=alpha is None, mse=loss, n_pairs=val.n))
                alpha, losses = choose_alpha(scores, alpha_grid)
                train = subset_moments(train_full, cov_ix, n_covariates)
                path = ridge_path(train, len(cov_ix), [None, alpha] if alpha is not None else [None],
                                  penalty_multiplier=multiplier)
                for label, selected in [('baseline', None), (kind, alpha)]:
                    model = f'{basis_name}_{label}'
                    c = path['coefficients'][selected]
                    # Map selected baseline coordinates back to common region/PC axes.
                    full_baseline = np.zeros(n_covariates)
                    full_baseline[cov_ix] = c[:len(cov_ix)]
                    pc_coefficient = full_baseline[region.shape[1]:]
                    component = region[rows]@full_baseline[:region.shape[1]]+x[rows]@c[len(cov_ix):]
                    pc_effect = p@pc_coefficient
                    predicted = component[:, None]+pc_effect[None, :]
                    if kind != feature_kinds[0] and label == 'baseline':
                        prior = np.load(out/'models'/f'outer_{fold}_{model}.npz')
                        if not np.allclose(prior['full_baseline'], full_baseline, atol=1e-9, rtol=1e-9):
                            raise AssertionError('Matched baseline differs between feature sets')
                        continue
                    direct = score_arrays(y[rows], predicted)
                    val = subset_moments(outer_parts[fold], cov_ix, n_covariates)
                    if not np.isclose(direct['sse']/direct['n'], moment_mse(val, c), rtol=1e-9):
                        raise AssertionError('Pair predictions differ from sufficient-statistic loss')
                    totals[model] = {k: totals.get(model, {}).get(k, 0)+v for k, v in direct.items()}
                    genotype_sse[model] = genotype_sse.get(model, np.zeros(len(p)))+np.sum((y[rows]-predicted)**2, axis=0)
                    means.loc[rows, model+'_CPM'] = predicted.mean(axis=1)
                    means.loc[rows, model+'_mse'] = np.mean((y[rows]-predicted)**2, axis=1)
                    np.savez_compressed(out/'models'/f'outer_{fold}_{model}.npz', coefficients=c,
                        full_baseline=full_baseline, covariate_indices=cov_ix,
                        training_mean=path['mean'], training_scale=path['scale'],
                        constant=path['constant'], pc_effect=pc_effect,
                        alpha=np.nan if selected is None else selected)
                    for k, effect in enumerate(pc_effect):
                        genotype_contributions.append(dict(outer_fold=fold, model=model,
                            genotype=data['genotypes'][k], ancestry_contribution=float(effect),
                            centered_ancestry_contribution=float(effect-pc_effect.mean())))
                    names = [data['covariate_names'][i] for i in cov_ix]+feature_names
                    coefficients.extend(dict(outer_fold=fold, model=model, parameter=n, raw_coefficient=float(v))
                        for n, v in zip(names, c))
                    folds.append(dict(outer_fold=fold, model=model, selected_alpha=selected,
                        n_training_pairs=train.n, n_validation_pairs=direct['n'], n_validation_acrs=len(rows),
                        nuisance_rank=path['nuisance_rank'], **summarize_totals(direct)))
                print(f'{kind}, {basis_name}, fold {fold+1}: alpha={alpha}; '
                    f'pair RMSE={np.sqrt(direct["sse"]/direct["n"]):.4f} CPM', flush=True)
            means.to_parquet(out/'acr_predictions.parquet', index=False)
            pd.DataFrame(folds).to_csv(out/'fold_results.csv', index=False)
        del x, outer_parts, inner_parts
    if means.outer_fold.lt(0).any() or means.filter(regex='_CPM$').isna().any().any():
        raise AssertionError('Every ACR must have matched out-of-fold predictions')
    metrics = []
    for model, value in totals.items():
        summary = summarize_totals(value)
        predicted_mean = means[model+'_CPM'].to_numpy()
        observed_mean = means.observed_CPM.to_numpy()
        mean_sse = float(np.sum((observed_mean-predicted_mean)**2))
        summary.update(model=model, mean_acr_rmse=float(np.sqrt(mean_sse/len(y))),
            mean_acr_r2=1-mean_sse/np.sum((observed_mean-observed_mean.mean())**2),
            mean_acr_pearson_r=float(np.corrcoef(observed_mean, predicted_mean)[0, 1]))
        baseline_key = model.rsplit('_', 1)[0]+'_baseline'
        summary['mse_reduction_vs_matched_baseline'] = 1-summary['mse']/(totals[baseline_key]['sse']/totals[baseline_key]['n'])
        metrics.append(summary)
    metric_frame = pd.DataFrame(metrics)
    metric_frame.to_csv(out/'metrics.csv', index=False)
    pd.DataFrame(inner_scores).to_csv(out/'inner_scores.csv', index=False)
    pd.DataFrame(coefficients).to_csv(out/'coefficients.csv', index=False)
    pd.DataFrame(genotype_contributions).to_csv(out/'genotype_contributions.csv', index=False)
    pd.DataFrame([dict(model=model, genotype=g, mse=float(sse[j]/len(y)))
        for model, sse in genotype_sse.items() for j, g in enumerate(data['genotypes'])]).to_csv(out/'genotype_results.csv', index=False)
    records = []
    for chromosome in sorted(means.chromosome.unique()):
        f = means.loc[means.chromosome.eq(chromosome)]
        for model in totals:
            records.append(dict(chromosome=chromosome, model=model, n_acrs=len(f),
                mse=float(f[model+'_mse'].mean()), rmse=float(np.sqrt(f[model+'_mse'].mean()))))
    pd.DataFrame(records).to_csv(out/'chromosome_results.csv', index=False)
    return metric_frame
