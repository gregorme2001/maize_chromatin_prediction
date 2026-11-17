"""Labeled offline report for the approved 13 matched linear models."""

import base64
import html
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def label(name):
    if name == 'baseline':
        return 'Nuisance baseline'
    source, extent, feature = name.split('_')
    return f'{source} · {"ACR" if extent == "core" else "2,000 bp"} · '+{
        'TF': 'TFs', 'word': 'five-mers', 'both': 'both'}[feature]


def write_report(out):
    summary = json.loads((out/'summary.json').read_text())
    config = json.loads((out/'run_config.json').read_text())
    metrics = pd.read_csv(out/'metrics.csv')
    folds = pd.read_csv(out/'fold_metrics.csv')
    penalties = pd.read_csv(out/'selected_penalties.csv')
    primary = metrics.loc[metrics.subgroup.eq('all')].set_index('model').loc[summary['models']]
    labels = [label(name) for name in primary.index]
    colors = ['#777777']+['#477da8' if name.startswith('B73') else '#c47738' for name in primary.index[1:]]
    fig, axes = plt.subplots(1, 3, figsize=(17, 8), constrained_layout=True)
    for ax, column, title, xlabel in [
        (axes[0], 'pair_rmse', 'All matched observations', 'Out-of-fold pair RMSE (CPM)'),
        (axes[1], 'ACR_mean_rmse', 'Mean accessibility per ACR', 'Out-of-fold ACR-mean RMSE (CPM)'),
        (axes[2], 'within_ACR_r2', 'Genotype differences within ACRs', 'Within-ACR R²')]:
        ax.barh(np.arange(len(primary)), primary[column], color=colors)
        ax.set(yticks=np.arange(len(primary)), yticklabels=labels if ax is axes[0] else ['']*len(primary),
               xlabel=xlabel, title=title)
        ax.invert_yaxis(); ax.axvline(primary.loc['baseline', column], color='black', ls='--', lw=1)
    fig.savefig(out/'model_comparison.png', dpi=145); plt.close(fig)
    comparison = []
    for group in summary['subgroups']:
        part = metrics.loc[metrics.subgroup.eq(group)].set_index('model')
        for extent in ['core', 'context']:
            for feature in ['TF', 'word', 'both']:
                reference = part.loc[f'B73_{extent}_{feature}']
                personal = part.loc[f'personalized_{extent}_{feature}']
                comparison.append(dict(subgroup=group, extent=extent, features=feature,
                    B73_pair_RMSE=reference.pair_rmse, personalized_pair_RMSE=personal.pair_rmse,
                    pair_MSE_reduction_percent=100*(1-personal.pair_mse/reference.pair_mse),
                    within_ACR_RMSE_change=personal.within_ACR_rmse-reference.within_ACR_rmse))
    comparison = pd.DataFrame(comparison)
    comparison.to_csv(out/'personalization_comparisons.csv', index=False)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, group in zip(axes, summary['subgroups']):
        part = comparison.loc[comparison.subgroup.eq(group)]
        labels0 = [f'{"ACR" if r.extent == "core" else "2 kb"}\n{r.features}' for r in part.itertuples()]
        ax.bar(np.arange(6), part.pair_MSE_reduction_percent, color='#4e8b79')
        ax.axhline(0, color='black', lw=1)
        ax.set(xticks=np.arange(6), xticklabels=labels0, title=group.replace('_', ' '),
               ylabel='Pair MSE reduction from personalization (%)')
    fig.savefig(out/'personalization.png', dpi=145); plt.close(fig)

    def table(frame):
        return '<div style="overflow-x:auto">'+frame.to_html(index=False, float_format=lambda x: f'{x:.6f}')+'</div>'

    def image(name, caption):
        data = base64.b64encode((out/name).read_bytes()).decode()
        return f'<figure><img src="data:image/png;base64,{data}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>'

    def metric_table(part):
        columns = ['model', 'n_pairs', 'n_ACRs', 'pair_rmse', 'pair_r2', 'ACR_mean_rmse',
                   'ACR_mean_r2', 'within_ACR_rmse', 'within_ACR_r2']
        result = part.reset_index()[columns].copy()
        result['model'] = result.model.map(label)
        result.columns = ['Model', 'Pairs', 'ACRs', 'Pair RMSE', 'Pair R²', 'ACR-mean RMSE',
                          'ACR-mean R²', 'Within-ACR RMSE', 'Within-ACR R²']
        return result

    parts = ['<!doctype html><html><meta charset="utf-8"><title>Matched sequence-ridge benchmark</title>',
        '<style>body{font:17px system-ui;max-width:1450px;margin:32px auto;padding:0 18px;line-height:1.55}'
        'td,th{padding:6px 10px;border-bottom:1px solid #ddd}table{border-collapse:collapse;font-size:14px}'
        'img{max-width:100%}figcaption{font-size:14px;color:#444}</style>',
        '<h1>Matched B73 versus personalized sequence: 13 linear models</h1>',
        f'<p>{summary["n_ACRs"]:,} frozen ACRs, {summary["n_pairs"]:,} ACR–genotype pairs, '
        f'{summary["n_genotypes"]} development genotypes. Every model uses identical observations '
        'and genomic folds. No CNN, DTS or permanent-holdout analysis.</p>',
        '<h2>Results on all matched pairs</h2>', table(metric_table(primary)),
        image('model_comparison.png', 'Predictions are out of fold at chromosome-specific 5-Mb block level. '
              'Dashed lines mark the matching nuisance baseline. Smaller RMSE is better; larger R² is better. '
              'Within-ACR R² can be negative when predicting genotype deviations is worse than predicting no deviation.'),
        '<h2>Does personalization help?</h2>',
        image('personalization.png', 'Positive bars mean lower out-of-fold squared error with SNP-substituted '
              'sequences than with B73, for the same extent/features/pairs. Subgroups use the same trained '
              'models; they are not separate fits or independent confirmations.'),
        table(comparison), '<h2>What is an observation and response?</h2>',
        '<p>i = original measured ACR; g = genotype; l = sequencing library. '
        'Y_igl counts cortex nuclei detecting ACR i. T_gl sums detections over all 108,843 ACRs '
        'in the same genotype/library cortex cells. C_ig = 1,000,000 × sum_l Y_igl / sum_l T_gl. '
        'Libraries are pooled within genotype before dividing. Genotypes are not pooled together. '
        'Every response remains original-core raw CPM, even when its sequence input is 2,000 bp. '
        'No log/rank response transform, clipping, or new sampling weights.</p>',
        '<h2>The fitted models</h2>',
        '<p>C_ig = b_igᵀ gamma + x_igᵀ beta + error_ig. All coefficients are globally shared. '
        'Unpenalized b_ig contains an intercept, four natural cubic B73-core-GC spline terms '
        '(fixed knots 0, .25, .5, .75, 1), original core length, nine chromosome indicators '
        'and ten existing ancestry PCs. There are no ACR-specific or library intercepts. '
        'B73 GC and length are fixed across all sequence models.</p>',
        '<p>x_ig contains TF scores, five-mer quasi-motif scores, or both, computed from B73 or '
        'SNP-substituted sequence. Extent is either the original core or the full 2,000-bp window. '
        'Full-window motifs have one coefficient regardless of where their matches occur; no '
        'core mask, flank-only predictors or separate core/flank coefficients enter these fits. '
        'Scores remain signed log2 mean likelihood ratios. The 927 TF columns and 1,024 conceptual '
        'five-mer columns use the same standardized-coefficient penalty; 512 exact reverse-complement '
        'word pairs are compressed with their algebraically equivalent half penalty.</p>',
        '<h2>Training and validation</h2>',
        '<p>Frozen five outer genomic-block folds and three inner folds. ACRs bring all their available '
        'genotype rows into the same fold. Training-only means/scales, nuisance adjustment and ridge fits. '
        'Each sequence model selects its own lambda from 1e-6 through 1e4, plus the exact nuisance-only '
        'candidate, using pooled inner validation SSE divided by pair count. Outer scores do not select '
        'lambda or a preferred fold. All outer predictions are combined. The same genotypes occur at '
        'different ACRs on both sides: this tests unseen regions, not unseen genotypes.</p>',
        '<p>Identical sequences are grouped only for exact computation. Every actual available pair '
        'contributes to the loss and ancestry cross-terms. Uneven genotype coverage is retained; '
        'the old balanced ACR-mean training shortcut is not used.</p>',
        '<h2>Metric definitions</h2><p>Pair RMSE averages squared prediction error over actual pairs, '
        'then takes its square root. ACR-mean RMSE first averages observed and predicted CPM over '
        'the same available genotypes within each ACR, then scores those means equally across ACRs. '
        'Within-ACR RMSE subtracts each ACR’s observed mean from its observations and its predicted '
        'mean from its predictions, then scores those genotype deviations. These are evaluation '
        'summaries of the same fits, not new models. All RMSEs are in CPM.</p>']
    for group in summary['subgroups'][1:]:
        part = metrics.loc[metrics.subgroup.eq(group)].set_index('model').loc[summary['models']]
        parts += ['<h2>'+group.replace('_', ' ').title()+'</h2>',
            '<p>At least two distinct '+('core' if group.startswith('core') else 'full-context')+
            ' sequences each carried by at least five available genotypes. This fixed sequence-only '
            'annotation was saved before fitting. No model is refitted for this subset.</p>', table(metric_table(part))]
    penalty_table = penalties[['model', 'outer_fold', 'selected_alpha']].copy()
    penalty_table['model'] = penalty_table.model.map(label)
    penalty_table['selected_alpha'] = penalty_table.selected_alpha.map(lambda v: 'baseline' if pd.isna(v) else f'{v:g}')
    parts += ['<h2>Selected penalties</h2>', table(penalty_table),
        '<h2>Limits</h2><p>CPM is compositional and measurements share genotypes and experimental '
        'material. No independent-row confidence intervals, coefficient p-values or FDR discoveries '
        'are claimed. SNP-substituted B73 is not a full personal genome; missing/heterozygous calls '
        'were excluded and indels/SVs/unlisted variants are absent. Call-complete sampling is selective. '
        'Prediction gains do not establish TF binding, activity, occupancy or causality. Results apply '
        'to the frozen spatial sample conditional on its call completeness, not every maize region.</p>',
        '<p>Saved predictions.npy and observations.parquet retain all 13 matched out-of-fold vectors. '
        'Per-fold coefficients, scales, exact selections and inner losses are retained, alongside '
        'source/code hashes and the independent raw-CPM reconstruction check.</p></html>']
    (out/'report.html').write_text('\n'.join(parts))
