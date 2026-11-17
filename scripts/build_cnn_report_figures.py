#!/usr/bin/env python3
"""Export two static CNN report figures from saved evaluations; never fit a model."""

from pathlib import Path
import hashlib
import json

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/figures'
SOURCES = [
    'results/sampled_cnn_benchmark/metrics.csv',
    'results/sampled_cnn_benchmark/completion.json',
    'results/sampled_linear_benchmark/metrics.csv',
    'results/cnn_saved_prediction_comparison/metrics.csv',
    'results/cnn_saved_prediction_comparison/fold_metrics.csv',
    'results/cnn_saved_prediction_comparison/audit.json',
]
COLORS = dict(baseline='#626262', ridge='#21785a', current='#2867a0', earlier='#8855a3')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cnn = pd.read_csv(ROOT / SOURCES[0])
    complete = json.loads((ROOT / SOURCES[1]).read_text())
    ridge = pd.read_csv(ROOT / SOURCES[2])
    comparison = pd.read_csv(ROOT / SOURCES[3])
    folds = pd.read_csv(ROOT / SOURCES[4])
    audit = json.loads((ROOT / SOURCES[5]).read_text())
    assert complete['complete'] and complete['completed_inner_fits'] == 120
    assert complete['completed_outer_refits'] == 20
    assert audit['fold_manifests_identical']
    assert audit['all_evaluation_ACRs_absent_from_corresponding_earlier_training_fold']
    assert audit['models_fitted'] == 0 and not audit['holdout_or_DTS_read']
    for frame in [cnn, ridge, comparison]:
        assert not frame.duplicated(['model', 'subgroup']).any()
        part = frame.loc[frame.subgroup.eq('all')]
        assert set(part.n_pairs) == {495900} and set(part.n_ACRs) == {9666}

    c = cnn.loc[cnn.subgroup.eq('all')].set_index('model')
    r = ridge.loc[ridge.subgroup.eq('all')].set_index('model')
    previous = comparison.loc[comparison.subgroup.eq('all')].set_index('model')
    fields = ['pair_rmse', 'ACR_mean_rmse', 'ACR_mean_r2', 'within_ACR_r2']
    for name, row in c.iterrows():
        np.testing.assert_allclose(row[fields].to_numpy(dtype=float),
                                   previous.loc['current_' + name, fields].to_numpy(dtype=float),
                                   rtol=0, atol=1e-10)
    for name in ['baseline', 'B73_core_TF', 'personalized_core_TF']:
        np.testing.assert_allclose(r.loc[name, fields].to_numpy(dtype=float),
                                   previous.loc['current_ridge_' + name, fields].to_numpy(dtype=float),
                                   rtol=0, atol=1e-10)
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10.5,
        'axes.titlesize': 11.5, 'axes.labelsize': 10.5,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'axes.spines.top': False, 'axes.spines.right': False,
    })

    # Figure 1: one training/evaluation cohort for all seven rows.
    rows = [
        ('Covariates only', r.loc['baseline'], 'baseline', 's'),
        ('TF ridge: B73 core', r.loc['B73_core_TF'], 'ridge', 'o'),
        ('TF ridge: personalized core', r.loc['personalized_core_TF'], 'ridge', 'D'),
        ('CNN: B73 core', c.loc['CNN_B73_core'], 'current', 'o'),
        ('CNN: personalized core', c.loc['CNN_personalized_core'], 'current', 'D'),
        ('CNN: B73 full 2,000 bp', c.loc['CNN_B73_context'], 'current', 'o'),
        ('CNN: personalized full 2,000 bp', c.loc['CNN_personalized_context'], 'current', 'D'),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.9), layout='constrained', sharey=True)
    for j, (ax, field, title, limits) in enumerate(zip(
            axes, ['pair_rmse', 'ACR_mean_rmse'],
            ['A  Individual ACR–genotype observations', 'B  Average accessibility per ACR'],
            [(7.50, 7.84), (4.43, 5.04)])):
        for y, (label, row, family, marker) in enumerate(rows):
            value = row[field]
            ax.scatter(value, y, c=COLORS[family], marker=marker, s=48, zorder=3)
            ax.annotate(f'{value:.4f}', (value, y), xytext=(7, 0),
                        textcoords='offset points', va='center', fontsize=9.5,
                        bbox=dict(facecolor='white', edgecolor='none', pad=.5))
        ax.axvline(r.loc['baseline', field], ls='--', color=COLORS['baseline'], lw=1, zorder=1)
        for y in [.5, 2.5]:
            ax.axhline(y, color='#dddddd', lw=.7, zorder=0)
        ax.set(xlabel='Out-of-fold RMSE (CPM; lower is better)', title=title,
               xlim=limits, yticks=np.arange(len(rows)), ylim=(len(rows)-.4, -.6))
        ax.grid(axis='x', alpha=.18)
        ax.ticklabel_format(axis='x', useOffset=False)
        if j == 0:
            ax.set_yticklabels([row[0] for row in rows])
        else:
            ax.tick_params(axis='y', left=False, labelleft=False)
    fig.suptitle('Matched CNN comparison: 9,666 ACRs and 495,900 observations', fontsize=13)
    for suffix in ['pdf', 'png']:
        fig.savefig(OUT / f'cnn_matched_comparison.{suffix}', dpi=220)
    plt.close(fig)

    # Figure 2: fixed evaluation observations, different completed training procedures.
    rows = [
        ('Current TF ridge: personalized core', 'current_ridge_personalized_core_TF', 'ridge'),
        ('Earlier CNN: B73 core', 'earlier_CNN_acr_only', 'earlier'),
        ('Earlier CNN: B73 full 2,048 bp', 'earlier_CNN_context_2048', 'earlier'),
        ('Current CNN: B73 core', 'current_CNN_B73_core', 'current'),
        ('Current CNN: B73 full 2,000 bp', 'current_CNN_B73_context', 'current'),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), layout='constrained',
                             gridspec_kw={'width_ratios': [1.1, 1]})
    ax = axes[0]
    for y, (label, model, family) in enumerate(rows):
        value = 100 * previous.loc[model, 'ACR_mean_r2']
        ax.barh(y, value, height=.52, color=COLORS[family])
        ax.text(value+.55, y, f'{value:.1f}%', va='center', fontsize=10)
    ax.set(yticks=np.arange(len(rows)), yticklabels=[row[0] for row in rows],
           ylim=(len(rows)-.35, -.65), xlim=(0, 36.5),
           xlabel='ACR-mean $R^2$ (%; higher is better)',
           title='A  Predicting average accessibility per ACR')
    ax.grid(axis='x', alpha=.18)
    ax.set_axisbelow(True)

    ax = axes[1]
    lines = [
        ('Current CNN: B73 full', 'current_CNN_B73_context', 'current', 's', '-'),
        ('Current TF ridge: personalized core', 'current_ridge_personalized_core_TF', 'ridge', '^', '--'),
        ('Earlier CNN: B73 full', 'earlier_CNN_context_2048', 'earlier', 'o', '-'),
    ]
    for label, model, family, marker, style in lines:
        frame = folds.loc[folds.model.eq(model)].sort_values('outer_fold')
        np.testing.assert_array_equal(frame.outer_fold, np.arange(5))
        assert set(frame.n_pairs) != {0} and frame.n_pairs.sum() == 495900
        ax.plot(frame.outer_fold+1, frame.pair_rmse, label=label,
                color=COLORS[family], marker=marker, ls=style, lw=1.4, markersize=5)
    ax.set(xticks=np.arange(1, 6), xlabel='Outer held-out genomic-block fold',
           ylabel='ACR–genotype RMSE (CPM; lower is better)', ylim=(6.75, 8.95),
           title='B  Prediction error in each outer fold')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.19), fontsize=8.5,
              frameon=False, handlelength=2.2)
    ax.grid(axis='y', alpha=.18)
    fig.suptitle('Earlier CNNs rescored on the current evaluation observations', fontsize=13)
    fig.supxlabel('Same evaluation observations; different training cohorts, architectures and training procedures',
                  fontsize=10)
    for suffix in ['pdf', 'png']:
        fig.savefig(OUT / f'cnn_earlier_comparison.{suffix}', dpi=220)
    plt.close(fig)

    receipt = {
        'purpose': 'Report figures from existing out-of-fold predictions; no fitting or model selection',
        'n_ACRs': 9666, 'n_ACR_genotype_observations': 495900,
        'figure_1': 'Matched training cohort and evaluation observations',
        'figure_2': 'Matched evaluation observations only; earlier training cohort is larger',
        'inputs_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCES},
        'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (OUT / 'cnn_figure_provenance.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(f'Exported two CNN figures in PDF and PNG to {OUT}')


if __name__ == '__main__':
    main()
