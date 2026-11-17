#!/usr/bin/env python3
"""Export report figures/tables from frozen results; no scoring or model fitting."""

from pathlib import Path
import hashlib
import json
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/figures'
SOURCES = [
    'results/sampled_linear_benchmark/metrics.csv',
    'results/sampled_linear_benchmark/personalization_comparisons.csv',
    'results/sampled_sequence_features/selected_acrs.parquet',
    'results/cortex_cpm_variance/overview.png',
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / SOURCES[3], OUT / 'cortex_cpm_variance.png')
    metrics = pd.read_csv(ROOT / SOURCES[0])
    main_results = metrics.loc[metrics.subgroup.eq('all')].set_index('model')
    sample = pd.read_parquet(ROOT / SOURCES[2])
    assert len(sample) == 9666 and len(main_results) == 13
    assert set(main_results.n_pairs) == {495900}
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42})

    # Two readable metric panels; tiny within-ACR effects are given numerically
    # in the manuscript/table instead of magnified into a third panel.
    configurations = [('core', 'TF'), ('core', 'word'), ('core', 'both'),
                      ('context', 'TF'), ('context', 'word'), ('context', 'both')]
    labels = [f'{"Core" if extent == "core" else "Full 2,000 bp"}: '
              f'{dict(TF="TFs", word="five-mers", both="both")[feature]}'
              for extent, feature in configurations]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), layout='constrained')
    for ax, metric, title in zip(axes, ['pair_rmse', 'ACR_mean_rmse'],
                               ['Individual ACR/genotype pairs', 'Mean accessibility per ACR']):
        for source, offset, color in [('B73', -.12, '#366e9b'),
                                      ('personalized', .12, '#ce7939')]:
            values = [main_results.loc[f'{source}_{extent}_{feature}', metric]
                      for extent, feature in configurations]
            ax.scatter(values, np.arange(6) + offset, color=color, s=38,
                       label='B73' if source == 'B73' else 'SNP-substituted', zorder=3)
        baseline = main_results.loc['baseline', metric]
        ax.axvline(baseline, ls='--', color='black', lw=1, label='Nuisance baseline')
        ax.set(yticks=np.arange(6), yticklabels=labels if ax is axes[0] else [],
               xlabel='Out-of-fold RMSE (CPM; lower is better)', title=title)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=.2)
        ax.ticklabel_format(axis='x', useOffset=False)
    axes[1].legend(loc='lower right', fontsize=8)
    fig.savefig(OUT / 'linear_comparison.pdf')
    fig.savefig(OUT / 'linear_comparison.png', dpi=180)
    plt.close(fig)

    comparisons = pd.read_csv(ROOT / SOURCES[1])
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), layout='constrained', sharey=True)
    for ax, group, title in zip(axes,
            ['all', 'core_variation_rich', 'context_variation_rich'],
            ['All matched pairs', 'Core-variation-rich', 'Context-variation-rich']):
        part = comparisons.loc[comparisons.subgroup.eq(group)]
        assert len(part) == 6
        names = [f'{"Core" if row.extent == "core" else "Full"}\n'
                 f'{dict(TF="TF", word="5mer", both="Both")[row.features]}'
                 for row in part.itertuples()]
        ax.bar(np.arange(6), part.pair_MSE_reduction_percent, color='#43846f')
        ax.axhline(0, color='black', lw=.8)
        ax.set(xticks=np.arange(6), xticklabels=names, title=title)
        ax.tick_params(axis='x', labelsize=8)
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Pair MSE reduction from personalization (%)')
    fig.savefig(OUT / 'personalization_comparison.pdf')
    fig.savefig(OUT / 'personalization_comparison.png', dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.4), layout='constrained')
    ax.set(xlim=(0, 10), ylim=(0, 5.4)); ax.axis('off')

    def box(x, y, width, height, text, color='#edf3f7'):
        ax.add_patch(FancyBboxPatch((x, y), width, height,
                     boxstyle='round,pad=0.08', facecolor=color, edgecolor='#6a7c88'))
        ax.text(x+width/2, y+height/2, text, ha='center', va='center', fontsize=10)

    def arrow(start, end):
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', color='#465763', lw=1.4))

    box(.2, 4.1, 4.3, 1.0, '169 original matched genotypes\nFreeze 30 holdouts; exclude CML341 from development\n138 development genotypes')
    box(5.2, 4.1, 4.3, 1.0, '108,843 measured ACRs\nB73 reference + listed biallelic SNP calls\nCentered 2,000-bp sequence windows')
    box(.2, 2.55, 4.3, .85, 'At least 50 pooled cortex nuclei per genotype\nH95 (49 nuclei) omitted from prediction\n137 eligible development genotypes')
    box(5.2, 2.4, 4.3, 1.1, 'Complete homozygous calls throughout window\nUnambiguous DNA; no cross-block clipping\nAt least 20 usable genotypes per ACR\n19,201 eligible ACRs')
    arrow((2.35, 4.0), (2.35, 3.5))
    arrow((7.35, 4.0), (7.35, 3.6))
    arrow((4.6, 2.98), (5.08, 2.98))
    box(1.0, .5, 8.0, 1.05,
        'Uniformly sample up to 25 ACRs per occupied 5-Mb block (seed 42)\n'
        'Keep all available genotype rows at each selected ACR\n'
        '9,666 ACRs | 495,900 pairs | 432 blocks | 10 chromosomes', '#e7f1e8')
    arrow((7.35, 2.3), (6.5, 1.65))
    ax.text(5, .08, 'One common pair mask for all B73/personalized and core/full-window models',
            ha='center', fontsize=9)
    fig.savefig(OUT / 'sampling_funnel.pdf')
    fig.savefig(OUT / 'sampling_funnel.png', dpi=180)
    plt.close(fig)

    rows = [r'\begin{tabular}{lllrrrr}', r'\toprule',
            r'Sequence & Extent & Features & Pair RMSE & Mean RMSE & Mean $R^2$ & Within $R^2$\\',
            r'\midrule']
    for name, row in main_results.iterrows():
        if name == 'baseline':
            fields = ['Baseline', '--', '--']
        else:
            source, extent, feature = name.split('_')
            fields = ['B73' if source == 'B73' else 'Personalized',
                      'Core' if extent == 'core' else 'Full',
                      dict(TF='TFs', word='Words', both='Both')[feature]]
        fields += [f'{row.pair_rmse:.4f}', f'{row.ACR_mean_rmse:.4f}',
                   f'{row.ACR_mean_r2:.4f}', f'{row.within_ACR_r2:.5f}']
        if name == 'personalized_core_TF':
            fields = [r'\textbf{' + field + '}' for field in fields]
        rows.append(' & '.join(fields) + r'\\')
    rows += [r'\bottomrule', r'\end{tabular}']
    (OUT / 'linear_results_table.tex').write_text('\n'.join(rows) + '\n')
    receipt = {'purpose': 'Presentation of existing frozen results only; no model fitting',
               'sources_sha256': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                                  for p in SOURCES},
               'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'provenance.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(f'Wrote three report figures and the 13-model table to {OUT}')


if __name__ == '__main__':
    main()
