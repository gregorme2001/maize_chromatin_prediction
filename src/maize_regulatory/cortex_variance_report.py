"""Self-contained report of observed cortex detection variation, without inference."""

import base64
import html
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .cortex_variance import VARIANCE_COLUMNS


LABELS = ['Among nuclei in the\nsame genotype/library',
          'Among library means\nwithin genotype',
          'Among pooled\ngenotype means']
COLORS = ['#427aa1', '#e0913b', '#377d57']


def overview(frame, stats, path):
    main = frame.loc[frame.scope.eq('all_development')]
    selected = stats.loc[stats.scope.eq('all_development')].set_index('quantity').loc[VARIANCE_COLUMNS]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    heights = selected.rms_sd_percentage_points.to_numpy()
    axes[0].bar(np.arange(3), heights, color=COLORS)
    axes[0].set(xticks=np.arange(3), xticklabels=LABELS,
                ylabel='100 × √(mean ACR variance), percentage points',
                title='Three observed levels of variation\n(all 108,843 ACRs; equal ACR weight)')
    axes[0].tick_params(axis='x', labelsize=8)
    for j, value in enumerate(heights):
        axes[0].text(j, value, f'{value:.3g}', ha='center', va='bottom', fontsize=9)
    axes[0].set_ylim(0, max(heights)*1.16)
    for name, label, color in zip(VARIANCE_COLUMNS, LABELS, COLORS):
        x = np.sort(100*np.sqrt(main[name].to_numpy()))
        axes[1].plot(x, np.arange(1, len(x)+1)/len(x), label=label.replace('\n', ' '), color=color)
    axes[1].set(xlabel='SD at one ACR (percentage points)', ylabel='Fraction of ACRs at or below x',
                title='Distribution across ACRs\n(all zero variances included)')
    axes[1].legend(fontsize=7, loc='lower right')
    x = 100*np.sqrt(main.between_libraries_within_genotype.to_numpy())
    y = 100*np.sqrt(main.between_genotype_pooled_means.to_numpy())
    axes[2].hexbin(x, y, gridsize=55, bins='log', mincnt=1, cmap='Blues')
    limit = max(x.max(), y.max())*1.03
    axes[2].plot([0, limit], [0, limit], '--', color='black', lw=1, label='Equal observed SD')
    axes[2].set(xlabel='Between-library SD within genotype (pp)',
                ylabel='Between-genotype pooled-mean SD (pp)',
                title='Each location represents ACRs\n(darker = more ACRs)',
                xlim=(0, limit), ylim=(0, limit))
    axes[2].legend(fontsize=7)
    for ax in axes:
        ax.grid(alpha=.15)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def examples(points, genotype_order, path):
    ids = points.ACR.unique()
    fig, axes = plt.subplots(len(ids), 1, figsize=(17, 10), constrained_layout=True)
    positions = {g: j for j, g in enumerate(genotype_order)}
    for ax, acr in zip(np.atleast_1d(axes), ids):
        local = points.loc[points.ACR.eq(acr)]
        for genotype, rows in local.groupby('genotype', sort=False):
            at = positions[genotype]
            offset = np.linspace(-.24, .24, len(rows)) if len(rows) > 1 else [0]
            ax.scatter(at+np.asarray(offset), 100*rows.detection_fraction,
                       s=8+3*np.sqrt(rows.n_nuclei), color=COLORS[1], alpha=.55)
            ax.scatter(at, 100*rows.pooled_genotype_fraction.iloc[0], s=13, color='black', zorder=3)
        ax.scatter([], [], s=25, color=COLORS[1], label='One library mean; larger dots = more nuclei')
        ax.scatter([], [], s=13, color='black', label='Genotype mean after pooling nuclei across libraries')
        ticks = np.arange(0, len(genotype_order), 5)
        ax.set(xticks=ticks, xticklabels=[genotype_order[j] for j in ticks],
               xlabel='Genotype in frozen development order (labels shown every five)',
               ylabel='Nuclei detecting this ACR (%)', title=acr, ylim=(0, None))
        ax.tick_params(axis='x', labelrotation=50, labelsize=7)
        ax.grid(axis='y', alpha=.15)
        ax.legend(fontsize=8, loc='upper right')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def write_report(output):
    summary = json.loads((output/'summary.json').read_text())
    frame = pd.read_parquet(output/'acr_variances.parquet')
    stats = pd.read_csv(output/'summary_statistics.csv')
    overview(frame, stats, output/'overview.png')
    example_data = pd.read_csv(output/'example_measurements.csv', dtype={'genotype': str})
    examples(example_data, summary['genotype_order'], output/'examples.png')
    table = stats[['scope', 'quantity', 'mean_variance', 'median_variance',
                   'rms_sd_percentage_points', 'median_sd_percentage_points']].copy()
    table.columns = ['Genotype view', 'Quantity', 'Mean variance (fraction²)',
                     'Median variance (fraction²)', '100 × √mean variance (pp)', 'Median ACR SD (pp)']
    finding = (
        'This report describes three different levels of observed variation in binary cortex '
        'ACR detections. It does not estimate genetic variance or subtract sampling noise. '
        'Each development genotype has equal weight. Within a genotype, each nucleus has equal '
        'weight, so a library containing more nuclei receives more weight. ACRs have equal '
        'weight when summarizing the genome. All variances use population denominators, '
        'describing these observed cells rather than applying N−1 corrections.')
    support = (
        f'{summary["n_acrs"]:,} ACRs across chromosomes 1–10; 138 development genotypes; '
        f'{summary["scope_support"][0]["n_nuclei"]:,} cortex nuclei in '
        f'{summary["scope_support"][0]["n_library_groups"]} genotype/library groups. '
        f'{summary["single_nucleus_groups"]} groups contain one nucleus and '
        f'{summary["single_library_genotypes"]} genotypes have only one cortex library. '
        'The main view retains every observed development cortex group, including H95. '
        'There is no minimum-cell, SNP, MAF or accessibility screen. The second view uses '
        'only the 131 genotypes with at least two observed cortex libraries; it is an '
        'availability view, not a new permanent exclusion. CML341 and the frozen 30 holdouts '
        'are excluded by the existing development policy. No DTS is read.')
    definitions = (
        'Fix ACR i. Let g index genotype, l library and c a nucleus within that genotype/library. '
        'D_iglc is 1 if the nucleus detects ACR i, otherwise 0. N_gl is its group’s nucleus '
        'count, Y_igl = Σ_c D_iglc is the positive-nucleus count, and p_igl = Y_igl/N_gl '
        'is the library detection fraction. N_g = Σ_l N_gl. The pooled genotype fraction '
        'p_ig = Σ_l Y_igl/N_g weights library means by their nucleus counts. G is the number '
        'of genotypes in the selected view; μ_i = (1/G)Σ_g p_ig gives genotypes equal weight. '
        'All these quantities are on the 0/1 detection scale. No CPM denominator or other '
        'cell state enters this analysis.')
    formulas = '''<p><strong>1. Among nuclei in the same genotype/library.</strong>
For a binary measurement, the observed within-group variance is p_igl(1−p_igl).
Average over libraries using their nucleus counts, then equally over genotypes:</p>
<pre>V_within,i = (1/G) Σ_g Σ_l (N_gl/N_g) p_igl(1 − p_igl)</pre>
<p><strong>2. Among library means, keeping genotype fixed.</strong>
Measure each library fraction’s squared deviation from its genotype’s pooled fraction,
weight libraries by nucleus counts, then average equally over genotypes:</p>
<pre>V_library,i = (1/G) Σ_g Σ_l (N_gl/N_g) (p_igl − p_ig)²</pre>
<p><strong>3. Among pooled genotype means.</strong>
Pool nuclei across libraries within each genotype, then measure variance across those genotype fractions:</p>
<pre>V_genotype,i = (1/G) Σ_g (p_ig − μ_i)²</pre>
<p>These weights give an exact empirical identity:</p>
<pre>V_within,i + V_library,i + V_genotype,i = μ_i(1 − μ_i)</pre>
<p>This is a decomposition of the observed distribution obtained by choosing a genotype
uniformly, then a nucleus uniformly within that genotype. It requires no independent-error model.
It is not the variance obtained by giving every nucleus in the whole dataset equal weight,
because genotypes have different cell counts.</p>'''
    interpretation = (
        'The first quantity measures heterogeneity among individual 0/1 detections. The other '
        'two measure variation among averages of many nuclei. Averaging naturally reduces '
        'variance, so a larger first quantity does not show that library noise overwhelms '
        'genetic signal. Between-library variation includes finite-cell sampling, preparation '
        'and possible plant differences. Between-genotype variation also contains finite-cell '
        'sampling and library-composition effects. Neither is a purified biological variance '
        'component. A one-nucleus group has zero observed within-group variance; this does '
        'not mean its true detection probability is known. A single-library genotype has no '
        'observed between-library comparison and contributes zero to that empirical term; '
        'the multiple-library view makes this support limitation explicit. Large libraries '
        'have more influence within genotype, by design; this is not an equal-library average.')
    plot_caption = (
        'Left: square root of the mean per-ACR variance, multiplied by 100 to express the '
        'spread in percentage points. It is not the mean of per-ACR SDs. Middle: cumulative '
        'distribution of per-ACR SDs; zeros are retained. Right: observed between-library '
        'versus between-genotype SD; dashed line marks equality, not a statistical threshold. '
        'Orange and green describe noisy means at different averaging levels, not competing '
        'estimates of a common latent variance.')
    example_caption = (
        'Three regions selected by first/middle/last coordinate order in the previously frozen '
        'chromosome-10 pilot panel, not by the current variances. Each orange dot is a '
        'genotype/library detection fraction. Dot area increases with nucleus count. Black '
        'dots pool all libraries within genotype. Genotypes retain source order; no ranking '
        'by accessibility is used. Raw values and denominators are in example_measurements.csv.')
    validation = (
        f'The three variances reconstruct total empirical binary variance with maximum absolute '
        f'error {summary["max_absolute_decomposition_error"]:.3g}. Genome-wide group event totals '
        'match the independent earlier sparse audit. '
        f'{summary["chr10_library_count_values_verified"]:,} chromosome-10 group/ACR counts '
        'match its old cache exactly. Cell IDs, genotype/library membership, nucleus counts '
        'and ACR order were checked explicitly before calculation. No original report, '
        'response, sample exclusion or predictive CV protocol was changed.')
    paragraphs = [('What is summarized', finding), ('Coverage and weighting', support),
                  ('Definitions', definitions), ('Interpretation', interpretation),
                  ('Validation', validation)]
    text = '# Cortex accessibility: three descriptive variance levels\n\n'
    for title, paragraph in paragraphs:
        text += f'## {title}\n\n{paragraph}\n\n'
    text += table.to_string(index=False)+'\n\nSee report.html for formulas and figures.\n'
    (output/'report.md').write_text(text)
    contents = ['<!doctype html><html><head><meta charset="utf-8">'
        '<title>Cortex accessibility: descriptive variances</title>'
        '<style>body{font:17px/1.6 system-ui;max-width:1250px;margin:36px auto;padding:0 24px;'
        'color:#203040}h1,h2{line-height:1.25}img{max-width:100%}'
        'table{border-collapse:collapse;font-size:13px}td,th{padding:9px;border:1px solid #ccd5dd;'
        'text-align:left}pre{white-space:pre-wrap;background:#eef3f6;padding:16px}'
        '.caption{font-size:15px;color:#485867}.scroll{overflow:auto}</style></head><body>'
        '<h1>Cortex accessibility: three descriptive variance levels</h1>'
        '<p><strong>Observed 0/1 detections. No statistical tests or noise correction.</strong></p>']
    for title, paragraph in paragraphs[:3]:
        contents.append(f'<h2>{title}</h2><p>{html.escape(paragraph)}</p>')
    contents += [formulas, '<h2>Results</h2><p>“Fraction²” means variance on the 0–1 scale. '
                'One percentage point is 0.01 on that scale. Means and medians summarize ACRs, '
                'not pooled observations across different ACRs.</p>',
                '<div class="scroll">'+table.to_html(index=False, float_format=lambda x: f'{x:.6g}')+'</div>']
    for filename, caption in [('overview.png', plot_caption), ('examples.png', example_caption)]:
        encoded = base64.b64encode((output/filename).read_bytes()).decode()
        contents.append(f'<img src="data:image/png;base64,{encoded}"><p class="caption">'
                        f'{html.escape(caption)}</p>')
    for title, paragraph in paragraphs[3:]:
        contents.append(f'<h2>{title}</h2><p>{html.escape(paragraph)}</p>')
    contents.append('</body></html>')
    (output/'report.html').write_text('\n'.join(contents))
