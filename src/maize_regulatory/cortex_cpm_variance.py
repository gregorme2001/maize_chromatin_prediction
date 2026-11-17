"""Observed CPM variation among library and genotype means; no inference."""

import base64
import html
import json

import numpy as np
import pandas as pd


VARIANCES = ['weighted_library_deviations', 'variance_of_genotype_means']
LABELS = ['Library CPM deviations\nfrom pooled genotype CPM',
          'Variation among\npooled genotype CPMs']


def cpm_variances(counts, totals, genotype_indices):
    """Library-by-ACR counts, whole-genome T_gl, and explicit genotype indices.

    Genotypes have equal weight; libraries within genotype have T_gl weights.
    Descriptive population variances use denominator G, not G-1. No cell-level
    variance, sampling correction, random-effects model, or F statistic.
    """
    y = np.asarray(counts, dtype=float)
    t = np.asarray(totals, dtype=float)
    g = np.asarray(genotype_indices)
    if (y.ndim != 2 or len(y) == 0 or y.shape[1] == 0
            or t.shape != (len(y),) or g.shape != (len(y),)
            or not np.isfinite(y).all() or not np.isfinite(t).all()
            or np.any(t <= 0) or np.any(t != np.floor(t))
            or np.any(y < 0) or np.any(y != np.floor(y))
            or np.any(y > t[:, None])):
        raise ValueError('Require integer detections and positive whole-genome event totals')
    if (g.dtype.kind not in 'iu' or np.any(g < 0)
            or not np.array_equal(np.unique(g), np.arange(len(np.unique(g))))):
        raise ValueError('Genotype indices must be contiguous nonnegative integers')
    cpm = 1e6*y/t[:, None]
    means, deviations = [], []
    for genotype in np.unique(g):
        selected = g == genotype
        w = t[selected]/t[selected].sum()
        # Direct pooled counts define the center, not an arithmetic library mean.
        mean = 1e6*y[selected].sum(axis=0)/t[selected].sum()
        if not np.allclose(w @ cpm[selected], mean, rtol=1e-12, atol=1e-10):
            raise AssertionError('Weighted library CPM does not reconstruct pooled CPM')
        means.append(mean)
        deviations.append(w @ (cpm[selected]-mean)**2)
    means = np.asarray(means)
    center = means.mean(axis=0)
    within = np.mean(deviations, axis=0)
    between = np.mean((means-center)**2, axis=0)
    total = np.zeros(y.shape[1])
    for genotype in np.unique(g):
        selected = g == genotype
        w = t[selected]/t[selected].sum()
        total += w @ (cpm[selected]-center)**2 / len(means)
    if not np.allclose(within+between, total, rtol=1e-11, atol=1e-8):
        raise AssertionError('Library and genotype variances do not sum to total CPM variance')
    result = pd.DataFrame(dict(mean_genotype_cpm=center,
        weighted_library_deviations=within, variance_of_genotype_means=between,
        total_cpm_variance=total, decomposition_error=within+between-total))
    return result, means


def summarize(frame):
    rows = []
    for scope, part in frame.groupby('scope', sort=False):
        for quantity in VARIANCES:
            values = part[quantity].to_numpy()
            rows.append(dict(scope=scope, quantity=quantity, n_acrs=len(part),
                mean_variance_cpm2=float(values.mean()), median_variance_cpm2=float(np.median(values)),
                variance_q10=float(np.quantile(values, .1)), variance_q90=float(np.quantile(values, .9)),
                sqrt_mean_variance_cpm=float(np.sqrt(values.mean())),
                median_acr_sd_cpm=float(np.median(np.sqrt(values)))))
    return pd.DataFrame(rows)


def write_report(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    summary = json.loads((out/'summary.json').read_text())
    frame = pd.read_parquet(out/'acr_variances.parquet')
    stats = pd.read_csv(out/'summary_statistics.csv')
    main = frame.loc[frame.scope.eq('all_development')]
    rows = stats.loc[stats.scope.eq('all_development')].set_index('quantity').loc[VARIANCES]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    heights = rows.sqrt_mean_variance_cpm.to_numpy()
    axes[0].bar([0, 1], heights, color=['#d88b32', '#377d57'])
    axes[0].set(xticks=[0, 1], xticklabels=LABELS,
                ylabel='Square root of mean per-ACR variance (CPM)',
                title='Average observed variation across ACRs')
    axes[0].set_ylim(0, max(heights)*1.18)
    for j, value in enumerate(heights):
        axes[0].text(j, value, f'{value:.3f} CPM', ha='center', va='bottom')
    x = np.sqrt(main.weighted_library_deviations.to_numpy())
    y = np.sqrt(main.variance_of_genotype_means.to_numpy())
    positive = (x > 0) & (y > 0)
    density = axes[1].hexbin(x[positive], y[positive], xscale='log', yscale='log',
                            gridsize=55, bins='log', mincnt=1, cmap='Blues')
    fig.colorbar(density, ax=axes[1], label='Number of ACRs per hexagon')
    lo, hi = min(x[positive].min(), y[positive].min()), max(x.max(), y.max())
    axes[1].plot([lo, hi], [lo, hi], '--', color='black', label='Equal observed spread')
    axes[1].set(xlabel='√weighted library deviations (CPM; log axis)',
                ylabel='SD across pooled genotype CPMs (CPM; log axis)',
                title='ACR-by-ACR comparison')
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.15)
    fig.savefig(out/'overview.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    table = stats[['scope', 'quantity', 'mean_variance_cpm2', 'median_variance_cpm2',
                   'sqrt_mean_variance_cpm', 'median_acr_sd_cpm']].copy()
    table.columns = ['Genotype view', 'Quantity', 'Mean variance (CPM²)', 'Median variance (CPM²)',
                     '√mean variance (CPM)', 'Median ACR SD (CPM)']
    definitions = (
        'Fix ACR i. g indexes genotype and l library. All cells are cortex. Y_igl is the '
        'number of nuclei detecting ACR i. T_gl is the total number of positive nucleus–ACR '
        'detections across ALL 108,843 ACRs in the SAME genotype/library cortex cells. '
        'R_igl = 1,000,000 × Y_igl/T_gl is a library CPM. C_ig = 1,000,000 × Σ_lY_igl/Σ_lT_gl '
        'is the genotype CPM after pooling libraries. Library weights '
        'w_gl = T_gl/Σ_l′T_gl′, so C_ig = Σ_l w_gl R_igl. Weights use total detections, '
        'not nucleus counts. G genotypes have equal weight; Cbar_i=(1/G)Σ_g C_ig.')
    formulas = '''<p><strong>Library deviations within genotype:</strong> weight squared library
deviations from each genotype’s pooled CPM, then average equally over genotypes.</p>
<pre>V_library,i = (1/G) Σ_g Σ_l w_gl (R_igl − C_ig)²</pre>
<p><strong>Variance across genotype means:</strong> pool each genotype’s libraries first,
then calculate the population variance of those genotype CPMs.</p>
<pre>V_genotype,i = (1/G) Σ_g (C_ig − Cbar_i)²</pre>
<p>Both are in CPM². There is no G−1 correction. Their sum equals the total observed
CPM variance under equal-genotype and within-genotype total-detection weights.
No cell-level variance is calculated.</p>'''
    support = summary['scope_support'][0]
    coverage = (
        f'{summary["n_acrs"]:,} ACRs on all ten chromosomes; {support["n_genotypes"]} '
        f'development genotypes, {support["n_library_groups"]} genotype/library groups, '
        f'and {support["n_nuclei"]:,} cortex nuclei. No minimum-cell, SNP, MAF or accessibility '
        'filter is applied. H95 is retained in this descriptive analysis. CML341 and the '
        'fixed 30 holdouts remain excluded by existing policy. Seven genotypes have only one '
        'observed cortex library and therefore contribute zero observed library deviation. '
        'The separate multiple_libraries view uses the 131 genotypes with at least two '
        'libraries for BOTH quantities; this is not a new permanent exclusion.')
    interpretation = (
        'These are observed variations, not purified genetic or technical components. '
        'Library deviations include finite-cell sampling, preparation effects and possible '
        'plant differences. Genotype means also retain sampling and library-composition '
        'effects. Pooling libraries itself reduces variation, so a larger library quantity '
        'does not establish that genetic variance is absent. Very small groups can give '
        'extreme CPMs; event weighting reduces their influence within genotype but does not '
        'eliminate it. Every ACR has equal weight in the summary; high-variance ACRs can '
        'dominate the mean, so medians and quantiles are saved too. No regression adjustment, '
        'statistical test, noise subtraction, confidence interval, heritability or predictive '
        'performance claim is made. No DTS, holdout values or previous prediction folds are used.')
    verification = (
        'The library-count cache and source identities were verified before use. All '
        f'{summary["pooled_count_values_verified"]:,} genome-wide genotype/ACR pooled counts '
        'match the earlier independent genotype cache. Genome-wide totals match the earlier '
        'audit. Event-weighted means reproduce ratio-of-pooled-count CPMs. Weighted library '
        'deviations plus genotype variance reconstruct total observed variance to floating-point '
        'precision. Source and code hashes are retained in summary.json.')
    lead = ('Two descriptive CPM quantities, calculated separately at each ACR and then '
            'summarized equally across ACRs. No F statistic or significance test.')
    caption = (
        'Left: square root of mean variance across ACRs, not mean per-ACR SD. Right: each '
        'hexagon contains ACRs; both axes are logarithmic, and the dashed line marks equal '
        f'observed spread. {int((~positive).sum())} ACRs with a zero on either axis are omitted '
        'from the log plot but retained in summary statistics. The plot uses all 138 genotypes.')
    text = '# Cortex CPM: genotype means versus library deviations\n\n'+lead+'\n\n'
    for label, paragraph in [('Definitions', definitions), ('Coverage', coverage),
                             ('Interpretation', interpretation), ('Validation', verification)]:
        text += f'## {label}\n\n{paragraph}\n\n'
    text += table.to_string(index=False)+'\n\nExact formulas are in report.html and docs/cortex_cpm_variance.md.\n'
    (out/'report.md').write_text(text)
    encoded = base64.b64encode((out/'overview.png').read_bytes()).decode()
    contents = ['<!doctype html><html><head><meta charset="utf-8">'
        '<title>Cortex CPM variance</title><style>body{font:17px/1.6 system-ui;max-width:1200px;'
        'margin:36px auto;padding:0 24px;color:#203040}img{max-width:100%}table{border-collapse:collapse;'
        'font-size:13px}td,th{padding:9px;border:1px solid #ccd5dd;text-align:left}'
        'pre{white-space:pre-wrap;background:#eef3f6;padding:16px}.scroll{overflow:auto}'
        '</style></head><body><h1>Cortex CPM: genotype means versus library deviations</h1>',
        '<p>'+html.escape(lead)+'</p>', '<p>'+html.escape(coverage)+'</p>',
        '<h2>Definitions</h2><p>'+html.escape(definitions)+'</p>', formulas,
        '<h2>Results</h2><div class="scroll">'+table.to_html(index=False,
          float_format=lambda x: f'{x:.6g}')+'</div>',
        f'<img src="data:image/png;base64,{encoded}"><p>{html.escape(caption)}</p>',
        '<h2>Interpretation</h2><p>'+html.escape(interpretation)+'</p>',
        '<h2>Validation</h2><p>'+html.escape(verification)+'</p></body></html>']
    (out/'report.html').write_text('\n'.join(contents))
