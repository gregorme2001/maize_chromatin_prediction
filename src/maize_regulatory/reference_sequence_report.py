"""Plain-language report for matched reference-only sequence representations."""

import base64
import html
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABELS = {'baseline':'Covariates only','kmer':'Covariates + B73 five-mers','motif':'Covariates + B73 motifs'}
COLORS = {'baseline':'#8b9aa7','kmer':'#2779b4','motif':'#c97835'}


def write_report(out):
    s = json.loads((out/'summary.json').read_text())
    metrics = pd.read_csv(out/'metrics.csv')
    acr = pd.read_parquet(out/'acr_predictions.parquet')
    chrom = pd.read_csv(out/'chromosome_results.csv')
    folds = pd.read_csv(out/'fold_results.csv')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for offset, (basis, label, color) in enumerate([('linear','Linear GC baseline','#2779b4'),
        ('gc_spline','Flexible GC baseline','#c97835')]):
        m = metrics.set_index('model').loc[[f'{basis}_{kind}' for kind in LABELS]]
        x = np.arange(3)+(offset-.5)*.34
        axes[0,0].bar(x,m.rmse,.34,label=label,color=color)
        axes[0,1].bar(x,100*m.r2,.34,label=label,color=color)
    for ax in axes[0]:
        ax.set_xticks(np.arange(3),['Covariates only','+ B73 five-mers','+ B73 motifs'])
        ax.legend(fontsize=8)
    axes[0,0].set(ylabel='Out-of-fold pair RMSE (CPM)', title='Prediction error on held-out genomic blocks')
    axes[0,1].set(ylabel='Out-of-fold R² × 100 (%)', title='Fraction of pair-level variation explained')
    for ax,basis,title in zip(axes[1],['linear','gc_spline'],['Linear GC adjustment','Flexible GC adjustment']):
        base=chrom.loc[chrom.model.eq(basis+'_baseline')].set_index('chromosome')
        for offset,kind in enumerate(['kmer','motif']):
            f=chrom.loc[chrom.model.eq(basis+'_'+kind)].set_index('chromosome')
            benefit=100*(1-f.mse/base.mse)
            ax.bar(benefit.index.to_numpy()+(offset-.5)*.36,benefit,.36,label=LABELS[kind],color=COLORS[kind])
        ax.axhline(0,color='black',lw=1)
        ax.set(xticks=sorted(chrom.chromosome.unique()),xlabel='Chromosome',
            ylabel='MSE reduction versus matching covariate baseline (%)',title=title)
        ax.legend(fontsize=8)
    fig.savefig(out/'model_comparison.png',dpi=145)
    plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
    # Display zoom is fixed at the observed 99.5th percentile; all points score.
    high=float(acr.observed_CPM.quantile(.995))
    for row,basis in enumerate(['linear','gc_spline']):
        for ax,kind in zip(axes[row],LABELS):
            prediction=acr[f'{basis}_{kind}_CPM']
            plotted=ax.hexbin(acr.observed_CPM,prediction,gridsize=55,mincnt=1,bins='log',cmap='Blues',
                extent=(0,high,-2,high))
            ax.plot([0,high],[0,high],'--',color='black',lw=1)
            ax.set(xlim=(0,high),ylim=(-2,high),xlabel='Mean observed cortex CPM at ACR',
                ylabel='Mean out-of-fold predicted CPM at ACR',
                title=f'{LABELS[kind]}\n'+('Linear GC' if basis=='linear' else 'Flexible GC'))
            fig.colorbar(plotted,ax=ax,label='Number of ACRs (log color scale)')
    fig.savefig(out/'acr_mean_predictions.png',dpi=145)
    plt.close(fig)
    primary=metrics.loc[metrics.model.str.startswith('linear_')].copy()
    sensitivity=metrics.loc[metrics.model.str.startswith('gc_spline_')].copy()
    def display(frame):
        f=frame[['model','rmse','r2','mse_reduction_vs_matched_baseline','mean_acr_rmse','mean_acr_pearson_r']].copy()
        f['model']=[LABELS[v.rsplit('_',1)[-1]] for v in f.model]
        f['mse_reduction_vs_matched_baseline']*=100
        f.columns=['Model','Pair RMSE (CPM)','Pair R²','MSE reduction vs baseline (%)','ACR-mean RMSE (CPM)','ACR-mean Pearson r']
        return f
    def table(frame):
        return '<div class="scroll">'+frame.to_html(index=False,border=0,float_format=lambda v:f'{v:.4f}')+'</div>'
    def figure(name,caption):
        data=base64.b64encode((out/name).read_bytes()).decode()
        return f'<figure><img src="data:image/png;base64,{data}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>'
    lead=(f'{s["n_acrs"]:,} B73-reference ACRs across all {len(s["chromosomes"])} chromosomes, crossed with '
        f'{s["n_genotypes"]} eligible development genotypes: {s["n_pairs"]:,} pooled cortex CPM observations. '
        'Every model uses exactly the same observations and genomic folds.')
    definition=('Indices: i = ACR, g = genotype, l = library, k = ancestry PC, q = five-mer, m = motif. '
        'Y_igl counts cortex nuclei detecting ACR i. T_gl sums detections at ALL 108,843 ACRs in the SAME '
        'genotype/library cortex cells. C_ig = 1,000,000 × Σ_l Y_igl / Σ_l T_gl. '
        'Pool libraries BEFORE dividing. The response is raw CPM, without log/rank transformations. '
        'Each eligible genotype has ≥50 pooled cortex nuclei (137 of the 138 development identities; H95 has 49). '
        'All ACRs have the SAME eligible genotype set. No SNP-call, MAF, ANOVA-F or accessibility screen is used. '
        'CML341 and the permanent holdout are excluded before cell-value aggregation; no DTS is read.')
    model=('Baseline: C_ig = a + f(GC_i) + b_L length_i + chromosome_indicator_i + Σ_k γ_k PC_gk + error_ig. '
        'All intercepts/slopes are shared globally. There are NO ACR-specific intercepts or local ancestry coefficients. '
        'The five-mer model adds Σ_q β_q K_iq; the motif model instead adds Σ_m θ_m M_im. '
        'GC_i, K_iq and M_im all use the SAME fixed B73 sequence and do not vary with genotype. '
        'K_iq is the raw count averaged across forward and reverse-complement strands (1,024 conceptual columns, '
        '512 exact duplicate-pair compression). M_im is the existing signed log2 mean likelihood ratio '
        'over motif placements; negative scores are retained. Only sequence coefficients are penalized. '
        'No library indicator is fitted after pooling; library-composition effects may remain.')
    gc=('Primary f(GC_i) is linear, matching the previous pooled pilot. The prespecified sensitivity uses '
        'four natural cubic GC spline terms with FIXED knots 0, 0.25, 0.5, 0.75, 1. '
        'No response or validation feature distribution chooses these knots. Length remains linear. '
        'Both sequence models are compared against their own matching covariate baseline; '
        'a flexible-GC improvement must not be compared only with the weaker linear baseline.')
    cv=('ACR midpoints define chromosome-specific 5 Mb blocks (block 0 on chromosome 1 is distinct from block 0 '
        'on chromosome 2). Five outer folds hold out whole blocks; three inner block folds independently choose '
        'each model’s ridge penalty, including the exact covariate-only baseline as a candidate. '
        'All observations of an ACR stay together. Every model uses the identical fold manifest. '
        'Training-only centering, scaling and nuisance adjustment prevent validation leakage. '
        'Fixed sequence annotations may be precomputed globally because they use no accessibility. '
        'The same development genotypes occur at different ACRs on both sides. This tests unseen genomic regions '
        'in cortex, not new genotypes, cell types or whole held-out chromosomes. There is no extra boundary buffer.')
    storage=('The fit and score retain every ACR–genotype observation. Because reference features are constant '
        'within an ACR, exact cross-products can be calculated without repeating feature rows millions of times. '
        'Prediction storage is also exact: predicted CPM at (i,g) equals the saved ACR predicted mean '
        'plus the saved CENTERED ancestry contribution for genotype g in that ACR’s outer fold and model. '
        'These ACR predicted means are calculated from shared coefficients; they are not fitted ACR intercepts. '
        'The full-genome cortex count cache reconstructs all observed responses.')
    limits=('This tests whether B73 sequence features predict differences in relative accessibility BETWEEN ACRs. '
        'Reference sequence models cannot explain genotype-specific sequence changes. Improvement may reflect '
        'sequence composition, TF-related preferences or technical/reference-mapping effects; it does not establish '
        'TF binding, occupancy, activity or causality. Five-mer counts and motifs are correlated, so individual '
        'ridge coefficients are not independent TF discoveries. Raw CPM is heavy-tailed and compositional. '
        'Negative linear predictions are retained without clipping. Shared genotypes/material and spatial dependence '
        'mean millions of rows are not millions of independent biological replicates. No coefficient p-values, '
        'FDR discoveries or independent-row confidence intervals are claimed. The expanded reference-only panel '
        'does not require SNP availability, so it differs from the earlier genotype-specific chr10 pilot.')
    comparisons=[]
    for basis in ['linear','gc_spline']:
        for kind in ['kmer','motif']:
            f=folds.loc[folds.model.eq(basis+'_'+kind)].set_index('outer_fold')
            b=folds.loc[folds.model.eq(basis+'_baseline')].set_index('outer_fold')
            comparisons.append(f'{basis}, {kind}: {int((f.rmse<b.rmse-1e-10).sum())}/5 outer folds improve.')
    page=('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>Matched genome-wide B73 sequence prediction</title><style>body{font:17px/1.6 system-ui;max-width:1400px;margin:30px auto;padding:0 20px}'
        'img{max-width:100%}table{border-collapse:collapse;font-size:14px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}'
        '.scroll{overflow-x:auto}figcaption{font-size:14px;color:#444}</style><h1>Matched B73-only sequence prediction across all chromosomes</h1>'
        f'<p>{html.escape(lead)}</p><h2>Primary comparison: linear GC adjustment</h2>'+table(display(primary))
        +'<h2>Prespecified control: flexible GC adjustment</h2>'+table(display(sensitivity))
        +'<p>'+html.escape(' '.join(comparisons))+'</p>'
        +figure('model_comparison.png','Top: pair-level out-of-fold RMSE and R² for each model under linear/flexible GC adjustment. '
            'Bottom: squared-error reduction by chromosome relative to the MATCHING baseline; positive values favor sequence features. '
            'Each chromosome summary combines predictions from genomic-block validation, not leave-one-chromosome-out training.')
        +figure('acr_mean_predictions.png','Each hexagon counts ACRs by mean observed and mean out-of-fold predicted CPM across the same 137 genotypes. '
            'Rows show linear/flexible GC adjustment. Diagonals mark perfect prediction. For visibility, axes zoom to the observed 99.5th percentile '
            '(with a small negative prediction margin); all ACRs and full ranges remain included in every numerical score. '
            'These averages describe pair-level predictions, not a separately fitted bulk-response model.')
        +f'<h2>Exact observations and models</h2><p>{html.escape(definition)}</p><p>{html.escape(model)}</p>'
        f'<p>{html.escape(gc)}</p><h2>Validation and storage</h2><p>{html.escape(cv)}</p><p>{html.escape(storage)}</p>'
        f'<h2>Interpretation and limits</h2><p>{html.escape(limits)}</p>'
        '<p>Saved: metrics.csv, fold_results.csv, fold_manifest.json, inner_scores.csv, chromosome/genotype summaries, '
        'acr_predictions.parquet, genotype_contributions.csv, coefficient tables, all fitted models/scales, '
        'sequence-feature identities, configuration, and input/code hashes. Earlier results, phenotype CV and permanent holdout remain untouched.</p></html>')
    (out/'report.html').write_text(page)
    (out/'report.md').write_text('\n\n'.join([lead,display(primary).to_string(index=False),
        display(sensitivity).to_string(index=False),' '.join(comparisons),definition,model,gc,cv,storage,limits])+'\n')
