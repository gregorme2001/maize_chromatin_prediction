"""Matched four-representation report with explicit pair/ACR-mean definitions."""

import base64
import html
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABELS = {'baseline':'Covariates only','kmer':'5-mer counts','frequency':'5-mer frequencies',
          'softlme':'5-mer PWM log-mean-exp','motif':'TF PWM log-mean-exp'}
COLORS = {'kmer':'#2779b4','frequency':'#389581','softlme':'#8657ab','motif':'#c97835'}


def write_report(out):
    summary=json.loads((out/'summary.json').read_text())
    metrics=pd.read_csv(out/'metrics.csv')
    chrom=pd.read_csv(out/'chromosome_results.csv')
    order=list(LABELS)
    fig,axes=plt.subplots(2,2,figsize=(16,10),constrained_layout=True)
    for offset,(basis,color,label) in enumerate([('linear','#2779b4','Linear GC adjustment'),
        ('gc_spline','#c97835','Flexible GC adjustment')]):
        f=metrics.set_index('model').loc[[basis+'_'+kind for kind in order]]
        x=np.arange(len(order))+(offset-.5)*.34
        for ax,column,multiplier in [(axes[0,0],'rmse',1),(axes[0,1],'mean_acr_rmse',1),
            (axes[1,0],'r2',100),(axes[1,1],'mean_acr_r2',100)]:
            ax.bar(x,f[column]*multiplier,.34,color=color,label=label)
    short=['Covariates','5-mer\ncounts','5-mer\nfrequencies','5-mer PWM\nlog-mean-exp','TF PWM\nlog-mean-exp']
    for ax in axes.ravel():
        ax.set_xticks(np.arange(len(order)),short)
        ax.legend(fontsize=8)
    axes[0,0].set(ylabel='Pair RMSE (CPM)',title='Error across individual ACR–genotype observations')
    axes[0,1].set(ylabel='ACR-mean RMSE (CPM)',title='Error after averaging genotypes within each ACR')
    axes[1,0].set(ylabel='Pair-level out-of-fold R² × 100 (%)',title='Explained variation across ACR–genotype observations')
    axes[1,1].set(ylabel='ACR-mean out-of-fold R² × 100 (%)',title='Explained variation between ACR averages')
    fig.savefig(out/'summary_comparison.png',dpi=145)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(15,5),constrained_layout=True)
    for ax,basis,title in zip(axes,['linear','gc_spline'],['Linear GC adjustment','Flexible GC adjustment']):
        base=chrom.loc[chrom.model.eq(basis+'_baseline')].set_index('chromosome')
        for offset,kind in enumerate(order[1:]):
            f=chrom.loc[chrom.model.eq(basis+'_'+kind)].set_index('chromosome')
            change=100*(1-f.mse/base.mse)
            ax.bar(change.index.to_numpy()+(offset-1.5)*.18,change,.18,label=LABELS[kind],color=COLORS[kind])
        ax.axhline(0,color='black',lw=1)
        ax.set(xticks=range(1,11),xlabel='Chromosome',ylabel='Pair MSE reduction vs matching baseline (%)',title=title)
        ax.legend(fontsize=8)
    fig.savefig(out/'chromosome_comparison.png',dpi=145)
    plt.close(fig)
    def display(basis):
        f=metrics.set_index('model').loc[[basis+'_'+kind for kind in order]].reset_index()
        f=f[['model','rmse','r2','mean_acr_rmse','mean_acr_r2','mse_reduction_vs_matched_baseline']]
        f['model']=[LABELS[k] for k in order]
        for column in ['r2','mean_acr_r2','mse_reduction_vs_matched_baseline']:
            f[column]*=100
        f.columns=['Model','Pair RMSE (CPM)','Pair R² (%)','ACR-mean RMSE (CPM)','ACR-mean R² (%)','Pair MSE reduction vs baseline (%)']
        return f
    def table(f):
        return '<div class="scroll">'+f.to_html(index=False,border=0,float_format=lambda v:f'{v:.4f}')+'</div>'
    def figure(name,caption):
        data=base64.b64encode((out/name).read_bytes()).decode()
        return f'<figure><img src="data:image/png;base64,{data}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>'
    lead=(f'{summary["n_acrs"]:,} ACRs across all ten chromosomes × {summary["n_genotypes"]} development genotypes '
        f'= {summary["n_pairs"]:,} matched cortex observations. Only sequence representation changes. '
        'The original count and TF-motif predictions and genomic folds were reproduced.')
    observations=('For ACR i, genotype g and library l, Y_igl counts cortex nuclei detecting ACR i; '
        'T_gl is total nucleus–ACR detections across ALL 108,843 ACRs in the same genotype/library cortex cells. '
        'C_ig=1,000,000×Σ_lY_igl/Σ_lT_gl. Libraries are pooled WITHIN genotype before division. '
        'Genotypes are NOT pooled together. Every ACR has 137 observed CPM values. '
        'Pair RMSE is sqrt(mean_ig[(C_ig−predicted_C_ig)²]). '
        'For ACR-mean RMSE, first average observed and predicted CPM separately across the same 137 genotypes '
        'at each ACR, then take sqrt(mean_i[(mean_observed_i−mean_predicted_i)²]). '
        'These are equal-genotype averages, not ratios of counts pooled across genotypes. '
        'All models are trained on the same pair-level response; mean metrics summarize their predictions.')
    models=('All fits have one global intercept and globally shared unpenalized GC, linear length, chromosome '
        'indicators and ten ancestry-PC coefficients. There are no ACR-specific intercepts/slopes. '
        'Sequence coefficients are global and ridge-penalized. All sequence features are fixed B73 annotations, '
        'so they vary across ACRs but not across genotypes within an ACR. '
        'Primary GC adjustment is linear; the predefined control uses four natural cubic terms with fixed knots '
        '0,.25,.5,.75,1. Raw CPM, availability, covariates, penalties and outer/inner folds stay as before.')
    features=('The 1,024 five-mers form 512 identical reverse-complement pairs under strand averaging. '
        'Let K_iq be the strand-averaged exact count of word q in ACR i, L_i its length, and n_i=L_i−4 its '
        'number of five-base windows. Counts use K_iq; frequencies use K_iq/n_i. '
        'The solver stores canonical pair SUMS, twice these conceptual columns; training-only standardization '
        'and the duplicate-pair penalty correction make this exactly equivalent. '
        'Five-mer PWM features instead make a one-hot profile for each word, mix it with 1% uniform background, '
        'and use the SAME log2(mean LR) aggregation as TF motifs. Preferred-base probability is .9925, '
        'other-base probability .0025, and the background probability .25. '
        'Each window’s LR is the product of position-wise probability-to-background ratios; average LRs over all windows '
        'and both orientations, THEN take log2. Averaging log scores would be a different statistic. '
        'All LRs are positive, so absent exact words still have finite scores. No count pseudocount or '
        'score threshold is introduced. TF features use the original 927 signed log-mean-LR columns unchanged.')
    formula=('Efficient exact calculation: for canonical word pair r, C_ir counts either word on the forward '
        'reference strand. A_rq=[LR(word_r,target_q)+LR(reverse_complement(word_r),target_q)]/2. '
        'The new feature S_iq=log2[Σ_r C_ir A_rq/n_i]. The saved kernel contains A_rq. '
        'This is mathematically identical to a position-by-position, both-strand PWM scan. '
        'Counts sum to n_i and no position information beyond five-mer counts is used.')
    cv=('Five outer folds of chromosome-specific 5 Mb genomic blocks and three inner block folds are exactly '
        'the same as the previous genome-wide experiment. All genotype observations from an ACR stay together. '
        'Each representation selects its own ridge penalty inside training folds, with an exact no-sequence '
        'candidate. All scaling and nuisance residualization use training rows only. The same genotypes can '
        'appear at different ACRs on both sides; this tests unseen genomic regions in cortex. '
        'There is no extra boundary buffer. Fixed sequence transforms/kernel use no accessibility outcomes. '
        'The complete balanced design is scored exactly without duplicating each feature vector 137 times.')
    limits=('A gain from frequency features supports normalizing by the number of windows. A gain from word-PWM '
        'log-mean-exp beyond frequencies reflects soft matching and/or the nonlinear log representation together; '
        'this experiment does not isolate those two effects separately. No new sequence information or longer '
        'grammar is added. Applying the same aggregation makes the TF comparison cleaner, but TF motifs still '
        'have varying lengths and learned position preferences. Similar predictive performance does not prove '
        'the models use identical information. This is reference-region prediction, not genotype-specific variant '
        'prediction, TF binding/activity or causality. CPM is compositional; mapping/library effects may remain. '
        'Rows and genomic folds are dependent. No coefficient p-values, FDR discoveries or independent-row '
        'confidence intervals are claimed. Negative predictions remain unclipped.')
    cohort=('The existing cohort/exclusions and ≥50 pooled cortex nuclei requirement remain fixed. H95 has 49 '
        'and remains recorded; CML341 is excluded. There is no MAF, SNP-call, ANOVA-F or accessibility screen. '
        'The permanent 30-genotype holdout, DTS, original motif matrix and all earlier results remain untouched. '
        'Prediction storage is exact: ACR predicted mean plus fold/model/genotype CENTERED ancestry contribution. '
        'The ACR means are calculated from shared coefficients, not fitted local intercepts.')
    page=('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>Matched sequence-summary comparison</title><style>body{font:17px/1.6 system-ui;max-width:1450px;margin:30px auto;padding:0 20px}'
        'img{max-width:100%}table{border-collapse:collapse;font-size:14px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}'
        '.scroll{overflow-x:auto}figcaption{font-size:14px;color:#444}</style><h1>Five-mer counts, frequencies and PWM log-mean-exp</h1>'
        f'<p>{html.escape(lead)}</p><h2>Primary: linear GC adjustment</h2>'+table(display('linear'))
        +'<h2>Control: flexible GC adjustment</h2>'+table(display('gc_spline'))
        +figure('summary_comparison.png','Each bar summarizes the same out-of-fold predictions. Left panels evaluate all ACR–genotype pairs; '
            'right panels first average observed/predicted CPM across genotypes within each ACR. Lower RMSE and higher R² are better. '
            'ACR-mean metrics do not come from a separate fit. Both baseline adjustments are shown.')
        +figure('chromosome_comparison.png','Within each chromosome, every sequence representation is compared with its matching covariate baseline '
            'using the same observations. Positive values favor sequence features. These are held-out-block predictions, not leave-one-chromosome-out fits.')
        +f'<h2>What are the observations and errors?</h2><p>{html.escape(observations)}</p>'
        f'<h2>Models and feature definitions</h2><p>{html.escape(models)}</p><p>{html.escape(features)}</p>'
        f'<details><summary>Exact count-based PWM calculation</summary><p>{html.escape(formula)}</p></details>'
        f'<h2>Validation and interpretation</h2><p>{html.escape(cv)}</p><p>{html.escape(limits)}</p><p>{html.escape(cohort)}</p>'
        '<p>Saved: all metrics, ACR/genotype/chromosome summaries, exact fold manifest, inner scores, raw coefficients, '
        'models/scales, factorized predictions, word-PWM definition/kernel, feature identities and source/code hashes.</p></html>')
    (out/'report.html').write_text(page)
    (out/'report.md').write_text('\n\n'.join([lead,display('linear').to_string(index=False),
        display('gc_spline').to_string(index=False),observations,models,features,formula,cv,limits,cohort])+'\n')
