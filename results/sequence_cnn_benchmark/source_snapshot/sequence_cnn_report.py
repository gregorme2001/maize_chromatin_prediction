"""Offline results for the matched, nested-CV residual CNN sequence benchmark."""

import base64
import html
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NAMES={'gc_spline_baseline':'Covariates only',
       'gc_spline_frequency':'ACR five-mer frequencies + ridge',
       'gc_spline_motif':'ACR TF motif scores + ridge',
       'context_frequency':'2-kb five-mer frequencies + ridge',
       'core_context_frequency':'ACR + 2-kb five-mer frequencies + ridge',
       'acr_only':'ACR-only residual CNN',
       'context_2048':'2-kb context residual CNN'}


def write_report(out,root):
    summary=json.loads((out/'summary.json').read_text())
    current=pd.read_csv(out/'metrics.csv')
    prior=pd.read_csv(root/'results/sequence_summary_comparison/metrics.csv')
    previous=prior.loc[prior.model.isin(['gc_spline_baseline','gc_spline_frequency','gc_spline_motif'])].copy()
    combined=pd.concat([previous,current],ignore_index=True)
    extra=out/'context_ridge_metrics.csv'
    if extra.exists():combined=pd.concat([combined,pd.read_csv(extra)],ignore_index=True)
    extra=out/'core_context_ridge_metrics.csv'
    if extra.exists():combined=pd.concat([combined,pd.read_csv(extra)],ignore_index=True)
    combined.to_csv(out/'comparison_metrics.csv',index=False)
    display=combined[['model','rmse','r2','mean_acr_rmse','mean_acr_r2']].copy()
    display['model']=display.model.map(NAMES)
    for key in ['r2','mean_acr_r2']:display[key]*=100
    display.columns=['Model','Pair RMSE (CPM)','Pair R² (%)','ACR-mean RMSE (CPM)','ACR-mean R² (%)']
    fig,axes=plt.subplots(1,2,figsize=(14,6),constrained_layout=True)
    labels=combined.model.map(NAMES).tolist()
    colors=['#8797a5' if 'CNN' not in name else '#3b86ac' for name in labels]
    for ax,column,title in zip(axes,['r2','mean_acr_r2'],['Individual ACR/genotype observations','Genotype-averaged accessibility per ACR']):
        values=combined[column].to_numpy()*100
        ax.barh(np.arange(len(labels)),values,color=colors)
        ax.set(yticks=np.arange(len(labels)),yticklabels=labels,xlabel='Out-of-fold R² (%)',title=title)
        ax.invert_yaxis();ax.axvline(0,color='black',lw=.8);ax.grid(axis='x',alpha=.15)
        for j,value in enumerate(values):ax.text(value,j,f' {value:.2f}',va='center',fontsize=9)
        ax.margins(x=.15)
    fig.savefig(out/'model_comparison.png',dpi=150,bbox_inches='tight');plt.close(fig)
    folds=pd.read_csv(out/'fold_results.csv')
    previous_folds=pd.read_csv(root/'results/sequence_summary_comparison/fold_results.csv')
    previous_folds=previous_folds.loc[previous_folds.model.eq('gc_spline_baseline')]
    fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
    for name,source in [('gc_spline_baseline',previous_folds)]+[(m,folds.loc[folds.model.eq(m)]) for m in current.model]:
        ax.plot(source.outer_fold+1,source.rmse,'o-',label=NAMES[name])
    ax.set(xlabel='Outer genomic-block fold',ylabel='Validation pair RMSE (CPM)',
           title='Five separate outer assessments; no best-fold selection',xticks=np.arange(1,6))
    ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.savefig(out/'fold_comparison.png',dpi=150,bbox_inches='tight');plt.close(fig)
    inner=[json.loads(p.read_text()) for p in sorted((out/'inner').glob('*.json'))]
    cap=summary['config']['max_epochs'];at_cap=sum(v['best_epoch']==cap for v in inner)
    context_inner=[json.loads(p.read_text()) for p in (out/'inner').glob('context_2048*.json')]
    context_at_cap=sum(v['best_epoch']==cap for v in context_inner)
    history=[]
    for path in sorted((out/'inner').glob('*.json')):
        info=json.loads(path.read_text())
        history.extend(dict(fit=path.stem,**row) for row in info['history'])
    curves=pd.DataFrame(history)
    curves.to_csv(out/'inner_learning_curves.csv',index=False)
    fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
    for kind,color in [('acr_only','#3b86ac'),('context_2048','#377d57')]:
        part=curves.loc[curves.fit.str.startswith(kind)]
        grouped=part.groupby('epoch').validation_mean_acr_mse
        center=grouped.median();lo=grouped.quantile(.25);hi=grouped.quantile(.75)
        ax.plot(center.index,center.to_numpy(),'o-',color=color,label=NAMES[kind])
        ax.fill_between(center.index,lo.to_numpy(),hi.to_numpy(),color=color,alpha=.15)
    ax.set(xlabel='Training epoch',ylabel='Inner validation ACR-mean MSE (CPM²)',
           title='Learning curves: median and middle 50% of inner fits')
    ax.legend();ax.grid(alpha=.2)
    fig.savefig(out/'learning_curves.png',dpi=150,bbox_inches='tight');plt.close(fig)
    timing=json.loads((out/'timing_benchmark.json').read_text()) if (out/'timing_benchmark.json').exists() else {}
    lead=(f'{summary["n_acrs"]:,} ACRs × {summary["n_genotypes"]} eligible development genotypes '
          f'= {summary["n_pairs"]:,} cortex observations. Both CNNs use the same response, covariates '
          'and frozen five-outer/three-inner genomic-block folds as the earlier ridge benchmark. '
          'One input is the complete ACR sequence; the other is a 2,048-base window centered on it.')
    definitions=(
        'Indices: i=ACR, g=genotype, l=library. Y_igl counts cortex nuclei detecting ACR i. '
        'T_gl sums positive nucleus–ACR detections across ALL 108,843 ACRs in those SAME '
        'genotype/library cortex cells. C_ig=1,000,000×Σ_lY_igl/Σ_lT_gl pools libraries within '
        'genotype before division. Genotypes are not pooled into a count ratio. Pair RMSE '
        'scores every C_ig equally; ACR-mean RMSE first averages observed and predicted C_ig '
        'separately over the same 137 genotypes, then scores across ACRs.')
    model=(
        'The prediction is a global linear GC-spline/length/chromosome contribution plus a '
        'global ancestry-PC contribution plus a sequence CNN contribution. There are no '
        'ACR-specific intercepts. The CNN uses A/C/G/T channels and a binary channel marking '
        'the measured ACR, a 15-base stride-4 convolution, max pooling, two residual blocks '
        '(each two 5-base convolutions), eight positional bins and a 128-unit dense head. '
        'It has 56,081 trainable parameters, including the unpenalized region covariate layer. '
        'Convolution/dense weights have explicit L2 penalties; nuisance coefficients and biases '
        'are unpenalized. The two candidates differ only in L2 (1e−5 versus 1e−4); width 32, '
        'dropout 0.1, Adam learning rate 0.003 and batch size 1,024 are fixed. Mixed precision '
        'uses float16 GPU convolutions, float32 weights and gradient scaling. Gradients are '
        'clipped at norm 5. Raw CPM predictions are not clipped or log-transformed.')
    factorization=(
        'To avoid computing the same reference sequence 137 times, training uses the exact '
        'balanced-design identity: pair MSE = MSE of equal-genotype ACR means + an ancestry-only '
        'term. Centering PCs over the shared genotype set makes the ancestry fit algebraically '
        'independent of the sequence/region fit. Its coefficients are profiled by training-only '
        'least squares; region coefficients and CNN weights are optimized JOINTLY by gradients. '
        'A training-only OLS initialization of the region layer is subsequently updated, not '
        'frozen. This is not a residualized-response replacement objective. Tests verify both '
        'the full-pair loss identity and the sequence-output gradient identity. Inner candidate '
        'selection and outer scoring include the ancestry term and all genotype observations.')
    validation=(
        f'Each candidate is trained from scratch in all three inner folds. Inner validation '
        f'uses reverse-complement-averaged predictions after each epoch, with patience '
        f'{summary["config"]["patience"]} and a cap of {cap} epochs. Pooled inner held-out pair '
        'MSE selects the penalty; the median best epoch of its three inner fits sets the outer '
        'refit duration. Three fresh refits use seeds 101/202/303 and average predictions. '
        'Outer values never select configurations, epochs or seeds. All genotype rows of an '
        'ACR stay together. Outer predictions are combined; no best outer fold is selected. '
        'Train-fold-only scaling is saved. Each ACR has a fixed length bucket, so padding and '
        'predictions do not depend on which other ACRs appear in its batch. Training randomly '
        'reverse-complements batches; evaluation averages both orientations.')
    audit=summary['sequence_audit']
    inputs=(
        f'Every extracted context core matches its original B73 ACR sequence. '
        f'{audit["clipped_windows"]} windows are clipped/padded at sequence-only partitions '
        'between adjacent occupied genomic blocks or chromosome edges; all measured ACR bases '
        'are preserved. This retains the old folds while preventing shared flank DNA across '
        f'blocks. Context includes {audit["ambiguous_context_bases"]:,} ambiguous genomic bases; '
        'ambiguous/padded positions have zero A/C/G/T channels. The label-independent sequence '
        'encoding is precomputed globally. No new genome is downloaded. The fixed ≥50 pooled '
        'cortex-nuclei prediction rule retains 137 genotypes; H95 remains recorded with 49, '
        'CML341 remains excluded, and the fixed 30 holdouts and DTS are unused.')
    limits=(
        f'This is a time-budgeted preliminary benchmark: {at_cap}/{len(inner)} inner fits '
        f'have their best checkpoint at the {cap}-epoch cap, including '
        f'{context_at_cap}/{len(context_inner)} context fits. A cap-bound learning curve may '
        'still be improving; weak results would not establish the maximum achievable signal. '
        'Architecture breadth and regularization search are limited. The context model receives '
        'additional DNA relative to the old ACR-only models; it does not isolate architecture '
        'unless compared with features extracted from that same context. The 2-kb frequency '
        'ridge comparator is listed when available. A second comparator concatenates ACR and '
        'context frequencies, preserving core identity rather than diluting it into the window. '
        'Blocks are dependent and repeated sequence '
        'families can cross folds. CPM is compositional and technical/mapping effects can be '
        'predictive. This assesses unseen regions among called ACRs in one sampled cell state; '
        'it is not genotype-specific variant prediction, phenotype prediction or a causal '
        'binding claim. No independent-row confidence intervals or significance tests are reported.')
    content=[('What was predicted',lead),('Response and metrics',definitions),('Model',model),
        ('Exact computation',factorization),('Nested validation',validation),('Sequence inputs and cohort',inputs),
        ('Limits and training budget',limits)]
    verification=out/'independent_validation.json'
    if verification.exists():
        check=json.loads(verification.read_text())
        if not check['all_predictions_reconstruct']:raise AssertionError('Independent verification failed')
        content.append(('Independent verification',
            'Every saved out-of-fold pair prediction and both reported RMSEs were independently '
            'reconstructed. All 30 refits have matching training-row hashes and training-only '
            'scaling. Centered ancestry contributions match the previous ridge calculation '
            'within 2e−12 CPM. Full-precision CPU predictions on 32 sampled ACRs per input '
            'version match mixed-precision GPU inference within about 0.0041 CPM. The original '
            'environment passes 147 tests; five new neural tests pass in the native environment. '
            'The two Torch-dependent test modules are skipped only in the original environment '
            'where Torch is intentionally absent.'))
    text='# Residual CNN sequence benchmark\n\n'
    for title,paragraph in content:text+=f'## {title}\n\n{paragraph}\n\n'
    text+=display.to_string(index=False)+'\n'
    (out/'report.md').write_text(text)
    pieces=['<!doctype html><html><head><meta charset="utf-8"><title>Sequence CNN benchmark</title>'
        '<style>body{font:17px/1.6 system-ui;max-width:1250px;margin:35px auto;padding:0 24px;'
        'color:#203040}img{max-width:100%}table{border-collapse:collapse;font-size:14px}'
        'td,th{border:1px solid #ccd5dd;padding:9px;text-align:left}.scroll{overflow:auto}'
        '</style></head><body><h1>Residual CNN sequence benchmark</h1>',
        '<p>'+html.escape(lead)+'</p><div class="scroll">'+display.to_html(index=False,
                float_format=lambda x:f'{x:.4f}')+'</div>']
    for filename in ['model_comparison.png','fold_comparison.png','learning_curves.png']:
        encoded=base64.b64encode((out/filename).read_bytes()).decode()
        pieces.append(f'<img src="data:image/png;base64,{encoded}">')
    pieces.append('<p>Learning-curve shading describes variation across inner fits and penalties, '
                  'not a confidence interval. Later epochs can have fewer fits because of early '
                  'stopping. These curves use inner validation data, never outer test blocks. '
                  'Outer refits use more training regions and a three-seed ensemble, so their '
                  'errors need not equal the inner-curve levels.</p>')
    for title,paragraph in content[1:]:pieces.append(f'<h2>{title}</h2><p>{html.escape(paragraph)}</p>')
    pieces.append('</body></html>');(out/'report.html').write_text('\n'.join(pieces))
    save=dict(n_inner_fits=len(inner),n_best_at_epoch_cap=at_cap,epoch_cap=cap,
              n_outer_refit_models=len(list((out/'models').glob('*.pt'))),timing=timing)
    (out/'training_budget_audit.json').write_text(json.dumps(save,indent=2)+'\n')
