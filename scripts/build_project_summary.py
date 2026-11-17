#!/usr/bin/env python3
"""Export a one-page handout from published metrics; no data acquisition/fitting."""
from pathlib import Path
import hashlib
import json

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'results/cnn_saved_prediction_comparison/metrics.csv'
OUT = ROOT/'docs'
URL = 'https://github.com/gregorme2001/maize_chromatin_prediction'


def main():
    data = pd.read_csv(SOURCE)
    data = data.loc[data.subgroup.eq('all')].set_index('model')
    assert data.index.is_unique
    assert set(data.n_pairs) == {495900} and set(data.n_ACRs) == {9666}
    rows = [
        ('Covariates only', 'current_ridge_baseline'),
        ('Best matched motif ridge', 'current_ridge_personalized_core_TF'),
        ('Best matched CNN', 'current_CNN_B73_context'),
        ('Earlier padded CNN*', 'earlier_CNN_context_2048'),
    ]
    title = 'Predicting maize chromatin accessibility\nfrom DNA sequence'
    sections = [
        ('Question and data',
         'Can local DNA predict accessibility at unseen genomic regions, and do '
         'genotype-specific SNPs help? Using Marand et al. (2025) single-nucleus '
         'ATAC-seq data, the matched cortex benchmark contains 9,666 accessible '
         'chromatin regions (ACRs), 137 development genotypes and 495,900 '
         'ACR–genotype observations. Libraries are pooled within genotype. '
         'The response is CPM: each region’s detections divided by all-genome '
         'ACR detections in the same genotype, multiplied by one million.'),
        ('Evaluation',
         'Five outer and three inner cross-validation folds keep chromosome-specific '
         '5-Mb blocks together. Supervised preprocessing and tuning use training '
         'observations only. All rows below evaluate the same observations using '
         'held-out-region predictions. This tests new regions within the studied '
         'genotypes, not new genotypes. The permanent 30-genotype holdout is unused.'),
        ('What we learned',
         'Motif/five-mer ridge summaries modestly predict differences in average '
         'accessibility between ACRs. SNP-substituted inputs provide essentially '
         'no overall improvement in the matched comparisons, and within-ACR '
         'genotype-deviation R² remains near zero. The four matched CNNs barely '
         'improve over covariates alone. The earlier padded CNN performs better '
         'than the best current ridge model in all five outer folds.'),
        ('Scope and limitations',
         'CPM is relative accessibility. Complete homozygous SNP calls restrict '
         'eligibility; reconstructed sequences omit indels, structural variants '
         'and unlisted SNPs. Library variation does not separate genetic and '
         'measurement variance. Prediction does not establish TF binding or '
         'causality; weak genotype prediction does not contradict caQTL association.'),
    ]
    caveat = ('*The earlier B73 CNN used a larger 108,843-ACR training catalog, '
              '2,048-bp inputs, a different architecture and three-network averaging. '
              'Each evaluated ACR was excluded from its prediction networks’ training '
              'sets. This comparison fixes evaluation observations, not training '
              'cohorts or architectures. The cause of its advantage remains unresolved.')
    plt.rcParams.update({'font.family':'DejaVu Sans', 'pdf.fonttype':42})
    fig = plt.figure(figsize=(8.5,11), facecolor='white')
    ax = fig.add_axes([0,0,1,1]); ax.set(xlim=(0,1),ylim=(0,1)); ax.axis('off')
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    y = .965
    ax.text(.07,y,title,va='top',fontsize=18,fontweight='bold',linespacing=1.15)
    y -= .072
    ax.text(.07,y,'Gregory Orme  |  Research summary  |  November 2026',fontsize=10,color='#444444',va='top')
    y -= .023
    ax.plot([.07,.93],[y,y],color='#777777',lw=.7)
    y -= .018

    def paragraph(text, size=10.3, leading=.0174):
        nonlocal y
        font = FontProperties(family='DejaVu Sans', size=size)
        maximum_width = .86*fig.get_figwidth()*fig.dpi
        wrapped, line = [], ''
        for word in text.split():
            candidate = (line+' '+word).strip()
            measured = renderer.get_text_width_height_descent(candidate, font, ismath=False)[0]
            if line and measured > maximum_width:
                wrapped.append(line)
                line = word
            else:
                line = candidate
        if line:
            wrapped.append(line)
        ax.text(.07,y,'\n'.join(wrapped),va='top',fontsize=size,linespacing=1.25)
        y -= leading*len(wrapped)

    def section(heading, text):
        nonlocal y
        ax.text(.07,y,heading,va='top',fontsize=11.2,fontweight='bold')
        y -= .022
        paragraph(text)
        y -= .014

    section(*sections[0])
    section(*sections[1])
    ax.text(.07,y,'Headline comparison',va='top',fontsize=11.2,fontweight='bold'); y -= .026
    height=.127
    cells = [[label,f'{data.loc[name,"pair_rmse"]:.4f}',
              f'{data.loc[name,"ACR_mean_rmse"]:.4f}',f'{data.loc[name,"ACR_mean_r2"]:.4f}']
             for label,name in rows]
    table=ax.table(cellText=cells,
        colLabels=['Model','Observation\nRMSE ↓','ACR-average\nRMSE ↓','ACR-average\nR² ↑'],
        colWidths=[.39,.20,.21,.20],bbox=[.07,y-height,.86,height],cellLoc='right')
    table.auto_set_font_size(False);table.set_fontsize(9.1)
    for (r,c),cell in table.get_celld().items():
        cell.set_linewidth(.35);cell.set_edgecolor('#bbbbbb')
        if r==0:cell.set_facecolor('#edf1f4');cell.set_text_props(weight='bold')
        if c==0:cell.set_text_props(ha='left')
    y -= height+.012
    paragraph('RMSE is in CPM; lower is better. ACR-average metrics average observed and predicted CPM over the same available genotypes at each ACR.',size=9.2,leading=.0157)
    y -= .008
    paragraph(caveat,size=9.2,leading=.0157)
    y -= .015
    section(*sections[2])
    section(*sections[3])
    assert y>.06, f'Content exceeded one-page layout: remaining y={y}'
    ax.plot([.07,.93],[.059,.059],color='#777777',lw=.7)
    ax.text(.07,.047,URL,fontsize=8.5,va='top',url=URL)
    ax.text(.07,.030,'Source study: Marand et al., Science (2025), doi:10.1126/science.ads6601',fontsize=8.5,va='top',url='https://doi.org/10.1126/science.ads6601')
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT/'project_summary.pdf',metadata={'Title':title.replace('\n',' '),'Author':'Gregory Orme'})
    fig.savefig(OUT/'project_summary.png',dpi=180)
    plt.close(fig)

    md=['# '+title.replace('\n',' '),'','Gregory Orme | November 2026','']
    for heading,text in sections[:2]:md+=['## '+heading,'',text,'']
    md+=['## Headline comparison','', '| Model | Observation RMSE | ACR-average RMSE | ACR-average R² |', '|---|---:|---:|---:|']
    for row in cells:md.append('| '+' | '.join(row)+' |')
    md+=['','RMSE is in CPM; lower is better. ACR-average metrics weight ACRs equally.','',caveat,'']
    for heading,text in sections[2:]:md+=['## '+heading,'',text,'']
    md += [URL,'','Source study: https://doi.org/10.1126/science.ads6601','']
    (OUT/'project_summary.md').write_text('\n'.join(md))
    provenance={'purpose':'One-page handout generated from published summaries; no fitting',
        'metrics_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'project_summary_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('Wrote one-page PDF, PNG preview and Markdown summary to docs/.')


if __name__=='__main__':
    main()
