"""Matched five-mer frequency ridge on the CNN's exact context inputs."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .kmer_pilot import kmer_dictionary
from .pooled_sequence import Moments,add_moments,ridge_path,moment_mse,choose_alpha
from .sequence_cnn import ancestry_projection
from .reference_sequence_evaluation import score_arrays,summarize_totals


def word_frequencies(codes):
    """Canonical five-mer frequencies; windows containing N/padding are excluded."""
    x=np.asarray(codes)
    if x.ndim!=2 or x.shape[1]<5 or np.any(x>4):raise ValueError('Invalid encoded sequence')
    _,_,canonical,mapping=kmer_dictionary(5)
    result=np.zeros((len(x),len(canonical)),dtype=np.float32)
    for start in range(0,len(x),128):
        part=x[start:start+128];width=x.shape[1]-4
        words=np.zeros((len(part),width),dtype=np.int32);valid=np.ones_like(words,dtype=bool)
        for j in range(5):
            b=part[:,j:j+width]
            words=4*words+np.minimum(b,3);valid&=b<4
        for row in range(len(part)):
            if not valid[row].any():raise ValueError('Sequence has no valid five-mer window')
            counts=np.bincount(mapping[words[row,valid[row]]],minlength=len(canonical))
            result[start+row]=counts/counts.sum()
    return result


def fit_context_baseline(cache,out,root,include_core=False):
    cache,out,root=map(Path,[cache,out,root])
    codes=np.load(cache/'context.npy',mmap_mode='r')
    x_path=cache/'context_frequencies.npy'
    if x_path.exists():features=np.load(x_path,mmap_mode='r')
    else:
        print('Counting five-mers in the identical 2-kb context inputs',flush=True)
        features=word_frequencies(codes);np.save(x_path,features)
    model_name='core_context_frequency' if include_core else 'context_frequency'
    prefix='core_context_ridge' if include_core else 'context_ridge'
    if include_core:
        core_path=cache/'core_frequencies.npy'
        if core_path.exists():core=np.load(core_path,mmap_mode='r')
        else:
            core=word_frequencies(np.load(cache/'core.npy',mmap_mode='r'));np.save(core_path,core)
        features=np.column_stack([core,features])
    y=np.load(cache/'response.npy',mmap_mode='r');target=y.mean(axis=1)
    p=np.load(cache/'pcs.npy');z=np.load(cache/'covariates.npy')
    block=np.load(cache/'blocks.npy');folds=json.loads((out/'fold_manifest.json').read_text())
    alphas=json.loads((root/'configs/reference_sequence_comparison.json').read_text())['alphas']+[None]
    design=np.column_stack([z,features]);n_cov=z.shape[1]
    def moments(rows):
        a=design[rows];b=target[rows]
        return Moments(len(rows),a.T@a,a.T@b,float(b@b))
    print('Building exact genomic-block moments for context frequency ridge',flush=True)
    outer_parts=[moments(np.flatnonzero(np.isin(block,s['validation_blocks']))) for s in folds]
    prediction=np.full(len(y),np.nan);total={};records=[];inner_records=[];effects=[]
    for spec in folds:
        k=spec['number'];tr=np.flatnonzero(np.isin(block,spec['training_blocks']))
        va=np.flatnonzero(np.isin(block,spec['validation_blocks']))
        parts=[moments(np.flatnonzero(np.isin(block,b['validation_blocks']))) for b in spec['inner']]
        scores=[]
        for j,validation in enumerate(parts):
            training=add_moments([m for index,m in enumerate(parts) if index!=j])
            path=ridge_path(training,n_cov,alphas,penalty_multiplier=.5)
            for alpha in alphas:
                mse=moment_mse(validation,path['coefficients'][alpha])
                scores.append(dict(alpha=alpha,sse=mse*validation.n,n=validation.n))
                inner_records.append(dict(outer_fold=k,inner_fold=j,alpha=alpha,mean_acr_mse=mse))
        chosen,_=choose_alpha(scores,alphas)
        training=add_moments([m for index,m in enumerate(outer_parts) if index!=k])
        path=ridge_path(training,n_cov,[chosen],penalty_multiplier=.5)
        value=design[va]@path['coefficients'][chosen]
        ancestry=ancestry_projection(y,p,tr)
        fitted=value[:,None]+ancestry['effect'][None,:]
        scores=score_arrays(y[va],fitted);total={key:total.get(key,0)+v for key,v in scores.items()}
        prediction[va]=value
        records.append(dict(model=model_name,outer_fold=k,selected_alpha=chosen,**summarize_totals(scores)))
        np.savez_compressed(out/f'{prefix}_outer{k}.npz',coefficient=path['coefficients'][chosen],
                            ancestry_effect=ancestry['effect'],selected_alpha=np.nan if chosen is None else chosen)
        effects.extend(dict(outer_fold=k,genotype=g,centered_ancestry_contribution=float(v)) for g,v in zip(
            json.loads((out/'input_manifest.json').read_text())['genotype_order'],ancestry['effect']))
        print(f'{model_name} ridge outer {k+1}: pair RMSE {records[-1]["rmse"]:.4f}',flush=True)
    metric=summarize_totals(total);mse=float(np.mean((prediction-target)**2))
    metric.update(model=model_name,mean_acr_rmse=float(np.sqrt(mse)),
                  mean_acr_r2=1-mse/float(np.var(target)),
                  mean_acr_pearson_r=float(np.corrcoef(prediction,target)[0,1]))
    pd.DataFrame([metric]).to_csv(out/f'{prefix}_metrics.csv',index=False)
    pd.DataFrame(records).to_csv(out/f'{prefix}_folds.csv',index=False)
    pd.DataFrame(inner_records).to_csv(out/f'{prefix}_inner_scores.csv',index=False)
    pd.DataFrame(effects).to_csv(out/f'{prefix}_genotype_contributions.csv',index=False)
    np.save(out/f'{prefix}_mean_predictions.npy',prediction)
    import hashlib
    files=[cache/'receipt.json',cache/'context.npy',cache/'response.npy',out/'fold_manifest.json',Path(__file__)]
    if include_core:files.append(cache/'core.npy')
    hashes={}
    for file in files:
        with file.open('rb') as handle:hashes[str(file.relative_to(root))]=hashlib.file_digest(handle,'sha256').hexdigest()
    (out/f'{prefix}_manifest.json').write_text(json.dumps(dict(model=model_name,inputs_and_code=hashes,
        include_core=include_core,selection='Frozen three inner block folds; mean-label loss exactly profiles full balanced pair loss'),indent=2)+'\n')
    print(metric,flush=True)
    return metric
