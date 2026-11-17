#!/usr/bin/env python
"""Run matched ACR-only and 2048-bp residual CNNs with frozen nested block CV."""

import argparse
from datetime import datetime,timezone
import gc
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from maize_regulatory.cohort import cohort_metadata
from maize_regulatory.reference_sequence import prepare_reference,digest
from maize_regulatory.pooled_sequence import blocked_manifest
from maize_regulatory.reference_sequence_evaluation import score_arrays,summarize_totals
from maize_regulatory.sequence_cnn import (encode_inputs,EncodedSequences,fit_network,predict,
    ancestry_projection,SequenceCNN,synchronize)


def save_json(path,value):
    temporary=path.with_suffix(path.suffix+'.part')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def prepare(root,out,cache):
    config=json.loads((root/'configs/reference_sequence_comparison.json').read_text())
    data=prepare_reference(root,root/'data/intermediate/reference_cortex',config)
    folds=blocked_manifest(data['coordinates'].block,5,3,42)
    frozen=json.loads((root/'results/sequence_summary_comparison/fold_manifest.json').read_text())
    if folds!=frozen:raise ValueError('CNN folds differ from frozen sequence comparison')
    cache.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=True)
    paths=[root/p for p in ['data/maize_acr_sequences.parquet','data/maize_ancestry_pcs.parquet',
        'data/intermediate/Zm-B73-REFERENCE-NAM-5.0.fa','data/intermediate/Zm-B73-REFERENCE-NAM-5.0.fa.fai',
        'data/intermediate/reference_cortex/receipt.json','results/sequence_summary_comparison/fold_manifest.json']]
    source=dict(inputs={str(p.relative_to(root)):digest(p) for p in paths},
                genotype_order=data['genotypes'].tolist(),n_acrs=len(data['acrs']))
    if (cache/'receipt.json').exists():
        if json.loads((cache/'receipt.json').read_text())['source']!=source:
            raise ValueError('Sequence cache source inputs changed')
    else:
        print('Encoding original ACRs and indexed B73 context; validating every core',flush=True)
        encoded=encode_inputs(data,root/'data/intermediate/Zm-B73-REFERENCE-NAM-5.0.fa')
        for name,value in encoded.items():
            if name!='audit':np.save(cache/f'{name}.npy',value)
        save_json(cache/'receipt.json',dict(source=source,audit=encoded['audit'],
            transform='Fixed sequence-only encoding; no response-derived selection',
            encoder_code_sha256=digest(root/'src/maize_regulatory/sequence_cnn.py')))
        del encoded
    basis=data['bases']['gc_spline'];basis=basis[basis<data['region'].shape[1]]
    np.save(cache/'covariates.npy',data['region'][:,basis])
    np.save(cache/'pcs.npy',data['pcs']);np.save(cache/'response.npy',data['y'])
    np.save(cache/'blocks.npy',data['coordinates'].block.to_numpy())
    data['coordinates'].to_csv(out/'acr_coordinates.csv')
    data['cohort'].to_csv(out/'genotype_coverage.csv',index=False)
    pd.DataFrame({'ACR':data['acrs'],'observed_CPM':data['y'].mean(axis=1)}).to_parquet(
        out/'observed_means.parquet',index=False)
    save_json(out/'fold_manifest.json',folds);save_json(out/'input_manifest.json',source)
    save_json(out/'sequence_audit.json',json.loads((cache/'receipt.json').read_text())['audit'])
    print(f'Prepared {len(data["acrs"]):,} ACRs × {len(data["genotypes"])} development genotypes',flush=True)


def load(cache,out):
    arrays={name:np.load(cache/f'{name}.npy',mmap_mode='r') for name in ['covariates','pcs','response','blocks']}
    arrays['folds']=json.loads((out/'fold_manifest.json').read_text())
    arrays['target']=arrays['response'].mean(axis=1)
    return arrays


def sequences(cache,kind,device):
    prefix='core' if kind=='acr_only' else 'context'
    return EncodedSequences(np.load(cache/f'{prefix}.npy',mmap_mode='r'),
        np.load(cache/f'{prefix}_start.npy'),np.load(cache/f'{prefix}_end.npy'),device)


def benchmark(cache,out,device,batch):
    data=load(cache,out);first=data['folds'][0]['inner'][0]
    train=np.flatnonzero(np.isin(data['blocks'],first['training_blocks']))
    validation=np.flatnonzero(np.isin(data['blocks'],first['validation_blocks']))
    setting=dict(width=32,depth=2,dropout=.1,learning_rate=.003,l2=1e-5)
    rows=[]
    for kind in ['acr_only','context_2048']:
        seq=sequences(cache,kind,device)
        print(f'Timing two training epochs: {kind}',flush=True)
        model,z,transform,info=fit_network(seq,data['covariates'],data['target'],train,validation,
            setting,seed=42,core_only=kind=='acr_only',max_epochs=2,patience=3,batch_size=batch)
        rows.append(dict(kind=kind,seconds_per_epoch=info['seconds']/2,n_parameters=info['n_parameters']))
        print(rows[-1],flush=True)
        del seq,model,z;gc.collect()
        if device=='mps':torch.mps.empty_cache()
    save_json(out/'timing_benchmark.json',dict(device=device,batch_size=batch,rows=rows,
        role='Training-only runtime planning on first inner split; no outer outcomes examined'))


def run(cache,out,root,config,device):
    data=load(cache,out);blocks=data['blocks'];y=data['response'];target=data['target']
    manifest=json.loads((out/'input_manifest.json').read_text())
    state=dict(config=config,device=device,torch=torch.__version__,
        code_sha256={str(p.relative_to(root)):digest(p) for p in [Path(__file__),
            root/'src/maize_regulatory/sequence_cnn.py']})
    if (out/'run_config.json').exists() and json.loads((out/'run_config.json').read_text())!=state:
        raise ValueError('Resume code or training configuration changed')
    save_json(out/'run_config.json',state)
    (out/'inner').mkdir(exist_ok=True);(out/'models').mkdir(exist_ok=True)
    means=pd.read_parquet(out/'observed_means.parquet')
    mean_prediction={};genotype_effects=[];all_metrics=[];fold_records=[]
    start=time.monotonic()
    for kind in ['acr_only','context_2048']:
        seq=sequences(cache,kind,device);core=kind=='acr_only'
        predicted_mean=np.full(len(y),np.nan);totals={}
        for spec in data['folds']:
            outer=spec['number'];train=np.flatnonzero(np.isin(blocks,spec['training_blocks']))
            validation=np.flatnonzero(np.isin(blocks,spec['validation_blocks']))
            if np.intersect1d(train,validation).size:raise AssertionError('Outer split overlap')
            candidate_scores=[]
            for h,setting in enumerate(config['candidates']):
                scores=[]
                for j,part in enumerate(spec['inner']):
                    path=out/'inner'/f'{kind}_outer{outer}_candidate{h}_inner{j}.json'
                    if path.exists():info=json.loads(path.read_text())
                    else:
                        tr=np.flatnonzero(np.isin(blocks,part['training_blocks']))
                        va=np.flatnonzero(np.isin(blocks,part['validation_blocks']))
                        if (np.intersect1d(tr,va).size or np.intersect1d(va,validation).size
                                or np.intersect1d(tr,validation).size):raise AssertionError('Inner split overlap')
                        print(f'{kind} outer {outer+1}/5 candidate {h+1} inner {j+1}/3',flush=True)
                        model,z,transform,info=fit_network(seq,data['covariates'],target,tr,va,setting,
                            seed=42+outer*100+j,core_only=core,max_epochs=config['max_epochs'],
                            patience=config['patience'],batch_size=config['batch_size'])
                        value=predict(model,seq,z,va,config['batch_size'],core)*transform['target_sd']+transform['target_mean']
                        ancestry=ancestry_projection(y,data['pcs'],tr)
                        error=y[va]-value[:,None]-ancestry['effect'][None,:]
                        info.update(sse=float(np.sum(error**2)),n=int(error.size),setting=setting)
                        save_json(path,info)
                        del model,z;gc.collect()
                    scores.append(info)
                candidate_scores.append(dict(candidate=h,sse=sum(v['sse'] for v in scores),
                    n=sum(v['n'] for v in scores),best_epochs=[v['best_epoch'] for v in scores]))
            selected=min(candidate_scores,key=lambda v:(v['sse']/v['n'],-config['candidates'][v['candidate']]['l2']))
            h=selected['candidate'];setting=config['candidates'][h]
            epochs=max(1,int(np.median(selected['best_epochs'])))
            save_json(out/'models'/f'{kind}_outer{outer}_selection.json',dict(
                candidates=candidate_scores,selected_candidate=h,refit_epochs=epochs,
                selection='Pooled inner out-of-fold pair MSE; stronger L2 breaks exact ties'))
            ancestry=ancestry_projection(y,data['pcs'],train)
            seed_predictions=[]
            for seed in config['ensemble_seeds']:
                checkpoint=out/'models'/f'{kind}_outer{outer}_seed{seed}.pt'
                prediction_path=checkpoint.with_suffix('.predictions.npy')
                if checkpoint.exists() and prediction_path.exists():
                    value=np.load(prediction_path)
                else:
                    print(f'{kind} outer {outer+1}/5 refit seed {seed}, {epochs} epochs',flush=True)
                    model,z,transform,info=fit_network(seq,data['covariates'],target,train,None,setting,
                        seed=seed+outer*1000,core_only=core,fixed_epochs=epochs,batch_size=config['batch_size'])
                    value=predict(model,seq,z,validation,config['batch_size'],core)*transform['target_sd']+transform['target_mean']
                    np.save(prediction_path,value)
                    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                        transform=transform,setting=setting,info=info,ancestry=ancestry,
                        training_rows_sha256=__import__('hashlib').sha256(train.tobytes()).hexdigest()),checkpoint)
                    del model,z;gc.collect()
                seed_predictions.append(value)
            value=np.mean(seed_predictions,axis=0);predicted_mean[validation]=value
            predicted=value[:,None]+ancestry['effect'][None,:]
            direct=score_arrays(y[validation],predicted)
            totals={k:totals.get(k,0)+v for k,v in direct.items()}
            score=summarize_totals(direct)
            fold_records.append(dict(model=kind,outer_fold=outer,selected_candidate=h,refit_epochs=epochs,**score))
            genotype_effects.extend(dict(model=kind,outer_fold=outer,genotype=g,centered_ancestry_contribution=float(v))
                for g,v in zip(manifest['genotype_order'],ancestry['effect']))
            print(f'{kind} outer {outer+1}: pair RMSE {score["rmse"]:.4f}; '
                  f'ACR-mean RMSE {np.sqrt(np.mean((value-target[validation])**2)):.4f}',flush=True)
            means[kind+'_CPM']=predicted_mean
            means.to_parquet(out/'acr_predictions.parquet',index=False)
            pd.DataFrame(fold_records).to_csv(out/'fold_results.csv',index=False)
        if not np.isfinite(predicted_mean).all():raise AssertionError('Missing out-of-fold predictions')
        metric=summarize_totals(totals);mean_mse=float(np.mean((predicted_mean-target)**2))
        metric.update(model=kind,mean_acr_rmse=float(np.sqrt(mean_mse)),
            mean_acr_r2=1-mean_mse/float(np.var(target)),
            mean_acr_pearson_r=float(np.corrcoef(target,predicted_mean)[0,1]))
        all_metrics.append(metric);mean_prediction[kind]=predicted_mean
        pd.DataFrame(all_metrics).to_csv(out/'metrics.csv',index=False)
        del seq;gc.collect()
        if device=='mps':torch.mps.empty_cache()
    pd.DataFrame(genotype_effects).to_csv(out/'genotype_contributions.csv',index=False)
    summary=dict(completed_utc=datetime.now(timezone.utc).isoformat(),cohort=cohort_metadata(),
        n_acrs=len(y),n_genotypes=y.shape[1],n_pairs=int(y.size),metrics=all_metrics,
        genotype_order=manifest['genotype_order'],evaluation_seconds=time.monotonic()-start,
        config=config,device=device,no_DTS_read=True,no_holdout_values_used=True,
        old_results_changed=False,previous_phenotype_CV_changed=False,
        folds_identical_to_reference=True,
        objective='Exact balanced pair MSE; global ancestry profiled analytically; region covariates and CNN fitted jointly',
        early_stopping='Inner validation only; refit epoch budget is median best epoch of winning inner fits; outer values never tune',
        sequence_audit=json.loads((out/'sequence_audit.json').read_text()))
    save_json(out/'summary.json',summary)
    from maize_regulatory.sequence_cnn_report import write_report
    write_report(out,root)
    print(pd.DataFrame(all_metrics).to_string(index=False),flush=True)
    print(f'Open {out/"report.html"}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('results/sequence_cnn_benchmark'))
    parser.add_argument('--cache-dir',type=Path,default=Path('data/intermediate/sequence_cnn'))
    parser.add_argument('--stage',choices=['prepare','benchmark','run','report'],default='run')
    parser.add_argument('--max-epochs',type=int,default=8)
    parser.add_argument('--batch-size',type=int,default=2048)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    out=args.output_dir.resolve();cache=args.cache_dir.resolve()
    if root not in out.parents or root not in cache.parents:raise ValueError('Use project-local paths')
    torch.set_num_threads(4)
    device='mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Torch {torch.__version__}; device={device}',flush=True)
    if args.stage=='prepare':prepare(root,out,cache);return
    if args.stage=='report':
        from maize_regulatory.sequence_cnn_report import write_report
        write_report(out,root);return
    if not (cache/'receipt.json').exists():prepare(root,out,cache)
    if args.stage=='benchmark':benchmark(cache,out,device,args.batch_size);return
    config=dict(max_epochs=args.max_epochs,patience=3,batch_size=args.batch_size,
        ensemble_seeds=[101,202,303],candidates=[dict(width=32,depth=2,dropout=.1,
            learning_rate=.003,l2=l2) for l2 in [1e-5,1e-4]],
        outer_splits=5,inner_splits=3,sequence_kinds=['acr_only','context_2048'],
        planning_note='Bounded preliminary GPU benchmark; compact network and finite epoch cap to meet requested runtime; no outer-directed selection')
    run(cache,out,root,config,device)


if __name__=='__main__':main()
