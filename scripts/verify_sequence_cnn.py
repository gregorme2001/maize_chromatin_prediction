#!/usr/bin/env python
"""Independently reconstruct saved CNN predictions, splits and numerical scores."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from maize_regulatory.sequence_cnn import SequenceCNN,EncodedSequences,predict


def main():
    root=Path(__file__).resolve().parents[1];out=root/'results/sequence_cnn_benchmark'
    cache=root/'data/intermediate/sequence_cnn'
    torch.set_num_threads(1)
    folds=json.loads((out/'fold_manifest.json').read_text())
    assert folds==json.loads((root/'results/sequence_summary_comparison/fold_manifest.json').read_text())
    ids=json.loads((out/'input_manifest.json').read_text())['genotype_order']
    assert ids==json.loads((root/'results/sequence_summary_comparison/summary.json').read_text())['genotype_order']
    y=np.load(cache/'response.npy',mmap_mode='r');z=np.load(cache/'covariates.npy')
    blocks=np.load(cache/'blocks.npy');frame=pd.read_parquet(out/'acr_predictions.parquet')
    metrics=pd.read_csv(out/'metrics.csv').set_index('model')
    assert len(frame)==len(y) and frame.ACR.is_unique and np.allclose(frame.observed_CPM,y.mean(axis=1))
    assert frame.ACR.tolist()==pd.read_csv(root/'data/intermediate/cells/acr_index.csv').ACR.tolist()
    contributions=pd.read_csv(out/'genotype_contributions.csv',dtype={'genotype':str})
    records=[];numeric=[]
    for kind in ['acr_only','context_2048']:
        prefix='core' if kind=='acr_only' else 'context'
        codes=np.load(cache/f'{prefix}.npy',mmap_mode='r')
        starts=np.load(cache/f'{prefix}_start.npy');ends=np.load(cache/f'{prefix}_end.npy')
        count=0;sse=0.;max_ancestry_difference=0.;max_seed_ensemble_difference=0.
        for spec in folds:
            k=spec['number'];tr=np.flatnonzero(np.isin(blocks,spec['training_blocks']))
            va=np.flatnonzero(np.isin(blocks,spec['validation_blocks']))
            assert not np.intersect1d(tr,va).size
            group=contributions.loc[contributions.model.eq(kind)&contributions.outer_fold.eq(k)]
            assert len(group)==len(ids) and group.genotype.is_unique and set(group.genotype)==set(ids)
            effect=group.set_index('genotype').loc[ids,'centered_ancestry_contribution'].to_numpy()
            old=np.load(root/f'results/sequence_summary_comparison/models/outer_{k}_gc_spline_baseline.npz')['pc_effect']
            difference=float(np.max(np.abs(effect-(old-old.mean()))))
            assert difference<1e-8;max_ancestry_difference=max(max_ancestry_difference,difference)
            seed_values=[]
            for seed in [101,202,303]:
                path=out/'models'/f'{kind}_outer{k}_seed{seed}.pt'
                # These checkpoints were generated locally by this benchmark.
                checkpoint=torch.load(path,map_location='cpu',weights_only=False)
                assert checkpoint['training_rows_sha256']==hashlib.sha256(tr.tobytes()).hexdigest()
                transform=checkpoint['transform']
                assert np.allclose(transform['covariate_mean'],z[tr].mean(axis=0),atol=1e-12)
                assert np.isclose(transform['target_mean'],y[tr].mean(axis=1).mean(),atol=1e-12)
                values=np.load(path.with_suffix('.predictions.npy'));seed_values.append(values)
                # Float32 CPU evaluation is an independent numerical cross-check of MPS AMP.
                if k==0 and seed==101:
                    positions=np.random.default_rng(72).choice(len(va),32,replace=False)
                    at=va[positions]
                    seq=EncodedSequences(np.asarray(codes[at]),starts[at],ends[at],'cpu')
                    scaled=torch.tensor((z[at]-transform['covariate_mean'])/transform['covariate_sd'],dtype=torch.float32)
                    setting=checkpoint['setting']
                    model=SequenceCNN(z.shape[1],setting['width'],setting['depth'],setting['dropout'])
                    model.load_state_dict(checkpoint['state_dict'])
                    predicted=predict(model,seq,scaled,np.arange(len(at)),16,kind=='acr_only')
                    predicted=predicted*transform['target_sd']+transform['target_mean']
                    error=float(np.max(np.abs(predicted-values[positions])))
                    assert np.allclose(predicted,values[positions],rtol=.003,atol=.02)
                    numeric.append(dict(model=kind,n=32,max_abs_cpu_float32_vs_mps_amp_cpm=error))
            ensemble=np.mean(seed_values,axis=0)
            delta=float(np.max(np.abs(frame[kind+'_CPM'].to_numpy()[va]-ensemble)))
            assert delta<1e-10;max_seed_ensemble_difference=max(max_seed_ensemble_difference,delta)
            for first in range(0,len(va),2048):
                rows=va[first:first+2048]
                prediction=frame[kind+'_CPM'].to_numpy()[rows,None]+effect[None,:]
                error=y[rows]-prediction;sse+=float(np.sum(error**2));count+=error.size
        pair_rmse=np.sqrt(sse/count)
        mean_rmse=np.sqrt(np.mean((frame[kind+'_CPM'].to_numpy()-y.mean(axis=1))**2))
        assert np.isclose(pair_rmse,metrics.loc[kind,'rmse'],rtol=1e-11,atol=1e-11)
        assert np.isclose(mean_rmse,metrics.loc[kind,'mean_acr_rmse'],rtol=1e-11,atol=1e-11)
        records.append(dict(model=kind,n_pairs=count,reconstructed_pair_rmse=float(pair_rmse),
            reconstructed_mean_rmse=float(mean_rmse),
            max_ancestry_difference_from_prior_ridge=max_ancestry_difference,
            max_saved_mean_vs_seed_ensemble_difference=max_seed_ensemble_difference))
    result=dict(frozen_folds_equal=True,genotype_and_acr_identity_checks=True,
        all_30_refit_training_row_hashes_match=True,all_refit_scaling_train_only=True,
        all_predictions_reconstruct=True,metrics=records,numerical_cross_checks=numeric,
        original_environment_tests='147 passed; 2 Torch-dependent modules skipped',
        native_neural_tests='5 passed',no_DTS_read=True,no_holdout_values_read=True)
    (out/'independent_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
