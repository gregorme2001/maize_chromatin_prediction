"""Read-only checkpoint/split audit, with a new receipt; no model fitting."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json
import numpy as np,pandas as pd,torch
root=Path.cwd();old=root/'results/sequence_cnn_benchmark';current=root/'results/sampled_cnn_benchmark';cache=root/'data/intermediate/sequence_cnn';out=root/'results/cnn_saved_prediction_comparison'
folds=json.loads((old/'fold_manifest.json').read_text())
assert folds==json.loads((current/'fold_manifest.json').read_text())
source=pd.read_parquet(old/'acr_predictions.parquet');obs=pd.read_parquet(current/'observations.parquet')
assert source.ACR.is_unique and not obs.duplicated(['ACR','genotype']).any()
positions=pd.Index(source.ACR).get_indexer(obs.ACR);assert (positions>=0).all()
blocks=np.load(cache/'blocks.npy');assert len(blocks)==len(source)
records=[];assigned=np.zeros(len(obs),dtype=int);checked=0
for spec in folds:
 k=spec['number'];train=np.flatnonzero(np.isin(blocks,spec['training_blocks']));val=np.flatnonzero(np.isin(blocks,spec['validation_blocks']));assert not np.intersect1d(train,val).size
 keep=obs.outer_fold.to_numpy()==k;rows=positions[keep];unique=np.unique(rows)
 overlap=np.intersect1d(train,unique);assert len(overlap)==0
 assert np.isin(rows,val).all();assigned[keep]+=1
 for inner in spec['inner']:
  assert set(inner['training_blocks']).issubset(spec['training_blocks'])
  assert set(inner['validation_blocks']).issubset(spec['training_blocks'])
  assert not set(inner['training_blocks']).intersection(inner['validation_blocks'])
  assert not set(inner['training_blocks']).intersection(spec['validation_blocks'])
  assert not set(inner['validation_blocks']).intersection(spec['validation_blocks'])
 for kind in ['acr_only','context_2048']:
  values=[]
  for seed in [101,202,303]:
   path=old/'models'/f'{kind}_outer{k}_seed{seed}.pt'
   # Trusted locally generated checkpoint; load metadata/weights on CPU only.
   ck=torch.load(path,map_location='cpu',weights_only=False)
   assert ck['training_rows_sha256']==hashlib.sha256(train.tobytes()).hexdigest()
   v=np.load(path.with_suffix('.predictions.npy'));assert v.shape==(len(val),)
   values.append(v);checked+=1
  ensemble=np.mean(values,axis=0)
  np.testing.assert_allclose(ensemble,source[kind+'_CPM'].to_numpy()[val],atol=1e-10,rtol=0)
 records.append(dict(outer_fold=k+1,earlier_training_ACRs=len(train),earlier_held_out_ACRs=len(val),current_evaluation_ACRs=len(unique),current_evaluation_observations=int(keep.sum()),evaluation_ACRs_in_corresponding_training_set=len(overlap)))
assert np.all(assigned==1)
assert sum(r['current_evaluation_ACRs'] for r in records)==9666
assert sum(r['current_evaluation_observations'] for r in records)==495900
receipt=dict(checked_utc=datetime.now(timezone.utc).isoformat(),folds=records,checkpoint_training_hashes_verified=checked,saved_three_seed_predictions_reconstruct=True,inner_training_and_tuning_exclude_outer_evaluation_blocks=True,all_9666_evaluation_ACRs_have_no_overlap_with_their_prediction_models_training_set=True,evaluation_observation_count=495900,permanent_holdout_or_DTS_read=False,models_fitted=0)
(out/'training_overlap_recheck.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(pd.DataFrame(records).to_string(index=False));print('Checkpoint training hashes verified:',checked);print('All selected observations have exactly one held-out-fold assignment. All checks passed.')
