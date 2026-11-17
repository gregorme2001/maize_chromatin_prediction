"""Evaluate existing out-of-fold CNN predictions; never train or select a model."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, html, json
import numpy as np
import pandas as pd
from maize_regulatory.sampled_linear import prediction_metrics

root = Path.cwd()
old = root/'results/sequence_cnn_benchmark'
current = root/'results/sampled_cnn_benchmark'
linear = root/'results/sampled_linear_benchmark'
cache = root/'data/intermediate/sequence_cnn'
out = root/'results/cnn_saved_prediction_comparison'
used = []
def read_json(path):
    used.append(path)
    return json.loads(path.read_text())
def read_frame(path, **kwargs):
    used.append(path)
    return pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path, **kwargs)
def read_array(path):
    used.append(path)
    return np.load(path, mmap_mode='r')

a = read_json(old/'fold_manifest.json')
b = read_json(current/'fold_manifest.json')
assert a == b, 'Frozen fold manifests differ'
obs = read_frame(current/'observations.parquet')
pd.testing.assert_frame_equal(obs, read_frame(linear/'observations.parquet'))
assert not obs.duplicated(['ACR','genotype']).any()
np.testing.assert_array_equal(obs.pair_index, np.arange(len(obs)))
source = read_frame(old/'acr_predictions.parquet')
assert source.ACR.is_unique
acr_rows = pd.Index(source.ACR).get_indexer(obs.ACR)
assert (acr_rows >= 0).all(), 'Unmatched ACR'
genotypes = read_json(old/'input_manifest.json')['genotype_order']
genotype_rows = pd.Index(genotypes).get_indexer(obs.genotype)
assert (genotype_rows >= 0).all(), 'Unmatched genotype'
y = obs.observed_CPM.to_numpy()
original_y = read_array(cache/'response.npy')
np.testing.assert_allclose(original_y[acr_rows,genotype_rows], y, rtol=0, atol=1e-12)
blocks = read_array(cache/'blocks.npy')
fold_ids = obs.outer_fold.to_numpy()
old_fold = np.full(len(source), -1, dtype=int)
for spec in a:
    train = np.isin(blocks, spec['training_blocks'])
    val = np.isin(blocks, spec['validation_blocks'])
    assert not (train & val).any()
    assert np.all(old_fold[val] == -1)
    old_fold[val] = spec['number']
    keep = fold_ids == spec['number']
    assert val[acr_rows[keep]].all()
    assert not train[acr_rows[keep]].any()
assert (old_fold >= 0).all()
np.testing.assert_array_equal(old_fold[acr_rows], fold_ids)

contributions = read_frame(old/'genotype_contributions.csv', dtype={'genotype':str})
assert not contributions.duplicated(['model','outer_fold','genotype']).any()
keys = pd.MultiIndex.from_arrays([fold_ids,obs.genotype], names=['outer_fold','genotype'])
old_predictions = {}
ensemble_errors = {}
for kind in ['acr_only','context_2048']:
    for spec in a:
        k = spec['number']
        val = np.flatnonzero(old_fold == k)
        ensemble = np.mean([read_array(old/'models'/f'{kind}_outer{k}_seed{seed}.predictions.npy')
                            for seed in [101,202,303]], axis=0)
        np.testing.assert_allclose(ensemble,source[kind+'_CPM'].to_numpy()[val],rtol=0,atol=1e-10)
        ensemble_errors[f'{kind}_outer{k}'] = float(np.max(np.abs(ensemble-source[kind+'_CPM'].to_numpy()[val])))
    frame = contributions.loc[contributions.model.eq(kind)].set_index(['outer_fold','genotype'])
    effect = frame.reindex(keys).centered_ancestry_contribution.to_numpy()
    assert np.isfinite(effect).all()
    predicted = source[kind+'_CPM'].to_numpy()[acr_rows] + effect
    assert np.isfinite(predicted).all()
    old_predictions['earlier_CNN_'+kind] = predicted

new_pred = read_array(current/'predictions.npy')
config = read_json(current/'run_config.json')
panels = config.get('panels') or config.get('config',{}).get('panels')
assert panels is not None
assert new_pred.shape == (len(obs),len(panels))
lin_pred = read_array(linear/'predictions.npy')
lin_names = read_json(linear/'fit_completion.json')['models']
assert lin_pred.shape == (len(obs),len(lin_names))
all_predictions = {**old_predictions,
    **{'current_CNN_'+name:np.asarray(new_pred[:,j]) for j,name in enumerate(panels)},
    **{'current_ridge_'+name:np.asarray(lin_pred[:,j]) for j,name in enumerate(lin_names)}}
sample = read_frame(linear/'selected_acrs.parquet').reset_index()
np.testing.assert_array_equal(sample.sample_acr_index, np.arange(len(sample)))
acr = obs.sample_acr_index.to_numpy()
assert sample.ACR.is_unique
np.testing.assert_array_equal(sample.ACR.to_numpy()[acr],obs.ACR)
rows, fold_rows = [], []
for model,pred in all_predictions.items():
    for subgroup in ['all','core_variation_rich','context_variation_rich']:
        keep = np.ones(len(obs),bool) if subgroup == 'all' else sample[subgroup].to_numpy()[acr]
        result = prediction_metrics(y[keep],pred[keep],acr[keep])
        independent = float(np.sqrt(np.sum((y[keep]-pred[keep])**2)/keep.sum()))
        assert abs(independent-result['pair_rmse']) < 1e-12
        rows.append(dict(model=model,subgroup=subgroup,**result))
    for k in range(5):
        keep = fold_ids == k
        fold_rows.append(dict(model=model,outer_fold=k,**prediction_metrics(y[keep],pred[keep],acr[keep])))
metrics = pd.DataFrame(rows)
fold_metrics = pd.DataFrame(fold_rows)
for folder,prefix in [(current,'current_'),(linear,'current_ridge_')]:
    saved = read_frame(folder/'metrics.csv')
    for r in saved.to_dict('records'):
        model = prefix+r['model'] if folder == linear else 'current_'+r['model']
        actual = metrics.loc[(metrics.model==model)&(metrics.subgroup==r['subgroup'])].iloc[0]
        for field in ['pair_rmse','ACR_mean_rmse','within_ACR_r2']:
            assert abs(actual[field]-r[field]) < 1e-10

main_names = ['current_ridge_baseline','current_ridge_B73_core_TF','current_ridge_personalized_core_TF',
    'earlier_CNN_acr_only','earlier_CNN_context_2048',
    'current_CNN_B73_core','current_CNN_personalized_core','current_CNN_B73_context','current_CNN_personalized_context']
labels = ['Current covariates only','Current TF ridge: B73 core','Current TF ridge: personalized core',
    'Earlier CNN: B73 core','Earlier CNN: B73 2,048 bp',
    'Current CNN: B73 core','Current CNN: personalized core','Current CNN: B73 2,000 bp','Current CNN: personalized 2,000 bp']
main = metrics.loc[metrics.subgroup=='all'].set_index('model').loc[main_names].reset_index()
main.insert(1,'description',labels)
main['MSE_reduction_vs_current_baseline_percent'] = 100*(1-main.pair_mse/main.pair_mse.iloc[0])
notes = [
    'No model was trained, tuned, refitted, or selected in this comparison. No phenotype or permanent-holdout values were read.',
    f'Every row evaluates the same {len(obs):,} ACR-genotype observations at {obs.ACR.nunique():,} ACRs and {obs.genotype.nunique()} development genotypes. The response is unchanged original-core, library-pooled cortex CPM.',
    'Earlier predictions are saved outer-held-out predictions. ACRs and genotypes were explicitly matched by name. Frozen outer and inner fold manifests are identical; each evaluation ACR was absent from its corresponding earlier outer training fold. Recorded observed CPM agrees to 1e-12.',
    'Earlier predictions equal the saved three-seed ACR prediction plus that outer fold\'s genotype-specific centered ancestry contribution. All ten saved ensembles were reconstructed from their three saved seed prediction arrays.',
    'This equalizes evaluation observations, not training data or architectures: earlier networks trained on roughly 108,843 ACRs across all folds, versus 9,666 in the current sample. Earlier models use B73 DNA, 2,048-bp context, an ACR-location channel, eight positional summary bins and three-seed averaging. Current models use 2,000-bp context, global mean/max pooling and one seed. Eligibility, available genotypes per ACR, learning rates and other settings also differ.',
    'The comparison therefore cannot attribute a performance difference specifically to architecture, training-set size, or personalization. It is a retrospective comparison of completed procedures, not independent confirmation after model selection.',
    'Observation RMSE averages squared errors over all available ACR-genotype observations. ACR-average RMSE averages observed and predicted CPM over the identical available genotypes at each ACR, then weights ACRs equally. Lower is better. Within-ACR R2 measures prediction of genotype deviations after centering each ACR for evaluation only.',
]
columns = ['description','pair_rmse','ACR_mean_rmse','ACR_mean_r2','within_ACR_r2']
shown = main[columns].rename(columns={'description':'Model','pair_rmse':'Observation RMSE','ACR_mean_rmse':'ACR-average RMSE','ACR_mean_r2':'ACR-average R2','within_ACR_r2':'Within-ACR R2'})
lines = ['# Earlier CNN predictions on the current evaluation sample','']
lines += [n+'\n' for n in notes[:4]]
lines += ['| '+' | '.join(shown.columns)+' |','|'+ '|'.join(['---']+['---:']*4)+'|']
for row in shown.itertuples(index=False,name=None):
    lines.append('| '+row[0]+' | '+' | '.join(f'{v:.6f}' for v in row[1:])+' |')
lines += ['']+[n+'\n' for n in notes[4:]]
md='\n'.join(lines)
out.mkdir(exist_ok=True)
metrics.to_csv(out/'metrics.csv',index=False)
fold_metrics.to_csv(out/'fold_metrics.csv',index=False)
main.to_csv(out/'main_comparison.csv',index=False)
np.savez_compressed(out/'earlier_predictions_on_current_observations.npz',
    pair_index=obs.pair_index.to_numpy(),acr_only=old_predictions['earlier_CNN_acr_only'],
    context_2048=old_predictions['earlier_CNN_context_2048'])
(out/'report.md').write_text(md+'\n')
body='<h1>Earlier CNN predictions on the current evaluation sample</h1>'
body+=''.join('<p>'+html.escape(n)+'</p>' for n in notes[:4])
body+=shown.to_html(index=False,float_format=lambda v:f'{v:.6f}',border=0)
body+=''.join('<p>'+html.escape(n)+'</p>' for n in notes[4:])
(out/'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Saved CNN comparison</title><style>body{font:17px/1.5 system-ui;max-width:1200px;margin:35px auto;padding:0 20px}table{border-collapse:collapse;font-size:15px}th,td{padding:8px 12px;border-bottom:1px solid #ddd;text-align:right}th:first-child,td:first-child{text-align:left}h1{font-size:27px}</style>'+body+'</html>')
receipt=dict(created_utc=datetime.now(timezone.utc).isoformat(),n_observations=len(obs),n_ACRs=obs.ACR.nunique(),
    n_genotypes=obs.genotype.nunique(),all_evaluation_observations_matched=True,fold_manifests_identical=True,
    all_evaluation_ACRs_absent_from_corresponding_earlier_training_fold=True,all_CPM_values_match=True,
    ensemble_max_abs_errors=ensemble_errors,current_metrics_reproduced=True,models_fitted=0,
    holdout_or_DTS_read=False,notes=notes,
    input_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(used))})
(out/'audit.json').write_text(json.dumps(receipt,indent=2)+'\n')
(out/'analysis_source.txt').write_text(Path(__file__).read_text())
print(shown.to_string(index=False),flush=True)
print('All checks passed; outputs:',out,flush=True)
print('Per-fold earlier context vs current baseline, current B73 context:')
print(fold_metrics.loc[fold_metrics.model.isin(['earlier_CNN_context_2048','current_ridge_baseline','current_CNN_B73_context']),['model','outer_fold','pair_rmse','ACR_mean_r2']].to_string(index=False),flush=True)
