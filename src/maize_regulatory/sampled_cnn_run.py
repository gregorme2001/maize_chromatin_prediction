"""Resumable nested training on frozen matched ACR/genotype observations."""

from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch

from .sampled_cnn import (GroupedLoss, MatchedCNN, SequenceBank, Trainer,
    apply_transform, group_observations, predict_groups, training_transform)
from .sampled_cnn_precision import MixedTrainer, predict_mixed
from .sampled_linear import prediction_metrics


class PauseRequested(Exception):
    pass


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


def save_json(path, value):
    def convert(v):
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, np.generic): return v.item()
        raise TypeError(type(v).__name__)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path)+'.part')
    temporary.write_text(json.dumps(value, indent=2, default=convert, allow_nan=False)+'\n')
    temporary.replace(path)


class RunControl:
    """Execution-only controls; they never select scientific settings."""
    def __init__(self, out, stop_at, gentle_until=0, batch_pause=.05):
        self.out, self.stop_at = Path(out), stop_at
        self.gentle_until, self.batch_pause = gentle_until, batch_pause
        self.stop_reason = None; self.last_disk_check = 0; self.last_status = 0
        self.last_details = {}

    def tick(self):
        now = time.time()
        if self.stop_reason: raise PauseRequested(self.stop_reason)
        if now >= self.stop_at-45: raise PauseRequested('Scheduled 05:00 stop; checkpointing in advance')
        if (self.out/'STOP').exists(): raise PauseRequested('STOP file requested')
        if now-self.last_disk_check >= 30:
            self.last_disk_check = now
            if shutil.disk_usage(self.out).free < 768*1024**2:
                raise PauseRequested('Low disk space; preserving checkpoint')
        if now < self.gentle_until and self.batch_pause: time.sleep(self.batch_pause)

    def emit(self, **details):
        self.last_details.update(details)
        now = time.time()
        if now-self.last_status < 10 and not details.get('force'): return
        self.last_status = now
        value = dict(self.last_details, updated_utc=datetime.now(timezone.utc).isoformat(),
                     stop_at_utc=datetime.fromtimestamp(self.stop_at, timezone.utc).isoformat(),
                     gentle_mode=now < self.gentle_until)
        value.pop('force', None); save_json(self.out/'status.json', value)


def score_groups(model, bank, groups, transform, batch, control):
    if str(bank.device) == 'mps':
        values = predict_mixed(model, bank, groups['pattern_ids'], batch, between_batches=control.tick)
    else:
        control.tick(); values = predict_groups(model, bank, groups['pattern_ids'], batch)
    criterion = GroupedLoss(groups, bank.device)
    with torch.no_grad():
        loss = criterion.full(torch.as_tensor(values, device=bank.device), model.linear)
    score = float(loss.cpu())*transform['target_sd']**2
    if not np.isfinite(score): raise FloatingPointError('Nonfinite validation error')
    return score


def run_fit(out, fit_id, bank, a, y, pattern_ids, train_rows, validation_rows,
            setting, config, control, *, seed, fixed_epochs=None):
    """One fit, with exact continuation of partially completed epochs.

    Outer refits receive validation_rows=None. Their held-out responses are
    never consulted here. Inner fits select checkpoints/epochs only internally.
    """
    directory = Path(out)/'fits'/fit_id; directory.mkdir(parents=True, exist_ok=True)
    done, last, best = directory/'fit.json', directory/'last.pt', directory/'best.pt'
    if done.exists():
        record = json.loads(done.read_text())
        if digest(Path(out)/record['checkpoint']) != record['checkpoint_sha256']:
            raise ValueError('Completed checkpoint changed: '+fit_id)
        return record
    transform = training_transform(a, y, train_rows)
    z, yy = apply_transform(a[train_rows], y[train_rows], transform)
    groups = group_observations(z, yy, pattern_ids[train_rows])
    validation = None
    if validation_rows is not None:
        vz, vy = apply_transform(a[validation_rows], y[validation_rows], transform)
        validation = group_observations(vz, vy, pattern_ids[validation_rows])
    factory = MixedTrainer if str(bank.device) == 'mps' else Trainer
    trainer = factory(bank, groups, setting, seed=seed)
    state = dict(fit_id=fit_id, phase='train', order=None, offset=0, history=[],
                 best_score=None, best_epoch=0, stale=0, transform=transform,
                 fixed_epochs=fixed_epochs, seed=seed, elapsed_training_seconds=0.)
    if last.exists():
        state = trainer.restore(last)
        if (state['fit_id'] != fit_id or state['fixed_epochs'] != fixed_epochs
                or state['seed'] != seed):
            raise ValueError('Checkpoint fit identity differs')
        for key in transform: np.testing.assert_array_equal(transform[key], state['transform'][key])
    cap = fixed_epochs if fixed_epochs is not None else config['max_epochs']
    try:
        while state['phase'] != 'done':
            control.tick()
            if state['phase'] == 'train':
                if state['order'] is None:
                    state['order'] = trainer.rng.permutation(groups['n_patterns'])
                    state['offset'] = 0; state['epoch_seconds'] = 0.
                order = np.asarray(state['order'], int)
                while state['offset'] < len(order):
                    control.tick()
                    before = time.monotonic()
                    at = order[state['offset']:state['offset']+config['batch_size']]
                    trainer.step(at)
                    duration = time.monotonic()-before
                    state['epoch_seconds'] += duration
                    state['elapsed_training_seconds'] += duration
                    state['offset'] += len(at)
                    control.emit(phase='training', active_fit=fit_id, epoch=trainer.epoch+1,
                                 completed_patterns=state['offset'], total_patterns=len(order))
                trainer.epoch += 1
                state['order'] = None; state['offset'] = 0
                state['phase'] = 'validation' if validation is not None else 'finish_epoch'
                trainer.save(last, state)
            if state['phase'] == 'validation':
                score = score_groups(trainer.model, bank, validation, transform,
                                     config['batch_size'], control)
                improved = state['best_score'] is None or score < state['best_score']
                if improved:
                    state['best_score'], state['best_epoch'] = score, trainer.epoch
                if trainer.epoch <= config['warmup_epochs']:
                    state['stale'] = 0
                elif improved:
                    state['stale'] = 0
                else:
                    state['stale'] += 1
                state['history'].append(dict(epoch=trainer.epoch, validation_pair_mse=score,
                    training_seconds=state['epoch_seconds'], best_epoch=state['best_epoch'],
                    stale=state['stale'], skipped_updates=getattr(trainer, 'skipped_updates', 0)))
                state['phase'] = ('done' if trainer.epoch >= cap or state['stale'] >= config['patience']
                                  else 'train')
                if improved: trainer.save(best, state)
                trainer.save(last, state)
                print(f'{fit_id}: epoch {trainer.epoch}/{cap}, inner MSE {score:.6f}, '
                      f'best epoch {state["best_epoch"]}, patience {state["stale"]}', flush=True)
                save_json(directory/'progress.json', dict(epochs_run=trainer.epoch,
                    best_epoch=state['best_epoch'], history=state['history']))
            if state['phase'] == 'finish_epoch':
                state['history'].append(dict(epoch=trainer.epoch, training_seconds=state['epoch_seconds']))
                state['phase'] = 'done' if trainer.epoch >= cap else 'train'
                trainer.save(last, state)
                print(f'{fit_id}: outer training epoch {trainer.epoch}/{cap}', flush=True)
        chosen = best if validation is not None else last
        record = dict(fit_id=fit_id, role='inner' if validation is not None else 'outer',
            seed=seed, setting=setting, epochs_run=trainer.epoch,
            best_epoch=state['best_epoch'] if validation is not None else fixed_epochs,
            validation_pair_mse=state['best_score'],
            n_validation_observations=validation['n_observations'] if validation is not None else 0,
            n_training_observations=groups['n_observations'], n_training_patterns=groups['n_patterns'],
            history=state['history'], transform=transform,
            checkpoint=str(chosen.relative_to(out)), checkpoint_sha256=digest(chosen),
            elapsed_training_seconds=state['elapsed_training_seconds'],
            skipped_updates=getattr(trainer, 'skipped_updates', 0))
        save_json(done, record)
        return record
    except PauseRequested:
        trainer.save(last, state)
        save_json(directory/'paused.json', dict(epochs_completed=trainer.epoch,
            phase=state['phase'], completed_patterns_in_epoch=state['offset'],
            checkpoint_sha256=digest(last)))
        raise
    finally:
        del trainer
        gc.collect()
        if str(bank.device) == 'mps': torch.mps.empty_cache()


def choose_candidate(inner_records, alphas):
    candidates = []
    for j, alpha in enumerate(alphas):
        records = inner_records[j]
        total_n = sum(r['n_validation_observations'] for r in records)
        mse = sum(r['validation_pair_mse']*r['n_validation_observations'] for r in records)/total_n
        candidates.append(dict(candidate=j, l2=alpha, pooled_inner_mse=mse,
            best_epochs=[r['best_epoch'] for r in records], n_validation_observations=total_n))
    winner = min(candidates, key=lambda r: (r['pooled_inner_mse'], -r['l2']))
    return winner, candidates


def outer_predictions(bank, a, pattern_ids, rows, record, out, control, batch):
    """Only called after the candidate and refit duration are frozen and fitted."""
    state = torch.load(Path(out)/record['checkpoint'], map_location='cpu', weights_only=False)
    setting = record['setting']
    model = MatchedCNN(a.shape[1]-1, setting['width'], setting['depth'], setting['dropout']).to(bank.device)
    model.stem_pool = setting['stem_pool']; model.load_state_dict(state['model'])
    patterns, inverse = np.unique(pattern_ids[rows], return_inverse=True)
    if str(bank.device) == 'mps':
        sequence = predict_mixed(model, bank, patterns, batch, between_batches=control.tick)
    else:
        control.tick(); sequence = predict_groups(model, bank, patterns, batch)
    transform = record['transform']
    z = (a[rows, 1:]-np.asarray(transform['covariate_mean']))/np.asarray(transform['covariate_sd'])
    gamma = model.linear.weight.detach().cpu().numpy()[0]
    intercept = float(model.linear.bias.detach().cpu()[0])
    predicted = (sequence[inverse]+z@gamma+intercept)*transform['target_sd']+transform['target_mean']
    if not np.isfinite(predicted).all(): raise FloatingPointError('Nonfinite outer predictions')
    del model; gc.collect()
    return predicted


def summarize(out, config, observations, sample):
    """Never fills missing predictions or reports incomplete cohorts as matched."""
    out = Path(out)
    folds = [json.loads(p.read_text()) for p in sorted((out/'outer').glob('*.json'))]
    fits = [json.loads(p.read_text()) for p in sorted((out/'fits').glob('*/fit.json'))]
    complete = len(folds) == len(config['panels'])*config['outer_folds']
    status = dict(completed_inner_fits=sum(r['role'] == 'inner' for r in fits),
                  completed_outer_refits=len(folds), total_inner_fits=120, total_outer_refits=20,
                  complete=complete)
    save_json(out/'completion.json', status)
    if folds: pd.DataFrame(folds).to_csv(out/'fold_metrics.csv', index=False)
    learning = [dict(fit=r['fit_id'], **h) for r in fits for h in r['history']]
    if learning: pd.DataFrame(learning).to_csv(out/'learning_curves.csv', index=False)
    if complete:
        predicted = np.load(out/'predictions.npy', mmap_mode='r')
        if not np.isfinite(predicted).all(): raise ValueError('Completed run has missing predictions')
        rows = []
        for subgroup in ['all', 'core_variation_rich', 'context_variation_rich']:
            keep = (np.ones(len(observations), bool) if subgroup == 'all' else
                    sample[subgroup].to_numpy()[observations.sample_acr_index.to_numpy()])
            for j, panel in enumerate(config['panels']):
                metric = prediction_metrics(observations.observed_CPM.to_numpy()[keep],
                    predicted[keep, j], observations.sample_acr_index.to_numpy()[keep])
                rows.append(dict(model='CNN_'+panel, subgroup=subgroup, **metric))
        pd.DataFrame(rows).to_csv(out/'metrics.csv', index=False)
    import html
    text = f'Completed inner fits: {status["completed_inner_fits"]}/120. '
    text += f'Completed outer refits and evaluations: {status["completed_outer_refits"]}/20.'
    body = '<h1>Matched sequence CNN benchmark</h1><p>'+html.escape(text)+'</p>'
    if complete:
        table = pd.read_csv(out/'metrics.csv')
        body += table.loc[table.subgroup.eq('all'), ['model', 'pair_rmse', 'ACR_mean_rmse',
            'ACR_mean_r2', 'within_ACR_r2']].to_html(index=False, float_format=lambda x:f'{x:.6f}')
    else:
        body += '<p>Overall matched performance is pending; no missing predictions are imputed.</p>'
    (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Matched CNN benchmark</title>'
        '<style>body{max-width:1200px;margin:32px auto;padding:0 20px;font:16px/1.6 system-ui}'
        'td,th{padding:8px;border-bottom:1px solid #ddd}table{border-collapse:collapse}</style>'+body)
    return status
