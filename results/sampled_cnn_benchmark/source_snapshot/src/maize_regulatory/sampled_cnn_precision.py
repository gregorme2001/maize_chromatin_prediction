"""Mixed-precision sequence operations; float32 weights, nuisance branch and loss."""
from pathlib import Path

import numpy as np
import torch

from .sampled_cnn import Trainer


class MixedTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scaler = torch.amp.GradScaler('mps')
        self.skipped_updates = 0
        self.consecutive_skips = 0

    def step(self, rows):
        self.model.train()
        x, mask = self.bank.batch(self.loss.groups['pattern_ids'][rows], self.rng.random() < .5)
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast('mps', dtype=torch.float16):
            prediction = self.model.sequence(x, mask)
        data_loss = self.loss.batch(prediction.float(), self.model.linear, rows)
        total = data_loss+self.setting['l2']*self.model.weight_penalty()
        if not torch.isfinite(total): raise FloatingPointError('Nonfinite mixed-precision loss')
        self.scaler.scale(total).backward(); self.scaler.unscale_(self.optimizer)
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
        if not torch.isfinite(norm):
            # Finite loss can still overflow when gradients are scaled into
            # float16. Skip that update and reduce the scale, preserving weights.
            self.skipped_updates += 1; self.consecutive_skips += 1
            self.scaler.update(new_scale=self.scaler.get_scale()/2)
            if self.consecutive_skips >= 20:
                raise FloatingPointError('Repeated mixed-precision gradient overflows')
            return float(data_loss.detach().cpu())
        self.consecutive_skips = 0
        self.scaler.step(self.optimizer); self.scaler.update()
        return float(data_loss.detach().cpu())

    def save(self, path, metadata=None):
        # The scaler and optimizer must be committed in the SAME atomic write.
        super().save(path, metadata, extra_state=dict(
            grad_scaler=self.scaler.state_dict(), skipped_updates=self.skipped_updates,
            consecutive_skips=self.consecutive_skips))

    def restore(self, path):
        metadata = super().restore(path)
        state = torch.load(path, map_location='cpu', weights_only=False)
        self.scaler.load_state_dict(state['grad_scaler'])
        self.skipped_updates = state.get('skipped_updates', 0)
        self.consecutive_skips = state.get('consecutive_skips', 0)
        return metadata


@torch.no_grad()
def predict_mixed(model, bank, ids, batch=256, between_batches=None):
    model.eval(); result = []
    for first in range(0, len(ids), batch):
        if between_batches is not None: between_batches()
        x, m = bank.batch(ids[first:first+batch]); rc, rm = bank.batch(ids[first:first+batch], True)
        with torch.autocast('mps', dtype=torch.float16):
            value = (model.sequence(x, m).float()+model.sequence(rc, rm).float())/2
        result.append(value.float().cpu())
    return torch.cat(result).numpy()


