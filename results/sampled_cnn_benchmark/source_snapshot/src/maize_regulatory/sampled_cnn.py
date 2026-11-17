"""Matched sequence CNN, exact unbalanced-observation loss, and resumable training.

No phenotype loading, genotype filtering, fold construction, or motif rescoring.
Identical DNA can share a forward pass without dropping its observation weights
or genotype-covariate cross-terms.
"""

from pathlib import Path
import copy
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def training_transform(a, y, rows):
    """Fit means/scales on actual training observations, excluding the intercept."""
    a, y, rows = np.asarray(a, float), np.asarray(y, float), np.asarray(rows, int)
    if a.ndim != 2 or not np.all(a[:, 0] == 1) or not len(rows):
        raise ValueError('Require explicit intercept and nonempty training observations')
    mean, sd = a[rows, 1:].mean(0), a[rows, 1:].std(0)
    sd[sd < 1e-10] = 1
    ym, ys = float(y[rows].mean()), float(y[rows].std())
    if not ys > 0:
        raise ValueError('Training target is constant')
    return dict(covariate_mean=mean, covariate_sd=sd, target_mean=ym, target_sd=ys)


def apply_transform(a, y, transform):
    z = (np.asarray(a)[:, 1:]-transform['covariate_mean'])/transform['covariate_sd']
    yy = (np.asarray(y)-transform['target_mean'])/transform['target_sd']
    return z, yy


def group_observations(z, y, pattern_ids):
    """Sufficient moments of actual rows, including within-DNA ancestry variation.

    MSE = sum_p n_p (f_p + intercept + mean(z)_p gamma - mean(y)_p)^2/N
          + (gamma' Wzz gamma - 2 gamma' Wzy + Wyy)/N.
    """
    z, y, ids = np.asarray(z, float), np.asarray(y, float), np.asarray(pattern_ids, int)
    if (z.ndim != 2 or len(z) != len(y) or len(y) != len(ids) or not len(y)
            or np.any(ids < 0) or not np.isfinite(z).all() or not np.isfinite(y).all()):
        raise ValueError('Invalid grouped observation axes or values')
    patterns, inverse, count = np.unique(ids, return_inverse=True, return_counts=True)
    mean_z = np.zeros((len(patterns), z.shape[1]))
    np.add.at(mean_z, inverse, z)
    mean_z /= count[:, None]
    mean_y = np.bincount(inverse, weights=y)/count
    dz, dy = z-mean_z[inverse], y-mean_y[inverse]
    return dict(pattern_ids=patterns, counts=count, mean_z=mean_z, mean_y=mean_y,
                within_zz=dz.T@dz, within_zy=dz.T@dy, within_yy=float(dy@dy),
                n_observations=len(y), n_patterns=len(patterns))


class GroupedLoss:
    def __init__(self, groups, device='cpu', dtype=torch.float32):
        self.groups = groups
        self.n, self.p = groups['n_observations'], groups['n_patterns']
        self.device = torch.device(device)
        for name in ['counts', 'mean_z', 'mean_y', 'within_zz', 'within_zy', 'within_yy']:
            setattr(self, name, torch.as_tensor(groups[name], device=device, dtype=dtype))

    def within(self, gamma):
        return (gamma@self.within_zz@gamma-2*gamma@self.within_zy+self.within_yy)/self.n

    def batch(self, sequence_prediction, linear, rows):
        # Uniformly shuffled DNA groups, weighted back to the actual pair loss.
        residual = sequence_prediction+linear(self.mean_z[rows]).squeeze(-1)-self.mean_y[rows]
        between = self.p/self.n*(self.counts[rows]*residual.square()).mean()
        return between+self.within(linear.weight[0])

    def full(self, sequence_prediction, linear):
        residual = sequence_prediction+linear(self.mean_z).squeeze(-1)-self.mean_y
        return (self.counts*residual.square()).sum()/self.n+self.within(linear.weight[0])


def masked_pool(x, mask, width):
    numerator = F.avg_pool1d(x*mask, width, width, ceil_mode=True, count_include_pad=False)
    denominator = F.avg_pool1d(mask, width, width, ceil_mode=True, count_include_pad=False)
    return numerator/denominator.clamp_min(1e-8), (denominator > 0).to(mask.dtype)


class MaskedResidualBlock(nn.Module):
    def __init__(self, width, dilation, dropout):
        super().__init__()
        self.first = nn.Conv1d(width, width, 5, padding=2*dilation, dilation=dilation)
        self.second = nn.Conv1d(width, width, 5, padding=2*dilation, dilation=dilation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        update = self.dropout(F.gelu(self.first(x))*mask)
        update = self.second(update)*mask
        return F.gelu(x+update)*mask


class MatchedCNN(nn.Module):
    """Four DNA channels; padding mask is computational, never a core indicator.

    Stem: 15-bp stride-one convolution -> masked pooling by four.
    Three residual blocks: two width-five feature convolutions each, dilations 1/2/4,
    followed by masked pooling by two. Final masked mean/max pool -> 128-unit
    hidden layer -> scalar sequence effect. One global linear nuisance branch.
    """
    def __init__(self, n_covariates, width=64, depth=3, dropout=.1):
        super().__init__()
        self.stem = nn.Conv1d(4, width, 15, padding=7)
        self.blocks = nn.ModuleList([MaskedResidualBlock(width, 2**j, dropout)
                                     for j in range(depth)])
        self.head = nn.Sequential(nn.Linear(width*2, 128), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(128, 1))
        self.linear = nn.Linear(n_covariates, 1)
        self.stem_pool = 4
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)

    def sequence(self, x, mask):
        x = F.gelu(self.stem(x))*mask
        x, mask = masked_pool(x, mask, self.stem_pool)
        for block in self.blocks:
            x = block(x, mask)
            x, mask = masked_pool(x, mask, 2)
        mean = (x*mask).sum(-1)/mask.sum(-1).clamp_min(1)
        maximum = x.masked_fill(mask == 0, torch.finfo(x.dtype).min).amax(-1)
        return self.head(torch.cat([mean, maximum], dim=1)).squeeze(-1)

    def forward(self, x, mask, z):
        return self.sequence(x, mask)+self.linear(z).squeeze(-1)

    def weight_penalty(self):
        return sum(p.square().sum() for name, p in self.named_parameters()
                   if p.ndim >= 2 and not name.startswith('linear.'))


class SequenceBank:
    def __init__(self, codes, device='cpu'):
        codes = np.asarray(codes)
        if codes.ndim != 2 or codes.dtype != np.uint8 or np.any(codes > 4):
            raise ValueError('Expected sequence codes A/C/G/T=0/1/2/3, padding=4')
        self.codes = torch.tensor(codes, device=device, dtype=torch.uint8)
        self.device = torch.device(device)

    def batch(self, rows, reverse=False):
        values = self.codes[torch.as_tensor(rows, device=self.device, dtype=torch.long)].long()
        mask = (values < 4).to(torch.float32)[:, None, :]
        x = F.one_hot(values, num_classes=5)[..., :4].transpose(1, 2).to(torch.float32)
        if reverse:
            x, mask = x[:, [3, 2, 1, 0], :].flip(-1), mask.flip(-1)
        return x, mask


def synchronize(device):
    if str(device) == 'mps':
        torch.mps.synchronize()
    elif str(device).startswith('cuda'):
        torch.cuda.synchronize()


def seed_everything(seed, device):
    torch.manual_seed(seed)
    if str(device) == 'mps':
        torch.mps.manual_seed(seed)


class Trainer:
    def __init__(self, bank, groups, setting, *, seed=42):
        self.bank, self.setting = bank, dict(setting)
        self.rng = np.random.default_rng(seed)
        seed_everything(seed, bank.device)
        self.loss = GroupedLoss(groups, bank.device)
        self.model = MatchedCNN(groups['mean_z'].shape[1], setting['width'], setting['depth'],
                                setting['dropout']).to(bank.device)
        self.model.stem_pool = setting.get('stem_pool', 4)
        if self.model.stem_pool not in [4, 8]:
            raise ValueError('Preflight supports stem pooling by four or eight')
        # Exact unpenalized OLS initialization on the actual training rows.
        z, y, n = groups['mean_z'], groups['mean_y'], groups['counts']
        design = np.c_[np.ones(len(z)), z]
        gram = design.T@(n[:, None]*design)
        gram[1:, 1:] += groups['within_zz']
        cross = design.T@(n*y); cross[1:] += groups['within_zy']
        coefficient = np.linalg.lstsq(gram, cross, rcond=1e-12)[0]
        with torch.no_grad():
            self.model.linear.bias.copy_(torch.tensor(coefficient[:1], device=bank.device,
                                                     dtype=torch.float32))
            self.model.linear.weight.copy_(torch.tensor(coefficient[1:][None], device=bank.device,
                                                       dtype=torch.float32))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=setting['learning_rate'])
        self.epoch = 0

    def step(self, rows):
        self.model.train()
        x, mask = self.bank.batch(self.loss.groups['pattern_ids'][rows], self.rng.random() < .5)
        self.optimizer.zero_grad(set_to_none=True)
        prediction = self.model.sequence(x, mask)
        data_loss = self.loss.batch(prediction, self.model.linear, rows)
        total_loss = data_loss+self.setting['l2']*self.model.weight_penalty()
        if not torch.isfinite(total_loss):
            raise FloatingPointError('Nonfinite loss')
        total_loss.backward()
        norm = nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
        if not torch.isfinite(norm):
            raise FloatingPointError('Nonfinite gradient')
        self.optimizer.step()
        return float(data_loss.detach().cpu())

    def train_epoch(self, batch_size):
        order = self.rng.permutation(self.loss.p)
        started = time.monotonic(); values = []
        for start in range(0, len(order), batch_size):
            values.append(self.step(order[start:start+batch_size]))
        self.epoch += 1; synchronize(self.bank.device)
        return dict(epoch=self.epoch, seconds=time.monotonic()-started,
                    stochastic_batch_loss_mean=float(np.mean(values)), steps=len(values))

    def save(self, path, metadata=None, extra_state=None):
        synchronize(self.bank.device)
        state = dict(model=copy.deepcopy(self.model.state_dict()), optimizer=self.optimizer.state_dict(),
                     numpy_rng=copy.deepcopy(self.rng.bit_generator.state), torch_rng=torch.get_rng_state(),
                     mps_rng=torch.mps.get_rng_state() if str(self.bank.device) == 'mps' else None,
                     epoch=self.epoch, setting=self.setting, metadata=metadata)
        if extra_state:
            state.update(extra_state)
        # Portable CPU checkpoints; source objects remain resident on their device.
        def cpu(value):
            if isinstance(value, torch.Tensor): return value.detach().cpu()
            if isinstance(value, dict): return {k: cpu(v) for k, v in value.items()}
            if isinstance(value, list): return [cpu(v) for v in value]
            if isinstance(value, tuple): return tuple(cpu(v) for v in value)
            return value
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix+'.part')
        torch.save(cpu(state), temporary); temporary.replace(path)

    def restore(self, path):
        state = torch.load(path, map_location='cpu', weights_only=False)
        if state['setting'] != self.setting:
            raise ValueError('Checkpoint configuration differs')
        self.model.load_state_dict(state['model']); self.optimizer.load_state_dict(state['optimizer'])
        self.rng.bit_generator.state = state['numpy_rng']; torch.set_rng_state(state['torch_rng'])
        if state['mps_rng'] is not None:
            if str(self.bank.device) != 'mps':
                raise ValueError('MPS checkpoint requires the same device for exact continuation')
            torch.mps.set_rng_state(state['mps_rng'])
        self.epoch = state['epoch']
        return state['metadata']


@torch.no_grad()
def predict_groups(model, bank, pattern_ids, batch_size=256):
    model.eval(); values = []
    for start in range(0, len(pattern_ids), batch_size):
        rows = pattern_ids[start:start+batch_size]
        forward, mask = bank.batch(rows)
        reverse, reverse_mask = bank.batch(rows, True)
        values.append(((model.sequence(forward, mask)+model.sequence(reverse, reverse_mask))/2).cpu())
    return torch.cat(values).numpy()


@torch.no_grad()
def evaluate(model, bank, groups, target_sd, batch_size=256):
    started = time.monotonic()
    prediction = predict_groups(model, bank, groups['pattern_ids'], batch_size)
    criterion = GroupedLoss(groups, bank.device)
    value = criterion.full(torch.as_tensor(prediction, device=bank.device), model.linear)
    synchronize(bank.device)
    return dict(pair_mse_CPM2=float(value.cpu())*target_sd**2,
                seconds=time.monotonic()-started, n_observations=groups['n_observations'],
                n_patterns=groups['n_patterns'])
