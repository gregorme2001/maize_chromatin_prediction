import json
import numpy as np
import pytest

torch = pytest.importorskip('torch')

from maize_regulatory.sampled_cnn import SequenceBank
from maize_regulatory.sampled_cnn_run import PauseRequested, choose_candidate, run_fit


class Control:
    def __init__(self, stop_after=None): self.calls = 0; self.stop_after = stop_after
    def tick(self):
        if self.stop_after is not None and self.calls >= self.stop_after:
            raise PauseRequested('Synthetic interruption')
        self.calls += 1
    def emit(self, **details): pass


def fixture():
    torch.set_num_threads(1)
    rng = np.random.default_rng(112)
    codes = rng.integers(0, 4, (9, 96), dtype=np.uint8)
    ids = np.repeat(np.arange(9), 3)
    a = np.c_[np.ones(len(ids)), rng.normal(size=(len(ids), 2))]
    y = a[:, 1]+rng.normal(size=len(ids))
    setting = dict(width=4, depth=1, dropout=.1, stem_pool=8, learning_rate=.003, l2=1e-5)
    config = dict(batch_size=3, max_epochs=3, warmup_epochs=1, patience=2)
    return SequenceBank(codes), a, y, ids, setting, config


def test_mid_epoch_pause_and_resume_match_uninterrupted_training(tmp_path):
    bank, a, y, ids, setting, config = fixture()
    rows = np.arange(len(y))
    full = run_fit(tmp_path/'full', 'outer', bank, a, y, ids, rows, None,
        setting, config, Control(), seed=17, fixed_epochs=2)
    with pytest.raises(PauseRequested):
        run_fit(tmp_path/'resumed', 'outer', bank, a, y, ids, rows, None,
            setting, config, Control(stop_after=3), seed=17, fixed_epochs=2)
    state = torch.load(tmp_path/'resumed/fits/outer/last.pt', weights_only=False)
    assert state['metadata']['offset'] == 6 and state['epoch'] == 0
    resumed = run_fit(tmp_path/'resumed', 'outer', bank, a, y, ids, rows, None,
        setting, config, Control(), seed=17, fixed_epochs=2)
    x = torch.load(tmp_path/'full'/full['checkpoint'], weights_only=False)
    z = torch.load(tmp_path/'resumed'/resumed['checkpoint'], weights_only=False)
    for key, value in x['model'].items(): assert torch.equal(value, z['model'][key])
    assert x['numpy_rng'] == z['numpy_rng']
    assert torch.equal(x['torch_rng'], z['torch_rng'])


def test_inner_selection_counts_observations_and_prefers_stronger_penalty_on_tie():
    first = [dict(validation_pair_mse=1., n_validation_observations=90, best_epoch=7),
             dict(validation_pair_mse=10., n_validation_observations=10, best_epoch=9)]
    second = [dict(validation_pair_mse=2., n_validation_observations=90, best_epoch=5),
              dict(validation_pair_mse=2., n_validation_observations=10, best_epoch=6)]
    chosen, _ = choose_candidate([first, second], [1e-5, 1e-4])
    assert chosen['l2'] == 1e-5 and chosen['pooled_inner_mse'] == 1.9
    chosen, _ = choose_candidate([first, first], [1e-5, 1e-4])
    assert chosen['l2'] == 1e-4


def test_inner_fit_records_best_epoch_and_never_uses_outer_responses(tmp_path):
    bank, a, y, ids, setting, config = fixture()
    training = np.arange(18); validation = np.arange(18, 24)
    one = run_fit(tmp_path/'one', 'inner', bank, a, y, ids, training, validation,
                  setting, config, Control(), seed=12)
    changed = y.copy(); changed[24:] += 1e6
    two = run_fit(tmp_path/'two', 'inner', bank, a, changed, ids, training, validation,
                  setting, config, Control(), seed=12)
    assert one['validation_pair_mse'] == two['validation_pair_mse']
    assert one['best_epoch'] == two['best_epoch']
    assert one['n_validation_observations'] == 6
    assert 1 <= one['best_epoch'] <= config['max_epochs']
