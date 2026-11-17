import numpy as np
import pytest

torch = pytest.importorskip('torch')
from maize_regulatory.sampled_cnn import (
    GroupedLoss, MatchedCNN, SequenceBank, Trainer, apply_transform, evaluate,
    group_observations, training_transform,
)


def test_grouped_loss_and_gradients_equal_explicit_unbalanced_observations():
    rng = np.random.default_rng(81)
    ids = np.array([0, 0, 0, 1, 2, 2, 3, 3, 3, 3])
    z, y = rng.normal(size=(10, 3)), rng.normal(size=10)
    groups = group_observations(z, y, ids)
    criterion = GroupedLoss(groups, dtype=torch.float64)
    linear = torch.nn.Linear(3, 1).double()
    sequence = torch.tensor(rng.normal(size=4), requires_grad=True)
    full = ((sequence[ids]+linear(torch.tensor(z)).squeeze()-torch.tensor(y))**2).mean()
    grouped = criterion.full(sequence, linear)
    assert torch.allclose(grouped, full, atol=1e-12)
    for a, b in zip(torch.autograd.grad(grouped, [sequence, *linear.parameters()], retain_graph=True),
                    torch.autograd.grad(full, [sequence, *linear.parameters()])):
        assert torch.allclose(a, b, atol=1e-12)
    # The weighted mean of minibatch losses also equals the full objective.
    batches = [np.arange(2), np.arange(2, 4)]
    parts = [criterion.batch(sequence[at], linear, at) for at in batches]
    assert torch.allclose(sum(parts)/2, grouped, atol=1e-12)


@pytest.mark.parametrize('stem_pool', [4, 8])
def test_scaling_uses_training_rows_and_padding_cannot_change_prediction(stem_pool):
    torch.set_num_threads(1)
    a = np.c_[np.ones(4), [0., 2., 100., 200.]]; y = np.array([1., 3., 100., 200.])
    transform = training_transform(a, y, np.array([0, 1]))
    changed = a.copy(); changed[2:, 1] = 1e6
    again = training_transform(changed, y, np.array([0, 1]))
    for key in transform: np.testing.assert_array_equal(transform[key], again[key])
    assert transform['covariate_mean'][0] == 1 and transform['target_mean'] == 2
    code = np.tile(np.arange(4, dtype=np.uint8), (2, 32))
    padded = np.pad(code, ((0, 0), (0, 64)), constant_values=4)
    model = MatchedCNN(1, width=4, depth=1, dropout=0).eval()
    model.stem_pool = stem_pool
    with torch.no_grad():
        model.head[-1].weight.fill_(.1)
        x, m = SequenceBank(code).batch([0, 1]); xx, mm = SequenceBank(padded).batch([0, 1])
        assert torch.allclose(model.sequence(x, m), model.sequence(xx, mm), atol=1e-7)
    assert model.stem.in_channels == 4
    assert torch.autograd.grad(model.weight_penalty(), model.linear.weight, allow_unused=True)[0] is None


def test_checkpoint_resumes_optimizer_dropout_and_random_orientation_exactly(tmp_path):
    torch.set_num_threads(1)
    rng = np.random.default_rng(82)
    codes = rng.integers(0, 4, (7, 96), dtype=np.uint8)
    ids = rng.integers(0, 7, 35); a = np.c_[np.ones(35), rng.normal(size=(35, 2))]
    y = 2*a[:, 1]+rng.normal(size=35)
    transform = training_transform(a, y, np.arange(35)); z, yy = apply_transform(a, y, transform)
    groups = group_observations(z, yy, ids); bank = SequenceBank(codes)
    setting = dict(width=4, depth=1, dropout=.2, learning_rate=.003, l2=1e-5)
    original = Trainer(bank, groups, setting, seed=10)
    original.train_epoch(3); original.save(tmp_path/'resume.pt', {'check': True})
    expected = original.train_epoch(3)
    weights = {k: v.clone() for k, v in original.model.state_dict().items()}
    restored = Trainer(bank, groups, setting, seed=99)
    assert restored.restore(tmp_path/'resume.pt') == {'check': True}
    actual = restored.train_epoch(3)
    assert expected['stochastic_batch_loss_mean'] == actual['stochastic_batch_loss_mean']
    assert restored.epoch == 2
    for k, v in restored.model.state_dict().items(): assert torch.equal(v, weights[k])


def test_network_can_learn_sequence_signal_beyond_linear_covariates():
    """Synthetic optimization check, not evidence about the maize dataset."""
    torch.set_num_threads(1)
    rng = np.random.default_rng(4)
    codes = np.repeat(np.arange(2, dtype=np.uint8)[:, None], 128, axis=1)
    ids = np.repeat(np.arange(2), 20)
    z = rng.normal(size=(40, 1)); y = 2*z[:, 0]+3*(ids == 1)
    groups = group_observations(z, y, ids); bank = SequenceBank(codes)
    setting = dict(width=4, depth=1, dropout=0., learning_rate=.01, l2=0.)
    trainer = Trainer(bank, groups, setting, seed=42)
    before = evaluate(trainer.model, bank, groups, 1., 2)['pair_mse_CPM2']
    for _ in range(120): trainer.train_epoch(2)
    after = evaluate(trainer.model, bank, groups, 1., 2)['pair_mse_CPM2']
    assert after < before*.02
