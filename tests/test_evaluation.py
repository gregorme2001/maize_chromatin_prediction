import numpy as np
from sklearn.model_selection import train_test_split
from maize_regulatory.evaluation import evaluate_nested, make_holdout_indices


def test_holdout_reproduces_original_split():
    train, test = make_holdout_indices(169)
    expected_train, expected_test = train_test_split(np.arange(169), test_size=30, random_state=42)
    np.testing.assert_array_equal(train, expected_train)
    np.testing.assert_array_equal(test, expected_test)
    assert not set(train) & set(test)


def test_evaluation_respects_development_only():
    rng = np.random.default_rng(1)
    n = 44
    C = rng.normal(size=(n, 2))
    Z = rng.normal(size=(n, 3))
    comp = rng.normal(size=(n, 2))
    residuals = rng.normal(size=(n, 3))
    y = 50 + 2*C[:, 0] + Z[:, 0] + rng.normal(size=n)
    mats = {
        "ancestry": C,
        "ancestry + composition": np.column_stack([C, comp]),
        "ancestry + raw motifs": np.column_stack([C, Z]),
        "ancestry + residual motifs": np.column_stack([C, comp, residuals]),
    }
    idx = np.arange(34)
    a = evaluate_nested(mats, y, idx, n_splits=3, n_repeats=2, alphas=np.array([1.,10.]))
    # Modifying held-out y cannot change nested CV results.
    y2 = y.copy(); y2[34:] += 10000
    b = evaluate_nested(mats, y2, idx, n_splits=3, n_repeats=2, alphas=np.array([1.,10.]))
    assert a.folds.equals(b.folds)
    assert a.repetitions.equals(b.repetitions)
    assert len(a.folds) == 12
    assert (a.repetitions["n_validation"] == 34).all()
