import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

from maize_regulatory.models import PartialRidge


def test_partial_ridge_keeps_baseline_when_motifs_shrunk():
    rng = np.random.default_rng(123)
    C = rng.normal(size=(90, 2))
    Z = rng.normal(size=(90, 6))
    y = 4 + 2*C[:, 0] - 3*C[:, 1] + rng.normal(size=90) * .03
    X = np.column_stack([C, Z])
    fitted = PartialRidge(n_covariates=2, alpha=1e10).fit(X, y)
    ols = LinearRegression().fit(C, y)
    np.testing.assert_allclose(fitted.predict(X), ols.predict(C), atol=1e-7)
    assert fitted.ridge_.coef_.shape == (6,)


def test_partial_ridge_can_be_cloned_and_predict_unseen():
    rng = np.random.default_rng(99)
    X = rng.normal(size=(25, 5))
    y = 6 + 2*X[:, 0] + X[:, 3] + rng.normal(size=25)*.1
    original = PartialRidge(n_covariates=1, alpha=2)
    fitted = clone(original).fit(X[:20], y[:20])
    assert np.isfinite(fitted.predict(X[20:])).all()


def test_partial_ridge_recovers_partialled_solution():
    rng = np.random.default_rng(17)
    X = rng.normal(size=(50, 4))
    y = 5 + X[:, 0] * 2 + X[:, 1] * 1.2 - X[:, 2] * .5
    model = PartialRidge(n_covariates=1, alpha=1).fit(X, y)
    C = np.column_stack([np.ones(len(X)), X[:, 0]])
    Z = model.scaler_.transform(X[:, 1:])
    expected = C @ model.gamma_y_ + (Z - C @ model.gamma_Z_) @ model.ridge_.coef_
    np.testing.assert_allclose(model.predict(X), expected, atol=1e-9)
