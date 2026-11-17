"""Partial ridge: baseline covariates unpenalized, motif coefficients penalized."""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


class PartialRidge(RegressorMixin, BaseEstimator):
    """Regress y on [baseline covariates | motif features].

    Fit intercept and baseline covariates without penalty; shrink standardized
    motif coefficients by L2 penalty. Training-only FWL residualization ensures
    predictions use the same adjusted baseline intercept / slopes.

    Assumes the first `n_covariates` columns of X are baseline covariates.
    """

    def __init__(self, n_covariates: int = 10, alpha: float = 1.0):
        self.n_covariates = n_covariates
        self.alpha = alpha

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim != 2 or X.shape[0] != len(y):
            raise ValueError("X must be a 2D matrix matching y")
        if not (0 <= self.n_covariates < X.shape[1]):
            raise ValueError("Need baseline covariates and at least one motif column")
        if self.alpha < 0:
            raise ValueError("alpha must be nonnegative")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("Input contains non-finite values")

        self.n_features_in_ = X.shape[1]
        C = np.column_stack([np.ones(X.shape[0]), X[:, :self.n_covariates]])
        self.scaler_ = StandardScaler()
        Z = self.scaler_.fit_transform(X[:, self.n_covariates:])

        self.gamma_y_ = np.linalg.lstsq(C, y, rcond=None)[0]
        self.gamma_Z_ = np.linalg.lstsq(C, Z, rcond=None)[0]
        y_resid = y - C @ self.gamma_y_
        Z_resid = Z - C @ self.gamma_Z_
        self.ridge_ = Ridge(alpha=self.alpha, fit_intercept=False)
        self.ridge_.fit(Z_resid, y_resid)
        return self

    def predict(self, X):
        check_is_fitted(self, ["ridge_", "scaler_", "gamma_y_", "gamma_Z_"])
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("Unexpected predictor count")
        C = np.column_stack([np.ones(X.shape[0]), X[:, :self.n_covariates]])
        Z = self.scaler_.transform(X[:, self.n_covariates:])
        beta = self.ridge_.coef_
        return C @ (self.gamma_y_ - self.gamma_Z_ @ beta) + Z @ beta
