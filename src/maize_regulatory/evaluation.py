"""Nested validation on development genotypes only."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, RepeatedKFold, train_test_split

from .models import PartialRidge


@dataclass(frozen=True)
class Evaluation:
    folds: pd.DataFrame
    repetitions: pd.DataFrame


def make_holdout_indices(n_samples: int, test_size: int = 30, seed: int = 42):
    """Preserves original row-order-dependent split; never peek at held-out y."""
    if n_samples <= test_size:
        raise ValueError("test_size must be smaller than sample count")
    return train_test_split(np.arange(n_samples), test_size=test_size, random_state=seed)


def evaluate_nested(
    model_matrices: dict[str, np.ndarray],
    y: np.ndarray,
    development_idx: np.ndarray,
    *,
    n_splits: int = 5,
    n_repeats: int = 5,
    alphas: np.ndarray | None = None,
    random_state: int = 42,
    n_jobs: int = 1,
) -> Evaluation:
    """Compare partial ridge with paired unpenalized OLS baselines.

    Does not use test indices; caller must supply development-only indices.
    For each outer fold, inner CV tunes motif penalty using its training data.
    """
    if alphas is None:
        alphas = np.logspace(2, 6, 17)
    development_idx = np.asarray(development_idx)
    y_dev = np.asarray(y, dtype=float).reshape(-1)[development_idx]
    if len(development_idx) < n_splits or n_splits < 2:
        raise ValueError("Not enough development samples for outer CV")
    specs = [
        ("raw motifs", "ancestry", "ancestry + raw motifs"),
        ("residual motifs", "ancestry + composition", "ancestry + residual motifs"),
    ]
    outer = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    inner = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    records = []
    for fold_index, (fit_idx, validation_idx) in enumerate(outer.split(y_dev)):
        for name, baseline, full in specs:
            X_base = np.asarray(model_matrices[baseline])[development_idx]
            X_full = np.asarray(model_matrices[full])[development_idx]
            n_covariates = X_base.shape[1]
            ols = LinearRegression().fit(X_base[fit_idx], y_dev[fit_idx])
            baseline_prediction = ols.predict(X_base[validation_idx])
            tuned = GridSearchCV(
                PartialRidge(n_covariates=n_covariates),
                {"alpha": alphas},
                cv=inner,
                scoring="neg_mean_squared_error",
                n_jobs=n_jobs,
            )
            tuned.fit(X_full[fit_idx], y_dev[fit_idx])
            motif_prediction = tuned.predict(X_full[validation_idx])
            records.append({
                "repeat": fold_index // n_splits,
                "fold": fold_index % n_splits,
                "model": name,
                "n_validation": len(validation_idx),
                "baseline_MSE": mean_squared_error(y_dev[validation_idx], baseline_prediction),
                "motif_MSE": mean_squared_error(y_dev[validation_idx], motif_prediction),
                "alpha": tuned.best_params_["alpha"],
            })
    folds = pd.DataFrame.from_records(records)
    # Weight each fold MSE by n_validation for exact overall held-out RMSE.
    summary_rows = []
    for (model, repetition), part in folds.groupby(["model", "repeat"], sort=True):
        total = part["n_validation"].sum()
        mse0 = np.average(part["baseline_MSE"], weights=part["n_validation"])
        mse1 = np.average(part["motif_MSE"], weights=part["n_validation"])
        summary_rows.append({
            "model": model,
            "repeat": repetition,
            "n_validation": total,
            "baseline_RMSE": np.sqrt(mse0),
            "motif_RMSE": np.sqrt(mse1),
            "improvement": np.sqrt(mse0) - np.sqrt(mse1),
        })
    return Evaluation(folds=folds, repetitions=pd.DataFrame(summary_rows))
