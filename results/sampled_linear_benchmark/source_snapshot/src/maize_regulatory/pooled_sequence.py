"""Global partial ridge from additive moments, without materializing repeated k-mers.

No ACR intercepts. All coefficients are shared. The 1,024 standardized
strand-averaged counts have 512 identical reverse-complement pairs. Compressing
each pair to coefficient b gives minimum full penalty alpha*b²/2.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.model_selection import KFold


@dataclass
class Moments:
    n: int
    xx: np.ndarray
    xy: np.ndarray
    yy: float


def add_moments(parts):
    if not parts:
        raise ValueError('No training blocks')
    return Moments(sum(p.n for p in parts), sum(p.xx for p in parts),
                   sum(p.xy for p in parts), sum(p.yy for p in parts))


def pattern_moments(covariates, pattern_ids, patterns, y):
    """Exact [unpenalized covariates | pattern counts] cross-products.

    Covariates include an explicit first-column intercept. Each row is one
    equally weighted ACR/genotype observation; repeated patterns are aggregated.
    """
    a, ids, x, y = (np.asarray(covariates, float), np.asarray(pattern_ids, int),
                    np.asarray(patterns, float), np.asarray(y, float))
    if (a.ndim != 2 or x.ndim != 2 or len(a) != len(ids) or len(ids) != len(y)
            or not len(y) or not np.all(a[:, 0] == 1) or np.any(ids < 0)
            or np.any(ids >= len(x)) or not all(np.isfinite(v).all() for v in [a, x, y])):
        raise ValueError('Invalid observation/pattern axes or nonfinite values')
    unique, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    px = x[unique]
    summed_a = np.zeros((len(unique), a.shape[1]))
    np.add.at(summed_a, inverse, a)
    ax = summed_a.T@px
    gram = np.block([[a.T@a, ax], [ax.T, px.T@(counts[:, None]*px)]])
    cross = np.r_[a.T@y, px.T@np.bincount(inverse, weights=y, minlength=len(unique))]
    return Moments(len(y), gram, cross, float(y@y))


def ridge_path(moments, n_covariates, alphas, *, penalty_multiplier=.5):
    """MSE + alpha*sum(full 1024 standardized coefficients²).

    All means, scales and residualization use only the supplied training moments.
    Returned coefficients act on RAW canonical counts and raw covariates.
    None is the exact matched baseline, not a very large finite approximation.
    penalty_multiplier=.5 handles duplicated strand-symmetric k-mers; use 1.
    for one coefficient per distinct feature, such as the 927 motif columns.
    """
    m = moments
    if m.n <= n_covariates or n_covariates < 1 or n_covariates >= len(m.xy):
        raise ValueError('Insufficient observations or invalid covariate count')
    if not np.isfinite(penalty_multiplier) or penalty_multiplier <= 0:
        raise ValueError('Penalty multiplier must be positive')
    if any(a is not None and (not np.isfinite(a) or a <= 0) for a in alphas):
        raise ValueError('Penalties must be positive or None for baseline')
    mean = m.xx[0, 1:]/m.n
    variance = np.diag(m.xx)[1:]/m.n-mean**2
    scale = np.sqrt(np.maximum(variance, 0))
    constant = scale <= 1e-10
    scale[constant] = 1.
    covariance = (m.xx[1:, 1:]/m.n-np.outer(mean, mean))/np.outer(scale, scale)
    target = (m.xy[1:]/m.n-mean*m.xy[0]/m.n)/scale
    covariance[constant, :], covariance[:, constant], target[constant] = 0., 0., 0.
    d = n_covariates-1
    aa, az, zz = covariance[:d, :d], covariance[:d, d:], covariance[d:, d:]
    ay, zy = target[:d], target[d:]
    if d:
        inverse = linalg.pinvh(aa, rtol=1e-12)
        residual = zz-az.T@inverse@az
        residual_target = zy-az.T@inverse@ay
        rank = int(np.linalg.matrix_rank(aa, tol=1e-10))
    else:
        inverse, residual, residual_target, rank = np.empty((0, 0)), zz, zy, 0
    residual = (residual+residual.T)/2
    eigenvalues, vectors = linalg.eigh(residual)
    if eigenvalues.min() < -1e-7:
        raise ValueError('Residual Gram matrix is not positive semidefinite')
    eigenvalues = np.maximum(eigenvalues, 0)
    projected = vectors.T@residual_target
    coefficients = {}
    standardized = {}
    for alpha in alphas:
        beta = (np.zeros(len(zy)) if alpha is None else
            vectors@(projected/(eigenvalues+alpha*penalty_multiplier)))
        beta[constant[d:]] = 0
        gamma = inverse@(ay-az@beta)
        standardized[alpha] = np.r_[gamma, beta]
        slopes = standardized[alpha]/scale
        coefficients[alpha] = np.r_[m.xy[0]/m.n-mean@slopes, slopes]
    return dict(coefficients=coefficients, standardized=standardized,
        mean=mean, scale=scale, constant=constant, nuisance_rank=rank+1,
        minimum_residual_eigenvalue=float(eigenvalues.min()))


def moment_mse(moments, coefficient):
    c = np.asarray(coefficient)
    result = (moments.yy-2*c@moments.xy+c@moments.xx@c)/moments.n
    if result < -1e-7:
        raise ValueError('Negative squared-error sum from numerical cancellation')
    return float(max(result, 0.))


def blocked_manifest(blocks, outer_splits=5, inner_splits=3, seed=42):
    """All observations in each fixed genomic block stay together at both levels."""
    blocks = np.unique(np.asarray(blocks, int))
    outer = KFold(outer_splits, shuffle=True, random_state=seed)
    manifest = []
    for number, (training, validation) in enumerate(outer.split(blocks)):
        train, val = blocks[training], blocks[validation]
        inner = KFold(inner_splits, shuffle=True, random_state=seed+number+1)
        manifest.append(dict(number=number, training_blocks=train.tolist(),
            validation_blocks=val.tolist(), inner=[dict(training_blocks=train[t].tolist(),
                validation_blocks=train[v].tolist()) for t, v in inner.split(train)]))
    return manifest


def choose_alpha(scores, alphas):
    """Aggregate validation SSE/count, prefer baseline then stronger shrinkage on ties."""
    losses = {alpha: sum(s['sse'] for s in scores if s['alpha'] == alpha)/
        sum(s['n'] for s in scores if s['alpha'] == alpha) for alpha in alphas}
    best = min(losses.values())
    tied = [a for a in alphas if np.isclose(losses[a], best, rtol=0, atol=1e-10)]
    return (None if None in tied else max(tied)), losses


def prediction_scores(frame):
    y = frame.observed_CPM.to_numpy()
    variance = float(np.var(y))
    records = []
    for name in ['baseline', 'sequence']:
        pred = frame[name+'_CPM'].to_numpy()
        errors = (y-pred)**2
        by_acr = pd.Series(errors).groupby(frame.ACR.to_numpy()).mean()
        by_genotype = pd.Series(errors).groupby(frame.genotype.to_numpy()).mean()
        records.append(dict(model=name, mse=float(errors.mean()), rmse=float(np.sqrt(errors.mean())),
            mae=float(np.mean(np.abs(y-pred))), r2=float(1-errors.mean()/variance),
            pearson_r=float(np.corrcoef(y, pred)[0, 1]),
            equal_acr_rmse=float(np.sqrt(by_acr.mean())),
            equal_genotype_rmse=float(np.sqrt(by_genotype.mean())),
            fraction_negative_predictions=float(np.mean(pred < 0))))
    return pd.DataFrame(records)
