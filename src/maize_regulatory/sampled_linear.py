"""Exact partial ridge on uneven matched pairs, with mixed duplicate penalties."""

import numpy as np
from scipy import linalg

from .pooled_sequence import Moments


def subset_moments(m, indices):
    ix = np.asarray(indices, int)
    return Moments(m.n, m.xx[np.ix_(ix, ix)], m.xy[ix], m.yy)


def partial_ridge_path(m, n_covariates, alphas, penalty):
    """Training-only scaling/FWL, then diagonal-penalty ridge in whitened space.

    `penalty` is 1 for TFs and .5 for compressed duplicate word pairs. Returned
    coefficients act on raw inputs. A feature-free path supplies the baseline.
    All moments must represent equally weighted actual pairs, not ACR means.
    """
    n_features = len(m.xy)-n_covariates
    penalty = np.broadcast_to(np.asarray(penalty, float), (n_features,)).copy()
    if (m.n <= n_covariates or n_covariates < 1 or n_features < 0
            or not np.isfinite(penalty).all() or np.any(penalty <= 0)
            or any(a is not None and (not np.isfinite(a) or a <= 0) for a in alphas)
            or (n_features == 0 and any(a is not None for a in alphas))):
        raise ValueError('Invalid moments, covariates or ridge penalties')
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
    inverse = linalg.pinvh(aa, rtol=1e-12) if d else np.empty((0, 0))
    residual = zz-az.T@inverse@az
    residual_target = zy-az.T@inverse@ay
    whitening = 1/np.sqrt(penalty)
    residual = ((residual+residual.T)/2)*np.outer(whitening, whitening)
    residual_target *= whitening
    if n_features:
        eigenvalues, vectors = linalg.eigh(residual)
        if eigenvalues.min() < -1e-7:
            raise ValueError('Residual covariance is not positive semidefinite')
        eigenvalues = np.maximum(eigenvalues, 0)
        projected = vectors.T@residual_target
        minimum = float(eigenvalues.min())
    else:
        minimum = 0.
    coefficients, standardized = {}, {}
    for alpha in alphas:
        beta = (np.zeros(n_features) if alpha is None else
                whitening*(vectors@(projected/(eigenvalues+alpha))))
        beta[constant[d:]] = 0
        gamma = inverse@(ay-az@beta)
        standardized[alpha] = np.r_[gamma, beta]
        slopes = standardized[alpha]/scale
        coefficients[alpha] = np.r_[m.xy[0]/m.n-mean@slopes, slopes]
    return dict(coefficients=coefficients, standardized=standardized,
                mean=mean, scale=scale, constant=constant,
                nuisance_rank=int(np.linalg.matrix_rank(aa, tol=1e-10))+1 if d else 1,
                minimum_residual_eigenvalue=minimum, penalty=penalty)


def aggregate_block_moments(a, ids, features, y, pair_blocks, partitions, progress=None):
    """Each block is multiplied once; identical pattern rows are exactly grouped."""
    from .pooled_sequence import pattern_moments
    width = a.shape[1]+features.shape[1]
    results = {name: Moments(0, np.zeros((width, width)), np.zeros(width), 0.)
               for name in partitions}
    memberships = {block: [name for name, blocks in partitions.items() if block in blocks]
                   for block in np.unique(pair_blocks)}
    order = np.argsort(pair_blocks, kind='stable')
    blocks, first, counts = np.unique(pair_blocks[order], return_index=True, return_counts=True)
    for k, (block, start, count) in enumerate(zip(blocks, first, counts)):
        rows = order[start:start+count]
        used, local_ids = np.unique(ids[rows], return_inverse=True)
        m = pattern_moments(a[rows], local_ids, features[used], y[rows])
        for name in memberships[block]:
            destination = results[name]
            destination.n += m.n; destination.xx += m.xx
            destination.xy += m.xy; destination.yy += m.yy
        if progress and (k % 50 == 0 or k == len(blocks)-1):
            progress(k+1, len(blocks))
    return results


def prediction_metrics(y, prediction, acr_index):
    """Pair, ACR-mean and within-ACR metrics on the same available genotype sets."""
    y, prediction, acr_index = map(np.asarray, (y, prediction, acr_index))
    if len(y) != len(prediction) or len(y) != len(acr_index) or not np.isfinite(prediction).all():
        raise ValueError('Prediction axes or finiteness invalid')
    unique, inverse, counts = np.unique(acr_index, return_inverse=True, return_counts=True)
    ym = np.bincount(inverse, weights=y)/counts
    pm = np.bincount(inverse, weights=prediction)/counts
    error = prediction-y
    centered_y, centered_prediction = y-ym[inverse], prediction-pm[inverse]
    within_error = centered_prediction-centered_y
    variance, mean_variance = float(np.var(y)), float(np.var(ym))
    within_variance = float(np.mean(centered_y**2))
    return dict(n_pairs=len(y), n_ACRs=len(unique), pair_rmse=float(np.sqrt(np.mean(error**2))),
        pair_mse=float(np.mean(error**2)), pair_mae=float(np.mean(np.abs(error))),
        pair_r2=float(1-np.mean(error**2)/variance) if variance else np.nan,
        ACR_mean_rmse=float(np.sqrt(np.mean((pm-ym)**2))),
        ACR_mean_r2=float(1-np.mean((pm-ym)**2)/mean_variance) if mean_variance else np.nan,
        within_ACR_rmse=float(np.sqrt(np.mean(within_error**2))),
        within_ACR_r2=float(1-np.mean(within_error**2)/within_variance) if within_variance else np.nan,
        fraction_negative_predictions=float(np.mean(prediction < 0)))
