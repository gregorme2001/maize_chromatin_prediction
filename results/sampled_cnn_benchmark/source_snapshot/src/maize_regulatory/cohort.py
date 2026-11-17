"""Explicit analysis exclusions layered over the immutable source/holdout identities."""

from pathlib import Path

import numpy as np
import pandas as pd

from .sample_identity import RESOURCE_DIR, load_identities, validate_analysis_order

EXCLUSIONS_PATH = RESOURCE_DIR / 'analysis_exclusions.json'


def analysis_exclusions():
    import json

    policy = json.loads(EXCLUSIONS_PATH.read_text())
    if policy['version'] != 1:
        raise ValueError('Unsupported analysis exclusion policy')
    records = policy['exclusions']
    ids = pd.Index([entry['genotype'] for entry in records])
    _, analysis, holdout, _, _ = load_identities()
    if not ids.is_unique or not ids.isin(analysis).all():
        raise ValueError('Excluded genotype identities must be unique and known')
    if ids.isin(holdout).any():
        raise ValueError('Exclusions must not change the permanent holdout')
    if any(not entry['reason'] or not entry['authorization'] for entry in records):
        raise ValueError('Every exclusion requires an explicit reason and authorization')
    return records


def development_genotypes() -> pd.Index:
    """Eligible development IDs in original source order, after explicit QC exclusions."""
    _, analysis, holdout, _, _ = load_identities()
    excluded = [entry['genotype'] for entry in analysis_exclusions()]
    return analysis[~analysis.isin(holdout) & ~analysis.isin(excluded)]


def cohort_metadata():
    _, analysis, holdout, _, _ = load_identities()
    return {
        'source_genotypes': len(analysis), 'original_development_genotypes': len(analysis) - len(holdout),
        'active_development_genotypes': len(development_genotypes()),
        'permanent_holdout_genotypes': len(holdout), 'exclusions': analysis_exclusions(),
    }


def read_development_parquet(path: Path, *, columns=None) -> pd.DataFrame:
    """Validate the frozen source axis, then request eligible rows only; never reorder."""
    identity_only = pd.read_parquet(path, columns=[])
    validate_analysis_order(identity_only.index)
    if identity_only.index.name != 'genotype':
        raise ValueError('Expected persisted genotype index for parquet row filtering')
    expected = development_genotypes()
    selected = pd.read_parquet(path, columns=columns,
                               filters=[('genotype', 'in', expected.tolist())])
    if not selected.index.equals(expected):
        raise ValueError('Development genotype identity/order mismatch; no silent reordering')
    return selected


def development_cv_indices(genotype_order: pd.Index) -> np.ndarray:
    """Preserve original development split order, omitting QC exclusions before CV.

    The source/holdout split is never recomputed on the reduced cohort. Returned
    positions index a development-only frame in original relative source order.
    """
    if not genotype_order.equals(development_genotypes()):
        raise ValueError('CV requires eligible development genotypes in validated order')
    _, analysis, _, original_development, _ = load_identities()
    original_names = analysis[original_development]
    cv_names = original_names[original_names.isin(genotype_order)]
    positions = genotype_order.get_indexer(cv_names)
    if len(positions) != len(genotype_order) or np.any(positions < 0):
        raise ValueError('CV development identities do not align')
    return positions
