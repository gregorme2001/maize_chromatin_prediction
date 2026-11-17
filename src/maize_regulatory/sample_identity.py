"""Permanent sample identities, independent of phenotype values and large GEO files."""

import gzip
import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import make_holdout_indices
from .preparation import normalize_genotype, require_unique

RESOURCE_DIR = Path(__file__).with_name("resources")
GEO_MEMBER = (
    "normalized_chromatin_accessibility_and_covariate_matrices/"
    "Merged_Genotype.INT_ACRs.08.08.2023.fastQTL.txt.bed.gz"
)


def genotype_header_from_archive(stream) -> bytes:
    """Stream the archive until the known member, then read only its gzip header line."""
    with tarfile.open(fileobj=stream, mode="r|gz") as archive:
        for member in archive:
            if member.name != GEO_MEMBER:
                continue
            if not member.isfile():
                raise ValueError("Expected a regular GEO matrix member")
            with archive.extractfile(member) as matrix, gzip.GzipFile(fileobj=matrix) as handle:
                header = handle.readline(1024 * 1024)
            if not header.endswith(b"\n"):
                raise ValueError("Missing or unexpectedly large matrix header")
            return header
    raise ValueError("Published genotype matrix not found in GEO archive")


def select_analysis_genotypes(published_names: pd.Index, phenotype_names: pd.Series):
    """Legacy identity-only matching; never needs DTS values."""
    published = pd.Index([normalize_genotype(n) for n in published_names], name="genotype")
    require_unique(published, "normalized published genotype")
    distinct_names = phenotype_names.dropna().unique()
    require_unique(pd.Index([normalize_genotype(n) for n in distinct_names]),
                   "normalized phenotype name")
    normalized = phenotype_names.map(normalize_genotype)
    counts = normalized.value_counts()
    selected = published[published.isin(counts.index[counts == 1])]
    excluded = published[~published.isin(selected)].tolist()
    return selected, excluded


def freeze_identities(header: bytes, phenotype_names: pd.Series, destination: Path,
                      source: str) -> None:
    columns = header.decode("utf-8").rstrip("\r\n").split("\t")
    published = pd.Index([c.removeprefix("G282set_") for c in columns
                          if c.startswith("G282set_")], name="genotype")
    require_unique(published, "published genotype")
    if len(published) != 172:
        raise ValueError("Expected 172 published genotypes")
    analysis, excluded = select_analysis_genotypes(published, phenotype_names)
    if len(analysis) != 169:
        raise ValueError("Legacy identity matching must retain exactly 169 genotypes")
    _, holdout_idx = make_holdout_indices(len(analysis))
    texts = {
        "published_genotypes.txt": "\n".join(published) + "\n",
        "analysis_genotypes.txt": "\n".join(analysis) + "\n",
        "holdout_genotypes.txt": "\n".join(analysis[holdout_idx]) + "\n",
    }
    manifest = {
        "source": source, "archive_member": GEO_MEMBER,
        "header_sha256": hashlib.sha256(header).hexdigest(),
        "matching_rule": "Legacy normalized names, exactly one phenotype row, published order",
        "excluded_published_genotypes": excluded,
        "test_size": 30, "random_state": 42,
        "phenotype_values_read": False,
        "identity_sha256": {name: hashlib.sha256(content.encode()).hexdigest()
                            for name, content in texts.items()},
    }
    # Overwrite of ordinary data products must never redefine the frozen split.
    for name, content in texts.items():
        path = destination / name
        if path.exists() and path.read_text() != content:
            raise ValueError(f"Refusing to change permanent sample identities: {path}")
    destination.mkdir(parents=True, exist_ok=True)
    for name, content in texts.items():
        (destination / name).write_text(content)
    target = destination / "sample_identity.json"
    if not target.exists():
        target.write_text(json.dumps(manifest, indent=2) + "\n")


def load_identities(directory: Path = RESOURCE_DIR):
    manifest = json.loads((directory / "sample_identity.json").read_text())
    indexes = []
    for filename, count in [("published_genotypes.txt", 172), ("analysis_genotypes.txt", 169),
                            ("holdout_genotypes.txt", 30)]:
        content = (directory / filename).read_bytes()
        if hashlib.sha256(content).hexdigest() != manifest["identity_sha256"][filename]:
            raise ValueError(f"Permanent identity checksum mismatch: {filename}")
        index = pd.Index(content.decode().splitlines(), name="genotype")
        require_unique(index, filename)
        if len(index) != count:
            raise ValueError(f"Unexpected sample count in {filename}")
        indexes.append(index)
    published, analysis, holdout = indexes
    if not analysis.isin([normalize_genotype(n) for n in published]).all():
        raise ValueError("Analysis genotypes are not a subset of the original published names")
    development_idx, holdout_idx = make_holdout_indices(len(analysis))
    if not analysis[holdout_idx].equals(holdout):
        raise ValueError("Permanent holdout differs from the original seed-42 split")
    if manifest["test_size"] != 30 or manifest["random_state"] != 42:
        raise ValueError("Permanent split metadata changed")
    return published, analysis, holdout, development_idx, holdout_idx


def validate_analysis_order(genotype_order: pd.Index, directory: Path = RESOURCE_DIR) -> None:
    _, expected, _, _, _ = load_identities(directory)
    if not np.array_equal(genotype_order.to_numpy(), expected.to_numpy()):
        raise ValueError("Genotype identities/order differ from the permanent 169-genotype list")
